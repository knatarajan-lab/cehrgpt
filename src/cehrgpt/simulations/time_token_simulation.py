from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel

from cehrgpt.simulations.time_embedding_simulation import generate_simulation_data


class ModelTimeToken(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super(ModelTimeToken, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 16)
        self.bert = BertModel(
            BertConfig(
                vocab_size=vocab_size,
                hidden_size=16,
                num_attention_heads=2,
                num_hidden_layers=2,
                intermediate_size=32,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                max_position_embeddings=3,
            ),
            add_pooling_layer=False,
        )
        self.linear = torch.nn.Linear(48, 2)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bz = input_ids.shape[0]
        x = self.embedding(input_ids)
        bert_output = self.bert.forward(inputs_embeds=x, return_dict=True)
        output = bert_output.last_hidden_state.reshape((bz, 48))
        y = self.linear(output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(y, labels)
        return loss, y


def create_time_token_tokenizer(simulated_data):
    vocab = []
    for row in simulated_data:
        x1, x2, t1, t2, y = row
        x1 = f"c-{x1}"
        x2 = f"c-{x2}"
        t = f"t-{t2 - t1}"
        if x1 not in vocab:
            vocab.append(x1)
        if x2 not in vocab:
            vocab.append(x2)
        if t not in vocab:
            vocab.append(t)
    return {c: i + 1 for i, c in enumerate(vocab)}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_step(simulated_data, time_token_tokenizer, time_embedding_model):
    time_embedding_model.eval()
    eval_input_ids = []
    eval_y = []
    for row in simulated_data:
        x1, x2, t1, t2, y = row
        x1 = f"c-{x1}"
        x2 = f"c-{x2}"
        t = f"t-{t2 - t1}"
        eval_input_ids.append(
            [
                time_token_tokenizer[x1],
                time_token_tokenizer[t],
                time_token_tokenizer[x2],
            ]
        )
        eval_y.append(y)
    with torch.no_grad():
        batched_input_ids = torch.tensor(eval_input_ids, dtype=torch.long).to(device)
        batched_y = np.asarray(eval_y)
        # Compute loss and forward pass
        _, y_pred = time_embedding_model(batched_input_ids)
        y_probs = torch.nn.functional.softmax(y_pred, dim=1)
        y_probs = y_probs.detach().cpu().numpy()
        roc_auc = roc_auc_score(batched_y, y_probs[:, 1])
        accuracy = accuracy_score(batched_y, y_probs[:, 1] > y_probs[:, 0])
        print(f"ROC AUC: {roc_auc}")
        print(f"Accuracy: {accuracy}")
    return accuracy, roc_auc


def train_step(
    simulated_data, time_token_tokenizer, time_embedding_model, time_embedding_optimizer
):
    batched_input_ids = []
    batched_y = []
    indices = np.random.choice(simulated_data.shape[0], size=8, replace=False)
    for row in simulated_data[indices, :]:
        x1, x2, t1, t2, y = row
        x1 = f"c-{x1}"
        x2 = f"c-{x2}"
        t = f"t-{t2 - t1}"
        batched_input_ids.append(
            [
                time_token_tokenizer[x1],
                time_token_tokenizer[t],
                time_token_tokenizer[x2],
            ]
        )
        batched_y.append(y)
    batched_input_ids = torch.tensor(batched_input_ids, dtype=torch.long).to(device)
    batched_y = torch.tensor(batched_y, dtype=torch.long).to(device)
    # Zero the gradients
    time_embedding_optimizer.zero_grad()
    # Compute loss and forward pass
    loss, _ = time_embedding_model(batched_input_ids, batched_y)
    # Backward pass (compute gradients)
    loss.backward()
    # Update model parameters
    time_embedding_optimizer.step()
    return loss


def main(args):
    simulated_data = generate_simulation_data(args.n_samples)
    time_token_tokenizer = create_time_token_tokenizer(simulated_data)
    time_embedding_model = ModelTimeToken(len(time_token_tokenizer) + 1).to(device)
    time_embedding_optimizer = optim.Adam(time_embedding_model.parameters(), lr=0.001)
    steps = []
    roc_aucs = []
    accuracies = []
    for step in range(args.n_steps):
        loss = train_step(
            simulated_data,
            time_token_tokenizer,
            time_embedding_model,
            time_embedding_optimizer,
        )
        print(f"Step {step}: Loss = {loss.item()}")
        # Evaluation
        if (
            args.n_steps % args.eval_frequency == 0
            and args.n_steps > args.eval_frequency
        ):
            accuracy, roc_auc = eval_step(
                simulated_data, time_token_tokenizer, time_embedding_model
            )
            steps.append(step)
            roc_aucs.append(roc_auc)
            accuracies.append(accuracy)
    return {"steps": steps, "roc_auc": roc_aucs, "accuracy": accuracies}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser("Model with time token simulation")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_frequency", type=int, default=100)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics = main(args)
    with open(output_dir / "time_token_metrics.json", "w") as f:
        json.dump(metrics, f)
