from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel


class ModelTimeEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super(ModelTimeEmbedding, self).__init__()
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
                max_position_embeddings=2,
            ),
            add_pooling_layer=False,
        )
        self.linear = torch.nn.Linear(32, 2)

    def forward(
        self,
        input_ids: torch.LongTensor,
        time_stamps: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bz = input_ids.shape[0]
        x = self.embedding(input_ids)
        t = self.embedding(time_stamps)
        x = x + t
        bert_output = self.bert.forward(inputs_embeds=x, return_dict=True)
        output = bert_output.last_hidden_state.reshape((bz, 32))
        y = self.linear(output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(y, labels)
        return loss, y


def generate_simulation_data(sample_size: int = 1000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)  # Set the seed for reproducibility

    # Define input values and time stamps
    x_values = [0, 1]
    time_stamp_values = list(range(0, 21))

    # Generate random choices for features and time stamps
    x1 = np.random.choice(x_values, size=sample_size)
    x2 = np.random.choice(x_values, size=sample_size)
    t1 = np.random.choice(time_stamp_values, size=sample_size)
    t2 = t1 + np.random.choice(time_stamp_values, size=sample_size)

    # Define conditions based on time differences
    time_diff = t2 - t1
    # Complex condition involving modulo operation
    is_custom_func_1 = (x1 == 1) & (time_diff % 4 == 0)
    is_custom_func_2 = (x1 == 0) & (time_diff % 3 == 0)
    is_xor = time_diff <= 7
    is_and = (time_diff > 7) & (time_diff <= 14)
    is_or = (time_diff > 14) & (time_diff <= 21)

    # Logical operations based on x1 and x2
    xor = (x2 != x1).astype(int)
    logical_and = (x2 & x1).astype(int)
    logical_or = (x2 | x1).astype(int)
    # Additional complexity: introduce a new rule based on a more complex condition
    custom_func_1_result = (x2 == 0).astype(int)  # For example, use a different rule
    custom_func_2_result = (x2 == 1).astype(int)  # For example, use a different rule

    # Determine output based on multiple conditions
    y = np.where(
        is_custom_func_1,
        custom_func_1_result,
        np.where(
            is_custom_func_2,
            custom_func_2_result,
            np.where(
                is_xor,
                xor,
                np.where(is_and, logical_and, np.where(is_or, logical_or, 0)),
            ),
        ),
    )

    # Return the data as a single numpy array with features and output
    return np.column_stack((x1, x2, t1, t2, y))


def create_time_embedding_tokenizer(simulated_data):
    vocab = []
    for row in simulated_data:
        x1, x2, t1, t2, y = row
        x1 = f"c-{x1}"
        x2 = f"c-{x2}"
        t1 = f"t-{t1}"
        t2 = f"t-{t2}"
        if x1 not in vocab:
            vocab.append(x1)
        if x2 not in vocab:
            vocab.append(x2)
        if t1 not in vocab:
            vocab.append(t1)
        if t2 not in vocab:
            vocab.append(t2)
    return {c: i + 1 for i, c in enumerate(vocab)}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_step(
    simulated_data,
    time_embedding_tokenizer,
    time_embedding_model,
    time_embedding_optimizer,
):
    time_embedding_optimizer.zero_grad()
    time_embedding_model.eval()
    eval_input_ids = []
    eval_time_stamps = []
    eval_y = []
    for row in simulated_data:
        x1, x2, t1, t2, y = row
        x1 = f"c-{x1}"
        x2 = f"c-{x2}"
        t1 = f"t-{t1}"
        t2 = f"t-{t2}"
        eval_input_ids.append(
            [time_embedding_tokenizer[x1], time_embedding_tokenizer[x2]]
        )
        eval_time_stamps.append(
            [time_embedding_tokenizer[t1], time_embedding_tokenizer[t2]]
        )
        eval_y.append(y)
    eval_input_ids = torch.tensor(eval_input_ids, dtype=torch.long).to(device)
    eval_time_stamps = torch.tensor(eval_time_stamps, dtype=torch.long).to(device)
    eval_y = np.asarray(eval_y)
    with torch.no_grad():
        # Compute loss and forward pass
        _, y_pred = time_embedding_model(eval_input_ids, eval_time_stamps)
        y_probs = torch.nn.functional.softmax(y_pred, dim=1)
        y_probs = y_probs.detach().cpu().numpy()
        # print(np.concatenate((y_probs, batched_y[:, None]), axis=1))
        roc_auc = roc_auc_score(eval_y, y_probs[:, 1])
        accuracy = accuracy_score(eval_y, y_probs[:, 1] > y_probs[:, 0])
        print(f"ROC AUC: {roc_auc}")
        print(f"Accuracy: {accuracy}")
    return roc_auc, accuracy


def train_step(
    simulated_data,
    time_embedding_tokenizer,
    time_embedding_model,
    time_embedding_optimizer,
):
    batched_input_ids = []
    batched_time_stamps = []
    batched_y = []
    indices = np.random.choice(simulated_data.shape[0], size=8, replace=False)
    for row in simulated_data[indices, :]:
        x1, x2, t1, t2, y = row
        x1 = f"c-{x1}"
        x2 = f"c-{x2}"
        t1 = f"t-{t1}"
        t2 = f"t-{t2}"
        batched_input_ids.append(
            [time_embedding_tokenizer[x1], time_embedding_tokenizer[x2]]
        )
        batched_time_stamps.append(
            [time_embedding_tokenizer[t1], time_embedding_tokenizer[t2]]
        )
        batched_y.append(y)
    batched_input_ids = torch.tensor(batched_input_ids, dtype=torch.long).to(device)
    batched_time_stamps = torch.tensor(batched_time_stamps, dtype=torch.long).to(device)
    batched_y = torch.tensor(batched_y, dtype=torch.long).to(device)
    # Zero the gradients
    time_embedding_optimizer.zero_grad()
    # Compute loss and forward pass
    loss, _ = time_embedding_model(batched_input_ids, batched_time_stamps, batched_y)
    # Backward pass (compute gradients)
    loss.backward()
    # Update model parameters
    time_embedding_optimizer.step()
    return loss


def main(args):
    simulated_data = generate_simulation_data(args.n_samples)
    time_embedding_tokenizer = create_time_embedding_tokenizer(simulated_data)
    time_embedding_model = ModelTimeEmbedding(len(time_embedding_tokenizer) + 1).to(
        device
    )
    time_embedding_optimizer = optim.Adam(time_embedding_model.parameters(), lr=0.001)
    steps = []
    roc_aucs = []
    accuracies = []
    for step in range(args.n_steps):
        loss = train_step(
            simulated_data,
            time_embedding_tokenizer,
            time_embedding_model,
            time_embedding_optimizer,
        )
        print(f"Step {step}: Loss = {loss.item()}")
        # Evaluation
        if (
            args.n_steps % args.eval_frequency == 0
            and args.n_steps > args.eval_frequency
        ):
            # Zero the gradients
            roc_auc, accuracy = eval_step(
                simulated_data,
                time_embedding_tokenizer,
                time_embedding_model,
                time_embedding_optimizer,
            )
            steps.append(step)
            roc_aucs.append(roc_auc)
            accuracies.append(accuracy)
    return {"steps": steps, "roc_auc": roc_aucs, "accuracy": accuracies}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser("Model with time embedding simulation")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_frequency", type=int, default=100)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics = main(args)
    with open(output_dir / "time_embedding_metrics.json", "w") as f:
        json.dump(metrics, f)
