from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase

PROMPT_TEMPLATE = "Question: {question} Answer:"


class PatientQACollator:
    """
    Turns {"patient_features", "question", "answer"} examples into model inputs.

    The sequence is [BOS] question-prompt answer [EOS]; only the answer and the EOS are
    labelled, so the loss is the cost of producing the right answer for this patient.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, question: str) -> List[int]:
        prompt = PROMPT_TEMPLATE.format(question=question)
        return [self.tokenizer.bos_token_id] + self.tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sequences, label_sequences = [], []
        for example in examples:
            prompt_ids = self.encode(example["question"])
            answer_ids = self.tokenizer(
                " " + example["answer"], add_special_tokens=False
            )["input_ids"] + [self.tokenizer.eos_token_id]
            # Keep the answer whole and trim the question first when too long
            prompt_ids = prompt_ids[: max(self.max_length - len(answer_ids), 1)]
            ids = (prompt_ids + answer_ids)[: self.max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
            sequences.append(ids)
            label_sequences.append(labels)
        longest = max(len(s) for s in sequences)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(examples), longest), pad_id, dtype=torch.long)
        labels = torch.full((len(examples), longest), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), longest), dtype=torch.long)
        for i, (ids, label_ids) in enumerate(zip(sequences, label_sequences)):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, : len(label_ids)] = torch.tensor(label_ids)
            attention_mask[i, : len(ids)] = 1
        return {
            "patient_features": torch.tensor(
                [example["patient_features"] for example in examples],
                dtype=torch.float32,
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def collate_prompts(
        self, questions: List[str], patient_features: List[List[float]]
    ) -> Dict[str, torch.Tensor]:
        """Prompts for generation, left padded so every answer starts at the same position."""
        sequences = [self.encode(question)[: self.max_length] for question in questions]
        longest = max(len(s) for s in sequences)
        input_ids = torch.full(
            (len(sequences), longest), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
        for i, ids in enumerate(sequences):
            input_ids[i, longest - len(ids) :] = torch.tensor(ids)
            attention_mask[i, longest - len(ids) :] = 1
        return {
            "patient_features": torch.tensor(patient_features, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
