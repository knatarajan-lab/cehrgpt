import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


class InstructCehrGptPPODataCollator:
    def __init__(
        self,
        encoder_tokenizer: PreTrainedTokenizer,
        cehrgpt_tokenizer: CehrGptTokenizer,
        max_length: int,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.cehrgpt_tokenizer = cehrgpt_tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        batch = {}
        # Pad sequences to the max length in the batch
        batch["encoder_input_ids"] = pad_sequence(
            [example["encoder_input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.encoder_tokenizer.pad_token_id,
        ).to(torch.int64)

        batch["encoder_attention_mask"] = pad_sequence(
            [example["encoder_attention_mask"] for example in examples],
            batch_first=True,
            padding_value=0.0,
        )

        batch["input_ids"] = pad_sequence(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.cehrgpt_tokenizer.pad_token_id,
        ).to(torch.int64)

        batch["attention_mask"] = pad_sequence(
            [example["attention_mask"] for example in examples],
            batch_first=True,
            padding_value=0.0,
        )

        assert (
            batch["input_ids"].shape[1] <= self.max_length
        ), f"Invalid input_ids length: {batch['input_ids'].shape[1]}"

        if "value_indicators" in examples[0]:
            batch["value_indicators"] = pad_sequence(
                [example["value_indicators"] for example in examples],
                batch_first=True,
                padding_value=False,
            )

        if "values" in examples[0]:
            batch["values"] = pad_sequence(
                [example["values"] for example in examples],
                batch_first=True,
                padding_value=self.cehrgpt_tokenizer.pad_value_token_id,
            )
            assert batch["value_indicators"].shape[1] <= self.max_length
            assert batch["values"].shape[1] <= self.max_length

        return batch
