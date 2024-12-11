import torch
from torch.nn.utils.rnn import pad_sequence

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator


class CehrGptDPODataCollator(CehrGptDataCollator):

    def create_preference_inputs(self, examples, prefix):
        batch = {}
        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [
            self._try_reverse_tensor(
                self._convert_to_tensor(example[f"{prefix}_input_ids"])
            )
            for example in examples
        ]
        batch_attention_mask = [
            self._try_reverse_tensor(
                torch.ones_like(
                    self._convert_to_tensor(example[f"{prefix}_input_ids"]),
                    dtype=torch.float,
                )
            )
            for example in examples
        ]
        # Pad sequences to the max length in the batch
        batch[f"{prefix}_input_ids"] = self._try_reverse_tensor(
            pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).to(torch.int64)
        )
        batch[f"{prefix}_attention_mask"] = self._try_reverse_tensor(
            pad_sequence(batch_attention_mask, batch_first=True, padding_value=0.0)
        )
        assert batch[f"{prefix}_input_ids"].shape[1] <= self.max_length
        assert batch[f"{prefix}_attention_mask"].shape[1] <= self.max_length

        if self.include_values:
            batch_value_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example[f"{prefix}_value_indicators"])
                )
                for example in examples
            ]
            batch_values = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example[f"{prefix}__values"])
                )
                for example in examples
            ]

            batch[f"{prefix}_value_indicators"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_value_indicators, batch_first=True, padding_value=False
                )
            )
            batch[f"{prefix}_values"] = self._try_reverse_tensor(
                pad_sequence(batch_values, batch_first=True, padding_value=-1.0)
            )
            assert batch[f"{prefix}_value_indicators"].shape[1] <= self.max_length
            assert batch[f"{prefix}_values"].shape[1] <= self.max_length
        return batch

    def __call__(self, examples):
        batch_chosen = self.create_preference_inputs(examples, "chosen")
        batch_rejected = self.create_preference_inputs(examples, "rejected")
        batch_chosen.update(batch_rejected)
        return batch_chosen
