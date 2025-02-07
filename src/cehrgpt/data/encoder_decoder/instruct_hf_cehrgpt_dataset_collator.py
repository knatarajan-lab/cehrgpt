import copy
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.gpt_utils import is_visit_end, random_slice_gpt_sequence
from cehrgpt.llm.clinical_statement_generator import ClinicalStatementGenerator


class InstructCehrGptDataCollator(CehrGptDataCollator):
    def __init__(
        self,
        encoder_tokenizer: PreTrainedTokenizer,
        clinical_statement_generator: ClinicalStatementGenerator,
        concept_name_mapping: Dict[str, str],
        concept_domain_mapping: Dict[str, str],
        *args,
        **kwargs,
    ):
        super(InstructCehrGptDataCollator, self).__init__(*args, **kwargs)
        self.encoder_tokenizer = encoder_tokenizer
        self.clinical_statement_generator = clinical_statement_generator
        self.concept_name_mapping = concept_name_mapping
        self.concept_domain_mapping = concept_domain_mapping

    def encoder_input_hook(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_encoder_input_ids = [
            self._try_reverse_tensor(
                self._convert_to_tensor(example["encoder_input_ids"])
            )
            for example in examples
        ]
        batch_encoder_attention_mask = [
            self._try_reverse_tensor(
                self._convert_to_tensor(example["encoder_attention_mask"])
            )
            for example in examples
        ]
        # Pad sequences to the max length in the batch
        batch["encoder_input_ids"] = self._try_reverse_tensor(
            pad_sequence(
                batch_encoder_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).to(torch.int64)
        )
        batch["encoder_attention_mask"] = self._try_reverse_tensor(
            pad_sequence(
                batch_encoder_attention_mask, batch_first=True, padding_value=0.0
            )
        )
        return batch

    def get_data_collector_hook(
        self,
    ) -> Optional[List[Callable[[List[Dict[str, Any]]], Dict[str, Any]]]]:
        return super().get_data_collector_hooks() + [self.encoder_input_hook]

    def generate_start_end_index(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Adding the start and end indices to extract a portion of the patient sequence."""
        # concept_ids will be used to for time to event predictions and identifying the visit starts
        input_ids = record["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().tolist()
        concept_ids = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        seq_length = len(record["input_ids"])
        new_max_length = (
            self.max_length - 2
        )  # Subtract one for the [START] and [END] token

        selected_concept_ids = copy.deepcopy(concept_ids)
        # Return the record directly if the actual sequence length is less than the max sequence
        if seq_length <= new_max_length:
            record["input_ids"] = torch.concat(
                [
                    self._convert_to_tensor([self.tokenizer.start_token_id]),
                    self._convert_to_tensor(record["input_ids"]),
                    self._convert_to_tensor([self.tokenizer.end_token_id]),
                ]
            )
            if self.include_values:
                record["value_indicators"] = torch.concat(
                    [
                        self._convert_to_tensor([False]),
                        self._convert_to_tensor(record["value_indicators"]),
                        self._convert_to_tensor([False]),
                    ]
                ).to(torch.bool)
                record["values"] = torch.concat(
                    [
                        self._convert_to_tensor([self.tokenizer.pad_value_token_id]),
                        self._convert_to_tensor(record["values"]),
                        self._convert_to_tensor([self.tokenizer.pad_value_token_id]),
                    ]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = torch.concat(
                    [
                        self._convert_to_tensor([-100.0]),
                        self._convert_to_tensor(
                            self._convert_time_to_event(concept_ids)
                        ),
                        self._convert_to_tensor([-100.0]),
                    ]
                )
        elif random.random() < 0.5:
            # There is a 50% chance we randomly slice out a portion of the patient history and update the demographic
            # prompt depending on the new starting point
            start_index, end_index, demographic_tokens = random_slice_gpt_sequence(
                concept_ids, new_max_length
            )
            # Update the selected concept ids for making the clinical statement
            selected_concept_ids = (
                demographic_tokens + concept_ids[start_index : end_index + 1]
            )
            record["input_ids"] = torch.concat(
                [
                    self._convert_to_tensor([self.tokenizer.start_token_id]),
                    self._convert_to_tensor(self.tokenizer.encode(demographic_tokens)),
                    self._convert_to_tensor(
                        record["input_ids"][start_index : end_index + 1]
                    ),
                ]
            )

            if self.include_values:
                record["value_indicators"] = torch.concat(
                    [
                        self._convert_to_tensor([False]),
                        self._convert_to_tensor(
                            np.zeros_like(demographic_tokens).astype(bool)
                        ),
                        record["value_indicators"][start_index : end_index + 1],
                    ]
                )
                record["values"] = torch.concat(
                    [
                        self._convert_to_tensor([self.tokenizer.pad_value_token_id]),
                        self._convert_to_tensor(
                            np.ones_like(demographic_tokens, dtype=np.int32)
                        ).fill_(self.tokenizer.pad_value_token_id),
                        record["values"][start_index : end_index + 1],
                    ]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = torch.concat(
                    [
                        self._convert_to_tensor([-100.0]),
                        self._convert_to_tensor(
                            np.ones_like(demographic_tokens, dtype=np.int32)
                        ).fill_(-100.0),
                        self._convert_to_tensor(
                            self._convert_time_to_event(
                                concept_ids[start_index : end_index + 1]
                            )
                        ),
                    ]
                )
        else:
            # The default employs a right truncation strategy, where the demographic prompt is reserved
            end_index = new_max_length
            for i in reversed(list(range(0, end_index))):
                current_concept = concept_ids[i]
                if current_concept == is_visit_end(current_concept):
                    end_index = i
                    break

            # Update the selected concept ids for making the clinical statement
            selected_concept_ids = concept_ids[0 : end_index + 1]
            record["input_ids"] = torch.concat(
                [
                    self._convert_to_tensor([self.tokenizer.start_token_id]),
                    self._convert_to_tensor(record["input_ids"][0 : end_index + 1]),
                ]
            )

            if self.include_values:
                record["value_indicators"] = torch.concat(
                    [
                        self._convert_to_tensor([False]),
                        self._convert_to_tensor(
                            record["value_indicators"][0 : end_index + 1]
                        ),
                    ]
                ).to(torch.bool)
                record["values"] = torch.concat(
                    [
                        self._convert_to_tensor([self.tokenizer.pad_value_token_id]),
                        self._convert_to_tensor(record["values"][0 : end_index + 1]),
                    ]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = torch.concat(
                    [
                        self._convert_to_tensor([-100.0]),
                        self._convert_to_tensor(
                            self._convert_time_to_event(concept_ids[0 : end_index + 1])
                        ),
                    ]
                )

        encoded_inputs = self.encoder_tokenizer(
            self.clinical_statement_generator.generate_clinical_statement(
                selected_concept_ids,
                self.concept_name_mapping,
                self.concept_domain_mapping,
            )
        )
        record["encoder_input_ids"] = encoded_inputs["input_ids"]
        record["encoder_attention_mask"] = encoded_inputs["attention_mask"]
        return record
