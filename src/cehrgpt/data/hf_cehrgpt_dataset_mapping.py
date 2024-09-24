import copy
from typing import Any, Dict

import numpy as np
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


class HFCehrGptTokenizationMapping(DatasetMapping):
    def __init__(
        self,
        concept_tokenizer: CehrGptTokenizer,
    ):
        self._concept_tokenizer = concept_tokenizer
        self._lab_token_ids = self._concept_tokenizer.lab_token_ids

    def remove_columns(self):
        return ["concept_value_masks", "concept_values"]

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = self._concept_tokenizer.encode(record["concept_ids"])
        record["input_ids"] = input_ids
        concept_value_masks = record["concept_value_masks"]
        concept_values = record["concept_values"]

        # If any concept has a value associated with it, we normalize the value
        if np.any(concept_value_masks > 0):
            normalized_concept_values = copy.deepcopy(concept_values)
            for i, (
                concept_id,
                token_id,
                concept_value_mask,
                concept_value,
            ) in enumerate(
                zip(
                    record["concept_ids"],
                    input_ids,
                    concept_value_masks,
                    concept_values,
                )
            ):
                if token_id in self._lab_token_ids:
                    normalized_concept_value = self._concept_tokenizer.normalize(
                        concept_id, concept_value
                    )
                    normalized_concept_values[i] = normalized_concept_value
            record["concept_values"] = normalized_concept_values

        # Overwrite the column names
        record["value_indicators"] = record["concept_value_masks"]
        record["values"] = record["concept_values"]
        return record
