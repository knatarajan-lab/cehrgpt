import copy
from typing import Any, Dict

import numpy as np
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


class HFCehrGptDPOTokenizationMapping(DatasetMapping):
    def __init__(
        self,
        concept_tokenizer: CehrGptTokenizer,
    ):
        self._concept_tokenizer = concept_tokenizer
        self._lab_token_ids = self._concept_tokenizer.lab_token_ids

    def transform_with_prefix(self, record: Dict[str, Any], prefix) -> Dict[str, Any]:
        concept_ids = record[f"{prefix}_concept_ids"]
        input_ids = self._concept_tokenizer.encode(concept_ids)
        record[f"{prefix}_input_ids"] = input_ids

        if f"{prefix}_concept_value_masks" in record:
            concept_value_masks = record[f"{prefix}_concept_value_masks"]
            concept_values = record[f"{prefix}_concept_values"]
            # If any concept has a value associated with it, we normalize the value
            if np.any(np.asarray(concept_value_masks) > 0):
                units = record[f"{prefix}_units"]
                normalized_concept_values = copy.deepcopy(concept_values)
                for i, (
                    concept_id,
                    unit,
                    token_id,
                    concept_value_mask,
                    concept_value,
                ) in enumerate(
                    zip(
                        concept_ids,
                        units,
                        input_ids,
                        concept_value_masks,
                        concept_values,
                    )
                ):
                    if token_id in self._lab_token_ids:
                        normalized_concept_value = self._concept_tokenizer.normalize(
                            concept_id, unit, concept_value
                        )
                        normalized_concept_values[i] = normalized_concept_value
                record[f"{prefix}_concept_values"] = normalized_concept_values
                # Overwrite the column names
                record[f"{prefix}_value_indicators"] = record[
                    f"{prefix}_concept_value_masks"
                ]
                record[f"{prefix}_values"] = record[f"{prefix}_concept_values"]
        return record

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = self.transform_with_prefix(record, prefix="chosen")
        record.update(self.transform_with_prefix(record, prefix="rejected"))
        return record
