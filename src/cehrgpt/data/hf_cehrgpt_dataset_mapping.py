import datetime
from typing import Any, Dict

import numpy as np
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


def convert_date_to_posix_time(index_date: datetime.date) -> float:
    return datetime.datetime.combine(
        index_date, datetime.datetime.min.time()
    ).timestamp()


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
        concept_ids = record["concept_ids"]
        input_ids = self._concept_tokenizer.encode(concept_ids)
        record["input_ids"] = input_ids
        concept_value_masks = record["concept_value_masks"]
        concept_values = record["concept_values"]
        # If any concept has a value associated with it, we normalize the value
        if np.any(np.asarray(concept_value_masks) > 0):
            updated_input_ids = []
            units = record["units"]
            num_of_labs = 0
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
                updated_input_ids.append(token_id)
                if token_id in self._lab_token_ids:
                    concept_value_bin = self._concept_tokenizer.normalize(
                        concept_id, unit, concept_value
                    )
                    concept_value_token_id = (
                        self._concept_tokenizer._tokenizer.token_to_id(
                            concept_value_bin
                        )
                    )
                    updated_input_ids.append(int(concept_value_token_id))
                    num_of_labs += 1
            assert len(updated_input_ids) == num_of_labs + len(concept_ids)
            record["input_ids"] = updated_input_ids
        return record


class HFFineTuningMapping(DatasetMapping):
    """Consider removing this transformation in the future."""

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "age_at_index": (
                record["age"] if "age" in record else record["age_at_index"]
            ),
            "classifier_label": record["label"],
            "index_date": (
                convert_date_to_posix_time(record["index_date"])
                if "index_date" in record
                else None
            ),
        }

    def remove_columns(self):
        return ["label"]
