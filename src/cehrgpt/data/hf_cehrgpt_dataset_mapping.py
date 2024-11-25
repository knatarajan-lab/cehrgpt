import datetime
from typing import Any, Dict

import numpy as np
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping

from cehrgpt.models.tokenization_hf_cehrgpt import (
    NONE_BIN,
    UNKNOWN_BIN,
    CehrGptTokenizer,
)


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
        # If any concept has a value associated with it, we normalize the value
        input_ids = self._concept_tokenizer.encode(record["concept_ids"])
        record["input_ids"] = input_ids
        record["value_indicators"] = record["concept_value_masks"]
        if np.any(np.asarray(record["concept_value_masks"]) > 0):
            values = []
            for i, (
                concept_id,
                unit,
                concept_value_mask,
                concept_value,
            ) in enumerate(
                zip(
                    record["concept_ids"],
                    record["units"],
                    record["concept_value_masks"],
                    record["concept_values"],
                )
            ):
                if concept_value_mask == 1:
                    if concept_id in self._concept_tokenizer.numeric_concept_ids:
                        concept_value_bin = self._concept_tokenizer.normalize(
                            concept_id, unit, concept_value
                        )
                        values.append(concept_value_bin)
                    elif isinstance(concept_value, str):
                        values.append(concept_value)
                    else:
                        values.append(UNKNOWN_BIN)
                else:
                    values.append(NONE_BIN)
            record["values"] = self._concept_tokenizer.encode_value(values)
        else:
            record["values"] = [
                self._concept_tokenizer.pad_value_token_id
                for _ in range(len(record["concept_values"]))
            ]
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
