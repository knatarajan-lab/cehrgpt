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
        return [
            "concept_value_masks",
            "is_numeric_types",
        ]

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # If any concept has a value associated with it, we normalize the value
        record["input_ids"] = self._concept_tokenizer.encode(record["concept_ids"])
        record["value_indicators"] = record["concept_value_masks"]
        if "number_as_values" not in record or "concept_as_values" not in record:
            record["number_as_values"] = [
                float(value) if isinstance(value, float) else None
                for value in record["concept_values"]
            ]
            record["is_numeric_types"] = [
                int(isinstance(value, float)) for value in record["concept_values"]
            ]
            record["concept_as_values"] = [
                value if isinstance(value, str) else None
                for value in record["concept_values"]
            ]
        if np.any(np.asarray(record["concept_value_masks"]) > 0):
            values = []
            for i, (
                concept_id,
                unit,
                concept_value_mask,
                number_as_value,
                concept_as_value,
                is_numeric_type,
            ) in enumerate(
                zip(
                    record["concept_ids"],
                    record["units"],
                    record["concept_value_masks"],
                    record["number_as_values"],
                    record["concept_as_values"],
                    record["is_numeric_types"],
                )
            ):
                if concept_value_mask == 1:
                    value = UNKNOWN_BIN
                    if is_numeric_type == 1:
                        if concept_id in self._concept_tokenizer.numeric_concept_ids:
                            value = self._concept_tokenizer.normalize(
                                concept_id, unit, number_as_value
                            )
                    elif isinstance(concept_as_value, str):
                        value = concept_as_value
                    values.append(value)
                else:
                    values.append(NONE_BIN)
            assert len(values) == len(record["input_ids"])
            record["values"] = self._concept_tokenizer.encode_value(values)
        else:
            record["values"] = self._concept_tokenizer.encode_value(
                [NONE_BIN for _ in range(len(record["concept_value_masks"]))]
            )
        # Delete these features because they contain null values and pyarrow cannot concatenate multiple records
        del record["number_as_values"]
        del record["concept_as_values"]
        return record


class HFFineTuningMapping(HFCehrGptTokenizationMapping):
    """Consider removing this transformation in the future."""

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = super().transform(record)
        record = {
            "age_at_index": (
                record["age"] if "age" in record else record["age_at_index"]
            ),
            "classifier_label": int(record["label"] > 0),
            "index_date": (
                convert_date_to_posix_time(record["index_date"])
                if "index_date" in record
                else None
            ),
        }
        return record

    def remove_columns(self):
        columns = super().remove_columns()
        columns.append("label")
        return columns
