import datetime as dt
import math
import unittest

from cehrbert.data_generators.hf_data_generator.hf_dataset import apply_cehrbert_dataset_mapping
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, Features, Sequence, Value

from cehrgpt.data.hf_cehrgpt_dataset_mapping import MedToCehrGPTDatasetMapping, _event_values

# The schema that the MEDS conversion writes (meds_dataset._create_cehrgpt_data_from_meds)
FEATURES = Features(
    {
        "patient_id": Value(dtype="int32"),
        "birth_datetime": Value(dtype="timestamp[us]"),
        "gender": Value(dtype="string"),
        "race": Value(dtype="string"),
        "ethnicity": Value(dtype="string"),
        "index_date": Value(dtype="timestamp[us]"),
        "age_at_index": Value(dtype="int16"),
        "label": Value(dtype="float32"),
        "visits": Sequence(
            feature={
                "visit_type": Value(dtype="string"),
                "visit_start_datetime": Value(dtype="timestamp[us]"),
                "visit_end_datetime": Value(dtype="timestamp[us]"),
                "discharge_facility": Value(dtype="string"),
                "is_inpatient_or_ed": Value(dtype="bool"),
                "events": Sequence(
                    feature={
                        "time": Value(dtype="timestamp[us]"),
                        "code": Value(dtype="string"),
                        "text_value": Value(dtype="string"),
                        "numeric_value": Value(dtype="float32"),
                        "unit": Value(dtype="string"),
                        "properties": {"visit_id": Value(dtype="int32"), "table": Value(dtype="string")},
                    }
                ),
            }
        ),
    }
)


def _event(code, hour, numeric_value=None, text_value=None, unit=None):
    return {
        "time": dt.datetime(2020, 1, 10, hour),
        "code": code,
        "text_value": text_value,
        "numeric_value": numeric_value,
        "unit": unit,
        "properties": {"visit_id": 1, "table": None},
    }


def _patient():
    return {
        "patient_id": 1,
        "birth_datetime": dt.datetime(1960, 1, 1),
        "gender": "GENDER//F",
        "race": "Race/0",
        "ethnicity": None,
        "index_date": None,
        "age_at_index": None,
        "label": None,
        "visits": [
            {
                "visit_type": "Visit/OP",
                "visit_start_datetime": dt.datetime(2020, 1, 10, 1),
                "visit_end_datetime": dt.datetime(2020, 1, 10, 23),
                "discharge_facility": None,
                "is_inpatient_or_ed": False,
                "events": [
                    _event("DIAGNOSIS//ICD//10//I10", 2),
                    _event("MEDICATION//Aspirin", 3),
                    _event("LAB//50912//mg/dL", 4, numeric_value=1.2, unit="mg/dL"),
                    _event("LAB//51006//UNK", 5, text_value="NEG"),
                ],
            }
        ],
    }


class EventValuesTest(unittest.TestCase):
    def test_nan_and_empty_text_are_no_value(self):
        self.assertEqual(_event_values({"numeric_value": math.nan, "text_value": None}), (None, None))
        self.assertEqual(_event_values({"numeric_value": None, "text_value": ""}), (None, None))
        self.assertEqual(_event_values({"numeric_value": math.nan, "text_value": math.nan}), (None, None))
        self.assertEqual(_event_values({}), (None, None))

    def test_real_values_are_kept(self):
        self.assertEqual(_event_values({"numeric_value": 0.0, "text_value": None}), (0.0, None))
        self.assertEqual(_event_values({"numeric_value": 1.5, "text_value": "NEG"}), (1.5, "NEG"))


class BatchedMedsMappingValueMaskTest(unittest.TestCase):
    """
    The batched mapping reads the events through pandas, where a missing numeric_value becomes NaN.

    Before the fix, every diagnosis and medication had a value mask of 1 and a "0" value, so they were
    counted as categorical labs and none of them could be a MOTOR task.
    """

    def test_events_without_a_value_have_no_value_mask(self):
        data_args = DataTrainingArguments(
            data_folder="unused", dataset_prepared_path="unused", is_data_in_meds=True, preprocessing_num_workers=1
        )
        dataset = Dataset.from_list([_patient()], features=FEATURES)
        mapped = apply_cehrbert_dataset_mapping(
            dataset,
            MedToCehrGPTDatasetMapping(data_args=data_args, include_inpatient_hour_token=True),
            num_proc=1,
            batch_size=8,
            streaming=False,
        )
        record = mapped[0]
        by_code = {
            code: (mask, number, value)
            for code, mask, number, value in zip(
                record["concept_ids"],
                record["concept_value_masks"],
                record["number_as_values"],
                record["concept_as_values"],
            )
        }
        self.assertEqual(by_code["DIAGNOSIS//ICD//10//I10"][0], 0)
        self.assertEqual(by_code["MEDICATION//Aspirin"][0], 0)
        self.assertEqual(by_code["LAB//50912//mg/dL"][0], 1)
        self.assertAlmostEqual(by_code["LAB//50912//mg/dL"][1], 1.2, places=5)
        self.assertEqual(by_code["LAB//51006//UNK"][0], 1)
        self.assertEqual(by_code["LAB//51006//UNK"][2], "NEG")
        for code, (_, number, _) in by_code.items():
            self.assertFalse(number is not None and math.isnan(number), code)


if __name__ == "__main__":
    unittest.main()
