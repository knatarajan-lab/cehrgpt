import datetime
import os
import shutil
import tempfile
import unittest

import pandas as pd
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict

from cehrgpt.runners.data_utils import extract_cohort_sequences
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

DAY = 24 * 3600
START = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
TOKENIZED_PERSON_IDS = {"train": [1, 2, 3], "validation": [4]}


def tokenized_record(person_id: int):
    concept_ids = ["year:2020", "age:50", "Gender/F", "Visit/OP", "1234", "5678"]
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "input_ids": list(range(len(concept_ids))),
        "epoch_times": [START + i * DAY for i in range(len(concept_ids))],
    }


class TestExtractCohortSequencesMissingPersons(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tokenized_path = os.path.join(self.tmp, "tokenized")
        DatasetDict(
            {
                split: Dataset.from_list([tokenized_record(p) for p in person_ids])
                for split, person_ids in TOKENIZED_PERSON_IDS.items()
            }
        ).save_to_disk(self.tokenized_path)

    def _cohort(self, person_ids):
        cohort_folder = os.path.join(self.tmp, "cohort")
        os.makedirs(cohort_folder, exist_ok=True)
        pd.DataFrame(
            {
                "person_id": person_ids,
                "index_date": [datetime.datetime(2020, 1, 4)] * len(person_ids),
                "label": [1] * len(person_ids),
            }
        ).to_parquet(os.path.join(cohort_folder, "cohort.parquet"))
        return cohort_folder

    def _extract(self, cohort_folder, allow_missing):
        data_args = DataTrainingArguments(
            data_folder=cohort_folder,
            dataset_prepared_path=self.tmp,
            cohort_folder=cohort_folder,
            preprocessing_num_workers=1,
        )
        cehrgpt_args = CehrGPTArguments(
            tokenized_full_dataset_path=self.tokenized_path,
            allow_missing_tokenized_persons=allow_missing,
        )
        return extract_cohort_sequences(data_args, cehrgpt_args)

    def test_missing_persons_raise_by_default(self):
        cohort_folder = self._cohort([1, 2, 3, 4, 98, 99])
        with self.assertRaises(RuntimeError) as ctx:
            self._extract(cohort_folder, allow_missing=False)
        self.assertIn("2 missing", str(ctx.exception))

    def test_missing_persons_are_skipped_when_allowed(self):
        cohort_folder = self._cohort([1, 2, 3, 4, 98, 99])
        with self.assertLogs("transformers", level="WARNING") as logs:
            result = self._extract(cohort_folder, allow_missing=True)
        self.assertTrue(any("2 of 6" in line for line in logs.output))
        extracted = [p for split in result.values() for p in split["person_id"]]
        self.assertEqual(sorted(extracted), [1, 2, 3, 4])

    def test_complete_cohort_is_unaffected_by_the_flag(self):
        cohort_folder = self._cohort([1, 2, 3, 4])
        for allow_missing in (False, True):
            result = self._extract(cohort_folder, allow_missing=allow_missing)
            extracted = [p for split in result.values() for p in split["person_id"]]
            self.assertEqual(sorted(extracted), [1, 2, 3, 4])

    def test_no_overlap_at_all_still_raises_when_allowed(self):
        cohort_folder = self._cohort([98, 99])
        with self.assertRaises(RuntimeError) as ctx:
            self._extract(cohort_folder, allow_missing=True)
        self.assertIn("None of the cohort persons", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
