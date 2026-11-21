import datetime
import glob
import os
import shutil
import tempfile
import unittest

import pandas as pd
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict, disable_caching, enable_caching, is_caching_enabled

from cehrgpt.runners.data_utils import (
    COHORT_CHUNKS_FILE,
    extract_cohort_sequences,
    is_cohort_sequence_chunks,
    load_cohort_sequence_chunks,
    save_cohort_sequences_in_chunks,
)
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

DAY = 24 * 3600
START = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
TOKENIZED_PERSON_IDS = {"train": list(range(1, 12)), "validation": [20, 21], "test": [30, 31, 32]}


def tokenized_record(person_id: int):
    concept_ids = ["year:2020", "age:50", "Gender/F", "Race/5"] + [f"c{i}" for i in range(12)]
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "input_ids": list(range(len(concept_ids))),
        "epoch_times": [START + i * DAY for i in range(len(concept_ids))],
    }


class SaveCohortSequencesInChunksTest(unittest.TestCase):
    """
    The cohort is sliced and saved a chunk of persons at a time, so the disk holds one chunk's cache files.

    Sliced at once, the cache files of the whole slice were on disk together with the saved copy of it: for
    the MIMIC readmission task 110 + 130 GB, more than was free.
    """

    def setUp(self):
        was_enabled = is_caching_enabled()
        enable_caching()
        self.addCleanup(enable_caching if was_enabled else disable_caching)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tokenized_path = os.path.join(self.tmp, "tokenized")
        DatasetDict(
            {
                split: Dataset.from_list([tokenized_record(p) for p in person_ids])
                for split, person_ids in TOKENIZED_PERSON_IDS.items()
            }
        ).save_to_disk(self.tokenized_path)
        self.tokenized_files = self._files(self.tokenized_path)
        # every person of the cohort but one has two index dates, so a chunk of persons gives more rows
        persons, dates = [], []
        for person_id in [p for ids in TOKENIZED_PERSON_IDS.values() for p in ids if p != 5]:
            for day in (6, 9) if person_id % 4 else (7,):
                persons.append(person_id)
                dates.append(datetime.datetime(2020, 1, day))
        self.cohort_folder = os.path.join(self.tmp, "cohort")
        os.makedirs(self.cohort_folder)
        pd.DataFrame({"person_id": persons, "index_date": dates, "label": [1] * len(persons)}).to_parquet(
            os.path.join(self.cohort_folder, "cohort.parquet")
        )
        self.data_args = DataTrainingArguments(
            data_folder=self.cohort_folder,
            dataset_prepared_path=self.tmp,
            cohort_folder=self.cohort_folder,
            # map runs in processes that open the cache files, as on a server: a deleted file that the
            # next chunk needs fails there, not in a single process that has it open already
            preprocessing_num_workers=2,
            preprocessing_batch_size=2,
        )
        self.cehrgpt_args = CehrGPTArguments(tokenized_full_dataset_path=self.tokenized_path)

    @staticmethod
    def _files(folder):
        return sorted(os.path.relpath(f, folder) for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True))

    def _save(self, num_persons_per_chunk, name="prepared"):
        output = os.path.join(self.tmp, name)
        chunks = save_cohort_sequences_in_chunks(
            self.data_args, self.cehrgpt_args, None, output, num_persons_per_chunk
        )
        return output, chunks

    def test_the_chunks_give_the_rows_of_one_slice(self):
        expected = extract_cohort_sequences(self.data_args, self.cehrgpt_args, None)
        for num_persons_per_chunk in (1, 3, 4, 100):
            output, chunks = self._save(num_persons_per_chunk, f"prepared_{num_persons_per_chunk}")
            loaded = load_cohort_sequence_chunks(output)
            self.assertEqual(set(loaded.keys()), set(expected.keys()))
            for split in expected:
                self.assertEqual(loaded[split].to_list(), expected[split].to_list(), (num_persons_per_chunk, split))
        # 10 train persons of the cohort in chunks of 3
        _, chunks = self._save(3, "prepared_3_again")
        self.assertEqual(chunks["train"], ["chunk_00000", "chunk_00001", "chunk_00002", "chunk_00003"])

    def test_no_cache_files_are_left_next_to_the_tokenized_dataset(self):
        output, _ = self._save(3)
        self.assertEqual(self._files(self.tokenized_path), self.tokenized_files)
        self.assertEqual(glob.glob(os.path.join(output, "**", "cache-*.arrow"), recursive=True), [])

    def test_the_list_of_chunks_marks_a_finished_save(self):
        output, chunks = self._save(4)
        self.assertTrue(is_cohort_sequence_chunks(output))
        self.assertTrue(os.path.exists(os.path.join(output, COHORT_CHUNKS_FILE)))
        self.assertFalse(is_cohort_sequence_chunks(self.tokenized_path))

    def test_a_split_without_cohort_persons_is_left_out(self):
        pd.DataFrame(
            {"person_id": [1, 2], "index_date": [datetime.datetime(2020, 1, 6)] * 2, "label": [0, 1]}
        ).to_parquet(os.path.join(self.cohort_folder, "cohort.parquet"))
        output, chunks = self._save(1)
        self.assertEqual(set(chunks), {"train"})
        self.assertEqual(len(load_cohort_sequence_chunks(output)["train"]), 2)

    def test_the_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            self._save(0)


if __name__ == "__main__":
    unittest.main()
