import os
import shutil
import tempfile
import unittest

import pandas as pd

from cehrgpt.data.parquet_loading import (
    UNUSED_PRETRAINING_COLUMNS,
    load_parquet_as_dataset,
)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sample_data", "pretrain")


class LoadParquetColumnsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.mkdtemp()
        os.environ["HF_DATASETS_CACHE"] = cls.cache
        import datasets

        datasets.config.HF_DATASETS_CACHE = cls.cache
        cls.frame = pd.read_parquet(os.path.join(SAMPLE, "patient_sequence.parquet"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cache, ignore_errors=True)

    def test_unused_columns_are_not_loaded_and_the_rest_is_unchanged(self):
        full = load_parquet_as_dataset(SAMPLE)
        slim = load_parquet_as_dataset(SAMPLE, exclude_columns=UNUSED_PRETRAINING_COLUMNS)
        dropped = set(full.column_names) - set(slim.column_names)
        self.assertTrue(dropped)
        self.assertTrue(dropped <= set(UNUSED_PRETRAINING_COLUMNS) | {"__index_level_0__"})
        for name in slim.column_names:
            self.assertEqual(slim[name], full[name], name)

    def test_the_columns_used_by_tokenization_are_kept(self):
        slim = load_parquet_as_dataset(SAMPLE, exclude_columns=UNUSED_PRETRAINING_COLUMNS)
        for name in ("person_id", "concept_ids", "concept_values", "concept_value_masks",
                     "units", "epoch_times", "ages", "num_of_concepts", "num_of_visits"):
            self.assertIn(name, slim.column_names)

    def test_columns_missing_from_the_files_are_ignored(self):
        slim = load_parquet_as_dataset(SAMPLE, exclude_columns=["not_a_column"])
        self.assertEqual(slim.num_rows, len(self.frame))

    def test_a_folder_without_parquet_files_is_an_error(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                load_parquet_as_dataset(empty)


if __name__ == "__main__":
    unittest.main()
