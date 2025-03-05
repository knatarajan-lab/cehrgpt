import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset

from cehrgpt.runners.hf_cehrgpt_finetune_runner import create_dataset_splits


def mock_load_parquet_as_dataset(folder_path, streaming=False):
    data = {
        "index_date": np.sort(
            np.random.choice(
                np.arange("2021-01-01", "2022-01-01", dtype="datetime64[D]"), 100
            )
        ),
        "person_id": np.asarray(range(0, 100)),
    }
    df = pd.DataFrame(data)
    if streaming:
        return Dataset.from_pandas(df).to_iterable_dataset()
    else:
        return Dataset.from_pandas(df)


class TestCreateDatasetSplits(unittest.TestCase):
    @patch(
        "cehrgpt.runners.hf_cehrgpt_finetune_runner.load_parquet_as_dataset",
        side_effect=mock_load_parquet_as_dataset,
    )
    def test_streaming_mode(self, mock_load_dataset):
        data_args = DataTrainingArguments(
            data_folder="data/",
            dataset_prepared_path="",
            streaming=True,
            validation_split_num=20,
            test_eval_ratio=0.5,
        )
        train_set, validation_set, test_set = create_dataset_splits(data_args, seed=42)
        self.assertEqual(len([_ for _ in validation_set]), 10)
        self.assertEqual(len([_ for _ in test_set]), 10)

        validation_person_ids = set([_["person_id"] for _ in validation_set])
        test_person_ids = set([_["person_id"] for _ in test_set])
        # Check if there is any overlap between validation and test person_ids
        self.assertTrue(len(validation_person_ids & test_person_ids) == 0)

    @patch(
        "cehrgpt.runners.hf_cehrgpt_finetune_runner.load_parquet_as_dataset",
        side_effect=mock_load_parquet_as_dataset,
    )
    def test_batch_mode(self, mock_load_dataset):
        data_args = DataTrainingArguments(
            data_folder="data/",
            dataset_prepared_path="",
            validation_split_percentage=0.2,
        )
        train_set, validation_set, test_set = create_dataset_splits(data_args, seed=42)
        total_size = len(train_set) + len(validation_set) + len(test_set)
        self.assertAlmostEqual(len(train_set) / total_size, 0.8, delta=0.05)
        self.assertAlmostEqual(len(validation_set) / total_size, 0.1, delta=0.05)
        self.assertAlmostEqual(len(test_set) / total_size, 0.1, delta=0.05)

        validation_person_ids = set([_["person_id"] for _ in validation_set])
        test_person_ids = set([_["person_id"] for _ in test_set])
        # Check if there is any overlap between validation and test person_ids
        self.assertTrue(len(validation_person_ids & test_person_ids) == 0)

    @patch(
        "cehrgpt.runners.hf_cehrgpt_finetune_runner.load_parquet_as_dataset",
        side_effect=mock_load_parquet_as_dataset,
    )
    def test_with_test_data_folder(self, mock_load_dataset):
        data_args = DataTrainingArguments(
            data_folder="data/",
            test_data_folder="test_data/",
            dataset_prepared_path="",
            validation_split_percentage=0.2,
        )
        train_set, validation_set, test_set = create_dataset_splits(data_args, seed=42)
        mock_load_dataset.assert_called_with("test_data/")
        self.assertTrue(len(test_set) > 0)


if __name__ == "__main__":
    unittest.main()
