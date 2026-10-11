import glob
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

os.environ["WANDB_MODE"] = "disabled"
os.environ["USE_TF"] = "0"

from datasets import load_from_disk

from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main


class PretrainTestDataFolderIntegrationTest(unittest.TestCase):
    NUM_TEST_ROWS = 20

    @classmethod
    def setUpClass(cls):
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        cls.data_folder = os.path.join(root_folder, "sample_data", "pretrain")
        cls.pretrained_embedding_folder = os.path.join(
            root_folder, "sample_data", "pretrained_embeddings"
        )
        cls.vocab_dir = os.path.join(root_folder, "sample_data", "omop_vocab")
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_folder = os.path.join(cls.temp_dir, "test_data")
        Path(cls.test_data_folder).mkdir()
        pd.read_parquet(
            os.path.join(cls.data_folder, "patient_sequence.parquet")
        ).head(cls.NUM_TEST_ROWS).to_parquet(
            os.path.join(cls.test_data_folder, "patient_sequence.parquet")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def _run(self, name, test_data_folder=None):
        model_folder = os.path.join(self.temp_dir, f"model_{name}")
        prepared = os.path.join(self.temp_dir, f"prepared_{name}")
        Path(model_folder).mkdir()
        Path(prepared).mkdir()
        for file_name in [
            PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
            PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
        ]:
            shutil.copy(
                os.path.join(self.pretrained_embedding_folder, file_name),
                os.path.join(model_folder, file_name),
            )
        sys.argv = [
            "hf_cehrgpt_pretraining_runner.py",
            "--model_name_or_path", model_folder,
            "--tokenizer_name_or_path", model_folder,
            "--output_dir", model_folder,
            "--data_folder", self.data_folder,
            "--dataset_prepared_path", prepared,
            "--pretrained_embedding_path", model_folder,
            "--vocab_dir", self.vocab_dir,
            "--max_steps", "2",
            "--save_steps", "2",
            "--save_strategy", "steps",
            "--hidden_size", "96",
            "--max_position_embeddings", "128",
            "--validation_split_percentage", "0.1",
            "--use_early_stopping", "false",
            "--report_to", "none",
            "--exclude_position_ids", "true",
            "--apply_rotary", "true",
        ]
        if test_data_folder:
            sys.argv += ["--test_data_folder", test_data_folder]
        train_main()
        return prepared

    def _prepared_datasets(self, prepared):
        return [
            p
            for p in glob.glob(os.path.join(prepared, "*"))
            if os.path.exists(os.path.join(p, "dataset_dict.json"))
        ]

    def test_test_data_folder_is_tokenized_into_a_test_split(self):
        prepared = self._run("with_test", self.test_data_folder)
        paths = self._prepared_datasets(prepared)
        self.assertEqual(len(paths), 1)
        dataset = load_from_disk(paths[0])
        self.assertEqual(set(dataset.keys()), {"train", "validation", "test"})
        self.assertGreater(len(dataset["test"]), 0)
        self.assertLessEqual(len(dataset["test"]), self.NUM_TEST_ROWS)
        # The test split is tokenized the same way as train/validation.
        self.assertEqual(
            set(dataset["test"].column_names), set(dataset["train"].column_names)
        )

    def test_without_test_data_folder_has_no_test_split(self):
        prepared = self._run("without_test")
        paths = self._prepared_datasets(prepared)
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            set(load_from_disk(paths[0]).keys()), {"train", "validation"}
        )

    def test_test_data_folder_changes_the_prepared_dataset_path(self):
        """Otherwise a cache prepared without a test split would be silently reused."""
        with_test = self._prepared_datasets(self._run("path_a", self.test_data_folder))
        without_test = self._prepared_datasets(self._run("path_b"))
        self.assertNotEqual(
            os.path.basename(with_test[0]), os.path.basename(without_test[0])
        )
        self.assertIn("_test", os.path.basename(with_test[0]))


if __name__ == "__main__":
    unittest.main()
