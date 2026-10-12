import glob
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

os.environ["WANDB_MODE"] = "disabled"
os.environ["USE_TF"] = "0"

from datasets import DatasetDict, load_from_disk

from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main
from cehrgpt.tools.linear_prob import compute_cehrgpt_features as cf


class LinearProbMultiWorkerIntegrationTest(unittest.TestCase):
    """
    Real model, real worker subprocesses: the features extracted by two sharded workers must
    equal the features extracted by a single worker over the whole dataset.
    """

    @classmethod
    def setUpClass(cls):
        root = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        data_folder = os.path.join(root, "sample_data", "pretrain")
        embeddings = os.path.join(root, "sample_data", "pretrained_embeddings")
        vocab_dir = os.path.join(root, "sample_data", "omop_vocab")
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_folder = os.path.join(cls.temp_dir, "model")
        prepared = os.path.join(cls.temp_dir, "prepared")
        test_folder = os.path.join(cls.temp_dir, "test_data")
        for folder in (cls.model_folder, prepared, test_folder):
            Path(folder).mkdir()
        for name in (
            PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
            PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
        ):
            shutil.copy(os.path.join(embeddings, name), cls.model_folder)
        sequences = pd.read_parquet(os.path.join(data_folder, "patient_sequence.parquet"))
        sequences.head(20).to_parquet(os.path.join(test_folder, "patient_sequence.parquet"))

        # 1. a tiny pretrained model, tokenizer and tokenized dataset
        sys.argv = [
            "pretrain",
            "--model_name_or_path", cls.model_folder,
            "--tokenizer_name_or_path", cls.model_folder,
            "--output_dir", cls.model_folder,
            "--data_folder", data_folder,
            "--test_data_folder", test_folder,
            "--dataset_prepared_path", prepared,
            "--pretrained_embedding_path", cls.model_folder,
            "--vocab_dir", vocab_dir,
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
        train_main()
        tokenized = load_from_disk(
            [p for p in glob.glob(os.path.join(prepared, "*")) if os.path.exists(os.path.join(p, "dataset_dict.json"))][0]
        )

        # 2. turn it into a linear-probe cohort dataset (labels and prediction times added)
        def add_cohort_columns(batch, indices):
            return {
                "age_at_index": [50.0 + i % 30 for i in indices],
                "classifier_label": [i % 2 for i in indices],
                "index_date": [1.6e9 + 86400.0 * i for i in indices],
            }

        cls.processed = DatasetDict(
            {
                split: tokenized[split].map(add_cohort_columns, batched=True, with_indices=True)
                for split in ("train", "validation", "test")
            }
        )
        persons = sorted(
            {i for split in cls.processed.values() for i in split["person_id"]}
        )
        cohort_folder = os.path.join(cls.temp_dir, "cohort")
        Path(cohort_folder).mkdir()
        pd.DataFrame(
            {
                "person_id": persons,
                "index_date": pd.Timestamp("2020-01-01"),
                "gender_concept_id": [8507 if i % 2 else 8532 for i in persons],
                "race_concept_id": [8527] * len(persons),
            }
        ).to_parquet(os.path.join(cohort_folder, "cohort.parquet"))
        cls.persons = persons
        cls.base_argv = [
            "compute_cehrgpt_features",
            "--model_name_or_path", cls.model_folder,
            "--tokenizer_name_or_path", cls.model_folder,
            "--data_folder", cohort_folder,
            "--test_data_folder", cohort_folder,
            "--dataset_prepared_path", os.path.join(cls.temp_dir, "lp_prepared"),
            "--max_position_embeddings", "128",
            "--torch_dtype", "float32",
            "--per_device_eval_batch_size", "1",
            "--preprocessing_num_workers", "1",
            "--report_to", "none",
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def _read_features(self, output_dir):
        frames = [
            pd.read_parquet(f)
            for f in glob.glob(os.path.join(output_dir, "features_with_label", "*_features", "*.parquet"))
        ]
        frame = pd.concat(frames, ignore_index=True)
        return frame.sort_values(["subject_id", "prediction_time_posix"]).reset_index(drop=True)

    def _run_single_worker(self, output_dir):
        shard = os.path.join(self.temp_dir, "whole_dataset_" + os.path.basename(output_dir))
        DatasetDict(
            {
                "train": cf.concatenate_datasets([self.processed["train"], self.processed["validation"]]),
                "test": self.processed["test"],
            }
        ).save_to_disk(shard)
        with mock.patch.object(
            sys, "argv", self.base_argv + ["--output_dir", output_dir, "--shard_dataset_path", shard]
        ):
            cf.main()

    def test_two_workers_match_a_single_worker(self):
        single_dir = os.path.join(self.temp_dir, "single")
        sharded_dir = os.path.join(self.temp_dir, "sharded")
        self._run_single_worker(single_dir)

        argv = self.base_argv + ["--output_dir", sharded_dir, "--gpu_ids", "0,1"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            cf,
            "prepare_processed_dataset",
            lambda *a, **k: (self.processed, None, self.model_folder),
        ):
            code = cf.run_on_multiple_gpus(
                cf.parse_runner_args()[0],
                cf.parse_runner_args()[1],
                cf.parse_runner_args()[2],
                cf.parse_runner_args()[3],
                cuda_devices=["0", "1"],
            )
        self.assertEqual(code, 0, "see worker logs in " + sharded_dir)

        single = self._read_features(single_dir)
        sharded = self._read_features(sharded_dir)
        self.assertEqual(len(sharded), len(single))
        self.assertGreater(len(single), 0)
        for column in ("subject_id", "prediction_time_posix", "boolean_value", "age_at_index", "gender_concept_id", "race_concept_id"):
            np.testing.assert_array_equal(sharded[column].to_numpy(), single[column].to_numpy(), err_msg=column)
        np.testing.assert_allclose(
            np.stack(sharded["features"].to_numpy()),
            np.stack(single["features"].to_numpy()),
            atol=1e-4,
        )
        # the demographics lookup really resolves the ids (they used to be 0)
        self.assertEqual(set(single["race_concept_id"]), {8527})
        self.assertTrue(set(single["gender_concept_id"]).issubset({8507, 8532}))
        self.assertNotIn(0, set(single["gender_concept_id"]))
        # both workers produced files and the shard folder was cleaned up
        self.assertEqual(len(glob.glob(os.path.join(sharded_dir, "worker_logs", "*.log"))), 2)
        self.assertFalse(os.path.exists(os.path.join(sharded_dir, "temp", "shards")))


if __name__ == "__main__":
    unittest.main()
