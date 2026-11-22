import glob
import json
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
from cehrgpt.tools.linear_prob import compute_attention_rollout as rollout


class AttentionRolloutIntegrationTest(unittest.TestCase):
    """The attention rollout tool on a real (tiny) model, tokenizer and cohort dataset."""

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
        cls.output_dir = os.path.join(cls.temp_dir, "output")
        cls.argv = [
            "compute_attention_rollout",
            "--model_name_or_path", cls.model_folder,
            "--tokenizer_name_or_path", cls.model_folder,
            "--data_folder", cls.temp_dir,
            "--dataset_prepared_path", os.path.join(cls.temp_dir, "lp_prepared"),
            "--output_dir", cls.output_dir,
            # Below the 128 of the model: sample packing cuts a patient at the length of the model
            "--max_position_embeddings", "64",
            "--torch_dtype", "float32",
            "--per_device_eval_batch_size", "3",
            "--preprocessing_num_workers", "1",
            "--report_to", "none",
            "--sample_packing",
            "--attention_rollout_max_samples", "6",
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_rollout_of_a_cohort(self):
        with mock.patch.object(sys, "argv", self.argv), mock.patch.object(
            rollout,
            "_load_or_create_prepared_dataset",
            lambda cehrgpt_args, data_args, model_args, training_args, tokenizer: (
                self.processed,
                tokenizer,
                self.model_folder,
                None,
            ),
        ):
            rollout.main()

        folder = Path(self.output_dir) / "attention_rollout_global"
        info = json.loads((folder / "attention_rollout_info.json").read_text())
        self.assertEqual(info["target"], "global")
        self.assertEqual(info["num_samples"], {"train": 6, "test": 6})
        lengths = {
            (person, date): length
            for split in self.processed.values()
            for person, date, length in zip(
                split["person_id"], split["index_date"], split["num_of_concepts"]
            )
        }
        num_tokens = []
        for split in ("train", "test"):
            frame = pd.concat(
                [pd.read_parquet(f) for f in glob.glob(str(folder / split / "*.parquet"))],
                ignore_index=True,
            )
            self.assertEqual(len(frame), 6)
            for row in frame.itertuples():
                self.assertEqual(len(row.concept_ids), len(row.importance))
                self.assertEqual(len(row.concept_ids), len(row.baseline_importance))
                # The tokens the model was given for the patient and no padding: the data
                # processor cuts a history to the context window and can drop its last token
                length = lengths[(row.subject_id, row.prediction_time_posix)]
                self.assertLessEqual(len(row.concept_ids), min(length, 128))
                self.assertGreaterEqual(len(row.concept_ids), min(length, 128) - 2)
                self.assertTrue((np.asarray(row.importance) >= 0).all())
                np.testing.assert_allclose(np.sum(row.importance), 1.0, rtol=1e-4)
                np.testing.assert_allclose(np.sum(row.baseline_importance), 1.0, rtol=1e-4)
                num_tokens.append(len(row.concept_ids))
            concepts = pd.read_csv(folder / f"{split}_concept_importance.csv")
            np.testing.assert_allclose(concepts["mean_importance"].sum(), 1.0, rtol=1e-4)
            self.assertTrue((concepts["num_samples"] <= 6).all())
            self.assertFalse(concepts["concept_id"].astype(str).str.fullmatch(r"\d{1,3}").all())
        # A patient longer than --max_position_embeddings is not cut there
        self.assertGreater(max(num_tokens), 64)


if __name__ == "__main__":
    unittest.main()
