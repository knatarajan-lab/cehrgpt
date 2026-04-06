"""
Integration test for the CEHR-GPT RL fine-tuning runner.

Test flow
---------
1. test_1_pretrain_model  – train a tiny model (non-streaming so the tokenized
   dataset is saved to disk) with MOTOR TTE enabled, producing:
     * a saved tokenizer with motor_time_to_event_codes
     * a saved DatasetDict with concept_ids / epoch_times / ages columns
     * model weights to warm-start the RL policy

2. test_2_rl_finetune     – run 2 steps of RL fine-tuning and verify that
     * the trainer runs without error
     * the RL model is saved to the output directory
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as pretrain_main
from cehrgpt.runners.hf_cehrgpt_rl_runner import main as rl_main


class HfCehrGptRLRunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        cls.data_folder = os.path.join(root_folder, "sample_data", "pretrain")
        cls.pretrained_embedding_folder = os.path.join(
            root_folder, "sample_data", "pretrained_embeddings"
        )
        cls.vocab_dir = os.path.join(root_folder, "sample_data", "omop_vocab")

        cls.temp_dir = tempfile.mkdtemp()
        cls.model_folder_path = os.path.join(cls.temp_dir, "pretrain_model")
        cls.rl_output_path = os.path.join(cls.temp_dir, "rl_model")
        cls.dataset_prepared_path = os.path.join(cls.temp_dir, "dataset_prepared")

        for d in [cls.model_folder_path, cls.rl_output_path, cls.dataset_prepared_path]:
            Path(d).mkdir(parents=True, exist_ok=True)

        for fname in [PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME, PRETRAINED_EMBEDDING_VECTOR_FILE_NAME]:
            shutil.copy(
                os.path.join(cls.pretrained_embedding_folder, fname),
                os.path.join(cls.model_folder_path, fname),
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    # ------------------------------------------------------------------
    # Step 1: pretrain a tiny model (non-streaming → saves tokenized dataset)
    # ------------------------------------------------------------------
    def test_1_pretrain_model(self):
        sys.argv = [
            "hf_cehrgpt_pretraining_runner.py",
            "--model_name_or_path", self.model_folder_path,
            "--tokenizer_name_or_path", self.model_folder_path,
            "--output_dir", self.model_folder_path,
            "--data_folder", self.data_folder,
            "--dataset_prepared_path", self.dataset_prepared_path,
            "--pretrained_embedding_path", self.model_folder_path,
            "--vocab_dir", self.vocab_dir,
            "--max_steps", "2",
            "--save_steps", "2",
            "--save_strategy", "steps",
            # Small model to keep the test fast
            "--hidden_size", "96",
            "--max_position_embeddings", "128",
            # MOTOR TTE required for RL reward targets
            "--include_motor_time_to_event", "true",
            "--linear_prob_n_layer", "2",
            "--motor_sampling_probability", "0.5",
            # Low threshold so sample data produces at least some motor codes
            "--apply_entropy_filter",
            "--min_prevalence", "0.001",
            "--apply_rotary", "true",
            "--exclude_position_ids", "true",
            "--activation_function", "silu",
            "--decoder_mlp", "LlamaMLP",
            "--use_early_stopping", "false",
            "--report_to", "none",
            # non-streaming → tokenized dataset is saved to disk
            "--streaming", "false",
        ]
        pretrain_main()

        # Verify the tokenized dataset was saved somewhere under dataset_prepared_path
        saved_datasets = list(Path(self.dataset_prepared_path).glob("**/dataset_dict.json"))
        self.assertTrue(
            len(saved_datasets) > 0,
            f"Expected a saved DatasetDict under {self.dataset_prepared_path}",
        )
        # Store the path for step 2
        self.__class__.tokenized_dataset_path = str(saved_datasets[0].parent)

    # ------------------------------------------------------------------
    # Step 2: RL fine-tuning for 2 steps
    # ------------------------------------------------------------------
    def test_2_rl_finetune(self):
        self.assertTrue(
            hasattr(self.__class__, "tokenized_dataset_path"),
            "test_1 must run before test_2",
        )
        sys.argv = [
            "hf_cehrgpt_rl_runner.py",
            "--model_name_or_path", self.model_folder_path,
            "--tokenizer_name_or_path", self.model_folder_path,
            "--tokenized_full_dataset_path", self.__class__.tokenized_dataset_path,
            "--output_dir", self.rl_output_path,
            # RL hyperparameters kept tiny for speed
            "--num_rollouts", "2",
            "--max_new_tokens", "32",
            "--kl_beta", "0.05",
            "--max_steps", "2",
            "--save_steps", "2",
            "--save_strategy", "steps",
            "--per_device_train_batch_size", "2",
            "--use_early_stopping", "false",
            "--report_to", "none",
            "--no_cuda", "true",
        ]
        rl_main()

        # Verify the RL model was saved
        self.assertTrue(
            any(Path(self.rl_output_path).glob("config.json")),
            f"Expected saved model config in {self.rl_output_path}",
        )

    # ------------------------------------------------------------------
    # Step 3: PPO fine-tuning for 2 steps (reuses pretrained model from step 1)
    # ------------------------------------------------------------------
    def test_3_ppo_finetune(self):
        self.assertTrue(
            hasattr(self.__class__, "tokenized_dataset_path"),
            "test_1 must run before test_3",
        )
        ppo_output_path = os.path.join(self.temp_dir, "ppo_model")
        Path(ppo_output_path).mkdir(parents=True, exist_ok=True)

        sys.argv = [
            "hf_cehrgpt_rl_runner.py",
            "--model_name_or_path", self.model_folder_path,
            "--tokenizer_name_or_path", self.model_folder_path,
            "--tokenized_full_dataset_path", self.__class__.tokenized_dataset_path,
            "--output_dir", ppo_output_path,
            # PPO-specific
            "--trainer_type", "ppo",
            "--ppo_clip_epsilon", "0.2",
            # RL hyperparameters kept tiny for speed
            "--num_rollouts", "2",
            "--max_new_tokens", "32",
            "--kl_beta", "0.05",
            "--max_steps", "2",
            "--save_steps", "2",
            "--save_strategy", "steps",
            "--per_device_train_batch_size", "2",
            "--use_early_stopping", "false",
            "--report_to", "none",
            "--no_cuda", "true",
        ]
        rl_main()

        # Verify the PPO model was saved
        self.assertTrue(
            any(Path(ppo_output_path).glob("config.json")),
            f"Expected saved model config in {ppo_output_path}",
        )


if __name__ == "__main__":
    unittest.main()
