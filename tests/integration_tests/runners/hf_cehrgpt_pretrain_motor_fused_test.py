import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["USE_TF"] = "0"

from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main


class PretrainMotorFusedLossIntegrationTest(unittest.TestCase):
    """Pretraining with MOTOR and --motor_fused_loss must train and record the setting."""

    @classmethod
    def setUpClass(cls):
        root = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        cls.data_folder = os.path.join(root, "sample_data", "pretrain")
        cls.vocab_dir = os.path.join(root, "sample_data", "omop_vocab")
        embeddings = os.path.join(root, "sample_data", "pretrained_embeddings")
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_folder = os.path.join(cls.temp_dir, "model")
        Path(cls.model_folder).mkdir()
        for name in (PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME, PRETRAINED_EMBEDDING_VECTOR_FILE_NAME):
            shutil.copy(os.path.join(embeddings, name), cls.model_folder)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_training_with_the_fused_motor_loss(self):
        sys.argv = [
            "pretrain",
            "--model_name_or_path", self.model_folder,
            "--tokenizer_name_or_path", self.model_folder,
            "--output_dir", self.model_folder,
            "--data_folder", self.data_folder,
            "--dataset_prepared_path", os.path.join(self.temp_dir, "prepared"),
            "--pretrained_embedding_path", self.model_folder,
            "--vocab_dir", self.vocab_dir,
            "--max_steps", "3",
            "--save_steps", "3",
            "--save_strategy", "steps",
            "--hidden_size", "96",
            "--max_position_embeddings", "128",
            "--validation_split_num", "10",
            "--streaming", "true",
            "--use_early_stopping", "false",
            "--report_to", "none",
            "--include_motor_time_to_event", "true",
            "--motor_sampling_probability", "0.5",
            "--motor_fused_loss", "true",
            "--profile_gpu_steps", "1",
            "--profile_gpu_start_step", "1",
            "--exclude_position_ids", "true",
            "--apply_rotary", "true",
            "--apply_entropy_filter",
            "--min_prevalence", "0.01",
        ]
        train_main()
        self.assertTrue(os.path.exists(os.path.join(self.model_folder, "gpu_profile_trace.json")))
        checkpoints = sorted(Path(self.model_folder).glob("checkpoint-*"))
        self.assertTrue(checkpoints, "no checkpoint was saved")
        config = json.load(open(checkpoints[-1] / "config.json"))
        self.assertTrue(config["include_motor_time_to_event"])
        self.assertTrue(config["motor_fused_loss"])
        # the loss was finite throughout training
        state = json.load(open(checkpoints[-1] / "trainer_state.json"))
        losses = [entry["loss"] for entry in state["log_history"] if "loss" in entry]
        self.assertTrue(all(loss == loss and abs(loss) != float("inf") for loss in losses))


if __name__ == "__main__":
    unittest.main()
