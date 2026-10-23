import glob
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["USE_TF"] = "0"

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main


class PretrainSpeedOptionsIntegrationTest(unittest.TestCase):
    """--optim adamw_torch_fused and --activation_function gelu_pytorch_tanh must train."""

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

    def test_fused_adamw_and_the_fused_gelu_train_and_are_recorded(self):
        sys.argv = [
            "pretrain",
            "--model_name_or_path", self.model_folder,
            "--tokenizer_name_or_path", self.model_folder,
            "--output_dir", self.model_folder,
            "--data_folder", self.data_folder,
            "--dataset_prepared_path", os.path.join(self.temp_dir, "prepared"),
            "--pretrained_embedding_path", self.model_folder,
            "--vocab_dir", self.vocab_dir,
            "--max_steps", "4",
            "--logging_steps", "1",
            "--profile_gpu_steps", "1",
            "--profile_gpu_start_step", "1",
            "--save_steps", "4",
            "--save_strategy", "steps",
            "--hidden_size", "96",
            "--max_position_embeddings", "128",
            "--validation_split_num", "10",
            "--streaming", "true",
            "--use_early_stopping", "false",
            "--report_to", "none",
            "--include_motor_time_to_event", "true",
            "--motor_sampling_probability", "0.5",
            "--exclude_position_ids", "true",
            "--apply_rotary", "true",
            "--apply_entropy_filter",
            "--min_patient_count", "1",
            # fused AdamW supports cuda and cpu, not mps, which the Trainer picks on a Mac
            "--use_cpu", "true",
            "--optim", "adamw_torch_fused",
            "--activation_function", "gelu_pytorch_tanh",
        ]
        train_main()
        self.assertTrue(os.path.exists(os.path.join(self.model_folder, "gpu_profile_trace.json")))
        checkpoint = sorted(Path(self.model_folder).glob("checkpoint-*"))[-1]

        config = json.load(open(checkpoint / "config.json"))
        self.assertEqual(config["activation_function"], "gelu_pytorch_tanh")

        model = CEHRGPT2LMHeadModel.from_pretrained(checkpoint)
        self.assertEqual(
            type(model.cehrgpt.h[0].mlp.act).__name__, "PytorchGELUTanh"
        )

        optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu")
        self.assertTrue(optimizer_state["param_groups"][0]["fused"])

        state = json.load(open(checkpoint / "trainer_state.json"))
        losses = [entry["loss"] for entry in state["log_history"] if "loss" in entry]
        self.assertTrue(losses)
        self.assertTrue(all(math.isfinite(loss) for loss in losses))


if __name__ == "__main__":
    unittest.main()
