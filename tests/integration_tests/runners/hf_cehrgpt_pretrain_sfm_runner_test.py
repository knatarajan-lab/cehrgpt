import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from datasets import disable_caching

from cehrgpt.generation.generate_batch_hf_gpt_sequence import create_arg_parser
from cehrgpt.generation.generate_batch_hf_gpt_sequence import main as generate_main
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main

disable_caching()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


class HfCehrGptCausalRunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        cls.data_folder = os.path.join(root_folder, "sample_data", "pretrain")
        # Create a temporary directory to store model and tokenizer
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_folder_path = os.path.join(cls.temp_dir, "model")
        Path(cls.model_folder_path).mkdir(parents=True, exist_ok=True)
        cls.dataset_prepared_path = os.path.join(cls.temp_dir, "dataset_prepared_path")
        Path(cls.dataset_prepared_path).mkdir(parents=True, exist_ok=True)
        cls.generation_folder_path = os.path.join(cls.temp_dir, "generation")
        Path(cls.generation_folder_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory
        shutil.rmtree(cls.temp_dir)

    def test_1_train_model(self):
        sys.argv = [
            "hf_cehrgpt_pretraining_runner.py",
            "--model_name_or_path",
            self.model_folder_path,
            "--tokenizer_name_or_path",
            self.model_folder_path,
            "--output_dir",
            self.model_folder_path,
            "--data_folder",
            self.data_folder,
            "--dataset_prepared_path",
            self.dataset_prepared_path,
            "--max_steps",
            "500",
            "--save_steps",
            "500",
            "--save_strategy",
            "steps",
            "--hidden_size",
            "32",
            "--max_position_embeddings",
            "128",
            "--use_sub_time_tokenization",
            "false",
            "--include_ttv_prediction",
            "false",
            "--causal_sfm",
            "--demographics_size",
            "4",
            "--use_early_stopping",
            "false",
            "--report_to",
            "none",
            "--exclude_position_ids",
            "true",
        ]
        train_main()
        # Teacher force the prompt to consist of [year][age][gender][race][VS] then inject the random vector before [VS]

    def test_2_generate_model(self):
        sys.argv = [
            "generate_batch_hf_gpt_sequence.py",
            "--model_folder",
            self.model_folder_path,
            "--tokenizer_folder",
            self.model_folder_path,
            "--output_folder",
            self.generation_folder_path,
            "--context_window",
            "128",
            "--num_of_patients",
            "16",
            "--batch_size",
            "4",
            "--buffer_size",
            "16",
            "--sampling_strategy",
            "TopPStrategy",
            "--demographic_data_path",
            self.data_folder,
        ]
        args = create_arg_parser().parse_args()
        generate_main(args)
        generated_sequences = pd.read_parquet(self.generation_folder_path)
        for concept_ids in generated_sequences.concept_ids:
            print(concept_ids)


if __name__ == "__main__":
    unittest.main()
