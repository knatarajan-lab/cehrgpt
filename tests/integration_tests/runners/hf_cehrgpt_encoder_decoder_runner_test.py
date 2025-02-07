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
from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main

disable_caching()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


class HfCehrGptRunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        cls.data_folder = os.path.join(root_folder, "sample_data", "pretrain")
        cls.knowledge_graph_folder = os.path.join(
            root_folder, "sample_data", "omop_vocab", "knowledge_graph_sample.pickle"
        )
        cls.omop_vocab_folder = os.path.join(root_folder, "sample_data", "omop_vocab")
        cls.pretrained_embedding_folder = os.path.join(
            root_folder, "sample_data", "pretrained_embeddings"
        )
        # Create a temporary directory to store model and tokenizer
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_folder_path = os.path.join(cls.temp_dir, "model")
        Path(cls.model_folder_path).mkdir(parents=True, exist_ok=True)
        cls.dataset_prepared_path = os.path.join(cls.temp_dir, "dataset_prepared_path")
        Path(cls.dataset_prepared_path).mkdir(parents=True, exist_ok=True)
        cls.generation_folder_path = os.path.join(cls.temp_dir, "generation")
        Path(cls.generation_folder_path).mkdir(parents=True, exist_ok=True)
        for file_name in [
            PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
            PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
        ]:
            shutil.copy(
                os.path.join(cls.pretrained_embedding_folder, file_name),
                os.path.join(cls.model_folder_path, file_name),
            )

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory
        shutil.rmtree(cls.temp_dir)

    def test_1_train_model(self):
        sys.argv = [
            "hf_cehrgpt_pretraining_runner.py",
            "--add_cross_attention",
            "true",
            "--encoder_model_name_or_path",
            "medicalai/ClinicalBERT",
            "--encoder_tokenizer_name_or_path",
            "medicalai/ClinicalBERT",
            "--knowledge_graph_path",
            self.knowledge_graph_folder,
            "--vocabulary_dir",
            self.omop_vocab_folder,
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
            "--pretrained_embedding_path",
            self.model_folder_path,
            "--max_steps",
            "100",
            "--save_steps",
            "100",
            "--save_strategy",
            "steps",
            "--hidden_size",
            "192",
            "--use_sub_time_tokenization",
            "false",
            "--include_ttv_prediction",
            "false",
            "--include_values",
            "true",
        ]
        train_main()


if __name__ == "__main__":
    unittest.main()
