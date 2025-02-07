import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from datasets import disable_caching
from transformers import AutoTokenizer, GenerationConfig

from cehrgpt.models.encoder_decoder.instruct_hf_cehrgpt import InstructCEHRGPTModel
from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
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
            "500",
            "--save_steps",
            "500",
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

    def test_2_generation(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(self.model_folder_path)
        encoder_tokenizer = AutoTokenizer.from_pretrained(self.model_folder_path)
        model = (
            InstructCEHRGPTModel.from_pretrained(self.model_folder_path)
            .eval()
            .to(device)
        )
        model.decoder.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
        model.decoder.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
        model.decoder.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

        query = """Race: White
        Gender: MALE

        1. Diagnosis age 31
        1. Condition: 4172829"""

        encoder_inputs = encoder_tokenizer(query, return_tensors="pt")
        encoder_input_ids = encoder_inputs["input_ids"]
        encoder_attention_mask = encoder_inputs["attention_mask"]
        batch_size = encoder_input_ids.shape[0]
        batched_inputs = torch.tile(
            torch.tensor([[cehrgpt_tokenizer.start_token_id]]), (batch_size, 1)
        ).to(device)

        generation_config = GenerationConfig(
            max_length=512,
            min_length=10,
            bos_token_id=cehrgpt_tokenizer.end_token_id,
            eos_token_id=cehrgpt_tokenizer.end_token_id,
            pad_token_id=cehrgpt_tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            renormalize_logits=True,
        )
        output = model.generate(
            input_ids=batched_inputs,
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask.to(device),
            generation_config=generation_config,
            lab_token_ids=cehrgpt_tokenizer.lab_token_ids,
        )
        sequences = [
            cehrgpt_tokenizer.decode(seq.cpu().numpy(), skip_special_tokens=False)
            for seq in output.sequences
        ]
        print(sequences)


if __name__ == "__main__":
    unittest.main()
