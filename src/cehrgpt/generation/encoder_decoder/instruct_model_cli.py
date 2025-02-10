import argparse
import os
from typing import Any, Dict, List, Union

import polars as pl
import torch
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer
from transformers.utils import is_flash_attn_2_available

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.generation.generate_batch_hf_gpt_sequence import (
    extract_output_from_model_response,
)
from cehrgpt.models.encoder_decoder.instruct_hf_cehrgpt import InstructCEHRGPTModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.omop.vocab_utils import generate_concept_maps


def setup_model(tokenizer_path, model_path, device):
    encoder_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_path)
    model = (
        InstructCEHRGPTModel.from_pretrained(
            model_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
        )
        .eval()
        .to(device)
    )
    return encoder_tokenizer, cehrgpt_tokenizer, model


def generate_responses(
    queries: List[str],
    encoder_tokenizer: PreTrainedTokenizer,
    cehrgpt_tokenizer: CehrGptTokenizer,
    model: InstructCEHRGPTModel,
    device: Union[torch.device, str],
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    encoder_inputs = encoder_tokenizer(queries, return_tensors="pt")
    encoder_input_ids = encoder_inputs["input_ids"]
    encoder_attention_mask = encoder_inputs["attention_mask"]
    batch_size = encoder_input_ids.shape[0]
    batched_inputs = torch.tile(
        torch.tensor([[cehrgpt_tokenizer.start_token_id]]), (batch_size, 1)
    ).to(device)

    output = model.generate(
        inputs=encoder_input_ids.to(device),
        attention_mask=encoder_attention_mask.to(device),
        decoder_input_ids=batched_inputs.to(device),
        generation_config=generation_config,
        lab_token_ids=cehrgpt_tokenizer.lab_token_ids,
    )
    return extract_output_from_model_response(
        results=output, cehrgpt_tokenizer=cehrgpt_tokenizer
    )


def create_instruct_cehrgpt_argparser():
    parser = argparse.ArgumentParser(
        description="Generate responses using InstructCEHRGPTModel"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        required=True,
        help="Path to InstructCEHRGPT folder containing the encoder and cehrgpt tokenizers",
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to the InstructCEHRGPT model checkpoint",
    )
    parser.add_argument(
        "--vocabulary_dir",
        required=True,
        help="Path to vocabulary dir containing concept and concept_ancestor data",
    )
    return parser.parse_args()


def main():
    args = create_instruct_cehrgpt_argparser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Setting up the model and tokenizer...")

    encoder_tokenizer, cehrgpt_tokenizer, model = setup_model(
        tokenizer_path=args.tokenizer_name_or_path,
        model_path=args.model_name_or_path,
        device=device,
    )

    # Configure model generation settings
    generation_config = GenerationConfig(
        max_length=1024,
        min_length=20,
        bos_token_id=cehrgpt_tokenizer.start_token_id,
        eos_token_id=cehrgpt_tokenizer.end_token_id,
        pad_token_id=cehrgpt_tokenizer.pad_token_id,
        decoder_start_token_id=cehrgpt_tokenizer.start_token_id,
        do_sample=True,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        renormalize_logits=True,
    )

    concept = pl.read_parquet(os.path.join(args.vocabulary_dir, "concept", "*parquet"))
    concept_name_map, concept_domain_map = generate_concept_maps(concept)
    while True:
        query = input(
            "Enter your query (type 'exit' to quit): \n"
            "Example: Race: White\nGender: MALE\n\n1. Age: 50\n1. Condition: Essential Hypertension\n"
        )
        if query.lower() == "exit":
            break

        sequences = generate_responses(
            queries=[query],
            encoder_tokenizer=encoder_tokenizer,
            cehrgpt_tokenizer=cehrgpt_tokenizer,
            model=model,
            device=device,
            generation_config=generation_config,
        )

        if sequences:
            patient_sequence_converter = get_cehrgpt_patient_converter(
                sequences[0], concept_domain_map
            )
            if patient_sequence_converter.is_validation_passed:
                cehrgpt_patient = patient_sequence_converter.get_patient(
                    concept_domain_map, concept_name_map
                )
                print("\nGenerated Response:\n", cehrgpt_patient.get_narrative())


if __name__ == "__main__":
    main()
