import os

import polars as pl
import torch
from flask import jsonify
from transformers import GenerationConfig

from cehrgpt.generation.encoder_decoder.instruct_cehrpgt_query import (
    parse_question_to_cehrgpt_query,
)
from cehrgpt.generation.encoder_decoder.instruct_model_cli import (
    generate_concept_maps,
    generate_responses,
    get_cehrgpt_patient_converter,
    setup_model,
)


def load_concept_domain_map(config):
    concept = pl.read_parquet(
        os.path.join(config.VOCABULARY_DIR, "concept", "*parquet")
    )
    concept_name_map, concept_domain_map = generate_concept_maps(concept)
    return concept_name_map, concept_domain_map


def load_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_tokenizer, cehrgpt_tokenizer, model = setup_model(
        config.TOKENIZER_PATH, config.MODEL_PATH, device
    )
    return encoder_tokenizer, cehrgpt_tokenizer, model, device


def get_generation_config(cehrgpt_tokenizer):
    return GenerationConfig(
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


def handle_query(
    user_input,
    encoder_tokenizer,
    cehrgpt_tokenizer,
    model,
    device,
    generation_config,
    concept_domain_map,
    concept_name_map,
):
    query = parse_question_to_cehrgpt_query(user_input)
    if not query:
        return "Failed to parse the query. Generating a default response..."

    model_responses = generate_responses(
        queries=[query],
        encoder_tokenizer=encoder_tokenizer,
        cehrgpt_tokenizer=cehrgpt_tokenizer,
        model=model,
        device=device,
        generation_config=generation_config,
    )

    sequences = model_responses["sequences"]
    if sequences:
        patient_sequence_converter = get_cehrgpt_patient_converter(
            sequences[0], concept_domain_map
        )
        if patient_sequence_converter.is_validation_passed:
            return jsonify(
                patient_sequence_converter.get_patient(
                    concept_domain_map, concept_name_map
                )
            )
        else:
            return jsonify(
                {
                    "message": f"Error in patient sequence: {'. '.join(patient_sequence_converter.get_error_messages())}"
                }
            )
    return jsonify({"message": "Failed to parse the query, please try again"})
