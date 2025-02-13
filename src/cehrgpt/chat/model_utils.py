import datetime
import json
import os
from pathlib import Path
from typing import List

import polars as pl
import torch
from flask import Response, jsonify
from transformers import GenerationConfig

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import (
    CehrGptPatient,
    load_cehrgpt_patient_from_json,
)
from cehrgpt.generation.encoder_decoder.instruct_cehrpgt_query import (
    parse_question_to_cehrgpt_query,
)
from cehrgpt.generation.encoder_decoder.instruct_model_cli import (
    generate_concept_maps,
    generate_responses,
    get_cehrgpt_patient_converter,
    setup_model,
)

from .config import Config

# Initialize model and configs
config = Config()

if config.DEV_MODE:
    encoder_tokenizer, cehrgpt_tokenizer, model, device = None, None, None, None
    concept_name_map, concept_domain_map = {}, {}
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_tokenizer, cehrgpt_tokenizer, model = setup_model(
        config.TOKENIZER_PATH, config.MODEL_PATH, device
    )
    concept = pl.read_parquet(
        os.path.join(config.VOCABULARY_DIR, "concept", "*parquet")
    )
    concept_name_map, concept_domain_map = generate_concept_maps(concept)


def load_test_patient() -> CehrGptPatient:
    root_folder_path = Path(os.path.abspath(__file__)).parent.parent.parent.parent
    test_patient_path = (
        root_folder_path / "sample_data" / "cehrgpt_patients" / "patient_data.json"
    )
    try:
        print(f"test_patient_path: {test_patient_path}")
        with open(test_patient_path, "rb") as f:
            return load_cehrgpt_patient_from_json(json.load(f))
    except Exception:
        return CehrGptPatient(
            birth_datetime=datetime.datetime.now(),
            gender_concept_id=8532,
            gender="Female",
            race_concept_id=8527,
            race="White",
        )


def handle_query(user_input: str) -> Response:
    if config.DEV_MODE:
        return jsonify(load_test_patient())
    query_tuple = parse_question_to_cehrgpt_query(user_input)
    if not query_tuple:
        return jsonify({"message": "Failed to parse the query, please try again"})
    query, _ = query_tuple
    cehrgpt_patients = prompt_model(query, 1)
    if cehrgpt_patients:
        return jsonify(cehrgpt_patients)
    return jsonify(
        {"message": f"Failed to generate patients for query: {query}, please try again"}
    )


def prompt_model(query: str, n_patients: int) -> List[CehrGptPatient]:
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
    model_responses = generate_responses(
        queries=[query] * n_patients,
        encoder_tokenizer=encoder_tokenizer,
        cehrgpt_tokenizer=cehrgpt_tokenizer,
        model=model,
        device=device,
        generation_config=generation_config,
    )

    cehrgpt_patients = []
    for seq in model_responses["sequences"]:
        patient_sequence_converter = get_cehrgpt_patient_converter(
            seq, concept_domain_map
        )
        if patient_sequence_converter.is_validation_passed:
            cehrgpt_patients.append(
                patient_sequence_converter.get_patient(
                    concept_domain_map, concept_name_map
                )
            )
    return cehrgpt_patients
