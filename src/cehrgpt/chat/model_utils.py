import datetime
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import polars as pl
import torch
from flask import Response, jsonify
from transformers import GenerationConfig, PreTrainedTokenizer

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
from cehrgpt.models.encoder_decoder.instruct_hf_cehrgpt import InstructCEHRGPTModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

from .config import Config


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


def load_concept_domain_map(config) -> Tuple[Dict[str, str], Dict[str, str]]:
    if config.DEV_MODE:
        return {}, {}
    concept = pl.read_parquet(
        os.path.join(config.VOCABULARY_DIR, "concept", "*parquet")
    )
    concept_name_map, concept_domain_map = generate_concept_maps(concept)
    return concept_name_map, concept_domain_map


def load_model(
    config,
) -> Tuple[
    Optional[PreTrainedTokenizer],
    Optional[CehrGptTokenizer],
    Optional[InstructCEHRGPTModel],
    Optional[Union[torch.device, str]],
]:
    if config.DEV_MODE:
        return None, None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_tokenizer, cehrgpt_tokenizer, model = setup_model(
        config.TOKENIZER_PATH, config.MODEL_PATH, device
    )
    return encoder_tokenizer, cehrgpt_tokenizer, model, device


def get_generation_config(
    cehrgpt_tokenizer: Optional[CehrGptTokenizer],
) -> Optional[GenerationConfig]:
    if cehrgpt_tokenizer is None:
        return None

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
    user_input: str,
    encoder_tokenizer: Optional[PreTrainedTokenizer],
    cehrgpt_tokenizer: Optional[CehrGptTokenizer],
    model: Optional[InstructCEHRGPTModel],
    device: Optional[Union[torch.device, str]],
    generation_config: Optional[GenerationConfig],
    concept_domain_map: Optional[Dict[str, str]],
    concept_name_map: Optional[Dict[str, str]],
    config: Config,
) -> Response:
    if config.DEV_MODE:
        return jsonify(load_test_patient())

    query_tuple = parse_question_to_cehrgpt_query(user_input)
    if not query_tuple:
        return jsonify(
            {"message": "Failed to parse the query. Generating a default response..."}
        )

    query, n_patients = query_tuple
    model_responses = generate_responses(
        queries=[query] * n_patients,
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
