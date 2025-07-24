import logging
import os
import sys
import uuid
from functools import partial
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
from tqdm import tqdm

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.generation.omop_converter_batch import (
    create_arg_parser,
    get_num_records,
    main_parallel,
    record_generator,
)
from cehrgpt.gpt_utils import random_slice_gpt_sequence
from cehrgpt.omop.vocab_utils import generate_concept_maps

logger = logging.getLogger(__name__)


def convert_concepts_to_patient_narrative(
    concept_ids: List[str],
    concept_name_mapping: Dict[str, str],
    concept_domain_mapping: Dict[str, str],
    context_window: int,
    person_id: Optional[int] = None,
    numeric_values: Optional[List[float]] = None,
    text_values: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
) -> Tuple[str, int, int]:
    pat_seq = list(concept_ids)
    starting_index = 0
    end_index = len(concept_ids)
    if len(concept_ids) > context_window:
        starting_index, end_index, demographic_tokens = random_slice_gpt_sequence(
            concept_ids=concept_ids, max_seq_len=context_window
        )
        pat_seq = demographic_tokens + pat_seq[starting_index:end_index]
        if numeric_values is not None:
            numeric_values = [0.0] * len(demographic_tokens) + list(
                numeric_values[starting_index:end_index]
            )
        if text_values is not None:
            text_values = [None] * len(demographic_tokens) + list(
                text_values[starting_index:end_index]
            )
        if units is not None:
            units = ["N/A"] * len(demographic_tokens) + list(
                units[starting_index:end_index]
            )

    patient_sequence_converter = get_cehrgpt_patient_converter(
        concept_ids=pat_seq,
        concept_domain_mapping=concept_domain_mapping,
        numeric_values=numeric_values,
        text_values=text_values,
        units=units,
    )

    narrative = None
    if patient_sequence_converter.is_validation_passed:
        patient = patient_sequence_converter.get_patient(
            domain_map=concept_domain_mapping, concept_map=concept_name_mapping
        )
        narrative = patient.get_narrative()
    else:
        logger.error(
            "person_id: %s, starting_index: %s, error: %s",
            person_id,
            starting_index,
            patient_sequence_converter.get_error_messages(),
        )

    return narrative, starting_index, end_index


def generate_patient_narratives(
    patient_sequence_parquet_files: List[str],
    concept_domain_map: Dict[str, str],
    output_folder: str,
    buffer_size: int,
    *args,
    **kwargs,
) -> None:
    context_window: Optional[int] = kwargs.get("context_window", None)
    concept_name_map: Optional[Dict[str, str]] = kwargs.get("concept_name_map", None)

    if context_window is None:
        raise RuntimeError("context_window must be specified")

    if concept_domain_map is None:
        raise RuntimeError("concept_domain_map must be specified")

    converted_narratives = []
    patient_record_generator = record_generator(patient_sequence_parquet_files)
    total_record = get_num_records(patient_sequence_parquet_files)
    for index, record in tqdm(enumerate(patient_record_generator), total=total_record):
        concept_ids = getattr(record, "concept_ids")
        numeric_values = getattr(record, "number_as_values", None)
        text_values = getattr(record, "concept_as_values", None)
        units = getattr(record, "units", None)
        person_id = getattr(record, "person_id", None)
        label = getattr(record, "label", None)
        index_date = getattr(record, "index_date", None)

        narrative, starting_index, end_index = convert_concepts_to_patient_narrative(
            concept_ids=concept_ids,
            concept_name_mapping=concept_name_map,
            concept_domain_mapping=concept_domain_map,
            context_window=context_window,
            person_id=person_id,
            numeric_values=numeric_values,
            text_values=text_values,
            units=units,
        )
        if narrative is not None:
            converted_narratives.append(
                {
                    "person_id": person_id,
                    "narrative": narrative,
                    "starting_index": starting_index,
                    "ending_index": end_index,
                    "label": label,
                    "index_date": index_date,
                }
            )
        if index != 0 and index % buffer_size == 0:
            if converted_narratives:
                pd.DataFrame(
                    converted_narratives,
                    columns=[
                        "person_id",
                        "index_date",
                        "label",
                        "narrative",
                        "starting_index",
                        "ending_index",
                    ],
                ).to_parquet(os.path.join(output_folder, f"{uuid.uuid4()}.parquet"))
                converted_narratives.clear()

    # Final flush to the disk if there are still records in the cache
    if converted_narratives:
        pd.DataFrame(
            converted_narratives,
            columns=[
                "person_id",
                "index_date",
                "label",
                "narrative",
                "starting_index",
                "ending_index",
            ],
        ).to_parquet(os.path.join(output_folder, f"batch_final_{uuid.uuid4()}.parquet"))
        converted_narratives.clear()


if __name__ == "__main__":
    args = create_arg_parser()
    concept_dataframe = pl.read_parquet(os.path.join(args.concept_path, "*parquet"))
    concept_name_map, _ = generate_concept_maps(concept_dataframe)
    main_parallel(
        args,
        partial(
            generate_patient_narratives,
            context_window=sys.maxsize,
            concept_name_map=concept_name_map,
        ),
    )
