import argparse
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from cehrbert.runners.runner_util import load_parquet_as_dataset

from cehrgpt.data.hf_cehrgpt_dataset import apply_cehrbert_dataset_mapping
from cehrgpt.data.hf_cehrgpt_dataset_mapping import DatasetMapping
from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.gpt_utils import random_slice_gpt_sequence
from cehrgpt.omop.vocab_utils import generate_concept_maps

logger = logging.getLogger(__name__)


class GeneratePatientNarrativeMapping(DatasetMapping):
    def __init__(
        self,
        context_window: int,
        concept_name_mapping: Dict[str, str],
        concept_domain_mapping: Dict[str, str],
    ):
        self.context_window = context_window
        self.concept_name_mapping = concept_name_mapping
        self.concept_domain_mapping = concept_domain_mapping

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        patient_narrative, start_index, end_index = (
            convert_concepts_to_patient_narrative(
                record["concept_ids"],
                self.concept_name_mapping,
                self.concept_domain_mapping,
                self.context_window,
                record.get("person_id", None),
            )
        )
        record["patient_narrative"] = {
            "narrative": patient_narrative,
            "start_index": start_index,
            "end_index": end_index,
        }
        return record


def convert_concepts_to_patient_narrative(
    concept_ids: List[str],
    concept_name_mapping: Dict[str, str],
    concept_domain_mapping: Dict[str, str],
    context_window: int,
    person_id: Optional[int] = None,
) -> Tuple[str, int, int]:
    pat_seq = list(concept_ids)
    starting_index = 0
    end_index = len(concept_ids)
    if len(concept_ids) > context_window:
        starting_index, end_index, demographic_tokens = random_slice_gpt_sequence(
            concept_ids=concept_ids, max_seq_len=context_window
        )
        pat_seq = demographic_tokens + pat_seq[starting_index:end_index]
    patient_sequence_converter = get_cehrgpt_patient_converter(
        concept_ids=pat_seq, concept_domain_mapping=concept_domain_mapping
    )
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
        narrative = None
    return narrative, starting_index, end_index


def main(args):
    pat_seq_dataset = load_parquet_as_dataset(args.patient_sequence_dir)
    concept_dataframe = pl.read_parquet(os.path.join(args.concept_dir, "*parquet"))
    concept_name_mapping, concept_domain_mapping = generate_concept_maps(
        concept_dataframe
    )

    transformed_pat_seq_dataset = apply_cehrbert_dataset_mapping(
        pat_seq_dataset,
        mapping_function=GeneratePatientNarrativeMapping(
            context_window=args.context_window,
            concept_name_mapping=concept_name_mapping,
            concept_domain_mapping=concept_domain_mapping,
        ),
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        streaming=args.streaming,
    )
    for i, batched_dataset_df in enumerate(
        transformed_pat_seq_dataset.to_pandas(batch_size=args.batch_size, batched=True)
    ):
        batched_dataset_df.to_parquet(
            os.path.join(args.output_dir, f"{str(uuid.uuid4())}.parquet")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate patient narratives")
    parser.add_argument(
        "--patient_sequence_dir",
        required=True,
        help="The patient sequence data dir",
    )
    parser.add_argument(
        "--concept_dir",
        required=True,
    )
    parser.add_argument(
        "--context_window",
        required=True,
        type=int,
        help="Context window size",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_proc",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    main(parser.parse_args())
