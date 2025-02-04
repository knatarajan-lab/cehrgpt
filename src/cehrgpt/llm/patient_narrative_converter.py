import argparse
import datetime
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
from cehrbert.runners.runner_util import load_parquet_as_dataset

from cehrgpt.data.hf_cehrgpt_dataset import apply_cehrbert_dataset_mapping
from cehrgpt.data.hf_cehrgpt_dataset_mapping import DatasetMapping
from cehrgpt.generation.omop_converter_batch import START_TOKEN_SIZE
from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_inpatient_att_token,
    random_slice_gpt_sequence,
)


class GeneratePatientNarrativeMapping(DatasetMapping):
    def __init__(
        self,
        context_window: int,
        concept_mapping: Dict[str, str],
    ):
        self.context_window = context_window
        self.concept_mapping = concept_mapping

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        patient_narrative, start_index, end_index = (
            convert_concepts_to_patient_narrative(
                record["concept_ids"], self.concept_mapping, self.context_window
            )
        )
        record["patient_narrative"] = {
            "patient_narrative": patient_narrative,
            "start_index": start_index,
            "end_index": end_index,
        }
        return record


def generate_concept_map(concept_pd: pd.DataFrame) -> Dict[str, str]:
    concept_map = {}
    for i in concept_pd.itertuples():
        concept_map[str(i.concept_id)] = i.concept_name
    return concept_map


def convert_concepts_to_patient_narrative(
    concept_ids: List[str], concept_mapping: Dict[str, str], context_window: int
) -> Tuple[str, int, int]:
    pat_seq = list(concept_ids)
    starting_index = 0
    end_index = len(concept_ids)
    if len(concept_ids) > context_window:
        starting_index, end_index, demographic_tokens = random_slice_gpt_sequence(
            concept_ids=concept_ids, max_seq_len=context_window
        )
        pat_seq = demographic_tokens + pat_seq[starting_index:end_index]

    [start_year, start_age, start_gender, start_race] = pat_seq[:START_TOKEN_SIZE]

    try:
        start_year = int(start_year.split(":")[1])
        start_age = int(start_age.split(":")[1])
        birth_year = start_year - start_age
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert {pat_seq[:START_TOKEN_SIZE]} due to {e}, skipping to the next record"
        )

    group_events_by_date = defaultdict(list)
    date_cursor = datetime.date(year=start_year, month=1, day=1)
    for clinical_event in pat_seq[START_TOKEN_SIZE:]:
        if is_att_token(clinical_event) or is_inpatient_att_token(clinical_event):
            day_delta = extract_time_interval_in_days(clinical_event)
            date_cursor += datetime.timedelta(days=day_delta)
        else:
            if clinical_event in concept_mapping:
                group_events_by_date[date_cursor].append(
                    concept_mapping[clinical_event]
                )

    narrative = (
        f"Patient Demographics:\n\tGender: {start_gender}\n\tRace: {start_race}\n"
    )
    starting_date = datetime.date(year=int(start_year), month=1, day=1)
    for current_date in sorted(group_events_by_date):
        age = current_date.year - birth_year
        narrative += f"\nOn day {(current_date - starting_date).days} (Date: {current_date}) (Age: {age})\n"
        narrative += "\n".join(
            [
                f"\t{i + 1}. {event}"
                for i, event in enumerate(group_events_by_date[current_date])
            ]
        )
    return narrative, starting_index, end_index


def main(args):
    pat_seq_dataset = load_parquet_as_dataset(args.patient_sequence_dir)
    concept_pd = pd.read_parquet(args.concept_dir)
    concept_mapping = generate_concept_map(concept_pd)

    transformed_pat_seq_dataset = apply_cehrbert_dataset_mapping(
        pat_seq_dataset,
        mapping_function=GeneratePatientNarrativeMapping(
            context_window=args.context_window, concept_mapping=concept_mapping
        ),
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        streaming=args.streaming,
    )
    for batched_dataset_df in transformed_pat_seq_dataset.to_pandas(
        batch_size=args.batch_size, batched=True
    ):
        batched_dataset_df.to_parquet(
            os.path.join(args.output_path, str(uuid.uuid4()), ".parquet")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate patient narratives")
    parser.add_argument(
        "--patient_sequence_dir", required=True, help="The patient sequence data dir"
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
