import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cehrbert.models.hf_models.tokenization_utils import agg_helper
from cehrbert.runners.runner_util import load_parquet_as_dataset
from datasets import Dataset
from tqdm import tqdm

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token
from cehrgpt.tools.generate_causal_patient_split_by_age import age_group_func


def nested_defaultdict():
    return defaultdict(int)


def calculate_sequence_length(
    batch: Dict[str, Any],
) -> Dict[Tuple[str, str, str, str], List[int]]:
    batched_concept_ids = batch["concept_ids"]
    sequence_length = defaultdict(list)
    for concept_ids in batched_concept_ids:
        year, patient_age, gender, race = concept_ids[:4]
        age = age_group_func(patient_age)
        sequence_length[(year, age, gender, race)].append(len(concept_ids[4:]))
    return sequence_length


def calculate_co_occurrence_batch_function(
    batch: Dict[str, Any], time_window: Optional[int]
) -> Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], int]]:
    batched_concept_ids = batch["concept_ids"]
    co_occurrence = defaultdict(nested_defaultdict)
    for concept_ids in batched_concept_ids:
        year, patient_age, gender, race = concept_ids[:4]
        age = age_group_func(patient_age)
        clinical_events = concept_ids[4:]

        for i, current_concept_id in enumerate(clinical_events):
            if not current_concept_id.isnumeric():
                continue

            time_interval = 0
            for j in range(i + 1, len(clinical_events)):
                future_concept_id = clinical_events[j]

                if is_att_token(future_concept_id):
                    time_interval += extract_time_interval_in_days(future_concept_id)
                    continue

                if not future_concept_id.isnumeric():
                    continue

                # Increment co-occurrence count
                co_occurrence[(year, age, gender, race)][
                    (current_concept_id, future_concept_id)
                ] += 1

                # Break if the time interval exceeds the time window
                if time_window is not None and time_interval > time_window:
                    break

    return co_occurrence


def create_co_occurrence_stats(dataset: Dataset, args, time_window: Optional[int]):
    parts = dataset.map(
        partial(
            agg_helper,
            map_func=partial(
                calculate_co_occurrence_batch_function, time_window=time_window
            ),
        ),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )
    concept_pair_stats: Dict[
        Tuple[str, str, str, str], Dict[Tuple[str, str], float]
    ] = defaultdict(lambda: defaultdict(int))
    for stat in tqdm(parts, desc="Aggregating the co-occurrence concept counts"):
        fixed_stat = pickle.loads(stat["data"])
        for demographics, stats in fixed_stat.items():
            for concept_pair, count in stats.items():
                concept_pair_stats[demographics][concept_pair] += count
            total_sum = sum(concept_pair_stats[demographics].values())
            for concept_pair, count in concept_pair_stats[demographics].items():
                concept_pair_stats[demographics][concept_pair] = count / total_sum
    with open(
        os.path.join(
            args.output_dir,
            f"matrix_{time_window if time_window else 'lifetime'}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(concept_pair_stats, f)
    return concept_pair_stats


def main(args):
    dataset = load_parquet_as_dataset(args.dataset_dir).filter(
        lambda batched: [
            args.context_window >= num_of_concepts > args.min_num_tokens
            for num_of_concepts in batched["num_of_concepts"]
        ],
        batched=True,
    )

    print("Creating 30 day co-occurrence")
    matrix_30 = create_co_occurrence_stats(dataset, args, time_window=30)
    print("Creating 60 day co-occurrence")
    matrix_60 = create_co_occurrence_stats(dataset, args, time_window=60)
    print("Creating 90 day co-occurrence")
    matrix_90 = create_co_occurrence_stats(dataset, args, time_window=90)
    print("Creating 180 day co-occurrence")
    matrix_180 = create_co_occurrence_stats(dataset, args, time_window=180)
    print("Creating 360 day co-occurrence")
    matrix_360 = create_co_occurrence_stats(dataset, args, time_window=360)
    print("Creating lifetime co-occurrence")
    matrix_lifetime = create_co_occurrence_stats(dataset, args, time_window=None)

    print("Creating paitent sequence length stats")
    parts = dataset.map(
        partial(agg_helper, map_func=calculate_sequence_length),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )
    sequence_aggregate_length: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(
        list
    )
    sequence_length_stats: Dict[Tuple[str, str, str, str], Dict[str, float]] = (
        defaultdict(lambda: defaultdict(float))
    )
    for stat in tqdm(parts, desc="Aggregating the patient sequence length"):
        fixed_stat = pickle.loads(stat["data"])
        for demographics, lengths in fixed_stat.items():
            sequence_aggregate_length[demographics].extend(lengths)
    for demographics, lengths in sequence_aggregate_length.items():
        sequence_length_stats[demographics]["mean"] = np.mean(lengths)
        sequence_length_stats[demographics]["q1"] = np.percentile(lengths, 25)
        sequence_length_stats[demographics]["median"] = np.percentile(lengths, 50)
        sequence_length_stats[demographics]["q3"] = np.percentile(lengths, 75)
        sequence_length_stats[demographics]["p90"] = np.percentile(lengths, 90)

    stats = {
        "matrix_30": matrix_30,
        "matrix_60": matrix_60,
        "matrix_90": matrix_90,
        "matrix_180": matrix_180,
        "matrix_360": matrix_360,
        "matrix_lifetime": matrix_lifetime,
        "sequence_length_stats": sequence_length_stats,
    }

    with open(os.path.join(args.output_dir, "all_statistics.pickle"), "wb") as f:
        pickle.dump(stats, f)


def create_arg_parser():
    import argparse

    base_arg_parser = argparse.ArgumentParser(
        description="Arguments for generating the stats for GRPO"
    )
    base_arg_parser.add_argument(
        "--context_window",
        dest="context_window",
        action="store",
        required=False,
        type=int,
        default=1024,
    )
    base_arg_parser.add_argument(
        "--batch_size",
        dest="batch_size",
        action="store",
        required=False,
        type=int,
        default=1024,
    )
    base_arg_parser.add_argument(
        "--num_proc",
        dest="num_proc",
        action="store",
        required=True,
        type=int,
    )
    base_arg_parser.add_argument(
        "--min_num_tokens",
        dest="min_num_tokens",
        action="store",
        required=False,
        type=int,
        default=20,
    )
    base_arg_parser.add_argument(
        "--dataset_dir",
        dest="dataset_dir",
        action="store",
        help="The path for your dataset",
        required=True,
    )
    base_arg_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        action="store",
        help="The path for your dataset",
        required=True,
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
