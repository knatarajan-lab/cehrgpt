import collections
import math
import pickle
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import femr.stat_utils
import numpy as np
from cehrbert.models.hf_models.tokenization_utils import agg_helper
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert_data.const.artificial_tokens import DEATH_TOKEN
from datasets import Dataset, IterableDataset
from femr.stat_utils import ReservoirSampler
from meds import death_code
from tqdm import tqdm

from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_clinical_event,
)
from cehrgpt.tokenization.tokenization_bin_utils import (
    create_bins_with_spline,
    truncated_sample,
)
from cehrgpt.tokenization.tokenization_constants import (
    DEGREE_OF_FREEDOM,
    NA,
    NUM_OF_BINS,
    SAMPLE_SIZE,
)
from cehrgpt.omop.ontology import Ontology


def get_dataset_len(dataset: Union[Dataset, IterableDataset]) -> int:
    if isinstance(dataset, Dataset):
        return len(dataset)
    elif isinstance(dataset, IterableDataset):
        return sum([1 for _ in dataset])
    raise RuntimeError(
        "The dataset must be one of the two types (Dataset, IterableDataset)"
    )


def get_allowed_motor_codes(
        original_concept_codes: List[str],
        data_args: DataTrainingArguments,
        ontology: Optional[Ontology],
) -> List[str]:
    filtered_original_concept_codes = [
        concept_code
        for concept_code in original_concept_codes
        if is_clinical_event(concept_code, data_args.is_data_in_meds)
    ]
    if ontology:
        allowed_motor_codes = []
        for concept in filtered_original_concept_codes:
            domain = ontology.get_domain(concept)
            if domain and domain in ["Condition", "Procedure", "Drug", "Visit"]:
                allowed_motor_codes.append(concept)
            elif concept in [DEATH_TOKEN, death_code]:
                allowed_motor_codes.append(concept)
        return allowed_motor_codes
    else:
        return list(filtered_original_concept_codes)


def map_motor_tte_statistics(
        batch: Dict[str, Any],
        allowed_motor_codes: List[str],
) -> Dict[str, Any]:
    allowed_motor_codes_set = set(allowed_motor_codes)
    motor_event_times = femr.stat_utils.ReservoirSampler(100_000)
    task_tte_stats: Dict[str, int] = collections.defaultdict(int)
    task_censor_stats: Dict[str, int] = collections.defaultdict(int)
    # Per-task OnlineStatistics for TTE (events only). Used together with
    # task_tte_stats / task_censor_stats to compute a per-task hazard rate
    # for MotorTaskHead bias initialisation:
    #   rate_k = frac_events_k / mean_event_tte_k
    task_tte_time_stats: Dict[str, femr.stat_utils.OnlineStatistics] = collections.defaultdict(
        femr.stat_utils.OnlineStatistics
    )
    for concept_ids, epoch_times in zip(batch["concept_ids"], batch["epoch_times"]):
        # Reverse walk through concept_ids to calculate TTE from each prediction point
        code_time_dict: Dict[str, float] = {}
        num_predictions = 0
        for concept_id, current_time in zip(reversed(concept_ids), reversed(epoch_times)):
            if is_att_token(concept_id) and extract_time_interval_in_days(concept_id) > 0:
                num_predictions += 1
                # Only iterate over observed motor codes — O(observed) not O(all motor codes)
                for motor_code, motor_time in code_time_dict.items():
                    tte = (motor_time - current_time) / 86400
                    motor_event_times.add(tte, 1)
                    task_tte_stats[motor_code] += 1
                    task_tte_time_stats[motor_code].add(1, tte)
            elif concept_id in allowed_motor_codes_set:
                if concept_id not in code_time_dict:
                    # First (nearest) occurrence in reverse walk = last occurrence in forward.
                    # num_predictions is the count of ATTs seen so far, all of which occur
                    # *after* this position in forward time — so the code was censored at each.
                    task_censor_stats[concept_id] += num_predictions
                code_time_dict[concept_id] = current_time

        # Motor codes never observed: every prediction point was censored
        for motor_code in allowed_motor_codes:
            if motor_code not in code_time_dict:
                task_censor_stats[motor_code] += num_predictions

    return {
        "motor_event_times": motor_event_times,
        "task_tte_stats": task_tte_stats,
        "task_censor_stats": task_censor_stats,
        "task_tte_time_stats": dict(task_tte_time_stats),
    }


def compute_motor_tte_statistics(
        dataset: Dataset,
        data_args: DataTrainingArguments,
        allowed_motor_codes: List[str],
        ontology: Optional[Ontology] = None,
) -> Dict[str, Any]:
    map_motor_tte_statistics_partial = partial(
        map_motor_tte_statistics,
        allowed_motor_codes=allowed_motor_codes,
    )
    if data_args.streaming:
        first_example = next(iter(dataset))
        parts = dataset.map(
            partial(agg_helper, map_func=map_motor_tte_statistics_partial),
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=first_example.keys(),
        )
    else:
        parts = dataset.map(
            partial(agg_helper, map_func=map_motor_tte_statistics_partial),
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
            keep_in_memory=True,
            new_fingerprint="invalid",
        )
    current = None
    for stat in tqdm(parts, desc="Aggregating the MOTOR TTE statistics"):
        fixed_stat = pickle.loads(stat["data"])
        if current is None:
            current = fixed_stat
        else:
            current["motor_event_times"].combine(fixed_stat["motor_event_times"])
            for k, v in fixed_stat["task_tte_stats"].items():
                current["task_tte_stats"][k] += v
            for k, v in fixed_stat["task_censor_stats"].items():
                current["task_censor_stats"][k] += v
            for k, v in fixed_stat["task_tte_time_stats"].items():
                if k in current["task_tte_time_stats"]:
                    current["task_tte_time_stats"][k].combine(v)
                else:
                    current["task_tte_time_stats"][k] = v

    # Aggregate the counts for the parent concepts
    if ontology is not None:
        for k in list(current["task_tte_stats"].keys()):
            for parent in ontology.get_all_parents(k):
                if parent != k:
                    current["task_tte_stats"][parent] += current["task_tte_stats"][k]
        for k in list(current["task_censor_stats"].keys()):
            for parent in ontology.get_all_parents(k):
                if parent != k:
                    current["task_censor_stats"][parent] += current[
                        "task_censor_stats"
                    ][k]
        for k in list(current["task_tte_time_stats"].keys()):
            for parent in ontology.get_all_parents(k):
                if parent != k:
                    if parent in current["task_tte_time_stats"]:
                        current["task_tte_time_stats"][parent].combine(
                            current["task_tte_time_stats"][k]
                        )
                    else:
                        import copy
                        current["task_tte_time_stats"][parent] = copy.deepcopy(
                            current["task_tte_time_stats"][k]
                        )
    return current


def agg_statistics(stats1, stats2):
    if stats1.get("numeric_stats_by_lab"):
        for k, v in stats2["numeric_stats_by_lab"].items():
            stats1["numeric_stats_by_lab"][k].combine(v)
    if stats1.get("categorical_stats_by_lab"):
        for (concept_id, concept_as_value), count in stats2[
            "categorical_stats_by_lab"
        ].items():
            stats1["categorical_stats_by_lab"][(concept_id, concept_as_value)] += count
    if stats1.get("concept_code_stats"):
        for concept_id, weight in stats2["concept_code_stats"].items():
            stats1["concept_code_stats"][concept_id] += weight
    if stats1.get("gender_list"):
        stats1.get("gender_list").update(stats2.get("gender_list"))
    if stats1.get("race_list"):
        stats1.get("race_list").update(stats2.get("race_list"))
    return stats1


def map_statistics(batch: Dict[str, Any], total_size, size=10_000) -> Dict[str, Any]:
    if "units" in batch:
        batch_value_units = batch["units"]
    else:
        batch_value_units = [[NA for _ in cons] for cons in batch["concept_ids"]]

    if "number_as_values" not in batch:
        batched_number_as_values = [
            [value if isinstance(value, float) else None for value in concept_values]
            for concept_values in batch["concept_values"]
        ]
    else:
        batched_number_as_values = batch["number_as_values"]

    if "concept_as_values" not in batch:
        batched_concept_as_values = [
            [value if isinstance(value, str) else None for value in concept_values]
            for concept_values in batch["concept_values"]
        ]
    else:
        batched_concept_as_values = batch["concept_as_values"]

    numeric_stats_by_lab = collections.defaultdict(partial(ReservoirSampler, size=size))
    categorical_stats_by_lab = collections.defaultdict(int)
    concept_code_stats = collections.defaultdict(int)
    gender_list = set()
    race_list = set()
    for (
            concept_ids,
            number_as_values,
            concept_as_values,
            concept_value_indicators,
            units,
    ) in zip(
        batch["concept_ids"],
        batched_number_as_values,
        batched_concept_as_values,
        batch["concept_value_masks"],
        batch_value_units,
    ):
        # Collecting demographics
        gender, race = concept_ids[2:4]
        gender_list.add(gender)
        race_list.add(race)

        unique_codes = set()
        for (
                concept_id,
                number_as_value,
                concept_as_value,
                concept_value_indicator,
                unit,
        ) in zip(
            concept_ids,
            number_as_values,
            concept_as_values,
            concept_value_indicators,
            units,
        ):
            if concept_value_indicator == 1:
                if number_as_value:
                    numeric_stats_by_lab[(concept_id, unit)].add(number_as_value, 1)
                if concept_as_value:
                    categorical_stats_by_lab[(concept_id, concept_as_value)] += 1
            unique_codes.add(concept_id)

        for code in unique_codes:
            concept_code_stats[code] += 1 / total_size

    return {
        "numeric_stats_by_lab": numeric_stats_by_lab,
        "categorical_stats_by_lab": categorical_stats_by_lab,
        "concept_code_stats": concept_code_stats,
        "gender_list": gender_list,
        "race_list": race_list,
    }


def compute_statistics(
        dataset: Dataset,
        data_args: DataTrainingArguments,
        ontology: Optional[Ontology] = None,
) -> Dict[str, Any]:
    total = get_dataset_len(dataset)
    map_statistics_partial = partial(map_statistics, total_size=total, size=SAMPLE_SIZE)
    if data_args.streaming:
        first_example = next(iter(dataset))
        parts = dataset.map(
            partial(agg_helper, map_func=map_statistics_partial),
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=first_example.keys(),
        )
    else:
        parts = dataset.map(
            partial(agg_helper, map_func=map_statistics_partial),
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
            keep_in_memory=True,
            new_fingerprint="invalid",
        )
    current = None
    for stat in tqdm(parts, desc="Aggregating the lab statistics"):
        fixed_stat = pickle.loads(stat["data"])
        if current is None:
            current = fixed_stat
        else:
            current = agg_statistics(current, fixed_stat)

    numeric_lab_stats = []
    for (concept_id, unit), online_stats in current["numeric_stats_by_lab"].items():
        if len(online_stats.samples) == 0:
            continue
        samples = truncated_sample(online_stats.samples, data_args.value_outlier_std)
        bins = create_bins_with_spline(samples, NUM_OF_BINS, DEGREE_OF_FREEDOM)
        if len(bins) > 0:
            numeric_lab_stats.append(
                {
                    "concept_id": concept_id,
                    "unit": unit,
                    "mean": np.mean(samples),
                    "std": np.std(samples),
                    "count": len(online_stats.samples),
                    "value_outlier_std": data_args.value_outlier_std,
                    "bins": bins,
                }
            )

    categorical_lab_stats = collections.defaultdict(int)
    for (concept_id, value_as_concept), count in current[
        "categorical_stats_by_lab"
    ].items():
        categorical_lab_stats[(concept_id, value_as_concept)] += count

    all_concept_code_stats = collections.defaultdict(float)
    for concept_id, count in current["concept_code_stats"].items():
        if ontology is not None:
            parents = ontology.get_all_parents(concept_id)
            for parent in parents:
                all_concept_code_stats[parent] += count
        else:
            all_concept_code_stats[concept_id] += count

    all_concept_code_entropies = collections.defaultdict(float)
    for concept_id, weight in all_concept_code_stats.items():
        baseline = (
            min(
                [1]
                + [
                    all_concept_code_stats[parent]
                    for parent in ontology.get_parents(concept_id)
                ]
            )
            if ontology is not None
            else 1
        )
        weight = weight / baseline
        weight = min(1.0, weight)
        if weight != 0 and weight != 1:
            weight = baseline * (
                    weight * math.log(weight) + (1 - weight) * math.log(1 - weight)
            )
            all_concept_code_entropies[concept_id] = weight

    return {
        "numeric_lab_stats": numeric_lab_stats,
        "categorical_lab_stats": categorical_lab_stats,
        "original_concept_codes": list(current["concept_code_stats"].keys()),
        "all_concept_code_stats": all_concept_code_stats,
        "all_concept_code_entropies": all_concept_code_entropies,
        "gender_list": current["gender_list"],
        "race_list": current["race_list"],
        "total": total,
    }


def create_numeric_concept_unit_mapping(
        lab_stats: List[Dict[str, Any]]
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    numeric_concept_unit_mapping = collections.defaultdict(list)
    for each_lab_stat in lab_stats:
        numeric_concept_unit_mapping[each_lab_stat["concept_id"]].append(
            (each_lab_stat["count"], each_lab_stat["unit"])
        )

    concept_prob_mapping = dict()
    concept_unit_mapping = dict()
    for concept_id in numeric_concept_unit_mapping.keys():
        counts, units = zip(*numeric_concept_unit_mapping[concept_id])
        total_count = sum(counts)
        probs = [float(c) / total_count for c in counts]
        concept_prob_mapping[concept_id] = probs
        concept_unit_mapping[concept_id] = units
    return concept_prob_mapping, concept_unit_mapping
