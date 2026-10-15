import collections
import logging
import math
import pickle
import time
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import femr.stat_utils
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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

LOG = logging.getLogger(__name__)


def get_dataset_len(dataset: Union[Dataset, IterableDataset]) -> int:
    if isinstance(dataset, Dataset):
        return len(dataset)
    elif isinstance(dataset, IterableDataset):
        return sum([1 for _ in dataset])
    raise RuntimeError(
        "The dataset must be one of the two types (Dataset, IterableDataset)"
    )


# OMOP's "No matching concept". It stands in for a missing value (an unknown discharge
# destination or race), not for something that happened, so it is never a MOTOR task.
NO_MATCHING_CONCEPT_ID = "0"


def get_allowed_motor_codes(
        original_concept_codes: List[str],
        data_args: DataTrainingArguments,
        ontology: Optional[Ontology],
        measurement_concept_ids: Optional[Set[str]] = None,
) -> List[str]:
    filtered_original_concept_codes = [
        concept_code
        for concept_code in original_concept_codes
        if concept_code != NO_MATCHING_CONCEPT_ID
        and is_clinical_event(concept_code, data_args.is_data_in_meds)
    ]
    if ontology:
        allowed_motor_codes = []
        for concept in filtered_original_concept_codes:
            domain = ontology.get_domain(concept)
            if domain and domain in ["Condition", "Procedure", "Drug", "Visit"]:
                allowed_motor_codes.append(concept)
            elif concept in [DEATH_TOKEN, death_code]:
                allowed_motor_codes.append(concept)
    else:
        allowed_motor_codes = list(filtered_original_concept_codes)

    if measurement_concept_ids:
        allowed_motor_codes = [
            concept for concept in allowed_motor_codes
            if concept not in measurement_concept_ids
        ]
    return allowed_motor_codes


def get_motor_code_ancestors(
        allowed_motor_codes: List[str],
        ontology: Optional[Ontology],
) -> Dict[str, Tuple[str, ...]]:
    """
    For every allowed code, the code itself and all its ancestors in the ontology.

    An event of a code is an event of each of these, which is also how the training labels are
    built (see CehrGptTokenizer.get_motor_parents). Without an ontology a code only covers
    itself.
    """
    if ontology is None:
        return {code: (code,) for code in allowed_motor_codes}
    return {
        code: tuple(sorted(ontology.get_all_parents(code) | {code}))
        for code in allowed_motor_codes
    }


def map_motor_tte_statistics(
        batch: Dict[str, Any],
        allowed_motor_codes: List[str],
        ancestors_by_code: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Dict[str, Any]:
    """
    Count events and censoring per MOTOR code, for a batch of patients.

    Every ATT token that spans a positive interval is a prediction point. A code has an event
    at a prediction point if it, or any code it covers (a descendant when ancestors_by_code is
    given), occurs later, and is censored otherwise. Counting the covered codes here, per
    prediction point, makes an ancestor count once per prediction point however many of its
    descendants occur, which adding the children's counts together afterwards would not.

    Args:
        batch: sequences with concept_ids and epoch_times
        allowed_motor_codes: the codes that can be tasks
        ancestors_by_code: code -> the code and its ancestors (see get_motor_code_ancestors);
            None means every code only covers itself
    """
    allowed_motor_codes_set = set(allowed_motor_codes)
    if ancestors_by_code is None:
        ancestors_by_code = {code: (code,) for code in allowed_motor_codes}
    # Every code that gets statistics: the allowed codes and their ancestors
    all_codes = sorted({a for code in allowed_motor_codes for a in ancestors_by_code[code]})
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
        # code (or one of its descendants) -> time of its nearest occurrence after the point
        code_time_dict: Dict[str, float] = {}
        # The allowed codes themselves, for the global sample of event times: this stays at
        # the level of the codes that occur, whatever the ontology
        token_time_dict: Dict[str, float] = {}
        num_predictions = 0
        for concept_id, current_time in zip(reversed(concept_ids), reversed(epoch_times)):
            if is_att_token(concept_id) and extract_time_interval_in_days(concept_id) > 0:
                num_predictions += 1
                # Only iterate over observed codes: O(observed), not O(all motor codes)
                for code, code_time in code_time_dict.items():
                    tte = (code_time - current_time) / 86400
                    task_tte_stats[code] += 1
                    task_tte_time_stats[code].add(1, tte)
                for token_time in token_time_dict.values():
                    motor_event_times.add((token_time - current_time) / 86400, 1)
            elif concept_id in allowed_motor_codes_set:
                token_time_dict[concept_id] = current_time
                for code in ancestors_by_code[concept_id]:
                    if code not in code_time_dict:
                        # First (nearest) occurrence in reverse walk = last occurrence in
                        # forward. num_predictions is the count of ATTs seen so far, all of
                        # which occur *after* this position in forward time, so the code was
                        # censored at each of them.
                        task_censor_stats[code] += num_predictions
                    code_time_dict[code] = current_time

        # Codes never observed: every prediction point was censored
        for code in all_codes:
            if code not in code_time_dict:
                task_censor_stats[code] += num_predictions

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
    # The statistics of an ancestor are counted directly (see map_motor_tte_statistics), so
    # there is nothing to add up afterwards
    map_motor_tte_statistics_partial = partial(
        map_motor_tte_statistics,
        allowed_motor_codes=allowed_motor_codes,
        ancestors_by_code=get_motor_code_ancestors(allowed_motor_codes, ontology),
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
    if stats1.get("concept_patient_counts"):
        for concept_id, count in stats2["concept_patient_counts"].items():
            stats1["concept_patient_counts"][concept_id] += count
    for token, count in stats2.get("concept_token_counts", {}).items():
        stats1["concept_token_counts"][token] += count
    for token, count in stats2.get("value_token_counts", {}).items():
        stats1["value_token_counts"][token] += count
    if stats1.get("gender_list"):
        stats1.get("gender_list").update(stats2.get("gender_list"))
    if stats1.get("race_list"):
        stats1.get("race_list").update(stats2.get("race_list"))
    return stats1


class _AncestorIndex:
    """
    A concept-to-ancestors mapping stored as flat NumPy arrays.

    This is equivalent to a ``dict[str, tuple[str, ...]]``, but is substantially cheaper for
    ``datasets.map`` to hash and pickle for each worker. It follows the same sorted-key/offset
    representation used by ``_PredictionTimeIndex``.
    """

    def __init__(self, ancestors_by_code: Dict[str, Tuple[str, ...]]):
        concept_ids = sorted(ancestors_by_code, key=int)
        offsets = [0]
        ancestors: List[int] = []
        for concept_id in concept_ids:
            ancestors.extend(int(ancestor) for ancestor in ancestors_by_code[concept_id])
            offsets.append(len(ancestors))
        self._concept_ids = np.fromiter(
            (int(concept_id) for concept_id in concept_ids),
            dtype=np.int64,
            count=len(concept_ids),
        )
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._ancestors = np.asarray(ancestors, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._concept_ids)

    @property
    def relationship_count(self) -> int:
        return len(self._ancestors)

    def get(self, concept_id: str, default=()):
        if not isinstance(concept_id, str) or not concept_id.isnumeric():
            return default
        numeric_concept_id = int(concept_id)
        position = int(np.searchsorted(self._concept_ids, numeric_concept_id))
        if (
            position >= len(self._concept_ids)
            or self._concept_ids[position] != numeric_concept_id
        ):
            return default
        start, end = self._offsets[position], self._offsets[position + 1]
        return [str(ancestor) for ancestor in self._ancestors[start:end].tolist()]

    def __getitem__(self, concept_id: str) -> List[str]:
        missing = object()
        result = self.get(concept_id, missing)
        if result is missing:
            raise KeyError(concept_id)
        return result

    def __contains__(self, concept_id: str) -> bool:
        missing = object()
        return self.get(concept_id, missing) is not missing


def get_ancestors_by_code(ontology: Optional[Ontology]) -> _AncestorIndex:
    """
    For every concept of the ontology that has parents, all its ancestors (not the concept itself).

    Used to count a patient once for every ancestor of the concepts the patient has. It is a
    flat-array index so that it can be sent cheaply to dataset-map worker processes.
    """
    if ontology is None:
        return _AncestorIndex({})
    ancestors: Dict[str, Tuple[str, ...]] = {}
    # A pruned ontology remembers which concepts actually occurred in the dataset. Ancestor-only
    # nodes never appear in a patient sequence, so including them in this worker lookup only makes
    # the multiprocessing callback more expensive to hash, pickle, copy, and retain.
    concept_ids = getattr(ontology, "dataset_concept_ids", None)
    if not concept_ids:
        concept_ids = ontology.parents_map.keys()
    for concept_id in concept_ids:
        found = tuple(sorted(a for a in ontology.get_all_parents(concept_id) if a != concept_id))
        if found:
            ancestors[concept_id] = found
    return _AncestorIndex(ancestors)


def map_statistics(
    batch: Dict[str, Any],
    total_size,
    size=10_000,
    ancestors_by_code: Optional[_AncestorIndex] = None,
    collect_token_counts: bool = False,
) -> Dict[str, Any]:
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
    # Iterable datasets cannot expose their Arrow columns for the optimized counting path below,
    # so collect their tokenizer counts as part of this single streaming pass.
    concept_token_counts = collections.Counter()
    value_token_counts = collections.Counter()
    # The number of patients (rows) that have a token, or a descendant of it when an ontology is
    # given, however often they have it
    concept_patient_counts = collections.defaultdict(int)
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

        if collect_token_counts:
            concept_token_counts.update(
                value for value in concept_ids if isinstance(value, str)
            )
            value_token_counts.update(
                value for value in concept_as_values if isinstance(value, str)
            )

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
        # A patient counts once for a concept when the patient has the concept or any descendant of
        # it. The ancestors are collected in a set first: adding up the counts of the children would
        # count a patient with two children twice.
        covered = set(unique_codes)
        if ancestors_by_code:
            for code in unique_codes:
                covered.update(ancestors_by_code.get(code, ()))
        for code in covered:
            concept_patient_counts[code] += 1

    return {
        "numeric_stats_by_lab": numeric_stats_by_lab,
        "categorical_stats_by_lab": categorical_stats_by_lab,
        "concept_code_stats": concept_code_stats,
        "concept_patient_counts": concept_patient_counts,
        "concept_token_counts": concept_token_counts,
        "value_token_counts": value_token_counts,
        "gender_list": gender_list,
        "race_list": race_list,
    }


def count_string_list_values(dataset: Dataset, feature_name: str) -> Dict[str, int]:
    """Count strings in an Arrow list column without a Python-level dataset map."""
    flattened = pc.list_flatten(dataset.data.column(feature_name))
    if not (pa.types.is_string(flattened.type) or pa.types.is_large_string(flattened.type)):
        return {}
    return {
        row["values"]: row["counts"]
        for row in pc.value_counts(flattened).to_pylist()
        if row["values"] is not None
    }


def compute_statistics(
        dataset: Dataset,
        data_args: DataTrainingArguments,
        ontology: Optional[Ontology] = None,
) -> Dict[str, Any]:
    total = get_dataset_len(dataset)
    ancestor_lookup_start = time.perf_counter()
    ancestors_by_code = get_ancestors_by_code(ontology)
    if ontology is not None:
        LOG.info(
            "Prepared ancestor lookup with %s observed concepts and %s relationships in %.2f seconds",
            len(ancestors_by_code),
            ancestors_by_code.relationship_count,
            time.perf_counter() - ancestor_lookup_start,
        )
    map_statistics_partial = partial(
        map_statistics,
        total_size=total,
        size=SAMPLE_SIZE,
        ancestors_by_code=ancestors_by_code,
        collect_token_counts=data_args.streaming,
    )
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

    if data_args.streaming:
        concept_token_counts = dict(current["concept_token_counts"])
        value_token_counts = dict(current["value_token_counts"])
    else:
        concept_token_counts = count_string_list_values(dataset, "concept_ids")
        value_feature = (
            "concept_as_values"
            if "concept_as_values" in dataset.column_names
            else "concept_values"
        )
        value_token_counts = count_string_list_values(dataset, value_feature)

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
        # How many patients have each concept, or a descendant of it when an ontology is given. A
        # patient is counted once however many descendants they have, unlike the prevalence in
        # all_concept_code_stats, which adds the descendants' prevalences up
        "concept_patient_counts": dict(current["concept_patient_counts"]),
        "concept_token_counts": concept_token_counts,
        "value_token_counts": value_token_counts,
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
