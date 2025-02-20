import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.tools.generate_causal_patient_split_by_age import age_group_func


@dataclass(frozen=True)
class DemographicGroup:
    age_group: str
    race: str
    gender: str


def reward_valid_sequences(
    prompts: List[List[str]], completions: List[List[str]], **kwargs
) -> List[float]:
    rewards = []
    concept_domain_map = kwargs.get("concept_domain_map")
    for prompt, completion in zip(prompts, completions):
        pat_seq = prompt + completion
        cehrgpt_patient_converter = get_cehrgpt_patient_converter(
            pat_seq, concept_domain_map
        )
        rewards.append(1.0 if cehrgpt_patient_converter.is_validation_passed else 0.0)
    return rewards


def reward_co_occurrence(
    prompts: List[List[str]], completions: List[List[str]], **kwargs
) -> List[float]:
    co_occurrence_list: List[
        Tuple[int, int, Dict[DemographicGroup, Dict[Tuple[str, str], float]]]
    ]
    co_occurrence_list = kwargs.get("co_occurrence_matrices")
    time_window_starts, time_windows, co_occurrence_matrices = zip(*co_occurrence_list)
    rewards = []
    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        age, gender, race = prompt[1:4]
        demographic_group = DemographicGroup(age_group_func(age), race, gender)
        for i, current_concept_id in enumerate(completion):
            if not current_concept_id.isnumeric():
                continue
            time_interval = 0
            for j in range(i + 1, len(completion)):
                future_concept_id = completion[j]
                if is_att_token(future_concept_id):
                    time_interval += extract_time_interval_in_days(future_concept_id)
                    continue

                if not future_concept_id.isnumeric():
                    continue

                for z, (time_window_start, time_window) in enumerate(
                    zip(time_window_starts, time_windows)
                ):
                    if (
                        time_window_start
                        <= time_interval
                        <= time_window_start + time_window
                    ):
                        co_occurrence = co_occurrence_matrices[z].get(
                            demographic_group, None
                        )
                        if co_occurrence:
                            reward += co_occurrence.get(
                                (current_concept_id, future_concept_id), 0.0
                            )
        rewards.append(reward / len(completion))
    return rewards


def reward_length(
    prompts: List[List[str]], completions: List[List[str]], **kwargs
) -> List[float]:
    length_stats: Dict[DemographicGroup, Dict[str, float]]
    length_stats = kwargs.get("length_stats")
    rewards = []
    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        age, gender, race = prompt[1:4]
        demographic_group = DemographicGroup(age_group_func(age), race, gender)
        if demographic_group in length_stats:
            log_mean = length_stats[demographic_group].get("log_mean")
            log_std = length_stats[demographic_group].get("log_std")
            if not pd.isnull(log_std) and not pd.isnull(log_mean):
                log_seq_length = math.log(len(completion))
                reward += np.exp(-np.abs(log_seq_length - log_mean) / log_std)
        rewards.append(reward)
    return rewards


def reward_concept_prevalence(
    prompts: List[List[str]], completions: List[List[str]], **kwargs
) -> List[float]:
    concept_prevalence: Dict[DemographicGroup, Dict[str, float]]
    concept_prevalence = kwargs.get("concept_prevalence")
    rewards = []
    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        age, gender, race = prompt[1:4]
        demographic_group = DemographicGroup(age_group_func(age), race, gender)
        if demographic_group in concept_prevalence:
            demographic_concept_prevalence = concept_prevalence[demographic_group]
            for concept_id in completion:
                if (
                    not is_visit_start(concept_id)
                    and not is_visit_end(concept_id)
                    and not is_att_token(concept_id)
                ):
                    reward += demographic_concept_prevalence.get(concept_id, 1e-9)
            reward = reward / len(completion)
        rewards.append(reward)
    return rewards
