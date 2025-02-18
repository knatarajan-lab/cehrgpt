from dataclasses import dataclass
from typing import Dict, List, Tuple

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token
from cehrgpt.tools.generate_causal_patient_split_by_age import age_group_func


@dataclass(frozen=True)
class DemographicGroup:
    age_group: str
    race: str
    gender: str


def reward_co_occurrence(
    prompts: List[List[str]], completions: List[List[str]], **kwargs
) -> List[float]:
    co_occurrence_list: List[
        Tuple[int, Dict[DemographicGroup, Dict[Tuple[str, str], float]]]
    ]
    co_occurrence_list = kwargs.get("co_occurrence_matrices")
    time_windows, co_occurrence_matrices = zip(*co_occurrence_list)
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

                for z, time_window in enumerate(time_windows):
                    if time_interval <= time_window:
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
        age, race, gender = prompt[1:4]
        demographic_group = DemographicGroup(age_group_func(age), race, gender)
        if demographic_group in length_stats:
            q1 = length_stats[demographic_group].get("q1")
            q3 = length_stats[demographic_group].get("q3")
            if q1 <= len(completion) <= q3:
                reward += 1.0
        rewards.append(reward)
    return rewards
