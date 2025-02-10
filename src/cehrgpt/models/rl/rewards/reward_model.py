import inspect
from typing import Dict, List, Optional, Tuple

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    PatientSequenceConverter,
    get_cehrgpt_patient_converter,
)

from .reward_function_base import RewardFunction


class CEHRGPTRewardModel:
    def __init__(self, **kwargs):
        self.reward_functions = []
        for subclass in RewardFunction.__subclasses__():
            sig = inspect.signature(subclass.__init__)
            # Filter kwargs to match only the keys that exist in the subclass's constructor
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            # Instantiate the subclass with the filtered kwargs
            instance = subclass(**filtered_kwargs)
            self.reward_functions.append(instance)

    def get_reward(
        self,
        query: str,
        patient_sequence: List[str],
        encoder_age_concept_prompt_tuples: List[Tuple[int, int, Optional[int]]],
        concept_name_map: Dict[str, str],
        concept_domain_map: Dict[str, str],
    ) -> float:
        patient_seq_converter: PatientSequenceConverter = get_cehrgpt_patient_converter(
            patient_sequence, concept_domain_map
        )
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(
                query,
                patient_seq_converter.get_patient(concept_domain_map, concept_name_map),
                encoder_age_concept_prompt_tuples=encoder_age_concept_prompt_tuples,
            )
        return reward
