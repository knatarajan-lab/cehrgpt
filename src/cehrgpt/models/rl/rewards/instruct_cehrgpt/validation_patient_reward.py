from typing import List, Optional, Tuple

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptPatient

from ..reward_function_base import RewardFunction


class PatientValidationReward(RewardFunction):
    def get_reward(
        self,
        query: str,
        cehrgpt_patient: Optional[CehrGptPatient],
        encoder_age_concept_prompt_tuples: List[Tuple[int, int, Optional[int]]] = None,
        **kwargs,
    ) -> float:
        return -1.0 if cehrgpt_patient is None else 0.0
