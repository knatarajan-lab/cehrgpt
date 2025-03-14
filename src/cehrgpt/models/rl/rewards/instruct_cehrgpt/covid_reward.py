import re
from typing import Dict, List, Optional, Tuple

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptPatient

from ..reward_function_base import RewardFunction

COVID_CONCEPT_ID = ["37311061"]


class CovidReward(RewardFunction):
    """
    A specialized reward function class that evaluates a CEHR-GPT patient scenario for mentions of COVID-related conditions.

    and modifies the reward based on specific criteria.

    Attributes:
        concept_name_map (Dict[str, str]): A dictionary mapping concept IDs to their corresponding names,
                                           used to lookup and evaluate conditions in the patient data.

    Methods:
        get_reward: Calculates a reward based on the occurrence and context of COVID-related conditions
                    in a patient's history.
    """

    def __init__(
        self,
        concept_name_map: Dict[str, str],
    ):
        """
        Initializes the CovidRewards instance with a concept name mapping.

        Args:
            concept_name_map (Dict[str, str]): A dictionary mapping concept IDs to concept names.
        """
        super().__init__()
        self.concept_name_map = concept_name_map

    def get_reward(
        self,
        query: str,
        cehrgpt_patient: Optional[CehrGptPatient],
        encoder_age_concept_prompt_tuples: List[Tuple[int, int, Optional[int]]] = None,
        **kwargs,
    ) -> float:
        """
        Computes the reward for a given query and patient data based on the presence of COVID-related conditions.

        The method evaluates whether any of the patient's conditions related to COVID (as identified in the
        encoder_age_concept_prompt_tuples) occurred before 2020, applying a penalty if this criterion is met.

        Args:
            query (str): The query string related to the patient's scenario.
            cehrgpt_patient (CehrGptPatient): An object representing detailed patient data.
            encoder_age_concept_prompt_tuples (List[Tuple[int, int, Optional[int]]], optional): A list of tuples where
                each tuple contains an age of diagnosis, a condition concept ID, and an optional drug concept ID.
                This list is used to identify relevant patient conditions and interventions for reward calculation.

        Returns:
            float: The calculated reward based on the analysis of COVID-related conditions in the patient's history.

        Note:
            This function assumes that COVID_CONCEPT_ID and related logic are defined outside of this method,
            potentially as global variables or constants within the module.
        """
        reward = 0.0
        if cehrgpt_patient and encoder_age_concept_prompt_tuples:
            for encoder_age_concept_prompt_tuple in encoder_age_concept_prompt_tuples:
                age_of_diagnosis, condition_concept_id, drug_concept_id = (
                    encoder_age_concept_prompt_tuple
                )
                if not condition_concept_id:
                    continue
                concept_name = self.concept_name_map.get(str(condition_concept_id), "")
                if re.findall(r"covid", concept_name, re.IGNORECASE):
                    for event in cehrgpt_patient.get_events():
                        if event.code in COVID_CONCEPT_ID and event.time.year < 2020:
                            reward -= 1.0
        return reward
