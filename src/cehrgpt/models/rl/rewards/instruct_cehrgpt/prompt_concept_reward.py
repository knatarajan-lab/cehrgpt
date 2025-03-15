from typing import Dict, List, Optional, Tuple

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptPatient

from ..reward_function_base import RewardFunction


class PromptConceptReward(RewardFunction):
    """
    A reward function class for evaluating the relevance and accuracy of patient data against specified.

    medical conditions and associated medications using ancestral relationships and ingredient mappings.

    Attributes:
        ancestor_descendent_map (Dict[str, List[str]]): Maps condition concept IDs to lists of their
                                                        descendant concept IDs.
        ingredient_to_drug_map (Dict[str, List[str]]): Maps ingredient concept IDs to lists of branded
                                                       drugs that contain these ingredients.
    """

    def __init__(
        self,
        ancestor_descendent_map: Dict[str, List[str]],
        ingredient_to_drug_map: Dict[str, List[str]],
    ):
        """
        Initializes a new instance of the PromptConceptRewards class.

        Args:
            ancestor_descendent_map (Dict[str, List[str]]): Maps condition concept IDs to their descendants.
            ingredient_to_drug_map (Dict[str, List[str]]): Maps ingredient concept IDs to their branded drugs.
        """
        super().__init__()
        self.ancestor_descendent_map = ancestor_descendent_map
        self.ingredient_to_drug_map = ingredient_to_drug_map

    def get_reward(
        self,
        query: str,
        cehrgpt_patient: Optional[CehrGptPatient],
        encoder_age_concept_prompt_tuples: List[Tuple[int, int, Optional[int]]] = None,
        **kwargs,
    ) -> float:
        """
        Computes a reward based on the match between the patient's medical records and the specified medical conditions.

        and medications.

        Args:
            query (str): Query string related to the patient's scenario.
            cehrgpt_patient (CehrGptPatient): An object containing detailed patient medical records.
            encoder_age_concept_prompt_tuples (List[Tuple[int, int, Optional[int]]], optional): A list of tuples,
                each with an age of diagnosis, condition concept ID, and optionally a drug concept ID, used to verify
                patient conditions and drugs.

        Returns:
            float: Calculated reward based on the match accuracy of the patient's events to the specified conditions
                   and drugs.
        """
        reward = 0.0
        if cehrgpt_patient and encoder_age_concept_prompt_tuples:
            for age, condition_id, drug_id in encoder_age_concept_prompt_tuples:
                if not condition_id:
                    continue
                condition_matches = self.ancestor_descendent_map.get(
                    str(condition_id), []
                ) + [str(condition_id)]
                drug_matches = (
                    self.ingredient_to_drug_map.get(str(drug_id), []) if drug_id else []
                )
                is_condition_matched, is_drug_matched = self.check_patient_data(
                    cehrgpt_patient, condition_matches, drug_matches, age
                )
                reward += self.calculate_reward(
                    is_condition_matched, is_drug_matched, bool(drug_matches)
                )
        return reward

    @staticmethod
    def check_patient_data(
        patient: CehrGptPatient,
        conditions: List[str],
        drugs: List[str],
        age_of_diagnosis: int,
    ) -> Tuple[bool, bool]:
        """
        Helper method to check patient data against conditions and drugs.

        Returns:
            Tuple[bool, bool]: Tuple indicating if conditions and drugs are found in patient data.
        """
        is_condition_found, is_drug_found = False, not drugs
        birth_year = patient.birth_datetime.year
        for event in patient.get_events():
            if (
                not is_condition_found
                and event.domain.lower() == "condition"
                and event.code in conditions
            ):
                if abs(event.time.year - birth_year - age_of_diagnosis) <= 5:
                    is_condition_found = True
            if (
                not is_drug_found
                and event.domain.lower() == "drug"
                and event.code in drugs
            ):
                is_drug_found = True
            if is_condition_found and is_drug_found:
                break
        return is_condition_found, is_drug_found

    @staticmethod
    def calculate_reward(
        condition_matched: bool, drug_matched: bool, drugs_specified: bool
    ) -> float:
        """
        Calculates the reward based on matching results.

        Returns:
            float: Reward value based on matching conditions and drugs.
        """
        if condition_matched and drug_matched:
            return 1.0 if drugs_specified else 0.8
        elif condition_matched:
            return 0.5
        return -1.0
