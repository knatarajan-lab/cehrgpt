import unittest
from datetime import datetime

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import (
    CehrGptEvent,
    CehrGptPatient,
    CehrGptVisit,
)
from cehrgpt.models.rl.rewards import PromptConceptReward


class TestPromptConceptReward(unittest.TestCase):
    def setUp(self):
        # Maps for testing
        self.ancestor_descendent_map = {"101": ["102", "103"], "201": ["202", "203"]}
        self.ingredient_to_drug_map = {"301": ["302", "303"], "401": ["402", "403"]}

        # Initialize the reward function
        self.reward_function = PromptConceptReward(
            ancestor_descendent_map=self.ancestor_descendent_map,
            ingredient_to_drug_map=self.ingredient_to_drug_map,
        )

        # Create mock patient data
        self.patient = CehrGptPatient(
            birth_datetime=datetime(1990, 1, 1),
            gender_concept_id=8507,
            gender="Male",
            race_concept_id=8527,
            race="Caucasian",
            patient_id=123,
            visits=[
                CehrGptVisit(
                    visit_type="Inpatient",
                    visit_concept_id=9201,
                    visit_start_datetime=datetime(2020, 5, 20),
                    events=[
                        CehrGptEvent(
                            time=datetime(2020, 5, 20), code="101", domain="condition"
                        ),
                        CehrGptEvent(
                            time=datetime(2020, 5, 21), code="302", domain="drug"
                        ),
                    ],
                )
            ],
        )

    def test_reward_calculation_full_match(self):
        """Test the reward calculation when both condition and drug match perfectly."""
        encoder_age_concept_prompt_tuples = [(30, 101, 301)]
        reward = self.reward_function.get_reward(
            "Test Query", self.patient, encoder_age_concept_prompt_tuples
        )
        self.assertEqual(reward, 1.0)

    def test_reward_calculation_no_drug_match(self):
        """Test the reward calculation when the condition matches but no drug match."""
        encoder_age_concept_prompt_tuples = [(30, 101, 401)]  # Non-matching drug ID
        reward = self.reward_function.get_reward(
            "Test Query", self.patient, encoder_age_concept_prompt_tuples
        )
        self.assertEqual(reward, 0.5)

    def test_reward_calculation_no_match(self):
        """Test the reward calculation when neither condition nor drug matches."""
        encoder_age_concept_prompt_tuples = [
            (30, 201, 401)
        ]  # Non-matching condition and drug IDs
        reward = self.reward_function.get_reward(
            "Test Query", self.patient, encoder_age_concept_prompt_tuples
        )
        self.assertEqual(reward, -1.0)


if __name__ == "__main__":
    unittest.main()
