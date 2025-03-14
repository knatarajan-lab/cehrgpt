import datetime
import unittest

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import (
    CehrGptEvent,
    CehrGptPatient,
    CehrGptVisit,
)
from cehrgpt.models.rl.rewards import CovidReward


class TestCovidReward(unittest.TestCase):
    def setUp(self):
        self.concept_name_map = {"37311061": "covid-19 viral infection"}
        self.reward_function = CovidReward(concept_name_map=self.concept_name_map)

    def test_covid_condition_pre_2020(self):
        """Tests if the COVID condition before 2020 correctly affects the reward."""
        patient = CehrGptPatient(
            birth_datetime=datetime.datetime(1985, 1, 1),
            gender_concept_id=0,
            race_concept_id=0,
            gender="Male",
            race="Caucasian",
            visits=[
                CehrGptVisit(
                    visit_type="Inpatient",
                    visit_concept_id=9201,
                    visit_start_datetime=datetime.datetime(2019, 12, 31),
                    events=[
                        CehrGptEvent(
                            time=datetime.datetime(2019, 12, 31),
                            code="37311061",
                            domain="condition",
                        )
                    ],
                )
            ],
        )
        encoder_tuples = [(34, 37311061, None)]
        reward = self.reward_function.get_reward("", patient, encoder_tuples)
        self.assertEqual(reward, -1.0)

    def test_covid_condition_post_2020(self):
        """Tests if the COVID condition post 2020 does not affect the reward."""
        patient = CehrGptPatient(
            birth_datetime=datetime.datetime(1985, 1, 1),
            gender_concept_id=0,
            race_concept_id=0,
            gender="Male",
            race="Caucasian",
            visits=[
                CehrGptVisit(
                    visit_type="Inpatient",
                    visit_concept_id=9201,
                    visit_start_datetime=datetime.datetime(2021, 1, 1),
                    events=[
                        CehrGptEvent(
                            time=datetime.datetime(2021, 1, 1),
                            code="37311061",
                            domain="condition",
                        )
                    ],
                )
            ],
        )
        encoder_tuples = [(36, 37311061, None)]
        reward = self.reward_function.get_reward("", patient, encoder_tuples)
        self.assertEqual(reward, 0.0)

    def test_non_covid_condition(self):
        """Tests that non-COVID conditions do not affect the reward."""
        patient = CehrGptPatient(
            birth_datetime=datetime.datetime(1985, 1, 1),
            gender_concept_id=0,
            race_concept_id=0,
            gender="Male",
            race="Caucasian",
            visits=[
                CehrGptVisit(
                    visit_type="Inpatient",
                    visit_concept_id=9201,
                    visit_start_datetime=datetime.datetime(2021, 1, 1),
                    events=[
                        CehrGptEvent(
                            time=datetime.datetime(2021, 1, 1),
                            code="123456",
                            domain="condition",
                        )
                    ],
                )
            ],
        )
        encoder_tuples = [(36, 123456, None)]
        reward = self.reward_function.get_reward("", patient, encoder_tuples)
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
