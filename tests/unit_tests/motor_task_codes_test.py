import unittest
from types import SimpleNamespace

from cehrbert_data.const.artificial_tokens import DEATH_TOKEN

from cehrgpt.tokenization.tokenization_statistics import (
    NO_MATCHING_CONCEPT_ID,
    get_allowed_motor_codes,
)

DOMAINS = {
    "0": "Metadata",
    "201826": "Condition",
    "4329847": "Condition",
    "1503297": "Drug",
    "2000": "Procedure",
    "9201": "Visit",
    "3004410": "Measurement",
}


class FakeOntology:
    def get_domain(self, concept_id):
        return DOMAINS.get(concept_id)


DATA_ARGS = SimpleNamespace(is_data_in_meds=False)
CODES = ["[VS]", "D5", "year:2020", "0", "201826", "1503297", "9201", "3004410", DEATH_TOKEN]


class MotorTaskCodesTest(unittest.TestCase):
    def test_no_matching_concept_is_never_a_task(self):
        self.assertEqual(NO_MATCHING_CONCEPT_ID, "0")
        for ontology in (None, FakeOntology()):
            with self.subTest(ontology=ontology is not None):
                codes = get_allowed_motor_codes(CODES, DATA_ARGS, ontology)
                self.assertNotIn("0", codes)

    def test_other_tasks_are_unchanged(self):
        # Without an ontology every clinical event is allowed, death included
        self.assertEqual(
            get_allowed_motor_codes(CODES, DATA_ARGS, None),
            ["201826", "1503297", "9201", "3004410", DEATH_TOKEN],
        )
        # With one, only conditions, procedures, drugs and visits remain, plus death
        self.assertEqual(
            get_allowed_motor_codes(CODES, DATA_ARGS, FakeOntology()),
            ["201826", "1503297", "9201", DEATH_TOKEN],
        )

    def test_measurements_are_still_removed(self):
        codes = get_allowed_motor_codes(
            CODES, DATA_ARGS, None, measurement_concept_ids={"3004410"}
        )
        self.assertNotIn("3004410", codes)
        self.assertNotIn("0", codes)

    def test_other_codes_that_contain_a_zero_are_kept(self):
        codes = get_allowed_motor_codes(["10", "100", "0", "201826"], DATA_ARGS, None)
        self.assertEqual(codes, ["10", "100", "201826"])


if __name__ == "__main__":
    unittest.main()
