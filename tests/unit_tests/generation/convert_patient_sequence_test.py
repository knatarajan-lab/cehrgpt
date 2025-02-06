import unittest

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    PatientSequenceConverter,
)
from cehrgpt.generation.cehrgpt_patient.typed_tokens import translate_to_cehrgpt_tokens


class TestConvertPatientSequence(unittest.TestCase):
    def test_case_one(self):
        """Test the function with typical input."""
        concept_ids = [
            "year:2021",
            "age:50",
            "8532",
            "8527",
            "[VS]",
            "9202",
            "concept1",
            "[VE]",
            "D10",
            "[VS]",
            "9201",
            "concept1",
            "i-D1",
            "i-H1",
            "concept2",
            "8536",
            "[VE]",
        ]
        domain_mapping = {"concept1": "Condition", "concept2": "Condition"}
        self.assertTrue(
            PatientSequenceConverter(
                translate_to_cehrgpt_tokens(concept_ids, domain_mapping)
            ).is_validation_passed
        )

    def test_case_two(self):
        """Test the function with typical input."""
        concept_ids = [
            "year:2021",
            "age:50",
            "0",
            "8527",
            "[VS]",
            "9202",
            "concept1",
            "[VE]",
            "D500",
            "[VS]",
            "[DEATH]",
            "[VE]",
        ]
        domain_mapping = {"concept1": "Condition", "concept2": "Condition"}
        self.assertTrue(
            PatientSequenceConverter(
                translate_to_cehrgpt_tokens(concept_ids, domain_mapping)
            ).is_validation_passed
        )

    def test_failure_cases(self):
        """Test the function with typical input."""
        domain_mapping = {"concept1": "Condition", "concept2": "Condition"}
        concept_ids = [
            "year:2021",
            "age:50",
            "0",
            "9202",
            "[VS]",
            "9202",
            "concept1",
            "[VE]",
            "D500",
            "[VS]",
            "[DEATH]",
            "[VE]",
        ]
        self.assertFalse(
            PatientSequenceConverter(
                translate_to_cehrgpt_tokens(concept_ids, domain_mapping)
            ).is_validation_passed
        )

        concept_ids = [
            "year:2021",
            "age:50",
            "0",
            "8527",
            "[VS]",
            "9202",
            "concept1",
            "[VE]",
            "D500",
            "[VS]",
            "[DEATH]",
        ]
        self.assertFalse(
            PatientSequenceConverter(
                translate_to_cehrgpt_tokens(concept_ids, domain_mapping)
            ).is_validation_passed
        )


if __name__ == "__main__":
    unittest.main()
