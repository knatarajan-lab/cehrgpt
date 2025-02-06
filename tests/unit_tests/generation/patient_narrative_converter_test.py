import unittest

from cehrgpt.generation.cehrgpt_patient.patient_narrative_generator import (
    convert_concepts_to_patient_narrative,
)


class TestConvertConceptsToPatientNarrative(unittest.TestCase):
    def test_normal_input(self):
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
            "D500",
            "[VS]",
            "9202",
            "concept1",
            "[VE]",
        ]
        concept_mapping = {
            "concept1": "Diabetes",
            "concept2": "Hypertension",
            "8532": "FEMALE",
            "8527": "White",
            "9202": "Outpatient Visit",
            "9201": "Inpatient Visit",
        }
        domain_mapping = {"concept1": "Condition", "concept2": "Condition"}
        context_window = 100
        expected_narrative = (
            "Patient Demographics:\n"
            "\tGender: FEMALE\n"
            "\tRace: White\n"
            "\nOutpatient Visit on 2021-01-01 (Age: 50)\n"
            "Condition: ['Diabetes']\n"
            "\nInpatient Visit on 2021-01-11 (Age: 50)\n"
            "\tOn day 0:\n"
            "\tCondition: ['Diabetes']\n"
            "\tOn day 1:\n"
            "\tCondition: ['Hypertension']\n"
            "\nOutpatient Visit on 2022-05-27 (Age: 51)\n"
            "Condition: ['Diabetes']\n"
        )
        expected_start_index = 0
        # Reflects the total number of elements in concept_ids if context window is not exceeded
        expected_end_index = len(concept_ids)

        narrative, start_index, end_index = convert_concepts_to_patient_narrative(
            concept_ids, concept_mapping, domain_mapping, context_window
        )

        self.assertEqual(expected_narrative, narrative)
        self.assertEqual(start_index, expected_start_index)
        self.assertEqual(end_index, expected_end_index)


if __name__ == "__main__":
    unittest.main()
