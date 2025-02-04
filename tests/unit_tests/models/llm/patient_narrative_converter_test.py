import unittest

from cehrgpt.llm.patient_narrative_converter import (
    convert_concepts_to_patient_narrative,
)


class TestConvertConceptsToPatientNarrative(unittest.TestCase):
    def test_normal_input(self):
        """Test the function with typical input."""
        concept_ids = [
            "year:2021",
            "age:50",
            "Female",
            "Caucasian",
            "[VS]",
            "concept1",
            "[VE]",
            "D10",
            "[VS]",
            "concept1",
            "concept2",
            "[VE]",
            "D500",
            "[VS]",
            "concept1",
            "[VE]",
        ]
        concept_mapping = {"concept1": "Diabetes", "concept2": "Hypertension"}
        context_window = 100
        expected_narrative = (
            "Patient Demographics:\n\tGender: Female\n\tRace: Caucasian\n\n"
            "On day 0 (Date: 2021-01-01) (Age: 50)\n\t1. Diabetes\n"
            "On day 10 (Date: 2021-01-11) (Age: 50)\n\t1. Diabetes\n\t2. Hypertension\n"
            "On day 510 (Date: 2022-05-26) (Age: 51)\n\t1. Diabetes"
        )
        expected_start_index = 0
        expected_end_index = 16  # Reflects the total number of elements in concept_ids if context window is not exceeded

        narrative, start_index, end_index = convert_concepts_to_patient_narrative(
            concept_ids, concept_mapping, context_window
        )

        self.assertEqual(expected_narrative, narrative)
        self.assertEqual(start_index, expected_start_index)
        self.assertEqual(end_index, expected_end_index)


if __name__ == "__main__":
    unittest.main()
