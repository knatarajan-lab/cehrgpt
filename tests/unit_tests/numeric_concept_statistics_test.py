import unittest

import numpy as np
from scipy.interpolate import UnivariateSpline

from cehrgpt.models.tokenization_hf_cehrgpt import (
    NA,
    UNKNOWN_BIN,
    NumericEventStatistics,
    create_numeric_concept_unit_mapping,
)


class TestNumericEventStatistics(unittest.TestCase):
    def setUp(self):
        self.spline = UnivariateSpline(list(range(10)), list(range(10)))
        self.spline_2 = UnivariateSpline(list(range(10)), list(range(100, 110)))
        # Mock lab stats data
        self.lab_stats = [
            {
                "concept_id": "concept_1",
                "unit": "unit_1",
                "mean": 10.0,
                "std": 2.0,
                "value_outlier_std": 3.0,
                "bins": [
                    {
                        "bin_index": 0,
                        "spline": self.spline,
                        "start_val": 0,
                        "end_val": 10,
                    }
                ],
                "count": 100,
            },
            {
                "concept_id": "concept_2",
                "unit": "unit_1",
                "mean": 20.0,
                "std": 5.0,
                "value_outlier_std": 2.5,
                "bins": [
                    {
                        "bin_index": 0,
                        "spline": self.spline_2,
                        "start_val": 100,
                        "end_val": 110,
                    }
                ],
                "count": 100,
            },
            {
                "concept_id": "concept_2",
                "unit": "unit_2",
                "mean": 15.0,
                "std": 3.0,
                "value_outlier_std": 2.0,
                "bins": [
                    {
                        "bin_index": 0,
                        "spline": self.spline,
                        "start_val": 0,
                        "end_val": 10,
                    }
                ],
                "count": 200,
            },
        ]
        # Create an instance of NumericEventStatistics
        self.numeric_event_statistics = NumericEventStatistics(self.lab_stats)

    def test_create_numeric_concept_unit_mapping(self):
        # Call the function
        concept_prob_mapping, concept_unit_mapping = (
            create_numeric_concept_unit_mapping(self.lab_stats)
        )

        # Check the concept_prob_mapping
        # For concept_1: Only one unit, so probability = 1.0
        self.assertEqual(concept_prob_mapping["concept_1"], [1.0])

        # For concept_2: Two units, unit_1 with count 100, unit_2 with count 200
        total_count_concept_2 = 100 + 200
        expected_probs_concept_2 = [
            100 / total_count_concept_2,
            200 / total_count_concept_2,
        ]
        self.assertEqual(concept_prob_mapping["concept_2"], expected_probs_concept_2)

        # Check the concept_unit_mapping
        self.assertEqual(concept_unit_mapping["concept_1"], ("unit_1",))
        self.assertEqual(concept_unit_mapping["concept_2"], ("unit_1", "unit_2"))

    def test_get_numeric_concept_ids(self):
        # Test for correct concept IDs
        expected_concept_ids = ["concept_1", "concept_2", "concept_2"]
        result = self.numeric_event_statistics.get_numeric_concept_ids()
        self.assertEqual(result, expected_concept_ids)

    def test_get_random_unit(self):
        # Test get_random_unit method for concept_1 (single unit)
        unit = self.numeric_event_statistics.get_random_unit("concept_1")
        self.assertEqual(unit, "unit_1")

        # Test get_random_unit method for concept_2 (multiple units)
        unit = self.numeric_event_statistics.get_random_unit("concept_2")
        self.assertIn(unit, ["unit_1", "unit_2"])

        # Test get_random_unit method for non-existent concept_3
        unit = self.numeric_event_statistics.get_random_unit("concept_3")
        self.assertEqual(unit, NA)

    def test_normalize(self):
        # Test normalization for concept_1, unit_1
        normalized_value = self.numeric_event_statistics.normalize(
            "concept_1", "unit_1", 9.0
        )
        # (12 - 10) / 2 = 1.0
        self.assertEqual(normalized_value, "BIN:0")

        # Test normalization with value beyond outlier bound
        normalized_value = self.numeric_event_statistics.normalize(
            "concept_1", "unit_1", 3.0
        )
        # Since (20 - 10) / 2 = 5.0, but it's clipped at value_outlier_std = 3.0
        self.assertEqual(normalized_value, "BIN:0")

        # Test normalization with a concept_id/unit pair not found
        normalized_value = self.numeric_event_statistics.normalize(
            "concept_2", "unit_1", 105.0
        )
        # Since the concept_id/unit doesn't exist, it should return the original value
        self.assertEqual(normalized_value, "BIN:0")

        # Test normalization with a concept_id/unit pair not found
        normalized_value = self.numeric_event_statistics.normalize(
            "concept_2", "unit_1", 9.9
        )
        # Since the concept_id/unit doesn't exist, it should return the original value
        self.assertEqual(normalized_value, UNKNOWN_BIN)

    def test_denormalize(self):
        # Mock np.random.choice to return a predictable unit
        np.random.choice = lambda *args, **kwargs: "unit_1"
        np.random.seed(42)
        expected_value = self.spline(
            np.random.uniform(self.spline.get_knots()[0], self.spline.get_knots()[-1])
        )
        np.random.seed(42)
        # Test denormalization for concept_1, unit_1
        denormalized_value, unit = self.numeric_event_statistics.denormalize(
            "concept_1", "BIN:0"
        )
        # value = 1.0 * 2.0 + 10.0 = 12.0
        self.assertEqual(denormalized_value, expected_value)
        self.assertEqual(unit, "unit_1")

        np.random.seed(42)
        expected_value = self.spline_2(
            np.random.uniform(
                self.spline_2.get_knots()[0], self.spline_2.get_knots()[-1]
            )
        )
        np.random.seed(42)
        # Test denormalization for concept_2, unit_1
        denormalized_value, unit = self.numeric_event_statistics.denormalize(
            "concept_2", "BIN:0"
        )
        self.assertEqual(denormalized_value, expected_value)
        self.assertEqual(unit, "unit_1")


# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()
