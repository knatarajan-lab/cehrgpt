import math
import unittest

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


class TestMergeLabStats(unittest.TestCase):

    def test_merge_lab_stats(self):
        # Sample data for existing lab stats
        lab_stats_existing = [
            {
                "concept_id": 101,
                "unit": "mg/dL",
                "mean": 5.0,
                "std": 2.0,
                "count": 10,
                "lower_bound": 3.0,
                "upper_bound": 7.0,
            },
            {
                "concept_id": 102,
                "unit": "mg/dL",
                "mean": 7.5,
                "std": 1.5,
                "count": 15,
                "lower_bound": 6.0,
                "upper_bound": 9.0,
            },
        ]

        # Sample data for new lab stats
        lab_stats_new = [
            {
                "concept_id": 101,
                "unit": "mg/dL",
                "mean": 6.0,
                "std": 1.0,
                "count": 5,
                "lower_bound": 4.5,
                "upper_bound": 7.5,
            },
            {
                "concept_id": 103,
                "unit": "mmol/L",
                "mean": 8.0,
                "std": 1.0,
                "count": 8,
                "lower_bound": 6.5,
                "upper_bound": 9.5,
            },
        ]

        # Expected output after merging
        expected_output = [
            {
                "concept_id": 101,
                "unit": "mg/dL",
                "mean": (10 * 5.0 + 5 * 6.0) / 15,
                "std": math.sqrt(3.2222),
                "count": 15,
                "lower_bound": 3.0,
                "upper_bound": 7.5,
            },
            {
                "concept_id": 102,
                "unit": "mg/dL",
                "mean": 7.5,
                "std": 1.5,
                "count": 15,
                "lower_bound": 6.0,
                "upper_bound": 9.0,
            },
            {
                "concept_id": 103,
                "unit": "mmol/L",
                "mean": 8.0,
                "std": 1.0,
                "count": 8,
                "lower_bound": 6.5,
                "upper_bound": 9.5,
            },
        ]

        # Perform the merge operation
        result = CehrGptTokenizer.merge_lab_stats(lab_stats_existing, lab_stats_new)

        # Sort results and expected output by concept_id and unit for comparison
        result_sorted = sorted(result, key=lambda x: (x["concept_id"], x["unit"]))
        expected_output_sorted = sorted(
            expected_output, key=lambda x: (x["concept_id"], x["unit"])
        )

        # Check if the result matches the expected output
        for res, exp in zip(result_sorted, expected_output_sorted):
            self.assertAlmostEqual(res["mean"], exp["mean"], places=4)
            self.assertAlmostEqual(res["std"], exp["std"], places=4)
            self.assertEqual(res["count"], exp["count"])
            self.assertEqual(res["lower_bound"], exp["lower_bound"])
            self.assertEqual(res["upper_bound"], exp["upper_bound"])


if __name__ == "__main__":
    unittest.main()
