import unittest

from scipy.interpolate import UnivariateSpline

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


class TestMergeLabStats(unittest.TestCase):

    def test_merge_lab_stats(self):
        spline = UnivariateSpline(list(range(100)), list(range(100)))
        spline_2 = UnivariateSpline(list(range(100)), list(range(100, 200)))
        # Sample data for existing lab stats
        lab_stats_existing = [
            {
                "concept_id": "101",
                "unit": "mg/dL",
                "mean": 10.0,
                "std": 2.0,
                "value_outlier_std": 3.0,
                "bins": [
                    {
                        "bin_index": 0,
                        "spline": spline,
                        "start_val": 0,
                        "end_val": 100,
                    }
                ],
                "count": 100,
            }
        ]

        # Sample data for new lab stats
        lab_stats_new = [
            {
                "concept_id": "101",
                "unit": "mg/dL",
                "mean": 6.0,
                "std": 1.0,
                "bins": [
                    {
                        "bin_index": 0,
                        "spline": spline_2,
                        "start_val": 100,
                        "end_val": 200,
                    }
                ],
                "count": 100,
            },
            {
                "concept_id": "102",
                "unit": "mmol/L",
                "mean": 8.0,
                "std": 1.0,
                "count": 100,
                "bins": [
                    {
                        "bin_index": 0,
                        "spline": spline_2,
                        "start_val": 100,
                        "end_val": 200,
                    }
                ],
            },
        ]

        # Expected output after merging
        expected_output = [
            {
                "concept_id": "101",
                "unit": "mg/dL",
                "mean": 8,
                "std": 2.5495,
                "count": 200,
            },
            {
                "concept_id": "102",
                "unit": "mg/dL",
                "mean": 8.0,
                "std": 1.0,
                "count": 100,
            },
        ]

        # Perform the merge operation
        result = CehrGptTokenizer.merge_numeric_lab_stats(
            lab_stats_existing, lab_stats_new
        )

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


if __name__ == "__main__":
    unittest.main()
