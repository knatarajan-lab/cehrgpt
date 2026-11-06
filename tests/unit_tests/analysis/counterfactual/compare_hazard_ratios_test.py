import tempfile
import unittest
from pathlib import Path

import polars as pl

from cehrgpt.analysis.counterfactual import compare_hazard_ratios as compare
from cehrgpt.analysis.counterfactual.study_definition import outcome_order

A, B = "thz", "acei"


def summary(rows, observed=False):
    """rows: (label, events_a, events_b, hr, lower, upper)"""
    return pl.DataFrame(
        {
            "outcome_concept_id": [r[0] for r in rows],
            f"n_events_{A}": [r[1] for r in rows],
            f"n_events_{B}": [r[2] for r in rows],
            "hr": [r[3] for r in rows],
            "hr_lower_95": [r[4] for r in rows],
            "hr_upper_95": [r[5] for r in rows],
        }
    )


class OrderTest(unittest.TestCase):
    def test_the_legend_study_lists_its_primary_outcomes_before_the_safety_outcomes(self):
        order = outcome_order("legend_hypertension")
        self.assertEqual(order["primary"], ["Acute myocardial infarction", "Hospitalization with heart failure", "Stroke"])
        self.assertEqual(order["safety"][:3], ["All-cause mortality", "Cardiovascular-related mortality", "Chest pain or angina"])
        self.assertNotIn("Stroke", order["safety"])

    def test_the_outcomes_of_the_study_come_in_that_order_and_the_controls_last(self):
        outcomes = compare.ordered_outcomes("legend_hypertension")
        names = [o[1] for o in outcomes]
        self.assertEqual(names[:4], ["Acute myocardial infarction", "Hospitalization with heart failure", "Stroke", "All-cause mortality"])
        groups = [o[2] for o in outcomes]
        self.assertEqual(groups[:3], ["primary"] * 3)
        self.assertLess(groups.index("safety"), groups.index("other"))
        self.assertLess(groups.index("other"), groups.index("negative_control"))
        self.assertIn("Venous thromboembolic events", names)  # the trailing blank of the file is gone


class ComparisonTest(unittest.TestCase):
    outcomes = [("stroke", "Stroke", "primary"), ("cough", "Cough", "safety"), ("rash", "Rash", "negative_control")]

    def test_the_rows_follow_the_order_and_join_both_summaries(self):
        generated = summary([("cough", 5, 9, 0.8, 0.5, 1.3), ("stroke", 10, 12, 0.9, 0.6, 1.4), ("rash", 3, 3, 1.0, 0.5, 2.0), ("999", 1, 1, 1.0, 0.1, 9.0)])
        observed = summary([("stroke", 100, 120, 0.85, 0.7, 1.0), ("cough", 50, 90, 0.7, 0.5, 0.9)])
        table = compare.build_comparison(generated, observed, A, B, self.outcomes)
        self.assertEqual(table["outcome"].to_list(), ["Stroke", "Cough", "Rash", "999"])
        self.assertEqual(table["group"].to_list(), ["primary", "safety", "negative_control", "other"])
        stroke = table.row(0, named=True)
        self.assertAlmostEqual(stroke["generated_over_observed"], 0.9 / 0.85)
        self.assertTrue(stroke["generated_ci_contains_observed"])
        self.assertTrue(table.row(1, named=True)["generated_ci_contains_observed"])
        self.assertIsNone(table.row(2, named=True)["observed_hr"])

    def test_the_console_table_has_the_primary_outcomes_first(self):
        generated = summary([("cough", 5, 9, 0.8, 0.5, 1.3), ("stroke", 10, 12, 0.9, 0.6, 1.4)])
        observed = summary([("stroke", 100, 120, 0.85, 0.7, 1.0), ("cough", 50, 90, 0.7, 0.5, 0.9)])
        text = compare.format_comparison(compare.build_comparison(generated, observed, A, B, self.outcomes), A, B)
        self.assertLess(text.index("Primary outcomes"), text.index("Stroke"))
        self.assertLess(text.index("Stroke"), text.index("Safety outcomes"))
        self.assertLess(text.index("Safety outcomes"), text.index("Cough"))
        self.assertIn("0.85 (0.70-1.00)", text)

    def test_the_comparison_is_written_next_to_the_summaries(self):
        with tempfile.TemporaryDirectory() as folder:
            summary([("stroke", 10, 12, 0.9, 0.6, 1.4)]).write_csv(Path(folder) / compare.GENERATED_FILE)
            self.assertIsNone(compare.compare(folder, A, B, "legend_hypertension"))  # no observed summary yet
            summary([("stroke", 100, 120, 0.85, 0.7, 1.0)]).write_csv(Path(folder) / compare.OBSERVED_FILE)
            table = compare.compare(folder, A, B, "legend_hypertension")
            self.assertEqual(table["outcome"].to_list(), ["Stroke"])
            self.assertTrue((Path(folder) / compare.COMPARISON_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
