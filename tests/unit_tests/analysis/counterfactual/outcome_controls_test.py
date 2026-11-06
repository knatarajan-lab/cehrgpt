import math
import tempfile
import unittest
from pathlib import Path

import polars as pl

from cehrgpt.analysis.counterfactual import hazard_ratio_estimation as hr

DAY = 86_400
START = 1_600_000_000.0


class ParseOutcomeSpecsTest(unittest.TestCase):
    def test_primary_outcomes_alone(self):
        ids, types, expected = hr.parse_outcome_specs("1, 2")
        self.assertEqual(ids, ["1", "2"])
        self.assertEqual(set(types.values()), {hr.PRIMARY})
        self.assertEqual(expected, {})

    def test_negative_controls_expect_a_hazard_ratio_of_one(self):
        ids, types, expected = hr.parse_outcome_specs("1", negative_controls="2,3")
        self.assertEqual(ids, ["1", "2", "3"])
        self.assertEqual(types["2"], hr.NEGATIVE_CONTROL)
        self.assertEqual(expected, {"2": 1.0, "3": 1.0})

    def test_positive_controls_take_an_optional_expected_hazard_ratio(self):
        _, types, expected = hr.parse_outcome_specs("1", positive_controls="4:1.5, 5")
        self.assertEqual(types["4"], hr.POSITIVE_CONTROL)
        self.assertEqual(types["5"], hr.POSITIVE_CONTROL)
        self.assertEqual(expected, {"4": 1.5})

    def test_an_outcome_cannot_have_two_roles(self):
        with self.assertRaisesRegex(ValueError, "both"):
            hr.parse_outcome_specs("1", negative_controls="1")
        with self.assertRaisesRegex(ValueError, "both"):
            hr.parse_outcome_specs("1", negative_controls="2", positive_controls="2:2")

    def test_a_non_positive_expected_hazard_ratio_raises(self):
        with self.assertRaises(ValueError):
            hr.parse_outcome_specs("1", positive_controls="4:0")


def summary(rows):
    return pl.DataFrame(
        rows,
        schema={
            "outcome_concept_id": pl.String,
            "hr": pl.Float64,
            "hr_lower_95": pl.Float64,
            "hr_upper_95": pl.Float64,
        },
        orient="row",
    )


class ControlSummaryTest(unittest.TestCase):
    def setUp(self):
        _, self.types, self.expected = hr.parse_outcome_specs(
            "1", negative_controls="2,3,4", positive_controls="5:2.0,6"
        )
        self.labelled = hr.label_outcomes(
            summary(
                [
                    ("1", 1.3, 1.1, 1.5),
                    ("2", 1.0, 0.8, 1.2),
                    ("3", 1.4, 1.2, 1.6),
                    ("4", 0.9, 0.7, 1.1),
                    ("5", 1.8, 1.5, 2.1),
                    ("6", 1.5, 1.2, 1.9),
                ]
            ),
            self.types,
            self.expected,
        )

    def row(self, outcome_id):
        return self.labelled.filter(pl.col("outcome_concept_id") == outcome_id).row(0, named=True)

    def test_outcomes_are_labelled_by_type(self):
        self.assertEqual(self.row("1")["outcome_type"], hr.PRIMARY)
        self.assertEqual(self.row("2")["outcome_type"], hr.NEGATIVE_CONTROL)
        self.assertEqual(self.row("5")["outcome_type"], hr.POSITIVE_CONTROL)
        self.assertEqual(self.labelled.columns[:4], [
            "outcome_concept_id", "outcome_type", "expected_hr", "ci_covers_expected"
        ])

    def test_the_interval_is_checked_against_the_expected_hazard_ratio(self):
        self.assertIsNone(self.row("1")["ci_covers_expected"])
        self.assertTrue(self.row("2")["ci_covers_expected"])
        self.assertFalse(self.row("3")["ci_covers_expected"])
        self.assertTrue(self.row("5")["ci_covers_expected"])
        self.assertIsNone(self.row("6")["ci_covers_expected"])

    def test_negative_controls_are_summarised_by_coverage_and_bias(self):
        controls = hr.summarize_controls(self.labelled)
        negative = controls.filter(pl.col("outcome_type") == hr.NEGATIVE_CONTROL).row(0, named=True)
        self.assertEqual(negative["n_controls"], 3)
        self.assertEqual(negative["n_ci_covers_expected"], 2)
        logs = [math.log(1.0), math.log(1.4), math.log(0.9)]
        self.assertAlmostEqual(negative["mean_log_hr_ratio"], sum(logs) / 3)

    def test_positive_controls_without_an_expected_value_do_not_count_towards_coverage(self):
        controls = hr.summarize_controls(self.labelled)
        positive = controls.filter(pl.col("outcome_type") == hr.POSITIVE_CONTROL).row(0, named=True)
        self.assertEqual(positive["n_controls"], 2)
        self.assertEqual(positive["n_with_expected_hr"], 1)
        self.assertEqual(positive["n_ci_covers_expected"], 1)

    def test_primary_only_analyses_have_no_control_summary(self):
        _, types, expected = hr.parse_outcome_specs("1")
        labelled = hr.label_outcomes(summary([("1", 1.3, 1.1, 1.5)]), types, expected)
        self.assertTrue(hr.summarize_controls(labelled).is_empty())

    def test_the_summaries_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            written = hr.write_labelled_summary(
                [r for r in self.labelled.select(["outcome_concept_id", "hr", "hr_lower_95", "hr_upper_95"]).to_dicts()],
                folder / "summary.csv",
                folder / "controls.csv",
                self.types,
                self.expected,
            )
            self.assertEqual(len(pl.read_csv(folder / "summary.csv")), 6)
            self.assertEqual(len(pl.read_csv(folder / "controls.csv")), 2)
        self.assertIn("outcome_type", written.columns)


class OverlapDropTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_dropped_patients_are_read_from_every_scores_file(self):
        for name, ids, retained in (("a", [1, 2], [True, False]), ("b", [3, 4], [False, True])):
            pl.DataFrame({"person_id": ids, "retained": retained}).write_parquet(
                self.root / f"{name}.parquet"
            )
        dropped = hr.read_dropped_person_ids(
            [str(self.root / "a.parquet"), str(self.root / "b.parquet")]
        )
        self.assertEqual(dropped, {2, 3})

    def test_the_observed_baseline_drops_the_same_patients(self):
        persons = list(range(1, 13))
        arm = ["source"] * 6 + ["comparator"] * 6
        pl.DataFrame(
            {
                "person_id": persons,
                "arm": arm,
                "era_end_epoch_time": [None] * 12,
                "last_observed_epoch_time": [START + 400 * DAY] * 12,
            },
            schema_overrides={"era_end_epoch_time": pl.Float64},
        ).write_parquet(self.root / "drug_info.parquet")
        events = [None if d is None else START + d * DAY for d in (10, 20, 30, 15, 25, None, 40, 50, None, 60, None, None)]
        pl.DataFrame(
            {"person_id": persons, "drug_epoch_time": [START] * 12, "99": events},
            schema_overrides={"99": pl.Float64},
        ).write_parquet(self.root / "observed.parquet")

        def counts(excluded):
            folder = self.root / ("out_with" if excluded else "out_without")
            folder.mkdir()
            hr.analyse_observed_outcomes(
                observed_outcomes_path=str(self.root / "observed.parquet"),
                drug_info_path=str(self.root / "drug_info.parquet"),
                outcome_concept_ids=["99"],
                follow_up_days=365,
                output_dir=folder,
                excluded_person_ids=excluded,
                outcome_types={"99": hr.NEGATIVE_CONTROL},
                expected_hr={"99": 1.0},
            )
            return pl.read_csv(folder / "observed_hazard_ratio_summary.csv").row(0, named=True)

        everyone = counts(None)
        trimmed = counts({1, 2, 7})
        self.assertEqual((everyone["n_acei"], everyone["n_thiazide"]), (6, 6))
        self.assertEqual((trimmed["n_acei"], trimmed["n_thiazide"]), (4, 5))
        self.assertEqual(trimmed["outcome_type"], hr.NEGATIVE_CONTROL)


if __name__ == "__main__":
    unittest.main()
