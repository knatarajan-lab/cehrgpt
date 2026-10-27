import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import polars as pl

from cehrgpt.analysis.counterfactual import hazard_ratio_estimation as hr
from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import process
from cehrgpt.analysis.counterfactual.hazard_ratio_estimation import compute_tte, observed_tte

DAY = 86_400
START = 1_600_000_000.0
SOURCE, SOURCE_2, COMPARATOR, EXCLUDED, COMBINATION, HYPERTENSION, OUTCOME = (
    100, 101, 200, 500, 999, 316, 700,
)  # fmt: skip


def patient(person_id, visits):
    """visits: list of (day, [tokens]) in the order given; every visit is an outpatient visit."""
    concept_ids = ["year:2020", "age:50", "8507", "0"]
    epoch_times = [0.0] * 4
    for day, tokens in visits:
        for token in ["[VS]", "9202", *tokens, "[VE]"]:
            concept_ids.append(token)
            epoch_times.append(START + day * DAY)
    n = len(concept_ids)
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "epoch_times": epoch_times,
        "ages": [50] * n,
        "input_ids": list(range(n)),
        "value_indicators": [0] * n,
        "values": [0] * n,
    }


class CohortRulesTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "vocab" / "concept_ancestor").mkdir(parents=True)
        pairs = [(i, i) for i in (SOURCE, SOURCE_2, COMPARATOR, EXCLUDED, HYPERTENSION, OUTCOME)]
        # a combination product with two ingredients
        pairs += [(SOURCE, COMBINATION), (EXCLUDED, COMBINATION), (COMBINATION, COMBINATION)]
        pl.DataFrame(
            {"ancestor_concept_id": [a for a, _ in pairs], "descendant_concept_id": [d for _, d in pairs]}
        ).write_parquet(self.root / "vocab" / "concept_ancestor" / "part.parquet")
        self.output = self.root / "out"

    def kept(self, patients, **kwargs):
        if self.output.exists():
            import shutil

            shutil.rmtree(self.output)
        pl.DataFrame(patients).write_parquet(self.root / "patients.parquet")
        process(
            patient_sequence_path=str(self.root / "patients.parquet"),
            vocab_path=str(self.root / "vocab"),
            source_ingredient_ids=[SOURCE, SOURCE_2],
            comparator_ingredient_ids=[COMPARATOR],
            exclusion_concept_ids=[EXCLUDED],
            output_dir=self.output,
            min_context_length=1,
            **kwargs,
        )
        return pl.read_parquet(str(self.output / "drug_info" / "*.parquet"))["person_id"].sort().to_list()

    def test_a_second_ingredient_in_the_first_week_leaves_the_cohort(self):
        patients = [
            patient(1, [(0, [str(SOURCE)])]),
            patient(2, [(0, [str(SOURCE)]), (5, [str(EXCLUDED)])]),
            patient(3, [(0, [str(SOURCE)]), (10, [str(EXCLUDED)])]),
            patient(4, [(0, [str(SOURCE)]), (3, [str(SOURCE_2)])]),
            patient(5, [(0, [str(SOURCE)]), (3, [str(SOURCE)])]),
            patient(6, [(0, [str(COMPARATOR)]), (7, [str(SOURCE)])]),
            patient(7, [(0, [str(COMPARATOR)]), (8, [str(SOURCE)])]),
        ]
        self.assertEqual(self.kept(patients, monotherapy_window_days=7), [1, 3, 5, 7])

    def test_the_index_visit_counts_as_part_of_the_first_week(self):
        patients = [patient(1, [(0, [str(EXCLUDED), str(SOURCE)])]), patient(2, [(0, [str(SOURCE)])])]
        self.assertEqual(self.kept(patients, monotherapy_window_days=7), [2])

    def test_a_combination_product_counts_for_each_of_its_ingredients(self):
        patients = [patient(1, [(0, [str(COMBINATION)])]), patient(2, [(0, [str(SOURCE)])])]
        self.assertEqual(self.kept(patients, monotherapy_window_days=7), [2])

    def test_exposure_on_an_earlier_date_leaves_the_cohort(self):
        patients = [
            patient(1, [(-30, [str(EXCLUDED)]), (0, [str(SOURCE)])]),
            patient(2, [(0, [str(SOURCE)])]),
        ]
        self.assertEqual(self.kept(patients, monotherapy_window_days=7), [2])

    def test_without_the_rule_a_second_ingredient_after_the_index_visit_is_kept(self):
        patients = [patient(1, [(0, [str(SOURCE)]), (3, [str(SOURCE_2)])])]
        self.assertEqual(self.kept(patients), [1])

    def test_the_diagnosis_has_to_be_in_the_window_before_the_index_date(self):
        patients = [
            patient(1, [(-100, [str(HYPERTENSION)]), (0, [str(SOURCE)])]),
            patient(2, [(-400, [str(HYPERTENSION)]), (0, [str(SOURCE)])]),
            patient(3, [(0, [str(HYPERTENSION), str(SOURCE)])]),
            patient(4, [(0, [str(SOURCE)])]),
            patient(5, [(0, [str(SOURCE)]), (5, [str(HYPERTENSION)])]),
        ]
        windowed = self.kept(
            patients, population_concept_ids=[HYPERTENSION], population_window_days=365
        )
        self.assertEqual(windowed, [1, 3])
        anywhere = self.kept(patients, population_concept_ids=[HYPERTENSION])
        self.assertEqual(anywhere, [1, 2, 3])

    def test_the_outcome_before_the_index_date_is_flagged(self):
        patients = [
            patient(1, [(-50, [str(OUTCOME)]), (0, [str(SOURCE)]), (30, [str(OUTCOME)])]),
            patient(2, [(0, [str(SOURCE)]), (30, [str(OUTCOME)])]),
            patient(3, [(0, [str(OUTCOME), str(SOURCE)])]),
        ]
        self.kept(patients, outcome_concept_ids=[OUTCOME])
        observed = pl.read_parquet(str(self.output / "observed_outcomes" / "*.parquet")).sort("person_id")
        self.assertEqual(observed[f"prior__{OUTCOME}"].to_list(), [True, False, False])
        self.assertEqual(
            observed[str(OUTCOME)].to_list(), [START + 30 * DAY, START + 30 * DAY, None]
        )


INDEX = datetime(2020, 1, 1)
INDEX_EPOCH = INDEX.replace(tzinfo=timezone.utc).timestamp()


def trajectories(rows):
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "trajectory_id": [0] * len(rows),
            "time": [INDEX + timedelta(days=r[1]) for r in rows],
            "code": [r[2] for r in rows],
        }
    ).with_row_index("position")


class TimeAtRiskTest(unittest.TestCase):
    def test_trajectories_with_less_than_the_minimum_time_at_risk_are_left_out(self):
        rows = [(1, 0, "x"), (2, 30, "x"), (3, 1, "x")]
        drug_info = pl.DataFrame({"person_id": [1, 2, 3], "drug_epoch_time": [INDEX_EPOCH] * 3})
        kept = compute_tte(trajectories(rows), drug_info, ["y"], 365, min_days_at_risk=1)
        self.assertEqual(sorted(kept["subject_id"].to_list()), [2, 3])
        everyone = compute_tte(trajectories(rows), drug_info, ["y"], 365)
        self.assertEqual(len(everyone), 3)

    def test_observed_patients_with_less_than_the_minimum_time_at_risk_are_left_out(self):
        start = INDEX_EPOCH
        observed = pl.DataFrame(
            {
                "drug_epoch_time": [start, start],
                "era_end_epoch_time": [start + 0.0, start + 100 * DAY],
                "last_observed_epoch_time": [start + 200 * DAY] * 2,
                "y": [None, start + 50 * DAY],
            },
            schema_overrides={"y": pl.Float64},
        )
        time_days, event = observed_tte(observed, "y", 365, min_days_at_risk=1)
        self.assertEqual((time_days.tolist(), event.tolist()), ([50.0], [1]))

    def test_intent_to_treat_ignores_the_end_of_the_drug_era(self):
        start = INDEX_EPOCH
        observed = pl.DataFrame(
            {
                "drug_epoch_time": [start],
                "era_end_epoch_time": [start + 20 * DAY],
                "last_observed_epoch_time": [start + 200 * DAY],
                "y": [start + 50 * DAY],
            }
        )
        self.assertEqual(observed_tte(observed, "y", 365)[1].tolist(), [0])
        time_days, event = observed_tte(observed, "y", 365, use_era_end=False)
        self.assertEqual((time_days.tolist(), event.tolist()), ([50.0], [1]))


class PriorOutcomeTest(unittest.TestCase):
    def test_patients_with_a_prior_outcome_are_left_out_of_that_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            persons = list(range(1, 61))
            start = INDEX_EPOCH
            pl.DataFrame(
                {
                    "person_id": persons,
                    "drug_concept_id": ["100"] * 60,
                    "drug_epoch_time": [start] * 60,
                    "arm": ["source"] * 60,
                    "era_end_epoch_time": [None] * 60,
                    "last_observed_epoch_time": [start + 400 * DAY] * 60,
                },
                schema_overrides={"era_end_epoch_time": pl.Float64},
            ).write_parquet(root / "drug_info.parquet")
            (root / "observed").mkdir()
            pl.DataFrame(
                {
                    "person_id": persons,
                    "drug_epoch_time": [start] * 60,
                    "600": [None] * 60,
                    "prior__600": [p <= 10 for p in persons],
                },
                schema_overrides={"600": pl.Float64},
            ).write_parquet(root / "observed" / "part.parquet")
            for arm, hit in (("a", 3), ("b", 4)):
                rows = []
                for p in persons:
                    events = [(30, "x"), (200, "x")] + ([(50, "600")] if p % hit == 0 else [])
                    rows += [(p, INDEX + timedelta(days=d), c) for d, c in events]
                arm_dir = root / "traj" / arm / "0"
                arm_dir.mkdir(parents=True)
                pl.DataFrame(
                    {
                        "subject_id": [r[0] for r in rows],
                        "trajectory_id": [0] * len(rows),
                        "time": [r[1] for r in rows],
                        "code": [r[2] for r in rows],
                    }
                ).write_parquet(arm_dir / "batch_00000.parquet")
            argv = [
                "prog", "--trajectories_dir", str(root / "traj"), "--drug_info_path", str(root / "drug_info.parquet"),
                "--outcome_concept_ids", "600", "--prior_outcome_path", str(root / "observed"),
                "--arm_a", "a", "--arm_b", "b", "--follow_up_days", "365", "--output_dir", str(root / "out"),
            ]  # fmt: skip
            with mock.patch("sys.argv", argv):
                hr.main()
            summary = pl.read_csv(root / "out" / "hazard_ratio_summary.csv").row(0, named=True)
            self.assertEqual(summary["n_patients_a"], 50)
            self.assertEqual(summary["n_patients_b"], 50)
            with mock.patch("sys.argv", argv + ["--time_at_risk", "intent_to_treat"]):
                hr.main()

    def test_the_flags_are_read_per_outcome_from_several_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, ids in (("one", [1, 2]), ("two", [3, 4])):
                (root / name).mkdir()
                pl.DataFrame(
                    {"person_id": ids, "prior__a": [True, False], "prior__b": [False, True], "a": [None] * 2}
                ).write_parquet(root / name / "part.parquet")
            prior = hr.read_prior_outcome_persons([str(root / "one"), str(root / "two")])
            self.assertEqual(prior, {"a": {1, 3}, "b": {2, 4}})


if __name__ == "__main__":
    unittest.main()
