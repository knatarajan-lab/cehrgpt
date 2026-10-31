import os
import tempfile
import unittest
from datetime import datetime, timedelta

import polars as pl

from cehrgpt.analysis.counterfactual import diagnose_original_sequences as diagnose

DAY = 86_400


def record(events):
    """events: (day, token); the first one is the index drug.  Returns concept_ids, epoch_times."""
    return [token for _, token in events], [float(day * DAY) for day, _ in events]


# index drug on day 0 at position 1 (one leading token), refilled on day 20, a visit on day 400
EVENTS = [
    (0, "[VS]"), (0, "drug"), (0, "[VE]"),
    (20, "[VS]"), (20, "drug"), (20, "[VE]"),
    (400, "[VS]"), (400, "x"), (400, "[VE]"),
]


class PatientDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.concept_ids, self.epoch_times = record(EVENTS)

    def diagnostics(self, prefix_length=2, **kwargs):
        return diagnose.patient_diagnostics(self.concept_ids, self.epoch_times, prefix_length, "drug", **kwargs)

    def test_the_window_holds_the_tokens_after_the_index_drug_within_a_year(self):
        result = self.diagnostics()
        self.assertEqual(result["tokens_after_index"], 7)
        self.assertEqual(result["tokens_in_window"], 4)  # [VE] of the index visit and the visit of day 20
        self.assertEqual(result["visits_in_window"], 1)
        self.assertEqual(result["events_in_window"], 1)  # the refill; [VS] and [VE] are not events
        self.assertEqual(result["event_days_in_window"], 1)
        self.assertEqual(result["same_drug_repeats"], 1)
        self.assertTrue(result["index_token_is_drug"])

    def test_the_follow_up_ends_with_the_last_observation(self):
        self.assertAlmostEqual(self.diagnostics()["follow_up_days"], 400.0)

    def test_the_follow_up_ends_at_the_end_token(self):
        concept_ids, epoch_times = record(EVENTS[:6] + [(100, "[END]")])
        result = diagnose.patient_diagnostics(concept_ids, epoch_times, 2, "drug")
        self.assertAlmostEqual(result["follow_up_days"], 100.0)

    def test_a_budget_smaller_than_the_window_is_binding(self):
        result = self.diagnostics(budget_lower=2, budget_upper=10)
        self.assertTrue(result["needs_more_than_lower_budget"])
        self.assertFalse(result["needs_more_than_upper_budget"])
        # two tokens after the index drug reach the end of the index visit
        self.assertAlmostEqual(result["days_covered_lower_budget"], 20.0)
        self.assertAlmostEqual(result["days_covered_upper_budget"], 400.0)

    def test_a_prefix_that_does_not_fit_the_record_gives_nothing(self):
        self.assertIsNone(self.diagnostics(prefix_length=100))
        self.assertIsNone(self.diagnostics(prefix_length=0))

    def test_a_prefix_that_does_not_end_with_the_drug_is_reported(self):
        self.assertFalse(self.diagnostics(prefix_length=3)["index_token_is_drug"])


class GeneratedTest(unittest.TestCase):
    def trajectories(self):
        start = datetime(2020, 1, 1)
        rows = [
            # patient 1, arm_a: keeps the drug, refills on day 10 and day 200, dies on day 250, an event follows
            (1, "arm_a", 0, 0, "x"), (1, "arm_a", 0, 10, "drug"), (1, "arm_a", 0, 200, "drug"),
            (1, "arm_a", 0, 250, "[DEATH]"), (1, "arm_a", 0, 300, "late"),
            # patient 2, arm_a: one event, no repeat of the drug
            (2, "arm_a", 0, 5, "x"),
        ]
        return pl.DataFrame(
            {
                "subject_id": [r[0] for r in rows],
                "arm": [r[1] for r in rows],
                "trajectory_id": [r[2] for r in rows],
                "prediction_time": [start] * len(rows),
                "time": [start + timedelta(days=r[3]) for r in rows],
                "code": [r[4] for r in rows],
            }
        ).lazy()

    def test_the_events_after_death_are_left_out_and_counted(self):
        arm_drugs = pl.DataFrame({"arm": ["arm_a"] * 2, "person_id": [1, 2], "drug_concept_id": ["drug", "drug"]})
        drug_info = pl.DataFrame({"person_id": [1, 2], "drug_concept_id": ["drug", "other"], "arm": ["source", "comparator"]})
        result = diagnose.generated_diagnostics(self.trajectories(), arm_drugs, drug_info, 365.0, 30).sort("subject_id")
        first, second = result.to_dicts()
        self.assertEqual(first["last_event_day"], 250.0)
        self.assertEqual(first["events_in_window"], 3)  # x and the two refills; [DEATH] is not an event
        self.assertEqual(first["same_drug_repeats"], 2)
        self.assertEqual(first["first_repeat_day"], 10.0)
        self.assertFalse(first["era_is_one_prescription"])
        self.assertTrue(first["has_death"])
        self.assertEqual(first["events_after_death"], 1)
        self.assertEqual(first["group"], "actual")
        self.assertEqual(second["group"], "swapped")  # the drug of the arm is not the drug the patient received
        self.assertTrue(second["era_is_one_prescription"])
        summary = diagnose.summarize_generated(result, 365.0)
        self.assertEqual(set(summary), {"arm_a / actual", "arm_a / swapped"})
        self.assertIn("arm_a / actual", diagnose.format_generated_report(summary, 365.0))


class CollectTest(unittest.TestCase):
    def test_the_patients_are_listed_by_the_arm_they_received(self):
        with tempfile.TemporaryDirectory() as folder:
            concept_ids, epoch_times = record(EVENTS)
            sequences = pl.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "concept_ids": [concept_ids] * 3,
                    "epoch_times": [epoch_times] * 3,
                }
            )
            sequences.write_parquet(os.path.join(folder, "sequences.parquet"))
            drug_info = pl.DataFrame(
                {
                    "person_id": [1, 2],
                    "drug_concept_id": ["drug", "drug"],
                    "drug_epoch_time": [0.0, 0.0],
                    "era_end_epoch_time": [30.0 * DAY, None],
                    "last_observed_epoch_time": [400.0 * DAY] * 2,
                    "arm": ["source", "comparator"],
                }
            )
            drug_info.write_parquet(os.path.join(folder, "drug_info.parquet"))
            info = diagnose.read_drug_info(os.path.join(folder, "drug_info.parquet"), 30, {"source": "thiazides", "comparator": "acei"})
            self.assertTrue(info.filter(pl.col("person_id") == 1)["era_is_one_prescription"].item())
            self.assertFalse(info.filter(pl.col("person_id") == 2)["era_ended"].item())

            result = diagnose.collect_diagnostics(
                [os.path.join(folder, "sequences.parquet")], {1: 2, 2: 2, 3: 2}, info, 365.0, 2048, 1024
            )
            self.assertEqual(sorted(result["person_id"].to_list()), [1, 2])  # 3 has no drug_info row
            summary = diagnose.summarize(result, 365.0)
            self.assertEqual(set(summary), {"thiazides", "acei"})
            self.assertEqual(summary["thiazides"]["share_era_is_one_prescription"], 1.0)
            self.assertEqual(summary["acei"]["share_era_ended"], 0.0)
            self.assertEqual(summary["acei"]["share_with_full_follow_up"], 1.0)
            self.assertIn("thiazides", diagnose.format_report(summary, 365.0, "budgets"))


if __name__ == "__main__":
    unittest.main()
