import unittest
import warnings
from datetime import datetime, timedelta, timezone

import polars as pl

from cehrgpt.analysis.counterfactual.hazard_ratio_estimation import (
    compute_follow_up,
    compute_tte,
)

INDEX = datetime(2020, 1, 1, tzinfo=timezone.utc)
TARGET = {"drug"}
COMPETING = {"other_drug"}


def trajectories(events, subject_id=1, trajectory_id=0):
    """events: list of (day since the index drug, code)."""
    return pl.DataFrame(
        {
            "subject_id": [subject_id] * len(events),
            "trajectory_id": [trajectory_id] * len(events),
            "time": [(INDEX + timedelta(days=day)).replace(tzinfo=None) for day, _ in events],
            "code": [code for _, code in events],
        }
    ).with_row_index("position")


def drug_info(*subject_ids):
    return pl.DataFrame(
        {
            "person_id": list(subject_ids),
            "drug_epoch_time": [int(INDEX.timestamp())] * len(subject_ids),
        }
    )


def tte(events, follow_up_days=365, target=None, competing=None, outcome="outcome", **kwargs):
    result = compute_tte(
        trajectories(events),
        drug_info(1),
        [outcome],
        follow_up_days,
        target_concept_ids=target,
        competing_concept_ids=competing,
        era_gap_days=30,
        **kwargs,
    )
    assert len(result) == 1
    return result["event"][0], result["time_days"][0]


class LastObservationCensoringTest(unittest.TestCase):
    def test_a_trajectory_without_the_outcome_is_censored_at_its_last_event(self):
        self.assertEqual(tte([(10, "a"), (50, "b")]), (0, 50))

    def test_the_last_event_of_any_code_is_the_last_observation(self):
        self.assertEqual(tte([(10, "a"), (50, "outcome"), (80, "b")]), (1, 50))
        self.assertEqual(tte([(10, "a"), (80, "b")], outcome="c"), (0, 80))

    def test_an_outcome_is_an_event_at_its_time(self):
        self.assertEqual(tte([(20, "outcome"), (50, "b")]), (1, 20))

    def test_the_follow_up_horizon_caps_a_longer_trajectory(self):
        self.assertEqual(tte([(10, "a"), (500, "b")], follow_up_days=365), (0, 365))

    def test_an_outcome_after_the_horizon_is_ignored(self):
        self.assertEqual(tte([(10, "a"), (500, "outcome")], follow_up_days=365), (0, 365))

    def test_events_before_the_index_drug_are_not_follow_up(self):
        self.assertEqual(tte([(-5, "a"), (7, "b")]), (0, 7))

    def test_a_trajectory_with_nothing_after_the_index_drug_has_no_follow_up(self):
        self.assertEqual(tte([(-5, "a")]), (0, 0))

    def test_trajectories_are_censored_separately(self):
        both = pl.concat(
            [
                trajectories([(10, "a"), (40, "b")], trajectory_id=0),
                trajectories([(10, "a"), (90, "outcome")], trajectory_id=1),
            ]
        )
        result = compute_tte(both, drug_info(1), ["outcome"], 365).sort("trajectory_id")
        self.assertEqual(result["event"].to_list(), [0, 1])
        self.assertEqual(result["time_days"].to_list(), [40, 90])


class EraGapTest(unittest.TestCase):
    def test_a_patient_who_never_refills_ends_the_era_one_gap_after_the_index_drug(self):
        self.assertEqual(tte([(10, "a"), (100, "b")], target=TARGET), (0, 30))

    def test_the_era_ends_one_gap_after_the_last_refill_when_the_trajectory_goes_on(self):
        events = [(10, "drug"), (20, "drug"), (100, "b")]
        self.assertEqual(tte(events, target=TARGET), (0, 50))

    def test_a_gap_between_refills_ends_the_era(self):
        events = [(10, "drug"), (80, "drug"), (90, "b")]
        self.assertEqual(tte(events, target=TARGET), (0, 40))

    def test_refills_within_the_gap_keep_the_era_open_until_the_last_observation(self):
        events = [(25, "drug"), (50, "drug"), (75, "b")]
        self.assertEqual(tte(events, target=TARGET), (0, 75))

    def test_a_trailing_gap_shorter_than_the_allowed_gap_is_not_a_discontinuation(self):
        self.assertEqual(tte([(10, "drug"), (35, "b")], target=TARGET), (0, 35))

    def test_an_outcome_after_the_era_ended_is_not_counted(self):
        events = [(10, "drug"), (60, "outcome"), (90, "b")]
        self.assertEqual(tte(events, target=TARGET), (0, 40))

    def test_an_outcome_inside_the_era_is_counted(self):
        events = [(10, "drug"), (30, "outcome"), (90, "b")]
        self.assertEqual(tte(events, target=TARGET), (1, 30))

    def test_the_era_gap_is_not_applied_without_target_concepts(self):
        self.assertEqual(tte([(10, "a"), (100, "b")]), (0, 100))


class CompetingAndRecordEndTest(unittest.TestCase):
    def test_a_competing_drug_censors_at_the_switch(self):
        events = [(10, "drug"), (25, "other_drug"), (100, "b")]
        self.assertEqual(tte(events, target=TARGET, competing=COMPETING), (0, 25))

    def test_a_competing_drug_after_the_last_event_changes_nothing(self):
        self.assertEqual(tte([(10, "a")], competing=COMPETING), (0, 10))

    def test_death_censors_at_its_time(self):
        self.assertEqual(tte([(10, "a"), (40, "[DEATH]"), (60, "b")]), (0, 40))


class FollowUpTest(unittest.TestCase):
    def follow_up(self, events, **kwargs):
        _, frame = compute_follow_up(
            trajectories(events), drug_info(1), 365, TARGET, None, 30, **kwargs
        )
        return frame.collect().row(0, named=True)

    def test_terminated_means_follow_up_ended_before_the_data_did(self):
        ended = self.follow_up([(10, "drug"), (100, "b")])
        self.assertTrue(ended["terminated"])
        self.assertEqual(ended["effective_follow_up_days"], 40)
        self.assertEqual(ended["last_observed_days"], 100)

        observed = self.follow_up([(10, "drug"), (30, "b")])
        self.assertFalse(observed["terminated"])
        self.assertEqual(observed["effective_follow_up_days"], 30)

    def test_a_subject_missing_from_drug_info_warns_and_gets_no_follow_up(self):
        with self.assertWarnsRegex(UserWarning, "not in drug_info"):
            result = compute_tte(
                trajectories([(10, "a")], subject_id=2), drug_info(1), ["outcome"], 365
            )
        self.assertEqual(result["time_days"].to_list(), [0])

    def test_no_warning_when_every_subject_is_in_drug_info(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_tte(trajectories([(10, "a")]), drug_info(1), ["outcome"], 365)


if __name__ == "__main__":
    unittest.main()
