import unittest

import numpy as np
import polars as pl

from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import (
    compute_drug_era_end_time,
    compute_last_observed_time,
)
from cehrgpt.analysis.counterfactual.hazard_ratio_estimation import observed_tte

DAY = 86_400
START = 1_600_000_000.0
TARGET = {"drug"}
COMPETING = {"other_drug"}


def record(*events):
    """A record whose index drug "drug" is at day 0; events are (day, token)."""
    tokens = ["[VS]", "9202", "drug", "[VE]"]
    times = [START] * 4
    for day, token in events:
        if token in ("[END]", "[DEATH]"):
            tokens.append(token)
            times.append(START + day * DAY)
        else:
            tokens += ["[VS]", "9202", token, "[VE]"]
            times += [START + day * DAY] * 4
    return tokens, times


def era_end_days(*events, gap=30):
    tokens, times = record(*events)
    end = compute_drug_era_end_time(tokens, times, TARGET, COMPETING, 2, gap)
    return None if end is None else (end - START) / DAY


class LastObservedTimeTest(unittest.TestCase):
    def test_it_is_the_last_event_of_the_record(self):
        tokens, times = record((10, "a"), (50, "b"))
        self.assertEqual(compute_last_observed_time(tokens, times), START + 50 * DAY)

    def test_it_is_the_earliest_of_the_last_event_end_and_death(self):
        tokens, times = record((10, "a"), (50, "b"), (40, "[END]"))
        self.assertEqual(compute_last_observed_time(tokens, times), START + 40 * DAY)
        tokens, times = record((10, "a"), (50, "b"), (30, "[DEATH]"), (40, "[END]"))
        self.assertEqual(compute_last_observed_time(tokens, times), START + 30 * DAY)
        tokens, times = record((10, "a"), (20, "[END]"), (50, "b"))
        self.assertEqual(compute_last_observed_time(tokens, times), START + 20 * DAY)

    def test_an_empty_record_has_none(self):
        self.assertIsNone(compute_last_observed_time([], []))


class DrugEraEndTimeTest(unittest.TestCase):
    def test_no_refill_and_a_record_that_goes_on_ends_the_era_one_gap_after_the_index(self):
        self.assertEqual(era_end_days((100, "a")), 30)

    def test_the_era_is_not_ended_when_the_record_stops_inside_the_gap(self):
        self.assertIsNone(era_end_days((20, "a")))

    def test_refills_inside_the_gap_keep_the_era_open(self):
        self.assertIsNone(era_end_days((20, "drug"), (40, "a")))

    def test_the_era_ends_one_gap_after_the_last_refill(self):
        self.assertEqual(era_end_days((20, "drug"), (100, "a")), 50)

    def test_a_gap_between_refills_ends_the_era(self):
        self.assertEqual(era_end_days((10, "drug"), (80, "drug")), 40)

    def test_a_switch_ends_the_era_at_the_switch(self):
        self.assertEqual(era_end_days((20, "other_drug"), (100, "a")), 20)

    def test_a_switch_after_the_era_lapsed_ends_it_when_it_lapsed(self):
        self.assertEqual(era_end_days((90, "other_drug")), 30)

    def test_death_ends_the_era_at_death(self):
        self.assertEqual(era_end_days((20, "drug"), (40, "[DEATH]")), 40)

    def test_death_after_the_era_lapsed_ends_it_when_it_lapsed(self):
        self.assertEqual(era_end_days((100, "[DEATH]")), 30)

    def test_end_is_not_a_discontinuation(self):
        self.assertIsNone(era_end_days((20, "drug"), (40, "[END]")))

    def test_end_after_the_era_lapsed_ends_it_when_it_lapsed(self):
        self.assertEqual(era_end_days((20, "drug"), (100, "[END]")), 50)


class ObservedTteTest(unittest.TestCase):
    def tte(self, outcome_day, era_end_day=None, last_day=None, follow_up=365):
        def epoch(day):
            return None if day is None else START + day * DAY

        frame = pl.DataFrame(
            {
                "drug_epoch_time": [START],
                "era_end_epoch_time": [epoch(era_end_day)],
                "last_observed_epoch_time": [epoch(last_day)],
                "outcome": [epoch(outcome_day)],
            },
            schema={
                "drug_epoch_time": pl.Float64,
                "era_end_epoch_time": pl.Float64,
                "last_observed_epoch_time": pl.Float64,
                "outcome": pl.Float64,
            },
        )
        time, event = observed_tte(frame, "outcome", follow_up)
        return int(event[0]), float(time[0])

    def test_without_an_outcome_a_patient_is_censored_at_the_last_observation(self):
        self.assertEqual(self.tte(None, last_day=80), (0, 80))

    def test_the_earliest_of_horizon_era_end_and_last_observation_censors(self):
        self.assertEqual(self.tte(None, era_end_day=50, last_day=80), (0, 50))
        self.assertEqual(self.tte(None, era_end_day=90, last_day=80), (0, 80))
        self.assertEqual(self.tte(None, era_end_day=500, last_day=600), (0, 365))

    def test_an_outcome_within_follow_up_is_an_event(self):
        self.assertEqual(self.tte(60, era_end_day=70, last_day=80), (1, 60))

    def test_an_outcome_after_censoring_is_not_counted(self):
        self.assertEqual(self.tte(75, era_end_day=70, last_day=80), (0, 70))
        self.assertEqual(self.tte(90, last_day=80), (0, 80))

    def test_an_outcome_on_or_before_the_index_is_not_counted(self):
        self.assertEqual(self.tte(0, last_day=80), (0, 80))
        self.assertEqual(self.tte(-5, last_day=80), (0, 80))

    def test_without_censoring_information_follow_up_runs_to_the_horizon(self):
        self.assertEqual(self.tte(None), (0, 365))


if __name__ == "__main__":
    unittest.main()
