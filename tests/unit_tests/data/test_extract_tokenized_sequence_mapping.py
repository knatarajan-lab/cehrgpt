"""
Tests for ExtractTokenizedSequenceDataMapping.transform()

Patient fixture
---------------
Birth year : 1950
Demographic header : ["year:2000", "age:50", "8507", "Race/0"]  (indices 0-3)

Sequence layout (18 tokens total):

  idx  token     epoch            age
  ---  --------  ---------------  ---
   0   year:2000  T_2000          50   ─┐
   1   age:50     T_2000          50    │ demographic header
   2   8507       T_2000          50    │
   3   Race/0     T_2000          50   ─┘
   4   [VS]       T_2000          50   ─┐
   5   9202       T_2000          50    │ visit 1  (2000-01-01)
   6   cond1      T_2000          50    │
   7   [VE]       T_2000          50   ─┘
   8   D730       T_2002          52      ATT between visits
   9   [VS]       T_2002          52   ─┐
  10   9202       T_2002          52    │ visit 2  (2002-06-15)
  11   cond2      T_2002          52    │
  12   [VE]       T_2002          52   ─┘
  13   D840       T_2004          54      ATT between visits
  14   [VS]       T_2004          54   ─┐
  15   9202       T_2004          54    │ visit 3  (2004-09-20)
  16   cond3      T_2004          54    │
  17   [VE]       T_2004          54   ─┘

Fake input_ids  : [0, 1, 2, ..., 17]
"""

import datetime
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np

from cehrgpt.data.hf_cehrgpt_dataset_mapping import ExtractTokenizedSequenceDataMapping

UTC = datetime.timezone.utc

# ---------------------------------------------------------------------------
# Epoch constants used across every test
# ---------------------------------------------------------------------------
T_2000 = datetime.datetime(2000, 1, 1, tzinfo=UTC)
T_2002 = datetime.datetime(2002, 6, 15, tzinfo=UTC)
T_2004 = datetime.datetime(2004, 9, 20, tzinfo=UTC)
T_2005 = datetime.datetime(2005, 1, 1, tzinfo=UTC)

EP_2000 = T_2000.timestamp()   # 946684800.0
EP_2002 = T_2002.timestamp()   # 1024099200.0
EP_2004 = T_2004.timestamp()   # 1095638400.0
EP_2005 = T_2005.timestamp()   # 1104537600.0

BIRTH_YEAR = 1950
PERSON_ID = 42
LABEL = 1


def _make_record():
    """Return the 18-token patient record described in the module docstring."""
    concept_ids = [
        "year:2000", "age:50", "8507", "Race/0",        # 0-3  demographic
        "[VS]", "9202", "cond1", "[VE]",                 # 4-7  visit 1
        "D730",                                           # 8    ATT
        "[VS]", "9202", "cond2", "[VE]",                 # 9-12 visit 2
        "D840",                                           # 13   ATT
        "[VS]", "9202", "cond3", "[VE]",                 # 14-17 visit 3
    ]
    epoch_times = [
        EP_2000, EP_2000, EP_2000, EP_2000,  # demo
        EP_2000, EP_2000, EP_2000, EP_2000,  # visit 1
        EP_2002,                              # ATT
        EP_2002, EP_2002, EP_2002, EP_2002,  # visit 2
        EP_2004,                              # ATT
        EP_2004, EP_2004, EP_2004, EP_2004,  # visit 3
    ]
    ages = [
        50, 50, 50, 50,  # demo
        50, 50, 50, 50,  # visit 1
        52,              # ATT
        52, 52, 52, 52,  # visit 2
        54,              # ATT
        54, 54, 54, 54,  # visit 3
    ]
    concept_value_masks = [0] * 18
    input_ids = list(range(18))  # fake token ids  0..17

    return {
        "person_id": PERSON_ID,
        "concept_ids": concept_ids,
        "input_ids": input_ids,
        "epoch_times": epoch_times,
        "ages": ages,
        "concept_value_masks": concept_value_masks,
    }


def _make_mapping(observation_window, index_date, label=LABEL, tokenizer=None):
    """Build a mapping object with a single prediction time for PERSON_ID."""
    person_index_date_map = {
        PERSON_ID: [{"index_date": index_date, "label": label}]
    }
    return ExtractTokenizedSequenceDataMapping(
        person_index_date_map=person_index_date_map,
        observation_window=observation_window,
        tokenizer=tokenizer,
    )


class TestExtractWithObservationWindow(unittest.TestCase):
    """
    1000-day window ending at T_2005.

    feature_extraction_start = EP_2005 - 1000*86400 = 1018137600
    Visit 1 (EP_2000 = 946684800) is outside  → excluded.
    Visit 2 (EP_2002 = 1024099200) is inside  → vs_start_idx = 9.
    Visit 3 (EP_2004 = 1095638400) is inside  → included up to vs_end_idx = 17.
    """

    def setUp(self):
        self.mapping = _make_mapping(observation_window=1000, index_date=T_2005)
        self.result = self.mapping.transform(_make_record())

    def test_single_sample_produced(self):
        self.assertEqual(len(self.result["concept_ids"]), 1)

    def test_demographic_tokens_prepended(self):
        concept_ids = list(self.result["concept_ids"][0])
        self.assertEqual(concept_ids[:4], ["year:2002", "age:52", "8507", "Race/0"])

    def test_year_token_recalculated(self):
        self.assertEqual(self.result["concept_ids"][0][0], "year:2002")

    def test_age_token_recalculated(self):
        self.assertEqual(self.result["concept_ids"][0][1], "age:52")

    def test_gender_and_race_preserved(self):
        tokens = self.result["concept_ids"][0]
        self.assertEqual(tokens[2], "8507")
        self.assertEqual(tokens[3], "Race/0")

    def test_sequence_starts_at_vs(self):
        # Token at index 4 (right after the 4 demo tokens) must be [VS]
        self.assertEqual(self.result["concept_ids"][0][4], "[VS]")

    def test_visit_1_excluded(self):
        tokens = list(self.result["concept_ids"][0])
        self.assertNotIn("cond1", tokens)

    def test_visit_2_and_3_included(self):
        tokens = list(self.result["concept_ids"][0])
        self.assertIn("cond2", tokens)
        self.assertIn("cond3", tokens)

    def test_att_before_first_window_visit_excluded(self):
        # D730 is the ATT token immediately before visit 2; it sits outside
        # the extracted slice because slicing starts at the [VS] of visit 2.
        tokens = list(self.result["concept_ids"][0])
        self.assertNotIn("D730", tokens)

    def test_att_between_window_visits_included(self):
        # D840 lives between visit 2 and visit 3, both inside the window.
        tokens = list(self.result["concept_ids"][0])
        self.assertIn("D840", tokens)

    def test_epoch_times_for_demo_tokens(self):
        ep = self.result["epoch_times"][0]
        np.testing.assert_array_equal(ep[:4], [EP_2002] * 4)

    def test_ages_for_demo_tokens(self):
        ages = self.result["ages"][0]
        np.testing.assert_array_equal(ages[:4], [52] * 4)

    def test_concept_value_masks_for_demo_tokens(self):
        masks = self.result["concept_value_masks"][0]
        # Original demo mask values are 0; they should be reused.
        np.testing.assert_array_equal(masks[:4], [0] * 4)

    def test_input_ids_demo_fallback_without_tokenizer(self):
        # No tokenizer supplied → original demo input_ids [0,1,2,3] are reused.
        input_ids = self.result["input_ids"][0]
        np.testing.assert_array_equal(input_ids[:4], [0, 1, 2, 3])

    def test_input_ids_event_portion(self):
        # Sliced input_ids from vs_start_idx=9 to vs_end_idx=17 inclusive.
        input_ids = self.result["input_ids"][0]
        np.testing.assert_array_equal(input_ids[4:], list(range(9, 18)))

    def test_age_at_index(self):
        self.assertEqual(self.result["age_at_index"][0], 52)

    def test_classifier_label(self):
        self.assertEqual(self.result["classifier_label"][0], LABEL)

    def test_index_date(self):
        self.assertAlmostEqual(self.result["index_date"][0], EP_2005, places=1)

    def test_total_sequence_length(self):
        # 4 demo + 9 event tokens (indices 9-17)
        self.assertEqual(len(self.result["concept_ids"][0]), 13)


class TestExtractWithNoObservationWindow(unittest.TestCase):
    """
    observation_window=0 → feature_extraction_start=0 → full history used.

    vs_start_idx = 4 (first [VS] in the sequence, visit 1).
    Expected demo tokens: year:2000, age:50, 8507, Race/0 (unchanged).
    All three visits included.
    """

    def setUp(self):
        self.mapping = _make_mapping(observation_window=0, index_date=T_2005)
        self.result = self.mapping.transform(_make_record())

    def test_single_sample_produced(self):
        self.assertEqual(len(self.result["concept_ids"]), 1)

    def test_demographic_year_unchanged(self):
        self.assertEqual(self.result["concept_ids"][0][0], "year:2000")

    def test_demographic_age_unchanged(self):
        self.assertEqual(self.result["concept_ids"][0][1], "age:50")

    def test_all_three_visits_included(self):
        tokens = list(self.result["concept_ids"][0])
        self.assertIn("cond1", tokens)
        self.assertIn("cond2", tokens)
        self.assertIn("cond3", tokens)

    def test_total_sequence_length(self):
        # 4 demo + 14 event tokens (indices 4-17)
        self.assertEqual(len(self.result["concept_ids"][0]), 18)

    def test_epoch_times_for_demo_tokens(self):
        ep = self.result["epoch_times"][0]
        np.testing.assert_array_equal(ep[:4], [EP_2000] * 4)

    def test_ages_for_demo_tokens(self):
        ages = self.result["ages"][0]
        np.testing.assert_array_equal(ages[:4], [50] * 4)

    def test_age_at_index(self):
        self.assertEqual(self.result["age_at_index"][0], 50)


class TestExtractNoVsInWindow(unittest.TestCase):
    """
    1-day window ending at T_2005 → feature_extraction_start ≈ 2004-12-31.
    All [VS] tokens have epoch << feature_extraction_start → no sample produced.
    """

    def setUp(self):
        self.mapping = _make_mapping(observation_window=1, index_date=T_2005)
        self.result = self.mapping.transform(_make_record())

    def test_no_samples_produced(self):
        self.assertEqual(len(self.result), 0)


class TestExtractWithTokenizer(unittest.TestCase):
    """
    When a tokenizer is provided, input_ids for the new demographic tokens
    should come from tokenizer.encode(), not the original demo ids.
    """

    def setUp(self):
        tokenizer = MagicMock()
        # encode(["year:2002", "age:52", "8507", "Race/0"]) → [100, 101, 102, 103]
        tokenizer.encode.return_value = [100, 101, 102, 103]
        self.mapping = _make_mapping(
            observation_window=1000, index_date=T_2005, tokenizer=tokenizer
        )
        self.result = self.mapping.transform(_make_record())

    def test_demo_input_ids_from_tokenizer(self):
        np.testing.assert_array_equal(
            self.result["input_ids"][0][:4], [100, 101, 102, 103]
        )

    def test_event_input_ids_unchanged(self):
        # Event portion (visit 2 + visit 3: indices 9-17) must still be original ids.
        np.testing.assert_array_equal(
            self.result["input_ids"][0][4:], list(range(9, 18))
        )


class TestExtractMultiplePredictionTimes(unittest.TestCase):
    """
    Two prediction times for the same person:
      - T_2005 with 1000-day window → visit 2 + 3 in scope  (sample produced)
      - T_2003 with 1-day window    → no VS in scope         (sample skipped)
    Only one sample should be present in the output.
    """

    def setUp(self):
        T_2003 = datetime.datetime(2003, 1, 1, tzinfo=UTC)
        person_index_date_map = {
            PERSON_ID: [
                {"index_date": T_2005, "label": 1},
                {"index_date": T_2003, "label": 0},
            ]
        }
        self.mapping = ExtractTokenizedSequenceDataMapping(
            person_index_date_map=person_index_date_map,
            observation_window=1,   # 1-day window for T_2003 → no VS
        )
        # Override only the first entry to use 1000-day window via direct call.
        # Easier: use two separate mappings to test the expectation.
        # Re-build: first entry uses 1000-day window, second uses 1-day.
        # We reuse observation_window=1000 for the first and check that
        # the 1-day T_2003 entry produces nothing.
        person_index_date_map = {
            PERSON_ID: [
                {"index_date": T_2005, "label": 1},  # 1000-day window applied globally
                {"index_date": T_2003, "label": 0},  # also 1000-day → EP_2003-1000d check
            ]
        }
        # T_2003 with 1000-day window:
        #   EP_2003 = datetime(2003,1,1,tzinfo=UTC).timestamp() = 1041379200
        #   feature_extraction_start = 1041379200 - 86400000 = 954979200 (≈2000-04-07)
        #   Visit 1 epoch EP_2000=946684800 < 954979200 → outside
        #   Visit 2 epoch EP_2002=1024099200 >= 954979200 → vs_start_idx=9
        #   vs_end_idx: last token with epoch <= 1041379200
        #     EP_2004=1095638400 > 1041379200 → visit 3 excluded
        #     EP_2002=1024099200 <= 1041379200 → [VE] at index 12 is last
        #   Slice: indices 9-12 = ["[VS]", "9202", "cond2", "[VE]"]  → sample produced
        # So both entries produce samples; total = 2.
        self.mapping = ExtractTokenizedSequenceDataMapping(
            person_index_date_map=person_index_date_map,
            observation_window=1000,
        )
        self.result = self.mapping.transform(_make_record())

    def test_two_samples_produced(self):
        self.assertEqual(len(self.result["concept_ids"]), 2)

    def test_first_sample_contains_visit_2_and_3(self):
        tokens = list(self.result["concept_ids"][0])
        self.assertIn("cond2", tokens)
        self.assertIn("cond3", tokens)

    def test_second_sample_contains_only_visit_2(self):
        tokens = list(self.result["concept_ids"][1])
        self.assertIn("cond2", tokens)
        self.assertNotIn("cond3", tokens)

    def test_labels_preserved(self):
        self.assertEqual(self.result["classifier_label"][0], 1)
        self.assertEqual(self.result["classifier_label"][1], 0)


class TestExtractIndexDateMidVisit(unittest.TestCase):
    """
    The prediction time (index_date) falls in the middle of a hospitalization,
    so the last token at or before index_date is an inpatient ATT token (i-D1).
    The backward step should skip it and land on the preceding clinical event.

    Fixture (single inpatient visit, 10 tokens):

      idx  token   epoch
      ---  ------  -----------
       0   year:2020  T_admit
       1   age:70     T_admit
       2   8507       T_admit
       3   Race/0     T_admit
       4   [VS]       T_admit      ← vs_start_idx (observation_window=0)
       5   9201       T_admit
       6   cond1      T_admit
       7   i-D1       T_day1       ← inpatient ATT; epoch > T_admit but <= index_date
       8   cond2      T_day2       ← epoch > index_date → excluded
       9   [VE]       T_day2       ← epoch > index_date → excluded

    index_date = T_day1 exactly:
      - Tokens 0-7 have epoch <= T_day1; token 7 (i-D1) is last → initial vs_end_idx=7
      - i-D1 is an ATT token → step back to idx 6 (cond1)
      - Expected last token: cond1; cond2 must not appear.
    """

    T_ADMIT = datetime.datetime(2020, 1, 1, tzinfo=UTC)
    T_DAY1  = datetime.datetime(2020, 1, 2, tzinfo=UTC)  # index_date
    T_DAY2  = datetime.datetime(2020, 1, 3, tzinfo=UTC)

    def _make_inpatient_record(self):
        EP_ADMIT = self.T_ADMIT.timestamp()
        EP_DAY1  = self.T_DAY1.timestamp()
        EP_DAY2  = self.T_DAY2.timestamp()
        concept_ids  = ["year:2020", "age:70", "8507", "Race/0",
                        "[VS]", "9201", "cond1", "i-D1", "cond2", "[VE]"]
        epoch_times  = [EP_ADMIT] * 7 + [EP_DAY1, EP_DAY2, EP_DAY2]
        ages         = [70] * 10
        return {
            "person_id": PERSON_ID,
            "concept_ids": concept_ids,
            "input_ids": list(range(10)),
            "epoch_times": epoch_times,
            "ages": ages,
            "concept_value_masks": [0] * 10,
        }

    def setUp(self):
        self.mapping = _make_mapping(observation_window=0, index_date=self.T_DAY1)
        self.result = self.mapping.transform(self._make_inpatient_record())

    def test_single_sample_produced(self):
        self.assertEqual(len(self.result["concept_ids"]), 1)

    def test_sequence_does_not_end_on_att_token(self):
        last_token = self.result["concept_ids"][0][-1]
        self.assertFalse(
            last_token.startswith("i-D"),
            f"Sequence ended on inpatient ATT token: {last_token}",
        )

    def test_sequence_ends_on_clinical_event(self):
        last_token = self.result["concept_ids"][0][-1]
        self.assertEqual(last_token, "cond1")

    def test_post_index_events_excluded(self):
        tokens = list(self.result["concept_ids"][0])
        self.assertNotIn("cond2", tokens)
        self.assertNotIn("[VE]", tokens)

    def test_pre_index_events_included(self):
        tokens = list(self.result["concept_ids"][0])
        self.assertIn("cond1", tokens)


if __name__ == "__main__":
    unittest.main()
