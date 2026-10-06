import os
import unittest
from unittest.mock import MagicMock

import yaml

from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel
from cehrgpt.time_to_event.time_to_event_prediction import (
    load_task_config_from_yaml,
    split_outcome_events,
)

CONFIG_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "src",
    "cehrgpt",
    "time_to_event",
    "config",
)


def make_model(outcome_events):
    return TimeToEventModel(
        tokenizer=MagicMock(),
        model=MagicMock(),
        outcome_events=outcome_events,
        generation_config=MagicMock(),
    )


class TestSplitOutcomeEvents(unittest.TestCase):
    def test_concept_ids(self):
        self.assertEqual(split_outcome_events(["9201", 262]), ([9201, 262], []))

    def test_token_sequences(self):
        self.assertEqual(
            split_outcome_events([["ICD10CM/0/I50"], ["ICD10CM/0/I50", "ICD10CM/1/9"]]),
            ([], [["ICD10CM/0/I50"], ["ICD10CM/0/I50", "ICD10CM/1/9"]]),
        )

    def test_bare_token_is_a_single_token_sequence(self):
        self.assertEqual(
            split_outcome_events(["CPT4/33510", ["ICD10CM/0/I50", "ICD10CM/1/9"]]),
            ([], [["CPT4/33510"], ["ICD10CM/0/I50", "ICD10CM/1/9"]]),
        )

    def test_invalid_outcome_events(self):
        for events in ([["A"], "9201"], ["CPT4/33510", "9201"], [[]], [""], []):
            with self.assertRaises(ValueError, msg=str(events)):
                split_outcome_events(events)


class TestTokenSequenceOutcomes(unittest.TestCase):
    def test_sequences_match_consecutive_tokens(self):
        model = make_model([["ICD10CM/0/I50", "ICD10CM/1/9"], ["Visit/IP"]])
        self.assertTrue(model.is_outcome_event(["x", "ICD10CM/0/I50", "ICD10CM/1/9"]))
        self.assertFalse(model.is_outcome_event(["ICD10CM/0/I50", "x", "ICD10CM/1/9"]))
        self.assertFalse(model.is_outcome_event(["ICD10CM/1/9"]))
        self.assertTrue(model.is_outcome_event(["x", "Visit/IP"]))

    def test_one_token_matches_every_code_with_it(self):
        model = make_model([["ICD10CM/0/I50"]])
        self.assertTrue(model.is_outcome_event(["ICD10CM/0/I50", "ICD10CM/1/84"]))

    def test_single_tokens_are_still_matched_as_before(self):
        model = make_model([9201, "262"])
        self.assertTrue(model.is_outcome_event("9201"))
        self.assertTrue(model.is_outcome_event("262"))
        self.assertFalse(model.is_outcome_event("9202"))

    def test_mixed_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            make_model([["Visit/IP"], "9201"])


class TestConfigs(unittest.TestCase):
    def test_no_config_has_the_removed_option(self):
        for name in sorted(os.listdir(CONFIG_FOLDER)):
            with open(os.path.join(CONFIG_FOLDER, name)) as f:
                self.assertNotIn("is_ethos_task", yaml.safe_load(f), name)

    def test_ethos_configs_list_their_tokens(self):
        for name in (
            "t2dm_hf_ethos.yaml",
            "30_day_readmission_ethos.yaml",
            "1_year_cabg_ethos.yaml",
        ):
            config = load_task_config_from_yaml(os.path.join(CONFIG_FOLDER, name))
            concept_ids, token_sequences = split_outcome_events(config.outcome_events)
            self.assertEqual(concept_ids, [], name)
            self.assertTrue(token_sequences, name)
            self.assertFalse(config.include_descendants, name)


if __name__ == "__main__":
    unittest.main()
