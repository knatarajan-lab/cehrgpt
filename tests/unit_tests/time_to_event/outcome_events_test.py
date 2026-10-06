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

    def test_tokens(self):
        self.assertEqual(
            split_outcome_events(["CPT4/33510", "ICD10CM/0/I50"]),
            ([], ["CPT4/33510", "ICD10CM/0/I50"]),
        )

    def test_invalid_outcome_events(self):
        for events in (
            ["CPT4/33510", "9201"],
            [["ICD10CM/0/I50", "ICD10CM/1/9"]],
            [[]],
            [""],
            [],
        ):
            with self.assertRaises(ValueError, msg=str(events)):
                split_outcome_events(events)


class TestOutcomeMatching(unittest.TestCase):
    def test_tokens_are_matched_one_by_one(self):
        model = make_model(["Visit/IP", "ICD10CM/0/I50"])
        self.assertTrue(model.is_outcome_event("Visit/IP"))
        self.assertTrue(model.is_outcome_event("ICD10CM/0/I50"))
        self.assertFalse(model.is_outcome_event("ICD10CM/1/9"))

    def test_concept_ids_are_matched_as_strings(self):
        model = make_model([9201, "262"])
        self.assertTrue(model.is_outcome_event("9201"))
        self.assertTrue(model.is_outcome_event("262"))
        self.assertFalse(model.is_outcome_event("9202"))


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
            concept_ids, tokens = split_outcome_events(config.outcome_events)
            self.assertEqual(concept_ids, [], name)
            self.assertTrue(tokens, name)
            self.assertFalse(config.include_descendants, name)


if __name__ == "__main__":
    unittest.main()
