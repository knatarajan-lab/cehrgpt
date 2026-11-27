import os
import unittest

from cehrgpt.cehrgpt_args import SamplingStrategy
from cehrgpt.time_to_event.time_to_event_prediction import (
    create_arg_parser,
    load_task_config_from_yaml,
)

READMISSION_YAML = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "src",
    "cehrgpt",
    "time_to_event",
    "config",
    "30_day_readmission.yaml",
)

REQUIRED_ARGS = [
    "--tokenizer_folder", "tokenizer",
    "--model_folder", "model",
    "--output_folder", "output",
    "--num_return_sequences", "10",
    "--batch_size", "4",
    "--context_window", "1024",
    "--sampling_strategy", list(SamplingStrategy)[0].value,
    "--task_config", READMISSION_YAML,
]


class TestTaskConfigOverride(unittest.TestCase):
    def test_no_override_keeps_the_yaml(self):
        config = load_task_config_from_yaml(READMISSION_YAML)
        self.assertEqual(config.prediction_window_end, 30)
        self.assertEqual(load_task_config_from_yaml(READMISSION_YAML, []), config)

    def test_values_are_parsed_as_yaml(self):
        config = load_task_config_from_yaml(
            READMISSION_YAML,
            [
                "prediction_window_end=730",
                "include_descendants=true",
                "outcome_events=[CPT4/33510, '9201']",
                "task_name=readmission_2_years",
            ],
        )
        self.assertEqual(config.prediction_window_end, 730)
        self.assertIs(config.include_descendants, True)
        self.assertEqual(config.outcome_events, ["CPT4/33510", "9201"])
        self.assertEqual(config.task_name, "readmission_2_years")
        # the fields that are not overridden come from the yaml
        self.assertEqual(config.max_new_tokens, 128)

    def test_the_last_override_of_a_field_wins(self):
        config = load_task_config_from_yaml(
            READMISSION_YAML, ["max_new_tokens=64", "max_new_tokens=256"]
        )
        self.assertEqual(config.max_new_tokens, 256)

    def test_a_value_may_contain_an_equals_sign(self):
        config = load_task_config_from_yaml(READMISSION_YAML, ["task_name=a=b"])
        self.assertEqual(config.task_name, "a=b")

    def test_an_unknown_field_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "prediction_window_ends"):
            load_task_config_from_yaml(READMISSION_YAML, ["prediction_window_ends=730"])

    def test_an_override_without_a_value_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "field=value"):
            load_task_config_from_yaml(READMISSION_YAML, ["prediction_window_end"])

    def test_the_flag_is_repeatable(self):
        args = create_arg_parser().parse_args(
            REQUIRED_ARGS
            + [
                "--task_config_override", "prediction_window_end=730",
                "--task_config_override", "max_new_tokens=256",
            ]
        )
        config = load_task_config_from_yaml(args.task_config, args.task_config_override)
        self.assertEqual((config.prediction_window_end, config.max_new_tokens), (730, 256))

    def test_the_flag_defaults_to_no_overrides(self):
        args = create_arg_parser().parse_args(REQUIRED_ARGS)
        self.assertEqual(args.task_config_override, [])


if __name__ == "__main__":
    unittest.main()
