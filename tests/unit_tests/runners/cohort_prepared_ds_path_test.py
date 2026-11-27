import os
import tempfile
import unittest
from types import SimpleNamespace

from cehrbert.runners.runner_util import generate_prepared_ds_path

from cehrgpt.runners.data_utils import generate_cohort_prepared_ds_path


def arguments(observation_window=None, tokenized_full_dataset_path="/data/tokenized"):
    data_args = SimpleNamespace(
        data_folder="/data/cohorts/AMI",
        cohort_folder="/data/cohorts/AMI",
        dataset_prepared_path="/data/prepared",
        validation_split_percentage=0.05,
        test_eval_ratio=0.5,
        split_by_patient=True,
        chronological_split=False,
        observation_window=observation_window,
    )
    model_args = SimpleNamespace(tokenizer_name_or_path="/data/tokenizer")
    cehrgpt_args = SimpleNamespace(
        tokenized_full_dataset_path=tokenized_full_dataset_path
    )
    return data_args, model_args, cehrgpt_args


class TestCohortPreparedDsPath(unittest.TestCase):
    def test_observation_windows_do_not_share_a_cache(self):
        whole_history = generate_cohort_prepared_ds_path(*arguments(0))
        two_years = generate_cohort_prepared_ds_path(*arguments(730))
        one_year = generate_cohort_prepared_ds_path(*arguments(365))
        self.assertEqual(3, len({whole_history, two_years, one_year}))
        self.assertIn("_obs730_", two_years.name)
        self.assertIn("_obs365_", one_year.name)

    def test_windows_that_keep_the_whole_history_share_a_cache(self):
        # ExtractTokenizedSequenceDataMapping treats None, 0 and a negative window alike
        paths = {
            generate_cohort_prepared_ds_path(*arguments(window))
            for window in (None, 0, -1)
        }
        self.assertEqual(1, len(paths))
        self.assertIn("_obsall_", paths.pop().name)

    def test_the_full_dataset_is_part_of_the_name(self):
        first = generate_cohort_prepared_ds_path(*arguments(730, "/data/tokenized"))
        second = generate_cohort_prepared_ds_path(*arguments(730, "/data/other"))
        self.assertNotEqual(first, second)
        # A trailing slash is the same dataset
        self.assertEqual(
            first, generate_cohort_prepared_ds_path(*arguments(730, "/data/tokenized/"))
        )

    def test_the_folder_stays_in_the_prepared_path_and_starts_with_the_old_name(self):
        data_args, model_args, cehrgpt_args = arguments(730)
        old = generate_prepared_ds_path(
            data_args, model_args, data_folder=data_args.cohort_folder
        )
        new = generate_cohort_prepared_ds_path(data_args, model_args, cehrgpt_args)
        self.assertEqual(old.parent, new.parent)
        self.assertTrue(new.name.startswith(old.name + "_obs730_"))
        self.assertEqual(os.path.abspath("/data/prepared"), str(new.parent))

    def test_a_cohort_tokenized_from_raw_data_keeps_the_old_name(self):
        data_args, model_args, cehrgpt_args = arguments(730, None)
        self.assertEqual(
            generate_prepared_ds_path(
                data_args, model_args, data_folder=data_args.cohort_folder
            ),
            generate_cohort_prepared_ds_path(data_args, model_args, cehrgpt_args),
        )

    def test_the_spelling_of_the_folders_does_not_change_the_name(self):
        expected = generate_cohort_prepared_ds_path(*arguments(730))
        for cohort_folder, tokenizer in (
            ("/data/cohorts/AMI/", "/data/tokenizer"),
            ("/data/cohorts//AMI", "/data/tokenizer"),
            ("/data/cohorts/AMI", "/data/tokenizer/"),
            ("/data/cohorts/./AMI/", "/data//tokenizer/"),
        ):
            data_args, model_args, cehrgpt_args = arguments(730)
            data_args.cohort_folder = cohort_folder
            model_args.tokenizer_name_or_path = tokenizer
            with self.subTest(cohort_folder=cohort_folder, tokenizer=tokenizer):
                self.assertEqual(
                    expected,
                    generate_cohort_prepared_ds_path(data_args, model_args, cehrgpt_args),
                )
            # the arguments of the caller are left as they were
            self.assertEqual(tokenizer, model_args.tokenizer_name_or_path)
        other = arguments(730)
        other[0].cohort_folder = "/data/cohorts/CLL/"
        self.assertNotEqual(expected, generate_cohort_prepared_ds_path(*other))

    def test_a_cache_named_after_the_paths_as_typed_is_still_found(self):
        with tempfile.TemporaryDirectory() as prepared:
            def spelled(cohort_folder):
                data_args, model_args, cehrgpt_args = arguments(730)
                data_args.dataset_prepared_path = prepared
                data_args.cohort_folder = cohort_folder
                return data_args, model_args, cehrgpt_args

            new = generate_cohort_prepared_ds_path(*spelled("/data/cohorts/AMI"))
            data_args, model_args, _ = spelled("/data/cohorts//AMI")
            # the name an earlier version gave the cache of this spelling
            old = generate_prepared_ds_path(
                data_args, model_args, data_folder=data_args.cohort_folder
            )
            old = old.with_name(old.name + new.name[len(old.name):])
            self.assertNotEqual(new, old)
            self.assertEqual(new, generate_cohort_prepared_ds_path(*spelled("/data/cohorts//AMI")))

            # a run that stopped while saving left a folder without its index: it is not used
            old.mkdir()
            (old / "train").mkdir()
            self.assertEqual(new, generate_cohort_prepared_ds_path(*spelled("/data/cohorts//AMI")))
            # a complete one is, for the spelling it was made with only
            (old / "dataset_dict.json").write_text("{}")
            self.assertEqual(old, generate_cohort_prepared_ds_path(*spelled("/data/cohorts//AMI")))
            self.assertEqual(new, generate_cohort_prepared_ds_path(*spelled("/data/cohorts/AMI")))
            # and a complete cache under the new name wins
            new.mkdir()
            (new / "dataset_dict.json").write_text("{}")
            self.assertEqual(new, generate_cohort_prepared_ds_path(*spelled("/data/cohorts//AMI")))


if __name__ == "__main__":
    unittest.main()
