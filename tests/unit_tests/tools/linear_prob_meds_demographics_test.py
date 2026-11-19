import unittest
from types import SimpleNamespace

from cehrgpt.tools.linear_prob.compute_cehrgpt_features import demographics_folders


def _args(is_data_in_meds=False, cohort_folder=None, tokenized_full_dataset_path=None):
    data_args = SimpleNamespace(
        is_data_in_meds=is_data_in_meds,
        cohort_folder=cohort_folder,
        data_folder="data",
        test_data_folder="test",
    )
    return data_args, SimpleNamespace(tokenized_full_dataset_path=tokenized_full_dataset_path)


class DemographicsFoldersTest(unittest.TestCase):
    def test_meds_cohort_reads_the_labels_not_the_reader(self):
        # data_folder is the MEDS reader, which pandas can not read, and there is no test_data_folder
        self.assertEqual(demographics_folders(*_args(is_data_in_meds=True, cohort_folder="labels")), ["labels"])

    def test_tokenized_full_dataset_reads_the_cohort(self):
        self.assertEqual(
            demographics_folders(*_args(cohort_folder="cohort", tokenized_full_dataset_path="tokenized")),
            ["cohort"],
        )

    def test_train_and_test_folders(self):
        self.assertEqual(demographics_folders(*_args()), ["data", "test"])
        self.assertEqual(demographics_folders(*_args(is_data_in_meds=True)), ["data", "test"])


if __name__ == "__main__":
    unittest.main()
