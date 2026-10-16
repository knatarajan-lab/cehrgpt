import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from cehrgpt.tools.linear_prob.compute_cehrgpt_features import _is_meds_cohort


class IsMedsCohortTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def cohort(self, name, columns):
        folder = os.path.join(self.root, name)
        os.makedirs(folder)
        pd.DataFrame({column: [1, 2] for column in columns}).to_parquet(
            os.path.join(folder, "part.parquet")
        )
        return folder

    def args(self, data_folder=None, test_data_folder=None, is_data_in_meds=False):
        return SimpleNamespace(
            data_folder=data_folder,
            test_data_folder=test_data_folder,
            is_data_in_meds=is_data_in_meds,
        )

    def test_subject_id_without_person_id_is_meds_even_without_the_flag(self):
        meds = self.cohort("meds", ["subject_id", "prediction_time", "boolean_value"])
        self.assertTrue(_is_meds_cohort(self.args(data_folder=meds, test_data_folder=meds)))

    def test_person_id_cohort_is_not_meds(self):
        plain = self.cohort("plain", ["person_id", "index_date", "gender_concept_id"])
        self.assertFalse(_is_meds_cohort(self.args(data_folder=plain, test_data_folder=plain)))

    def test_the_flag_forces_meds(self):
        plain = self.cohort("plain", ["person_id", "index_date"])
        self.assertTrue(_is_meds_cohort(self.args(data_folder=plain, is_data_in_meds=True)))

    def test_missing_or_unset_folders_are_not_meds(self):
        self.assertFalse(_is_meds_cohort(self.args()))
        self.assertFalse(_is_meds_cohort(self.args(data_folder=os.path.join(self.root, "nope"))))

    def test_a_cohort_with_both_columns_is_not_meds(self):
        both = self.cohort("both", ["subject_id", "person_id"])
        self.assertFalse(_is_meds_cohort(self.args(data_folder=both)))


if __name__ == "__main__":
    unittest.main()
