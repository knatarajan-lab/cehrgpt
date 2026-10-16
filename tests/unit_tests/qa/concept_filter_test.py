import os
import pickle
import tempfile
import unittest

import pandas as pd

from cehrgpt.qa.concept_filter import (
    eligible_concepts,
    is_artificial_token,
    load_concept_domains,
    load_value_concepts,
)

NAMES = {
    "201826": "Type 2 diabetes",
    "4329847": "Myocardial infarction",
    "3004410": "Hemoglobin A1c",
    "9202": "Outpatient visit",
    "[VS]": "[VS]",
    "[VE]": "[VE]",
    "D5": "5 days",
    "W2": "2 weeks",
    "LT": "long time",
    "year:2020": "2020",
    "age:50": "50",
    "Gender/F": "female",
    "Race/5": "white",
    "999": "",
}


class ConceptFilterTest(unittest.TestCase):
    def test_artificial_tokens(self):
        for token in ("[VS]", "[VE]", "[PAD]", "D5", "W2", "M3", "LT", "year:2020", "age:50", "Gender/F", "Race/5"):
            self.assertTrue(is_artificial_token(token), token)
        for token in ("201826", "9202", "0"):
            self.assertFalse(is_artificial_token(token), token)

    def test_only_concept_tokens_remain(self):
        eligible = eligible_concepts(NAMES)
        self.assertEqual(set(eligible), {"201826", "4329847", "3004410", "9202"})

    def test_measurements_are_removed(self):
        eligible = eligible_concepts(NAMES, value_concepts={"3004410"})
        self.assertNotIn("3004410", eligible)
        self.assertIn("201826", eligible)

    def test_domains_keep_conditions_procedures_and_drugs(self):
        domains = {
            "201826": "Condition",
            "4329847": "Condition",
            "3004410": "Measurement",
            "9202": "Visit",
        }
        self.assertEqual(
            set(eligible_concepts(NAMES, concept_domains=domains)), {"201826", "4329847"}
        )
        self.assertEqual(
            set(eligible_concepts(NAMES, concept_domains=domains, allowed_domains=["Visit"])),
            {"9202"},
        )

    def test_unknown_domain_is_not_eligible(self):
        self.assertEqual(eligible_concepts(NAMES, concept_domains={}), {})

    def test_value_concepts_from_the_tokenizer_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_value_concepts(folder), set())
            with open(os.path.join(folder, "cehrgpt_lab_stats.pickle"), "wb") as f:
                pickle.dump(
                    {
                        "numeric_lab_stats": [{"concept_id": "3004410"}, {"concept_id": 3000963}],
                        "categorical_lab_stats": {("4000", "positive"): 3, ("4000", "negative"): 2},
                    },
                    f,
                )
            self.assertEqual(load_value_concepts(folder), {"3004410", "3000963"})
            self.assertEqual(
                load_value_concepts(folder, include_categorical=True),
                {"3004410", "3000963", "4000"},
            )

    def test_concept_domains_from_files_and_folders(self):
        table = pd.DataFrame({"concept_id": [1, 2], "domain_id": ["Drug", "Visit"]})
        with tempfile.TemporaryDirectory() as folder:
            table.to_parquet(os.path.join(folder, "a.parquet"))
            table.to_csv(os.path.join(folder, "b.csv"), index=False)
            expected = {"1": "Drug", "2": "Visit"}
            self.assertEqual(load_concept_domains(folder), expected)
            self.assertEqual(load_concept_domains(os.path.join(folder, "a.parquet")), expected)
            self.assertEqual(load_concept_domains(os.path.join(folder, "b.csv")), expected)


if __name__ == "__main__":
    unittest.main()
