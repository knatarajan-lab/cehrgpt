import json
import os
import pickle
import shutil
import tempfile
import unittest

import pandas as pd
from datasets import Dataset, load_from_disk

from cehrgpt.omop.ontology import Ontology
from cehrgpt.qa.build_query_dataset import (
    build_query_dataset,
    coverage_report,
)
from cehrgpt.qa.query_generation import SECONDS_PER_DAY

INDEX = 1_600_000_000
# 30 Clinical finding > 10 Diabetes mellitus > (11 Type 2, 12 Type 1); 30 > 20 Asthma
NAMES = {
    "11": "Type 2 diabetes",
    "12": "Type 1 diabetes",
    "20": "Asthma",
    "3004410": "Hemoglobin A1c",  # a measurement
    "9202": "Outpatient visit",  # a visit
    "[VS]": "[VS]",
    "D3": "3 days",
    "year:2020": "2020",
    "10": "Diabetes mellitus",
    "30": "Clinical finding",
}
DOMAINS = {
    "11": "Condition", "12": "Condition", "20": "Condition", "10": "Condition",
    "30": "Condition", "3004410": "Measurement", "9202": "Visit",
}
FEATURE_DIM = 6


def ontology():
    # A real Ontology object without the Athena download its constructor needs
    result = Ontology.__new__(Ontology)
    result.parents_map = {"11": {"10"}, "12": {"10"}, "10": {"30"}, "20": {"30"}}
    result.concept_domain_map = DOMAINS
    result.concept_vocabulary_map = {}
    result.all_parents_map = {}
    return result


def sequence(person_id):
    concept_ids = ["year:2020", "age:60", "Gender/F", "Race/5", "[VS]", "9202", "12", "[VS]", "11", "3004410", "20"]
    days = [-400, -400, -400, -400, -30, -30, -30, 10, 10, 10, 50]
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "epoch_times": [float(INDEX + d * SECONDS_PER_DAY) for d in days],
    }


class BuildQueryDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        Dataset.from_list([sequence(1), sequence(2)]).save_to_disk(os.path.join(self.tmp, "tokenized"))
        tokenizer = os.path.join(self.tmp, "tokenizer")
        os.makedirs(tokenizer)
        with open(os.path.join(tokenizer, "concept_name_mapping.json"), "w") as f:
            json.dump(NAMES, f)
        with open(os.path.join(tokenizer, "ontology.pickle"), "wb") as f:
            pickle.dump(ontology(), f)
        with open(os.path.join(tokenizer, "cehrgpt_lab_stats.pickle"), "wb") as f:
            pickle.dump({"numeric_lab_stats": [{"concept_id": "3004410"}], "categorical_lab_stats": {}}, f)
        os.makedirs(os.path.join(self.tmp, "features"))
        pd.DataFrame(
            {
                "subject_id": [1, 2, 999],  # 999 has no tokenized sequence
                "prediction_time_posix": [float(INDEX)] * 3,
                "features": [[float(i)] * FEATURE_DIM for i in (1, 2, 3)],
            }
        ).to_parquet(os.path.join(self.tmp, "features", "part.parquet"))

    def build(self, **kwargs):
        return build_query_dataset(
            features_folder=os.path.join(self.tmp, "features"),
            tokenized_full_dataset_path=os.path.join(self.tmp, "tokenized"),
            tokenizer_path=os.path.join(self.tmp, "tokenizer"),
            output_path=os.path.join(self.tmp, "queries"),
            queries_per_patient=6,
            **kwargs,
        )

    def test_rows_carry_the_patient_the_query_and_the_label(self):
        dataset = self.build()
        self.assertGreater(len(dataset), 0)
        self.assertEqual(set(dataset["person_id"]), {1, 2})
        for row in dataset:
            self.assertEqual(len(row["patient_features"]), FEATURE_DIM)
            self.assertEqual(row["patient_features"][0], float(row["person_id"]))
            self.assertLess(row["start_days"], row["end_days"])
        reloaded = load_from_disk(os.path.join(self.tmp, "queries"))
        self.assertEqual(len(reloaded), len(dataset))

    def test_labels_follow_the_descendants(self):
        dataset = self.build(query_config={"positive_fraction": 1.0, "windows": ((0, 90),)})
        # Type 2 diabetes (day 10) and asthma (day 50) are inside (0, 90]; type 1 is history
        self.assertEqual(set(dataset["concept_id"]), {"11", "20", "10", "30"})
        self.assertTrue(all(dataset["label"]))

    def test_only_conditions_never_measurements_visits_or_artificial_tokens(self):
        dataset = self.build()
        queried = set(dataset["concept_id"])
        self.assertTrue(queried <= {"11", "12", "20", "10", "30"})
        for banned in ("3004410", "9202", "[VS]", "D3", "year:2020"):
            self.assertNotIn(banned, queried)

    def test_vocabulary_and_coverage_files(self):
        dataset = self.build()
        with open(os.path.join(self.tmp, "queries", "query_vocabulary.json")) as f:
            vocabulary = json.load(f)
        self.assertEqual(vocabulary["concept_ids"], sorted(["11", "12", "20", "10", "30"]))
        # the concept counts itself: it is a token as well as the parent of 11 and 12
        self.assertEqual(vocabulary["covered_tokens"]["10"], 3)
        with open(os.path.join(self.tmp, "queries", "coverage_report.json")) as f:
            report = json.load(f)
        self.assertEqual(report["query_concepts"], 5)
        self.assertEqual(report["concepts_queried"], len(set(dataset["concept_id"])))

    def test_broad_concepts_can_be_dropped(self):
        dataset = self.build(max_covered_tokens=2)
        self.assertNotIn("30", set(dataset["concept_id"]))

    def test_motor_scope_tag(self):
        with open(os.path.join(self.tmp, "tokenizer", "motor_time_to_event_info.pickle"), "wb") as f:
            pickle.dump({"motor_time_to_event_codes": ["20"]}, f)
        dataset = self.build()
        for row in dataset:
            self.assertEqual(row["is_motor_task"], row["concept_id"] == "20")

    def test_wrong_feature_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disable_combine_global_local"):
            self.build(expected_feature_dim=2 * FEATURE_DIM)
        self.assertGreater(len(self.build(expected_feature_dim=FEATURE_DIM)), 0)

    def test_max_patients(self):
        self.assertEqual(set(self.build(max_patients=1)["person_id"]), {1})

    def test_missing_features_raise(self):
        with self.assertRaises(FileNotFoundError):
            build_query_dataset(
                features_folder=os.path.join(self.tmp, "nowhere"),
                tokenized_full_dataset_path=os.path.join(self.tmp, "tokenized"),
                tokenizer_path=os.path.join(self.tmp, "tokenizer"),
                output_path=os.path.join(self.tmp, "q2"),
            )

    def test_without_an_ontology_a_concept_covers_only_itself(self):
        os.remove(os.path.join(self.tmp, "tokenizer", "ontology.pickle"))
        dataset = self.build(query_config={"positive_fraction": 1.0, "windows": ((0, 90),)})
        # Type 2 diabetes and asthma happen in the window; without the hierarchy their
        # parents (10, 30) are not positive, and there are no domains to drop the visit
        self.assertEqual(set(dataset["concept_id"]), {"11", "20"})
        self.assertNotIn("3004410", set(self.build()["concept_id"]))

    def test_coverage_report_numbers(self):
        data = Dataset.from_dict(
            {"concept_id": ["a", "a", "b", "c"], "label": [True, False, False, True]}
        )
        report = coverage_report(data, {"a": 1, "b": 1, "c": 1, "d": 1}, {"a"})
        self.assertEqual(report["concepts_queried"], 3)
        self.assertEqual(report["concepts_with_a_yes"], 2)
        self.assertEqual(report["concepts_outside_motor"], 3)
        self.assertEqual(report["concepts_queried_outside_motor"], 2)
        self.assertEqual(report["least_queried_concept_count"], 1)
        self.assertAlmostEqual(report["yes_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
