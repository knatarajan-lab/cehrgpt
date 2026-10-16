import json
import os
import pickle
import shutil
import tempfile
import unittest

import pandas as pd
from datasets import Dataset, load_from_disk

from cehrgpt.qa.build_qa_dataset import (
    build_qa_dataset,
    coverage_report,
    load_motor_codes,
)
from cehrgpt.qa.question_templates import SECONDS_PER_DAY

INDEX = 1_600_000_000
NAMES = {"100": "diabetes", "200": "asthma", "300": "fracture", "400": "migraine"}
FEATURE_DIM = 6


def sequence(person_id):
    concept_ids = ["year:2020", "age:60", "Gender/F", "Race/5", "[VS]", "100", "[VS]", "200", "300"]
    days = [-400, -400, -400, -400, -30, -30, 10, 10, 50]
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "epoch_times": [float(INDEX + d * SECONDS_PER_DAY) for d in days],
    }


class BuildQADatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        Dataset.from_list([sequence(1), sequence(2)]).save_to_disk(
            os.path.join(self.tmp, "tokenized")
        )
        os.makedirs(os.path.join(self.tmp, "tokenizer"))
        with open(os.path.join(self.tmp, "tokenizer", "concept_name_mapping.json"), "w") as f:
            json.dump(NAMES, f)
        os.makedirs(os.path.join(self.tmp, "features"))
        pd.DataFrame(
            {
                "subject_id": [1, 2, 999],  # 999 has no tokenized sequence
                "prediction_time_posix": [float(INDEX)] * 3,
                "features": [[float(i)] * FEATURE_DIM for i in (1, 2, 3)],
            }
        ).to_parquet(os.path.join(self.tmp, "features", "part.parquet"))

    def build(self, **kwargs):
        return build_qa_dataset(
            features_folder=os.path.join(self.tmp, "features"),
            tokenized_full_dataset_path=os.path.join(self.tmp, "tokenized"),
            tokenizer_path=os.path.join(self.tmp, "tokenizer"),
            output_path=os.path.join(self.tmp, "qa"),
            questions_per_patient=3,
            **kwargs,
        )

    def test_rows_carry_features_and_answers(self):
        dataset = self.build()
        self.assertGreater(len(dataset), 0)
        self.assertEqual(set(dataset["person_id"]), {1, 2})
        for row in dataset:
            self.assertEqual(len(row["patient_features"]), FEATURE_DIM)
            self.assertEqual(row["patient_features"][0], float(row["person_id"]))
            self.assertTrue(row["answer"])
        reloaded = load_from_disk(os.path.join(self.tmp, "qa"))
        self.assertEqual(len(reloaded), len(dataset))

    def test_answers_come_from_after_the_index_date(self):
        dataset = self.build()
        asthma = [r for r in dataset if "asthma" in r["question"] and r["qa_type"] == "time_to_event"]
        for row in asthma:
            self.assertEqual(row["answer"], "About 10 days.")

    def test_only_the_two_question_types(self):
        self.assertTrue(set(self.build()["qa_type"]) <= {"will_occur", "time_to_event"})

    def test_config_is_passed_through(self):
        dataset = self.build(qa_config={"will_occur_weight": 0.0})
        self.assertEqual(set(dataset["qa_type"]), {"time_to_event"})

    def test_wrong_feature_width_is_rejected(self):
        # e.g. 2x wide because the last-token state was appended to the linear probing state
        with self.assertRaisesRegex(ValueError, "disable_combine_global_local"):
            self.build(expected_feature_dim=2 * FEATURE_DIM)
        self.assertGreater(len(self.build(expected_feature_dim=FEATURE_DIM)), 0)

    def test_rows_are_tagged_with_concept_and_motor_scope(self):
        with open(
            os.path.join(self.tmp, "tokenizer", "motor_time_to_event_info.pickle"), "wb"
        ) as f:
            pickle.dump({"motor_time_to_event_codes": ["200", 300]}, f)
        dataset = self.build()
        for row in dataset:
            self.assertIn(row["concept_id"], NAMES)
            self.assertEqual(row["is_motor_task"], row["concept_id"] in {"200", "300"})
        self.assertEqual(
            load_motor_codes(os.path.join(self.tmp, "tokenizer")), {"200", "300"}
        )

    def test_missing_motor_info_means_nothing_is_in_scope(self):
        self.assertEqual(load_motor_codes(os.path.join(self.tmp, "tokenizer")), set())
        self.assertFalse(any(self.build()["is_motor_task"]))

    def test_measurements_and_artificial_tokens_are_never_asked_about(self):
        # 100 is a measurement: it has numeric lab statistics in the tokenizer folder
        with open(os.path.join(self.tmp, "tokenizer", "cehrgpt_lab_stats.pickle"), "wb") as f:
            pickle.dump(
                {"numeric_lab_stats": [{"concept_id": "100"}], "categorical_lab_stats": {}}, f
            )
        with open(os.path.join(self.tmp, "tokenizer", "concept_name_mapping.json"), "w") as f:
            json.dump({**NAMES, "[VS]": "[VS]", "D3": "3 days", "year:2020": "2020"}, f)
        dataset = self.build()
        self.assertGreater(len(dataset), 0)
        self.assertNotIn("100", set(dataset["concept_id"]))
        self.assertTrue(set(dataset["concept_id"]) <= {"200", "300", "400"})

    def test_only_allowed_domains_with_a_concept_table(self):
        pd.DataFrame(
            {
                "concept_id": [100, 200, 300, 400],
                "domain_id": ["Measurement", "Condition", "Visit", "Drug"],
            }
        ).to_parquet(os.path.join(self.tmp, "concept.parquet"))
        dataset = self.build(concept_table=os.path.join(self.tmp, "concept.parquet"))
        self.assertTrue(set(dataset["concept_id"]) <= {"200", "400"})
        self.assertTrue(len(set(dataset["concept_id"])) > 0)

    def test_no_eligible_concepts_is_an_error(self):
        pd.DataFrame({"concept_id": [100], "domain_id": ["Measurement"]}).to_parquet(
            os.path.join(self.tmp, "concept.parquet")
        )
        with self.assertRaisesRegex(ValueError, "No eligible concepts"):
            self.build(concept_table=os.path.join(self.tmp, "concept.parquet"))

    def test_coverage_report_is_written(self):
        dataset = self.build()
        with open(os.path.join(self.tmp, "qa", "coverage_report.json")) as f:
            report = json.load(f)
        self.assertEqual(report["eligible_concepts"], len(NAMES))
        self.assertEqual(report["concepts_asked"], len(set(dataset["concept_id"])))
        self.assertTrue(0 < report["concepts_asked_fraction"] <= 1)
        self.assertTrue(any(dataset["event_occurred"]))

    def test_coverage_report_numbers(self):
        data = Dataset.from_dict(
            {
                "concept_id": ["a", "a", "b", "c"],
                "event_occurred": [True, False, False, True],
            }
        )
        report = coverage_report(data, {"a": "A", "b": "B", "c": "C", "d": "D"}, {"a"})
        self.assertEqual(report["concepts_asked"], 3)
        self.assertEqual(report["concepts_with_an_event_answer"], 2)
        self.assertEqual(report["concepts_outside_motor"], 3)
        self.assertEqual(report["concepts_asked_outside_motor"], 2)
        self.assertEqual(report["least_asked_concept_count"], 1)

    def test_max_patients(self):
        dataset = self.build(max_patients=1)
        self.assertEqual(set(dataset["person_id"]), {1})

    def test_missing_features_raise(self):
        with self.assertRaises(FileNotFoundError):
            build_qa_dataset(
                features_folder=os.path.join(self.tmp, "nowhere"),
                tokenized_full_dataset_path=os.path.join(self.tmp, "tokenized"),
                tokenizer_path=os.path.join(self.tmp, "tokenizer"),
                output_path=os.path.join(self.tmp, "qa2"),
            )


if __name__ == "__main__":
    unittest.main()
