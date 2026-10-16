import json
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments

from cehrgpt.models.patient_query_model import PatientQueryModel
from cehrgpt.omop.ontology import Ontology
from cehrgpt.qa.build_query_dataset import build_query_dataset
from cehrgpt.qa.query_generation import SECONDS_PER_DAY
from cehrgpt.runners.hf_patient_query_runner import QueryRunnerArguments, run

INDEX = 1_600_000_000
NAMES = {str(i): f"concept {i}" for i in list(range(100, 112)) + [10, 30]}
FEATURE_DIM = 6


def ontology():
    result = Ontology.__new__(Ontology)
    result.parents_map = {str(i): {"10"} for i in range(100, 106)}
    result.parents_map.update({str(i): {"30"} for i in range(106, 112)})
    result.parents_map["10"] = {"30"}
    result.concept_domain_map = {c: "Condition" for c in NAMES}
    result.concept_vocabulary_map = {}
    result.all_parents_map = {}
    return result


class RunnerTest(unittest.TestCase):
    def test_end_to_end_build_train_and_evaluate(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rng = np.random.RandomState(0)
        sequences = []
        for person in range(60):
            tokens = ["year:2020", "age:60", "Gender/F", "Race/5"]
            days = [-400] * 4
            for _ in range(5):
                tokens.append(str(rng.randint(100, 112)))
                days.append(int(rng.randint(-300, 700)))
            order = np.argsort(days, kind="stable")
            sequences.append(
                {
                    "person_id": person,
                    "concept_ids": [tokens[i] for i in order],
                    "epoch_times": [float(INDEX + days[i] * SECONDS_PER_DAY) for i in order],
                }
            )
        Dataset.from_list(sequences).save_to_disk(os.path.join(tmp, "tokenized"))
        tokenizer = os.path.join(tmp, "tokenizer")
        os.makedirs(tokenizer)
        with open(os.path.join(tokenizer, "concept_name_mapping.json"), "w") as f:
            json.dump(NAMES, f)
        with open(os.path.join(tokenizer, "ontology.pickle"), "wb") as f:
            pickle.dump(ontology(), f)
        os.makedirs(os.path.join(tmp, "features"))
        pd.DataFrame(
            {
                "subject_id": list(range(60)),
                "prediction_time_posix": [float(INDEX)] * 60,
                "features": [rng.randn(FEATURE_DIM).tolist() for _ in range(60)],
            }
        ).to_parquet(os.path.join(tmp, "features", "part.parquet"))
        queries = os.path.join(tmp, "queries")
        build_query_dataset(
            os.path.join(tmp, "features"), os.path.join(tmp, "tokenized"), tokenizer,
            queries, queries_per_patient=6,
        )

        output = os.path.join(tmp, "run")
        report = run(
            QueryRunnerArguments(
                query_dataset_path=queries, tokenizer_path=tokenizer,
                hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                held_out_concept_fraction=0.2, validation_patient_fraction=0.25,
            ),
            TrainingArguments(
                output_dir=output, num_train_epochs=1, per_device_train_batch_size=16,
                use_cpu=True, report_to=[], save_strategy="no", remove_unused_columns=False,
                seed=1, logging_steps=1,
            ),
        )
        self.assertEqual(set(report), {"train", "validation", "test_unseen_concepts"})
        self.assertTrue(os.path.exists(os.path.join(output, "evaluation.json")))
        self.assertGreater(report["train"]["all"]["n"], 0)
        # The saved model reloads with its query vocabulary
        reloaded = PatientQueryModel.from_pretrained(output)
        self.assertEqual(reloaded.config.patient_feature_dim, FEATURE_DIM)
        self.assertTrue(len(reloaded.config.query_concept_ids) > 0)

    def test_empty_training_split_is_an_error(self):
        # one patient, all of them held out for evaluation: nothing to train on
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        Dataset.from_list(
            [{"patient_features": [0.0] * 4, "concept_id": "100", "start_days": 0, "end_days": 30,
              "label": True, "is_motor_task": False, "covered_tokens": 1, "person_id": 1,
              "prediction_time_posix": 0.0}]
        ).save_to_disk(tmp)
        with open(os.path.join(tmp, "query_vocabulary.json"), "w") as f:
            json.dump({"concept_ids": ["100"], "covered_tokens": {"100": 1}}, f)
        with self.assertRaisesRegex(ValueError, "training split is empty"):
            run(
                QueryRunnerArguments(query_dataset_path=tmp, tokenizer_path=tmp, validation_patient_fraction=1.0),
                TrainingArguments(output_dir=os.path.join(tmp, "o"), use_cpu=True, report_to=[]),
            )


if __name__ == "__main__":
    unittest.main()
