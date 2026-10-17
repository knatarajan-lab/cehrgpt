import argparse
import json
import os
import tempfile
import unittest

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from cehrgpt.open_vocab import question_embeddings as qe
from cehrgpt.open_vocab.question_encoder import EmbedderMismatchError
from cehrgpt.tools import zero_shot_qa_predictions as tool

from .model_integration_test import N_EMBD, QUESTION_DIM, tiny_model
from .question_encoder_test import FakeEncoder

IDS = list(range(100, 112))
ROWS = 7


def arguments(model_path, features, output, **kwargs):
    defaults = dict(
        model_path=model_path, features_folder=features, output_dir=output, questions=["asthma", "gout"],
        questions_file=None, concepts=None, embeddings_dir=None, windows=["30", "365"], gap_days=0.0,
        gap_days_column=None, label_column="boolean_value", evaluate=False, outcome_window=None,
        probe_similarity=0.999, fp16=False, batch_size=4,
        device="cpu",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class ZeroShotQATest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.embeddings_dir = os.path.join(self.tmp.name, "embeddings")
        universe = pl.DataFrame({"concept_id": IDS, "concept_name": [f"condition {i}" for i in IDS]})
        record = qe.embed_questions(universe, FakeEncoder(), self.embeddings_dir).embedder()
        model = tiny_model()
        model.config.open_vocab_embedder = record
        self.model_path = os.path.join(self.tmp.name, "model")
        model.save_pretrained(self.model_path)
        rng = np.random.default_rng(0)
        self.features = os.path.join(self.tmp.name, "features")
        os.makedirs(self.features)
        self.n = ROWS
        pq.write_table(pa.table({
            "subject_id": pa.array(np.arange(self.n), pa.int64()),
            "prediction_time_posix": pa.array(np.arange(self.n) * 1000.0),
            "boolean_value": pa.array([0, 1] * 3 + [0], pa.int8()),
            "features": pa.array([list(rng.normal(size=2 * N_EMBD).astype(np.float32)) for _ in range(self.n)]),
        }), os.path.join(self.features, "part-0.parquet"))
        self.output = os.path.join(self.tmp.name, "out")

    def run_tool(self, encoder=None, **kwargs):
        return tool.run(arguments(self.model_path, self.features, self.output, **kwargs), encoder or FakeEncoder())

    def test_it_writes_a_probability_per_patient_window_and_question(self):
        metrics = self.run_tool()
        table = pq.read_table(os.path.join(self.output, "predictions.parquet")).to_pandas()
        self.assertEqual(len(table), self.n * 2 * 2)
        self.assertEqual(set(table["question"]), {"asthma", "gout"})
        self.assertEqual(set(table["window_days"]), {"0-30", "0-365"})
        self.assertTrue(table["probability"].between(0, 1).all())
        self.assertEqual(metrics["num_patients"], self.n)
        self.assertEqual(metrics["question_embedder"]["model_name"], "fake")
        self.assertNotIn("probes", metrics["question_embedder"])
        with open(os.path.join(self.output, "metrics.json")) as f:
            self.assertEqual(json.load(f)["questions"], ["asthma", "gout"])

    def test_the_answer_does_not_depend_on_the_batch_size(self):
        self.run_tool(batch_size=2)
        small = pq.read_table(os.path.join(self.output, "predictions.parquet")).to_pandas()
        self.run_tool(batch_size=100)
        large = pq.read_table(os.path.join(self.output, "predictions.parquet")).to_pandas()
        np.testing.assert_allclose(small["probability"], large["probability"], rtol=1e-5)

    def test_concept_ids_are_asked_by_their_name(self):
        self.run_tool(questions=None, concepts=["100", "101"], embeddings_dir=self.embeddings_dir)
        table = pq.read_table(os.path.join(self.output, "predictions.parquet")).to_pandas()
        self.assertEqual(set(table["question"]), {"condition 100", "condition 101"})
        with self.assertRaisesRegex(ValueError, "not questions"):
            self.run_tool(questions=None, concepts=["5"], embeddings_dir=self.embeddings_dir)

    def test_evaluate_scores_the_outcome_window(self):
        metrics = self.run_tool(evaluate=True, outcome_window="365")
        outcome = metrics["outcome"]
        self.assertEqual(outcome["window_days"], "0-365")
        self.assertEqual(set(outcome["per_question"]), {"asthma", "gout"})
        self.assertEqual(outcome["per_question"]["asthma"]["n"], self.n)
        self.assertNotIn("any_of_the_questions", outcome)

    def test_another_encoder_is_refused(self):
        for other in (FakeEncoder(name="other"), FakeEncoder(salt="changed")):
            with self.assertRaises(EmbedderMismatchError):
                self.run_tool(other)
        self.assertFalse(os.path.exists(os.path.join(self.output, "predictions.parquet")))

    def test_a_model_without_a_record_is_refused(self):
        model = tiny_model()  # trained without the record of its embedder
        model.save_pretrained(os.path.join(self.tmp.name, "bare"))
        with self.assertRaises(EmbedderMismatchError):
            tool.run(arguments(os.path.join(self.tmp.name, "bare"), self.features, self.output), FakeEncoder())

    def test_features_of_the_wrong_width_are_refused(self):
        narrow = os.path.join(self.tmp.name, "narrow")
        os.makedirs(narrow)
        pq.write_table(pa.table({
            "subject_id": pa.array([1], pa.int64()), "prediction_time_posix": pa.array([0.0]),
            "boolean_value": pa.array([0], pa.int8()), "features": pa.array([[0.0] * N_EMBD]),
        }), os.path.join(narrow, "part-0.parquet"))
        with self.assertRaisesRegex(ValueError, "global and the local"):
            tool.run(arguments(self.model_path, narrow, self.output), FakeEncoder())
