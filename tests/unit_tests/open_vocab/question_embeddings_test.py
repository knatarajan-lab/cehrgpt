import hashlib
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import polars as pl

from cehrgpt.open_vocab import question_embeddings as qe


class FakeEncoder:
    """A deterministic stand-in for the language model: the vector depends on the text only."""

    def __init__(self, name="fake", dim=8):
        self.name, self.dim, self.calls = name, dim, 0

    def describe(self):
        return {"model_name": self.name, "pooling": "fake", "normalize": "l2"}

    def encode(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
            out[i] = np.random.RandomState(seed).randn(self.dim)
            out[i] /= np.linalg.norm(out[i])
        return out


UNIVERSE = pl.DataFrame(
    {"concept_id": [11, 12, 13, 12], "concept_name": ["Angina", "Arrhythmia", "Pneumonia", "Arrhythmia"]}
)


class QuestionEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_one_row_per_question_in_the_order_of_the_universe(self):
        table = qe.build_text_table(UNIVERSE)
        self.assertEqual(table["concept_id"].to_list(), [11, 12, 13])  # the duplicate is dropped
        self.assertEqual(table["row"].to_list(), [0, 1, 2])
        self.assertEqual(table["text"].to_list(), ["Angina", "Arrhythmia", "Pneumonia"])

    def test_embeddings_are_looked_up_by_concept_id(self):
        result = qe.embed_questions(UNIVERSE, FakeEncoder(), self.folder)
        self.assertEqual(result.embeddings.shape, (3, 8))
        expected = FakeEncoder().encode(["Pneumonia", "Angina"])
        np.testing.assert_allclose(result.vectors([13, 11]), expected, rtol=1e-6)
        self.assertEqual(result.rows([12, 11]).tolist(), [1, 0])
        self.assertIn(11, result)
        self.assertNotIn(99, result)
        with self.assertRaisesRegex(KeyError, "no embedding"):
            result.rows([99])

    def test_the_template_is_applied_to_the_name(self):
        result = qe.embed_questions(UNIVERSE, FakeEncoder(), self.folder, template="condition: {text}")
        np.testing.assert_allclose(result.vectors([11]), FakeEncoder().encode(["condition: Angina"]), rtol=1e-6)

    def test_load_returns_what_was_saved(self):
        saved = qe.embed_questions(UNIVERSE, FakeEncoder(), self.folder)
        loaded = qe.QuestionEmbeddings.load(self.folder)
        np.testing.assert_allclose(loaded.embeddings, saved.embeddings)
        self.assertEqual(loaded.texts["concept_id"].to_list(), [11, 12, 13])
        self.assertEqual(loaded.dim, 8)
        with open(os.path.join(self.folder, qe.DESCRIPTION_FILE)) as f:
            description = json.load(f)
        self.assertEqual(description["template"], "{text}")
        self.assertEqual(description["model_name"], "fake")

    def test_a_second_run_reuses_the_cache_and_a_new_encoder_recomputes(self):
        first = FakeEncoder()
        qe.embed_questions(UNIVERSE, first, self.folder)
        again = FakeEncoder()
        qe.embed_questions(UNIVERSE, again, self.folder)
        self.assertEqual(again.calls, 0)
        other = FakeEncoder(name="other model")
        qe.embed_questions(UNIVERSE, other, self.folder)
        self.assertEqual(other.calls, 3)

    def test_the_matrix_and_the_texts_must_agree(self):
        with self.assertRaisesRegex(ValueError, "embeddings for"):
            qe.QuestionEmbeddings(np.zeros((2, 4), dtype=np.float32), qe.build_text_table(UNIVERSE))


if __name__ == "__main__":
    unittest.main()
