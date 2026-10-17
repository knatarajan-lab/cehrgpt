import hashlib
import shutil
import tempfile
import unittest

import numpy as np
import polars as pl
import torch

from cehrgpt.open_vocab import question_embeddings as qe
from cehrgpt.open_vocab.head import OpenVocabularyHead
from cehrgpt.open_vocab.predictor import OpenVocabPredictor
from cehrgpt.open_vocab.question_encoder import (
    EmbedderMismatchError,
    QuestionEncoder,
    QuestionVectors,
    fingerprint,
    require_same_embedder,
)

DIM = 8
UNIVERSE = pl.DataFrame({"concept_id": list(range(20)), "concept_name": [f"condition {i}" for i in range(20)]})


class FakeEncoder:
    """A deterministic stand-in for the language model: the vector depends on the text and the salt only."""

    def __init__(self, name="fake", revision="abc", salt="", dim=DIM, max_length=128):
        self.name, self.revision, self.salt, self.dim, self.max_length = name, revision, salt, dim, max_length

    def describe(self):
        return {"model_name": self.name, "revision": self.revision, "pooling": "fake", "normalize": "l2", "max_length": self.max_length}

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha1((self.salt + text).encode()).hexdigest()[:8], 16)
            out[i] = np.random.RandomState(seed).randn(self.dim)
            out[i] /= np.linalg.norm(out[i])
        return out


def embedder_of(encoder=None, folder=None, template="{text}"):
    result = qe.embed_questions(UNIVERSE, encoder or FakeEncoder(), folder, template)
    return result.embedder()


class QuestionEncoderTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.embedder = embedder_of(folder=self.folder)

    def test_the_record_has_probes_that_are_the_rendered_text_and_its_vector(self):
        probes = self.embedder["probes"]
        self.assertEqual(len(probes), qe.PROBE_COUNT)
        record = embedder_of(FakeEncoder(), tempfile.mkdtemp(), template="Condition: {text}")
        self.assertTrue(all(p["text"].startswith("Condition: condition") for p in record["probes"]))
        text = record["probes"][0]["text"]
        np.testing.assert_allclose(record["probes"][0]["vector"], FakeEncoder().encode([text])[0], rtol=1e-6)
        self.assertEqual(record["dim"], DIM)

    def test_the_same_encoder_is_accepted_and_embeds_with_the_template(self):
        record = embedder_of(FakeEncoder(), tempfile.mkdtemp(), template="Condition: {text}")
        encoder = QuestionEncoder(record, FakeEncoder())
        questions = encoder.embed(["asthma"])
        self.assertIsInstance(questions, QuestionVectors)
        np.testing.assert_allclose(questions.vectors.numpy(), FakeEncoder().encode(["Condition: asthma"]), rtol=1e-6)

    def test_another_model_a_revision_or_a_length_is_refused(self):
        for other in (FakeEncoder(name="other"), FakeEncoder(revision="def"), FakeEncoder(max_length=64)):
            with self.assertRaises(EmbedderMismatchError, msg=other.describe()):
                QuestionEncoder(self.embedder, other)

    def test_an_encoder_with_the_same_name_that_gives_other_vectors_is_refused(self):
        with self.assertRaisesRegex(EmbedderMismatchError, "does not reproduce"):
            QuestionEncoder(self.embedder, FakeEncoder(salt="changed"))

    def test_another_width_is_refused(self):
        with self.assertRaises(EmbedderMismatchError):
            QuestionEncoder(self.embedder, FakeEncoder(dim=DIM + 2))

    def test_an_unknown_revision_is_not_a_mismatch_but_the_probes_still_decide(self):
        QuestionEncoder(self.embedder, FakeEncoder(revision=None))
        with self.assertRaises(EmbedderMismatchError):
            QuestionEncoder(self.embedder, FakeEncoder(revision=None, salt="x"))

    def test_a_model_without_a_record_cannot_be_asked(self):
        for record in (None, {}, {k: v for k, v in self.embedder.items() if k != "probes"}):
            with self.assertRaises(EmbedderMismatchError):
                QuestionEncoder(record, FakeEncoder())

    def test_the_fingerprint_ignores_the_probes_and_the_library_version(self):
        other = {**self.embedder, "transformers_version": "0.0", "probes": []}
        self.assertEqual(fingerprint(self.embedder), fingerprint(other))
        self.assertNotEqual(fingerprint(self.embedder), fingerprint({**self.embedder, "template": "x {text}"}))

    def test_require_same_embedder(self):
        require_same_embedder(self.embedder, dict(self.embedder))
        with self.assertRaisesRegex(EmbedderMismatchError, "template"):
            require_same_embedder(self.embedder, {**self.embedder, "template": "x {text}"})
        with self.assertRaisesRegex(EmbedderMismatchError, "no record"):
            require_same_embedder(None, self.embedder)


class OpenVocabPredictorTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        self.embedder = embedder_of(folder=folder)
        torch.manual_seed(0)
        head = OpenVocabularyHead(feature_dim=12, question_dim=DIM, hidden_dim=6, num_time_pieces=3, residual_dim=4)
        self.predictor = OpenVocabPredictor(head, [0.0, 1.0, 7.0, float("inf")], self.embedder)
        self.encoder = QuestionEncoder(self.embedder, FakeEncoder())
        self.features = torch.randn(5, 12)

    def test_verified_questions_are_answered_with_probabilities(self):
        questions = self.encoder.embed(["asthma", "gout"])
        p = self.predictor.probabilities(self.features, questions, [(0.0, 7.0), (0.0, 30.0)])
        self.assertEqual(tuple(p.shape), (5, 2, 2))
        self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))
        self.assertTrue(bool((p[:, 1] >= p[:, 0]).all()))  # a longer window is never less likely

    def test_raw_vectors_are_refused(self):
        with self.assertRaises(TypeError):
            self.predictor.probabilities(self.features, torch.randn(2, DIM), [(0.0, 7.0)])

    def test_vectors_of_another_embedder_are_refused(self):
        other = {**self.embedder, "template": "x {text}"}
        foreign = QuestionVectors(torch.randn(2, DIM), fingerprint(other))
        with self.assertRaises(EmbedderMismatchError):
            self.predictor.probabilities(self.features, foreign, [(0.0, 7.0)])

    def test_the_features_must_be_global_and_local(self):
        with self.assertRaisesRegex(ValueError, "global and the local"):
            self.predictor.probabilities(torch.randn(5, 6), self.encoder.embed(["gout"]), [(0.0, 7.0)])

    def test_a_predictor_needs_the_record(self):
        with self.assertRaises(EmbedderMismatchError):
            OpenVocabPredictor(self.predictor.head, [0.0, 1.0], None)
