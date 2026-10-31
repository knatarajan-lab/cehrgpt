import json
import os
import tempfile
import unittest

import numpy as np
import polars as pl

from cehrgpt.open_vocab import question_universe as qu
from cehrgpt.open_vocab.assets import (
    OpenVocabAssets,
    build_hierarchy,
    read_concept_counts,
    vocabulary_concept_ids,
)

from .question_universe_test import CATEGORIES, tables

# share of the patients (1000 of them), with the concepts below summed in; Angina (140) and Arrhythmia (150) are tokens
STATS = {"140": 0.2, "150": 0.05, "130": 0.2, "120": 0.25, "160": 0.05, "180": 0.001, "year:2010": 0.9}


def write_ohdsi(folder):
    concept, ancestor = tables()
    for name, frame in (("concept", concept), ("concept_ancestor", ancestor)):
        os.makedirs(os.path.join(folder, name))
        frame.write_parquet(os.path.join(folder, name, "part-0.parquet"))


def write_tokenizer_stats(folder, stats=STATS):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "cehrgpt_concept_stats.json"), "w") as f:
        json.dump(stats, f)


def write_embeddings(folder, ids, dim=4, seed=0):
    os.makedirs(folder, exist_ok=True)
    matrix = np.random.default_rng(seed).normal(size=(len(ids), dim)).astype(np.float32)
    np.save(os.path.join(folder, "embeddings.npy"), matrix)
    pl.DataFrame({"row": list(range(len(ids))), "concept_id": ids, "text": [str(i) for i in ids]}).write_parquet(
        os.path.join(folder, "question_texts.parquet")
    )
    with open(os.path.join(folder, "question_embedder.json"), "w") as f:
        json.dump({"model": "test"}, f)
    return matrix


class AssetsTest(unittest.TestCase):
    """The universe and the embeddings are made without a tokenizer; the tokenizer only prunes, when training starts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ohdsi = os.path.join(self.tmp.name, "ohdsi")
        write_ohdsi(self.ohdsi)
        # the universe, built from the vocabulary alone
        concept, ancestor = tables()
        self.universe_dir = os.path.join(self.tmp.name, "universes")
        os.makedirs(self.universe_dir)
        self.universe = os.path.join(self.universe_dir, "condition_universe.parquet")
        qu.build_pool_universe(concept, ancestor, "Condition", CATEGORIES).write_parquet(self.universe)
        self.universe_ids = pl.read_parquet(self.universe)["concept_id"].to_list()  # 120, 130, 140, 150, 160
        self.tokenizer_dir = os.path.join(self.tmp.name, "tokenizer")
        write_tokenizer_stats(self.tokenizer_dir)
        self.vocab = ["[PAD]", "140", "150", "180", "year:2010"]

    def hierarchy(self, vocab=None, min_patient_count=0):
        counts = read_concept_counts([self.tokenizer_dir]) if min_patient_count else None
        return build_hierarchy(
            [self.universe], self.ohdsi, vocabulary_concept_ids(vocab or self.vocab), counts, min_patient_count
        )

    def test_tokens_roll_up_to_the_questions_of_the_universe(self):
        h = self.hierarchy()
        self.assertEqual(h.questions_of(140), (120, 130, 140))
        self.assertEqual(h.questions_of(150), (120, 150, 160))
        self.assertEqual(sorted(h.question_ids().tolist()), [120, 130, 140, 150, 160])

    def test_the_vocabulary_is_the_digit_tokens(self):
        self.assertEqual(vocabulary_concept_ids(["[PAD]", "140", "year:2010", "D5", "9201"]), {140, 9201})

    def test_a_token_that_is_below_no_question_is_left_out(self):
        h = self.hierarchy()
        self.assertEqual(h.questions_of(180), ())  # the orphan disease is below no category
        self.assertEqual(h.questions_of("year:2010"), ())

    def test_a_question_without_a_token_below_it_is_pruned(self):
        # only Arrhythmia is in the vocabulary: Angina and Coronary disease have no token at or below them
        h = self.hierarchy(vocab=["150"])
        self.assertEqual(sorted(h.question_ids().tolist()), [120, 150, 160])

    def test_min_patient_count_uses_the_statistics_of_the_training_data(self):
        # 1000 patients: Angina 200, Coronary 200, Heart disease 250, Arrhythmia 50, Lung disease 50
        h = self.hierarchy(min_patient_count=100)
        self.assertEqual(sorted(h.question_ids().tolist()), [120, 130, 140])
        self.assertEqual(h.questions_of(150), (120,))  # Arrhythmia and Lung disease are too rare to be questions
        self.assertEqual(h.questions_of(140), (120, 130, 140))

    def test_a_concept_that_is_not_in_the_vocabulary_is_never_an_event(self):
        # Arrhythmia has 50 patients in the data but is no token: nothing rolls up to its own question
        h = self.hierarchy(vocab=["140"])
        self.assertEqual(h.questions_of(150), ())
        self.assertNotIn(150, h.question_ids().tolist())

    def test_a_minimum_needs_the_counts(self):
        with self.assertRaises(ValueError):
            build_hierarchy([self.universe], self.ohdsi, {140}, None, 10)

    def test_concept_counts_are_patients(self):
        counts = read_concept_counts([self.tokenizer_dir])
        self.assertEqual((counts[140], counts[120], counts[160], counts[180]), (200, 250, 50, 1))
        self.assertNotIn("year:2010", counts)

    def test_counts_of_several_tokenizers_take_the_maximum(self):
        other = os.path.join(self.tmp.name, "other")
        write_tokenizer_stats(other, {"140": 0.5, "120": 0.5})
        counts = read_concept_counts([self.tokenizer_dir, other])
        self.assertEqual(counts[140], 200)  # the larger of 200 and 1
        self.assertEqual(counts[120], 250)

    def build(self, embeddings_dir, **kwargs):
        return OpenVocabAssets.build(
            universe_dir=self.universe_dir, vocab_path=self.ohdsi, embeddings_dir=embeddings_dir,
            token_ids=vocabulary_concept_ids(self.vocab), **kwargs,
        )

    def test_the_embeddings_are_cut_to_the_questions_with_tokens(self):
        embeddings_dir = os.path.join(self.tmp.name, "embeddings")
        matrix = write_embeddings(embeddings_dir, self.universe_ids)  # the whole universe
        counts = read_concept_counts([self.tokenizer_dir])
        loaded = self.build(embeddings_dir, concept_counts=counts, min_patient_count=100)
        kept = [i for i in self.universe_ids if i in (120, 130, 140)]
        self.assertEqual(loaded.question_concept_ids.tolist(), kept)
        np.testing.assert_array_equal(loaded.embeddings, matrix[[self.universe_ids.index(i) for i in kept]])
        self.assertEqual(loaded.num_rows, 3)
        self.assertEqual(loaded.hierarchy.questions_of(140), (120, 130, 140))

    def test_without_a_minimum_every_question_with_a_token_below_is_used(self):
        embeddings_dir = os.path.join(self.tmp.name, "embeddings")
        write_embeddings(embeddings_dir, self.universe_ids)
        self.assertEqual(self.build(embeddings_dir).num_rows, 5)

    def test_the_same_universe_and_embeddings_serve_another_tokenizer(self):
        embeddings_dir = os.path.join(self.tmp.name, "embeddings")
        write_embeddings(embeddings_dir, self.universe_ids)
        other = OpenVocabAssets.build(
            self.universe_dir, self.ohdsi, embeddings_dir, token_ids=vocabulary_concept_ids(["150"])
        )
        self.assertEqual(sorted(other.question_concept_ids.tolist()), [120, 150, 160])
        self.assertEqual(self.build(embeddings_dir).num_rows, 5)

    def test_embeddings_of_another_universe_are_rejected(self):
        embeddings_dir = os.path.join(self.tmp.name, "embeddings")
        write_embeddings(embeddings_dir, [7001, 7002])
        with self.assertRaises(ValueError):
            self.build(embeddings_dir)

    def test_a_folder_without_universes_is_an_error(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        with self.assertRaises(FileNotFoundError):
            OpenVocabAssets.build(empty, self.ohdsi, empty, token_ids={140})

    def test_a_universe_file_must_be_named_after_its_domain(self):
        wrong = os.path.join(self.tmp.name, "things.parquet")
        pl.read_parquet(self.universe).write_parquet(wrong)
        with self.assertRaises(ValueError):
            build_hierarchy([wrong], self.ohdsi, {140})


if __name__ == "__main__":
    unittest.main()
