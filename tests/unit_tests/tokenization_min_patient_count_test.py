import os
import unittest
from collections import Counter

import pandas as pd
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset
from transformers import HfArgumentParser

from cehrgpt.gpt_utils import is_clinical_event
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer
from cehrgpt.tokenization.tokenization_statistics import compute_statistics

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data")


def data_args(batch_size=100):
    return DataTrainingArguments(
        data_folder="x", dataset_prepared_path="y",
        preprocessing_num_workers=1, preprocessing_batch_size=batch_size,
    )


def patient(person_id, tokens):
    n = len(tokens)
    return {
        "person_id": person_id,
        "concept_ids": ["year:2000", "age:30", "8532", "8527"] + tokens,
        "concept_value_masks": [0] * (n + 4),
        "concept_values": [0.0] * (n + 4),
        "units": ["N/A"] * (n + 4),
    }


class PatientCountsTest(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_list(
            [
                patient(1, ["100", "100", "200", "[VS]"]),  # 100 twice: still one patient
                patient(2, ["100", "300", "[VS]"]),
                patient(3, ["100", "[VS]", "[VE]"]),
                patient(4, ["300", "400"]),
            ]
        )

    def counts(self, batch_size, **kwargs):
        return compute_statistics(self.dataset, data_args(batch_size), **kwargs)["concept_patient_counts"]

    def test_a_token_is_counted_once_per_patient(self):
        counts = self.counts(100)
        self.assertEqual(counts["100"], 3)   # patients 1, 2, 3 although patient 1 has it twice
        self.assertEqual(counts["200"], 1)
        self.assertEqual(counts["300"], 2)
        self.assertEqual(counts["400"], 1)
        self.assertEqual(counts["[VS]"], 3)
        self.assertEqual(counts["8532"], 4)  # the demographic tokens are counted as well

    def test_counts_add_up_across_batches(self):
        reference = self.counts(100)
        for batch_size in (1, 2, 3):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(self.counts(batch_size), reference)

    def test_counts_are_integers_and_agree_with_the_prevalence(self):
        statistics = compute_statistics(self.dataset, data_args(2))
        for token, count in statistics["concept_patient_counts"].items():
            self.assertIsInstance(count, int)
            self.assertAlmostEqual(count / 4, statistics["all_concept_code_stats"][token], places=9)

    def test_counts_stay_at_the_token_with_an_ontology(self):
        # The prevalence is summed up the ontology; the patient count of a token is its own
        from cehrgpt.omop.ontology import Ontology

        ontology = Ontology.__new__(Ontology)
        ontology.parents_map = {"100": {"900"}, "200": {"900"}}
        ontology.all_parents_map = {}
        ontology.concept_domain_map = {}
        ontology.concept_vocabulary_map = {}
        statistics = compute_statistics(self.dataset, data_args(100), ontology)
        counts = statistics["concept_patient_counts"]
        self.assertNotIn("900", counts)
        self.assertEqual(counts["100"], 3)
        self.assertGreater(statistics["all_concept_code_stats"]["900"], 0)


class TokenizerFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_parquet(os.path.join(SAMPLE, "pretrain", "patient_sequence.parquet"))
        cls.dataset = Dataset.from_parquet(os.path.join(SAMPLE, "pretrain", "patient_sequence.parquet"))
        concept = pd.read_parquet(os.path.join(SAMPLE, "omop_vocab", "concept"))
        cls.names = {str(r.concept_id): r.concept_name for r in concept.itertuples()}
        cls.embeddings = PretrainedEmbeddings(os.path.join(SAMPLE, "pretrained_embeddings"))
        # The number of patients per token, counted independently of the code under test
        cls.patients_per_token = Counter()
        for tokens in cls.frame["concept_ids"]:
            cls.patients_per_token.update(set(map(str, tokens)))
        cls.baseline = cls.build()

    @classmethod
    def build(cls, dataset=None, **kwargs):
        return CehrGptTokenizer.train_tokenizer(
            dataset if dataset is not None else cls.dataset, cls.names, data_args(),
            cls.embeddings, **kwargs,
        )

    def vocabulary(self, tokenizer):
        return set(tokenizer.get_vocab())

    def test_the_default_changes_nothing(self):
        baseline = self.vocabulary(self.baseline)
        self.assertEqual(self.vocabulary(self.build(min_patient_count=0)), baseline)
        self.assertEqual(self.vocabulary(self.build(min_patient_count=None)), baseline)
        # every token occurs in at least one patient
        self.assertEqual(self.vocabulary(self.build(min_patient_count=1)), baseline)

    def test_only_clinical_tokens_with_too_few_patients_are_removed(self):
        threshold = 10
        kept = self.vocabulary(self.build(min_patient_count=threshold))
        baseline = self.vocabulary(self.baseline)
        removed = baseline - kept
        self.assertTrue(removed, "the sample has no token that rare, so the test says nothing")
        for token in removed:
            self.assertTrue(is_clinical_event(token), token)
            self.assertLess(self.patients_per_token[token], threshold, token)
        for token in kept:
            if is_clinical_event(token) and token in self.patients_per_token:
                self.assertGreaterEqual(self.patients_per_token[token], threshold, token)
        # nothing that qualifies was lost
        expected_removed = {
            t for t in baseline
            if is_clinical_event(t) and self.patients_per_token.get(t, 0) < threshold
        }
        self.assertEqual(removed, expected_removed)

    def test_structural_tokens_are_never_removed(self):
        kept = self.vocabulary(self.build(min_patient_count=10_000))
        baseline = self.vocabulary(self.baseline)
        structural = {t for t in baseline if not is_clinical_event(t)}
        self.assertTrue(structural)
        self.assertTrue(structural <= kept)
        self.assertIn("[VS]", kept)
        # with a threshold no token can reach, no clinical token is left
        self.assertFalse({t for t in kept if is_clinical_event(t)})

    def test_a_removed_token_is_out_of_vocabulary(self):
        tokenizer = self.build(min_patient_count=10)
        baseline = self.vocabulary(self.baseline)
        removed = sorted(baseline - self.vocabulary(tokenizer))
        kept_clinical = sorted(t for t in self.vocabulary(tokenizer) if is_clinical_event(t))
        oov = tokenizer.oov_token_id if hasattr(tokenizer, "oov_token_id") else tokenizer.encode(["not a token"])[0]
        self.assertEqual(tokenizer.encode([removed[0]])[0], oov)
        self.assertNotEqual(tokenizer.encode([kept_clinical[0]])[0], oov)

    def test_it_combines_with_the_prevalence_filter(self):
        by_prevalence = self.vocabulary(self.build(min_prevalence=0.08))
        by_count = self.vocabulary(self.build(min_patient_count=5))
        both = self.vocabulary(self.build(min_prevalence=0.08, min_patient_count=5))
        self.assertEqual(both, by_prevalence & by_count)
        self.assertLess(len(both), len(self.vocabulary(self.baseline)))

    def test_motor_tasks_come_only_from_the_kept_tokens(self):
        tokenizer = self.build(min_patient_count=30, num_motor_tasks=40)
        tasks = set(tokenizer._motor_time_to_event_codes)
        self.assertTrue(tasks)
        self.assertTrue(tasks <= self.vocabulary(tokenizer))
        for code in tasks:
            self.assertGreaterEqual(self.patients_per_token[code], 30, code)

    def test_negative_threshold(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            self.build(min_patient_count=-1)

    def test_an_expanded_tokenizer_applies_the_threshold_to_the_new_data(self):
        first = self.dataset.select(range(50))
        second = self.dataset.select(range(50, 100))
        base = self.build(dataset=first)
        expanded = CehrGptTokenizer.expand_trained_tokenizer(
            cehrgpt_tokenizer=base, dataset=second, concept_name_mapping=self.names,
            data_args=data_args(), pretrained_concept_embedding_model=self.embeddings,
            min_patient_count=5,
        )
        new_tokens = self.vocabulary(expanded) - self.vocabulary(base)
        second_counts = Counter()
        for tokens in self.frame["concept_ids"].iloc[50:]:
            second_counts.update(set(map(str, tokens)))
        clinical_new = [t for t in new_tokens if is_clinical_event(t)]
        self.assertTrue(clinical_new, "no clinical token was added, so the test says nothing")
        for token in clinical_new:
            self.assertGreaterEqual(second_counts[token], 5, token)
        # without the threshold the same expansion adds rarer tokens too
        unfiltered = CehrGptTokenizer.expand_trained_tokenizer(
            cehrgpt_tokenizer=base, dataset=second, concept_name_mapping=self.names,
            data_args=data_args(), pretrained_concept_embedding_model=self.embeddings,
        )
        self.assertGreater(len(self.vocabulary(unfiltered)), len(self.vocabulary(expanded)))


class ArgumentTest(unittest.TestCase):
    def test_default_and_parsing(self):
        self.assertEqual(CehrGPTArguments().min_patient_count, 0)
        parser = HfArgumentParser(CehrGPTArguments)
        parsed, = parser.parse_args_into_dataclasses(args=["--min_patient_count", "11"])
        self.assertEqual(parsed.min_patient_count, 11)
        self.assertEqual(parser.parse_args_into_dataclasses(args=[])[0].min_patient_count, 0)


if __name__ == "__main__":
    unittest.main()
