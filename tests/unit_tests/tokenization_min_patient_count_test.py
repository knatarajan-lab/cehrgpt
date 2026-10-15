import dataclasses
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

    @staticmethod
    def ontology(parents):
        from cehrgpt.omop.ontology import Ontology

        ontology = Ontology.__new__(Ontology)
        ontology.parents_map = parents
        ontology.all_parents_map = {}
        ontology.concept_domain_map = {}
        ontology.concept_vocabulary_map = {}
        return ontology

    def test_descendants_count_towards_their_ancestors_once_per_patient(self):
        # 100 and 200 are below 900; patient 1 has both, patient 2 has 100, patient 4 neither
        counts = compute_statistics(self.dataset, data_args(100), self.ontology({"100": {"900"}, "200": {"900"}}))["concept_patient_counts"]
        self.assertEqual(counts["100"], 3)           # a token still counts its own patients
        self.assertEqual(counts["200"], 1)
        self.assertEqual(counts["900"], 3)           # patients 1, 2, 3: not 3 + 1, patient 1 counts once
        self.assertEqual(counts["300"], 2)           # unrelated tokens are unchanged
        # the prevalence that is summed up the ontology would have said 4 / 4 patients
        statistics = compute_statistics(self.dataset, data_args(100), self.ontology({"100": {"900"}, "200": {"900"}}))
        self.assertAlmostEqual(statistics["all_concept_code_stats"]["900"], 4 / 4)

    def test_chains_and_diamonds(self):
        # 100 < 900 < 950, and 300 sits below both 900 and 500
        ontology = self.ontology({"100": {"900"}, "900": {"950"}, "300": {"900", "500"}})
        counts = compute_statistics(self.dataset, data_args(100), ontology)["concept_patient_counts"]
        self.assertEqual(counts["900"], 4)   # patients 1, 2, 3 (via 100) and 4 (via 300): patient 2 has both, once
        self.assertEqual(counts["950"], 4)   # a grandparent counts the same patients
        self.assertEqual(counts["500"], 2)   # only through 300: patients 2 and 4

    def test_a_token_that_is_also_an_ancestor_counts_its_own_patients_too(self):
        dataset = Dataset.from_list([patient(1, ["900"]), patient(2, ["100"]), patient(3, ["100", "900"])])
        counts = compute_statistics(dataset, data_args(100), self.ontology({"100": {"900"}}))["concept_patient_counts"]
        self.assertEqual(counts["900"], 3)   # patient 3 has the parent and the child, once

    def test_the_counts_do_not_depend_on_the_batching(self):
        ontology = self.ontology({"100": {"900"}, "200": {"900"}, "300": {"900", "500"}})
        reference = compute_statistics(self.dataset, data_args(100), ontology)["concept_patient_counts"]
        for batch_size in (1, 2, 3):
            self.assertEqual(compute_statistics(self.dataset, data_args(batch_size), ontology)["concept_patient_counts"], reference)

    def test_without_an_ontology_only_the_token_counts(self):
        counts = self.counts(100)
        self.assertNotIn("900", counts)
        self.assertEqual(counts["100"], 3)

    def test_the_ancestors_helper(self):
        from cehrgpt.tokenization.tokenization_statistics import get_ancestors_by_code

        ontology = self.ontology({"100": {"900"}, "900": {"950"}, "300": {"900", "500"}})
        found = get_ancestors_by_code(ontology)
        self.assertEqual(found["100"], ("900", "950"))
        self.assertEqual(found["300"], ("500", "900", "950"))
        self.assertEqual(found["900"], ("950",))
        self.assertNotIn("950", found)
        self.assertEqual(get_ancestors_by_code(None), {})


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

    def test_the_entropy_filter_takes_its_cutoff_from_the_patient_count(self):
        import numpy as np

        statistics = compute_statistics(self.dataset, data_args())
        total, entropies = statistics["total"], statistics["all_concept_code_entropies"]

        def expected(threshold):
            share = max(1e-8, threshold / total)
            cutoff = np.log(1 - share) * (1 - share) + np.log(share) * share
            return {
                t for t in statistics["original_concept_codes"]
                if not is_clinical_event(t)
                or (entropies.get(t, 0.0) <= cutoff and self.patients_per_token[t] >= threshold)
            }

        sizes = []
        for threshold in (0, 3, 10):
            vocabulary = self.vocabulary(self.build(apply_entropy_filter=True, min_patient_count=threshold))
            clinical = {t for t in vocabulary if is_clinical_event(t)}
            self.assertEqual(clinical, {t for t in expected(threshold) if is_clinical_event(t)}, threshold)
            sizes.append(len(clinical))
        self.assertGreaterEqual(sizes[0], sizes[1])
        self.assertGreaterEqual(sizes[1], sizes[2])
        # the filter does something: it keeps fewer clinical tokens than no filter at all
        self.assertLess(sizes[0], len([t for t in self.vocabulary(self.baseline) if is_clinical_event(t)]))

    def test_min_prevalence_is_gone(self):
        import inspect

        self.assertNotIn("min_prevalence", inspect.signature(CehrGptTokenizer.train_tokenizer).parameters)
        self.assertNotIn("min_prevalence", inspect.signature(CehrGptTokenizer.expand_trained_tokenizer).parameters)
        self.assertNotIn("min_prevalence", {f.name for f in dataclasses.fields(CehrGPTArguments)})
        with self.assertRaisesRegex(ValueError, "min_prevalence"):
            HfArgumentParser(CehrGPTArguments).parse_dict({"min_prevalence": 0.001})

    def test_a_prevalence_is_a_patient_count_of_the_same_share(self):
        # what min_prevalence=p kept (without an ontology) is what min_patient_count=ceil(p * patients) keeps
        n_patients = len(self.frame)
        for share in (0.05, 0.2):
            count = -(-int(round(share * n_patients * 1000)) // 1000)
            kept = {t for t in self.vocabulary(self.baseline)
                    if not is_clinical_event(t) or self.patients_per_token.get(t, 0) / n_patients >= share - 1e-12}
            self.assertEqual(self.vocabulary(self.build(min_patient_count=count)), kept)

    def test_a_parent_is_kept_through_its_descendants(self):
        # a rare clinical token above a common one is kept when descendants count towards it
        from cehrgpt.omop.ontology import Ontology

        clinical = [t for t in self.patients_per_token if is_clinical_event(t)]
        child = max(clinical, key=lambda t: self.patients_per_token[t])
        rare = min((t for t in clinical if t != child), key=lambda t: self.patients_per_token[t])
        threshold = self.patients_per_token[rare] + 1
        self.assertGreaterEqual(self.patients_per_token[child], threshold)
        ontology = Ontology.__new__(Ontology)
        ontology.parents_map = {child: {rare}}
        ontology.all_parents_map = {}
        ontology.concept_domain_map = {}
        ontology.concept_vocabulary_map = {}
        # on its own the token is too rare ...
        self.assertNotIn(rare, self.vocabulary(self.build(min_patient_count=threshold)))
        # ... with the descendant it is had by every patient who has either of them
        with_ontology = self.build(min_patient_count=threshold, ontology=ontology)
        self.assertIn(rare, self.vocabulary(with_ontology))
        union = sum(
            1 for tokens in self.frame["concept_ids"] if {rare, child} & set(map(str, tokens))
        )
        self.assertGreaterEqual(union, threshold)
        # a threshold above the union removes it again
        self.assertNotIn(rare, self.vocabulary(self.build(min_patient_count=union + 1, ontology=ontology)))

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
