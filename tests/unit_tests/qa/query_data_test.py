import unittest

import numpy as np
import torch
from datasets import Dataset

from cehrgpt.models.patient_query_model import PatientQueryModel, PatientQueryModelConfig
from cehrgpt.qa.concept_embeddings import (
    build_concept_embedding_matrix,
    token_vectors_from_weights,
)
from cehrgpt.qa.concept_hierarchy import ConceptHierarchy
from cehrgpt.qa.query_data import (
    QueryCollator,
    evaluate_queries,
    predict_dataset,
    related_concepts,
    split_queries,
)

PARENTS = {"11": {"10"}, "12": {"10"}, "10": {"30"}, "20": {"30"}, "21": {"30"}}
VOCAB = ["10", "11", "12", "20", "21", "30"]


def rows(n_patients=40, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for person in range(n_patients):
        for concept in rng.choice(VOCAB, 4, replace=False):
            out.append(
                {
                    "patient_features": rng.randn(4).tolist(),
                    "concept_id": str(concept),
                    "start_days": 0,
                    "end_days": 90,
                    "label": bool(rng.randint(2)),
                    "is_motor_task": concept in ("20", "21"),
                    "covered_tokens": 1,
                    "person_id": person,
                    "prediction_time_posix": 0.0,
                }
            )
    return Dataset.from_list(out)


def model():
    return PatientQueryModel(
        PatientQueryModelConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            patient_feature_dim=4, query_concept_ids=VOCAB,
        )
    )


class CollatorTest(unittest.TestCase):
    def test_tensors_and_order(self):
        collator = QueryCollator(VOCAB)
        batch = collator(
            [
                {"patient_features": [1.0] * 4, "concept_id": "12", "start_days": 0, "end_days": 30, "label": True},
                {"patient_features": [2.0] * 4, "concept_id": "30", "start_days": 30, "end_days": 90, "label": False},
            ]
        )
        self.assertEqual(batch["concept_index"].tolist(), [2, 5])
        self.assertEqual(batch["labels"].tolist(), [1.0, 0.0])
        self.assertEqual(tuple(batch["patient_features"].shape), (2, 4))
        self.assertEqual(batch["end_days"].tolist(), [30.0, 90.0])

    def test_unknown_concept(self):
        with self.assertRaises(KeyError):
            QueryCollator(VOCAB)([{"patient_features": [0.0] * 4, "concept_id": "x", "start_days": 0, "end_days": 1, "label": 1}])

    def test_from_a_model(self):
        self.assertEqual(QueryCollator(model()).index["11"], 1)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.hierarchy = ConceptHierarchy(PARENTS)

    def test_related_concepts(self):
        self.assertEqual(related_concepts({"11"}, self.hierarchy, VOCAB), {"11", "10", "30"})
        self.assertEqual(related_concepts({"10"}, self.hierarchy, VOCAB), {"10", "30", "11", "12"})
        # without the ancestors only the descendants (and the concept itself) remain
        self.assertEqual(related_concepts({"10"}, self.hierarchy, VOCAB, include_ancestors=False), {"10", "11", "12"})
        self.assertEqual(related_concepts(set(), self.hierarchy, VOCAB), set())

    def test_splits_do_not_leak(self):
        data = rows()
        for block_ancestors in (False, True):
            parts = split_queries(data, self.hierarchy, VOCAB, 0.25, 0.2, seed=1, block_ancestors=block_ancestors)
            train, validation, unseen = (set(parts[k]["person_id"]) for k in ("train", "validation", "test_unseen_concepts"))
            self.assertTrue(train.isdisjoint(validation | unseen))
            train_concepts = set(parts["train"]["concept_id"])
            held_out = set(parts["test_unseen_concepts"]["concept_id"])
            self.assertTrue(held_out)
            self.assertTrue(held_out.isdisjoint(train_concepts))
            descendants = related_concepts(held_out, self.hierarchy, VOCAB, include_ancestors=False)
            self.assertTrue(descendants.isdisjoint(train_concepts))
            if block_ancestors:
                self.assertTrue(related_concepts(held_out, self.hierarchy, VOCAB).isdisjoint(train_concepts))
            self.assertTrue(held_out.isdisjoint(set(parts["validation"]["concept_id"])))

    def test_only_narrow_concepts_are_held_out(self):
        covered = {"10": 2, "30": 6, "11": 1, "12": 1, "20": 1, "21": 1}
        for seed in range(10):
            parts = split_queries(rows(), self.hierarchy, VOCAB, 0.25, 0.5, seed=seed, covered_tokens=covered, max_held_out_covered_tokens=1)
            self.assertTrue(set(parts["test_unseen_concepts"]["concept_id"]) <= {"11", "12", "20", "21"})
            # the broad concepts stay available for training
            self.assertIn("30", set(parts["train"]["concept_id"]))

    def test_reproducible(self):
        data = rows()
        first = split_queries(data, self.hierarchy, VOCAB, 0.25, 0.2, seed=3)
        second = split_queries(data, self.hierarchy, VOCAB, 0.25, 0.2, seed=3)
        for key in first:
            self.assertEqual(first[key]["person_id"], second[key]["person_id"])

    def test_no_held_out_concepts(self):
        parts = split_queries(rows(), self.hierarchy, VOCAB, 0.25, 0.0)
        self.assertEqual(len(parts["test_unseen_concepts"]), 0)
        self.assertGreater(len(parts["validation"]), 0)


class EvaluationTest(unittest.TestCase):
    def test_report_groups_and_ranges(self):
        data = rows(30)
        report = evaluate_queries(model(), data, batch_size=7)
        self.assertEqual(report["all"]["n"], len(data))
        self.assertEqual(report["motor_tasks"]["n"] + report["outside_motor"]["n"], len(data))
        for group in report.values():
            self.assertTrue(0.0 <= group["auroc"] <= 1.0)
            self.assertTrue(0.0 <= group["accuracy"] <= 1.0)
        self.assertEqual(len(predict_dataset(model(), data, 5)), len(data))

    def test_auroc_is_undefined_with_one_class(self):
        data = rows(10).filter(lambda r: r["label"])
        report = evaluate_queries(model(), data)
        self.assertIsNone(report["all"]["auroc"])
        self.assertEqual(report["all"]["yes_fraction"], 1.0)

    def test_empty_dataset(self):
        report = evaluate_queries(model(), rows(5).select([]))
        self.assertEqual(report["all"]["n"], 0)
        self.assertIsNone(report["all"]["auroc"])


class ConceptEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.hierarchy = ConceptHierarchy(PARENTS)
        self.vectors = {
            "11": np.array([1.0, 0.0]),
            "12": np.array([3.0, 0.0]),
            "20": np.array([0.0, 4.0]),
        }

    def test_tokens_use_their_own_vector_and_parents_the_mean_of_descendants(self):
        matrix, found = build_concept_embedding_matrix(VOCAB, self.hierarchy, self.vectors)
        self.assertEqual(matrix.shape, (6, 2))
        np.testing.assert_allclose(matrix[VOCAB.index("11")], [1.0, 0.0])
        np.testing.assert_allclose(matrix[VOCAB.index("10")], [2.0, 0.0])  # mean of 11 and 12
        np.testing.assert_allclose(matrix[VOCAB.index("30")], [4.0 / 3, 4.0 / 3])
        # 21 has no vector and no described descendant, so it is noise
        self.assertLess(np.abs(matrix[VOCAB.index("21")]).max(), 0.2)
        self.assertAlmostEqual(found, 5 / 6)

    def test_matrix_can_initialise_the_model(self):
        matrix, _ = build_concept_embedding_matrix(VOCAB, self.hierarchy, self.vectors)
        query_model = PatientQueryModel(
            PatientQueryModelConfig(
                hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                patient_feature_dim=4, query_concept_ids=VOCAB, concept_embedding_dim=2,
            )
        )
        query_model.init_concept_embeddings(matrix, freeze=True)
        self.assertEqual(query_model.concept_projector.in_features, 2)

    def test_token_vectors_from_weights(self):
        weights = np.arange(12, dtype=float).reshape(6, 2)
        vectors = token_vectors_from_weights({"11": 3, "x": 0}, weights, ["11", "20"])
        self.assertEqual(list(vectors), ["11"])
        np.testing.assert_allclose(vectors["11"], [6.0, 7.0])

    def test_empty_vectors(self):
        with self.assertRaises(ValueError):
            build_concept_embedding_matrix(VOCAB, self.hierarchy, {})


if __name__ == "__main__":
    unittest.main()
