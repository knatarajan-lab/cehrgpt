import inspect
import tempfile
import unittest

import numpy as np
import torch

from cehrgpt.models.patient_query_model import (
    PatientQueryModel,
    PatientQueryModelConfig,
    window_features,
)

CONCEPTS = ["c0", "c1", "c2"]


def config(lm_type="gpt2", **kwargs):
    return PatientQueryModelConfig(
        lm_type=lm_type,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        patient_feature_dim=8,
        query_concept_ids=CONCEPTS,
        **kwargs,
    )


def batch(n=4, seed=0):
    torch.manual_seed(seed)
    return {
        "patient_features": torch.randn(n, 8),
        "concept_index": torch.randint(0, 3, (n,)),
        "start_days": torch.zeros(n),
        "end_days": torch.full((n,), 90.0),
    }


class PatientQueryModelTest(unittest.TestCase):
    def test_outputs_for_both_architectures(self):
        for lm_type in ("gpt2", "llama"):
            for prefix in (1, 3):
                with self.subTest(lm_type=lm_type, prefix=prefix):
                    model = PatientQueryModel(config(lm_type, num_prefix_tokens=prefix))
                    inputs = batch()
                    out = model(**inputs, labels=torch.tensor([1, 0, 1, 0]))
                    self.assertEqual(out.logits.shape, (4,))
                    self.assertTrue(torch.isfinite(out.loss))
                    proba = model.predict_proba(**inputs)
                    self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_each_input_changes_the_score(self):
        model = PatientQueryModel(config()).eval()
        base = batch(2)
        with torch.no_grad():
            reference = model(**base).logits
            for name, value in (
                # a different vector, not a constant shift, which the projector's LayerNorm removes
                ("patient_features", torch.randn_like(base["patient_features"])),
                ("concept_index", (base["concept_index"] + 1) % 3),
                ("end_days", base["end_days"] + 275.0),
                ("start_days", base["start_days"] + 30.0),
            ):
                changed = dict(base, **{name: value})
                self.assertFalse(torch.allclose(reference, model(**changed).logits), name)

    def test_inputs_are_only_the_patient_and_the_query(self):
        parameters = set(inspect.signature(PatientQueryModel.forward).parameters)
        self.assertEqual(
            parameters,
            {"self", "patient_features", "concept_index", "start_days", "end_days", "labels"},
        )
        with self.assertRaises(TypeError):
            PatientQueryModel(config())(**batch(), age_at_index=torch.zeros(4))

    def test_wrong_feature_width_is_rejected(self):
        inputs = batch()
        inputs["patient_features"] = torch.cat([inputs["patient_features"]] * 2, dim=-1)
        with self.assertRaisesRegex(ValueError, "disable_combine_global_local"):
            PatientQueryModel(config())(**inputs)

    def test_gradients_reach_every_part(self):
        model = PatientQueryModel(config())
        model(**batch(), labels=torch.tensor([1, 0, 1, 0])).loss.backward()
        for name in ("patient_projector", "window_projector", "head", "trunk"):
            grads = [p.grad for p in getattr(model, name).parameters()]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), name)
        self.assertGreater(model.concept_embedding.weight.grad.abs().sum(), 0)

    def test_save_and_load_round_trip(self):
        model = PatientQueryModel(config()).eval()
        inputs = batch()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            loaded = PatientQueryModel.from_pretrained(folder).eval()
        self.assertEqual(loaded.config.query_concept_ids, CONCEPTS)
        torch.testing.assert_close(model(**inputs).logits, loaded(**inputs).logits)

    def test_concept_indices(self):
        model = PatientQueryModel(config())
        self.assertEqual(model.concept_indices(["c2", "c0"]).tolist(), [2, 0])
        with self.assertRaisesRegex(KeyError, "outside the query vocabulary"):
            model.concept_indices(["nope"])

    def test_pretrained_concept_embeddings(self):
        model = PatientQueryModel(config(concept_embedding_dim=12))
        matrix = np.random.RandomState(0).randn(3, 12).astype("float32")
        model.init_concept_embeddings(matrix, freeze=True)
        np.testing.assert_allclose(model.concept_embedding.weight.detach().numpy(), matrix)
        self.assertFalse(model.concept_embedding.weight.requires_grad)
        self.assertEqual(model(**batch()).logits.shape, (4,))  # projected to the trunk width
        with self.assertRaisesRegex(ValueError, "Expected embeddings of shape"):
            model.init_concept_embeddings(matrix[:, :5])

    def test_window_features_are_ordered_and_bounded(self):
        short = window_features(torch.tensor([0.0]), torch.tensor([30.0]))
        long_ = window_features(torch.tensor([0.0]), torch.tensor([730.0]))
        shifted = window_features(torch.tensor([30.0]), torch.tensor([730.0]))
        self.assertTrue((long_[0, 1] > short[0, 1]).item())
        self.assertTrue((long_[0, 2] > shifted[0, 2]).item())
        self.assertTrue((window_features(torch.tensor([0.0]), torch.tensor([3650.0])) <= 1.0001).all())

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            PatientQueryModelConfig(lm_type="bert")
        with self.assertRaises(ValueError):
            PatientQueryModelConfig(num_prefix_tokens=0)


def train(model, make_batch, steps=300, lr=3e-3):
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        loss = model(**make_batch()).loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.eval()


class LearningTest(unittest.TestCase):
    """The label is a function of two inputs together, so a model that ignores either fails."""

    def test_patient_and_concept_are_both_used(self):
        torch.manual_seed(0)
        patterns = torch.randn(2, 8)  # two kinds of patient

        def make(n, seed=None):
            generator = torch.Generator().manual_seed(seed) if seed is not None else None
            kind = torch.randint(0, 2, (n,), generator=generator)
            concept = torch.randint(0, 2, (n,), generator=generator)
            noise = 0.05 * torch.randn(n, 8, generator=generator)
            return {
                "patient_features": patterns[kind] + noise,
                "concept_index": concept,
                "start_days": torch.zeros(n),
                "end_days": torch.full((n,), 90.0),
                # yes only when this kind of patient is asked about the matching concept
                "labels": (kind == concept).float(),
            }

        model = PatientQueryModel(config())
        train(model, lambda: make(32))
        test = make(200, seed=1)
        labels = test.pop("labels")
        predicted = model.predict_proba(**test) > 0.5
        self.assertGreater((predicted == labels.bool()).float().mean().item(), 0.97)

    def test_the_window_is_used(self):
        torch.manual_seed(0)
        patterns = torch.randn(8)

        def make(n):
            long_window = torch.randint(0, 2, (n,))
            return {
                "patient_features": patterns.repeat(n, 1) + 0.05 * torch.randn(n, 8),
                "concept_index": torch.zeros(n, dtype=torch.long),
                "start_days": torch.zeros(n),
                "end_days": torch.where(long_window == 1, torch.tensor(365.0), torch.tensor(30.0)),
                "labels": long_window.float(),  # an event is more likely in the longer window
            }

        model = PatientQueryModel(config())
        train(model, lambda: make(32), steps=200)
        test = make(100)
        labels = test.pop("labels")
        self.assertGreater(((model.predict_proba(**test) > 0.5) == labels.bool()).float().mean().item(), 0.97)


if __name__ == "__main__":
    unittest.main()
