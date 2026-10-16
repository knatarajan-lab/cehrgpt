"""Collation, train/test splits and evaluation for the patient query model."""

import random
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from cehrgpt.models.patient_query_model import PatientQueryModel
from cehrgpt.qa.concept_hierarchy import ConceptHierarchy


class QueryCollator:
    """Turns query rows into the tensors the model takes."""

    def __init__(self, model_or_vocabulary):
        concepts = (
            model_or_vocabulary.config.query_concept_ids
            if hasattr(model_or_vocabulary, "config")
            else list(model_or_vocabulary)
        )
        self.index = {c: i for i, c in enumerate(concepts)}

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        missing = {e["concept_id"] for e in examples} - set(self.index)
        if missing:
            raise KeyError(f"Concepts outside the query vocabulary: {sorted(missing)[:5]}")
        return {
            "patient_features": torch.tensor(
                [e["patient_features"] for e in examples], dtype=torch.float32
            ),
            "concept_index": torch.tensor(
                [self.index[e["concept_id"]] for e in examples], dtype=torch.long
            ),
            "start_days": torch.tensor([e["start_days"] for e in examples], dtype=torch.float32),
            "end_days": torch.tensor([e["end_days"] for e in examples], dtype=torch.float32),
            "labels": torch.tensor([float(e["label"]) for e in examples]),
        }


def related_concepts(
    held_out: Set[str],
    hierarchy: ConceptHierarchy,
    vocabulary: Sequence[str],
    include_ancestors: bool = True,
) -> Set[str]:
    """The held-out concepts' descendants, and with include_ancestors also their ancestors."""
    descendants: Dict[str, List[str]] = {}
    for concept in vocabulary:
        for ancestor in hierarchy.ancestors(concept):
            descendants.setdefault(ancestor, []).append(concept)
    in_vocabulary = set(vocabulary)
    related: Set[str] = set()
    for concept in held_out:
        related.update(descendants.get(concept, ()))
        if include_ancestors:
            related |= hierarchy.ancestors(concept) & in_vocabulary
    return related


def split_queries(
    dataset: Dataset,
    hierarchy: ConceptHierarchy,
    vocabulary: Sequence[str],
    validation_patient_fraction: float = 0.1,
    held_out_concept_fraction: float = 0.1,
    seed: int = 42,
    covered_tokens: Optional[Dict[str, int]] = None,
    max_held_out_covered_tokens: int = 3,
    block_ancestors: bool = False,
) -> Dict[str, Dataset]:
    """
    Split by patient, and hold out whole concepts to test what was never trained on.

    * train: patients outside the evaluation set, concepts that are not blocked
    * validation: evaluation patients, concepts that are not blocked (new patients only)
    * test_unseen_concepts: evaluation patients and the held-out concepts (new patients and
      new concepts)

    Only narrow concepts are held out (covering at most max_held_out_covered_tokens tokens,
    when covered_tokens is given), otherwise holding out a broad class would remove most of
    the data. The descendants of a held-out concept are blocked everywhere: a descendant with
    a yes would give the answer for the held-out concept away. Its ancestors stay in training
    by default, which is the realistic case of a new fine-grained concept under a known
    parent; block_ancestors removes them too for a stricter test.
    """
    rng = random.Random(seed)
    patients = sorted(set(dataset["person_id"]))
    rng.shuffle(patients)
    n_eval = max(1, round(len(patients) * validation_patient_fraction))
    eval_patients = set(patients[:n_eval])
    concepts = sorted(vocabulary)
    candidates = [
        c
        for c in concepts
        if covered_tokens is None or covered_tokens.get(c, 1) <= max_held_out_covered_tokens
    ]
    n_held = min(round(len(concepts) * held_out_concept_fraction), len(candidates))
    held_out = set(rng.sample(candidates, n_held)) if n_held else set()
    blocked = held_out | related_concepts(held_out, hierarchy, concepts, block_ancestors)

    person = dataset["person_id"]
    concept = dataset["concept_id"]
    train, validation, unseen = [], [], []
    for row, (p, c) in enumerate(zip(person, concept)):
        if p in eval_patients:
            if c in held_out:
                unseen.append(row)
            elif c not in blocked:
                validation.append(row)
        elif c not in blocked:
            train.append(row)
    return {
        "train": dataset.select(train),
        "validation": dataset.select(validation),
        "test_unseen_concepts": dataset.select(unseen),
    }


def _auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(labels) == 0 or labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def predict_dataset(
    model: PatientQueryModel, dataset: Dataset, batch_size: int = 256
) -> np.ndarray:
    """P(yes) for every row."""
    collator = QueryCollator(model)
    device = next(model.parameters()).device
    model.eval()
    scores = []
    for start in range(0, len(dataset), batch_size):
        batch = collator(list(dataset.select(range(start, min(start + batch_size, len(dataset))))))
        batch.pop("labels")
        scores.append(model.predict_proba(**{k: v.to(device) for k, v in batch.items()}).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def evaluate_queries(
    model: PatientQueryModel, dataset: Dataset, batch_size: int = 256
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    AUROC, accuracy and the base rate, overall and for the MOTOR and non-MOTOR concepts.

    The non-MOTOR line is the one that shows whether the model answers beyond the fixed
    task set.
    """
    scores = predict_dataset(model, dataset, batch_size)
    labels = np.asarray(dataset["label"], dtype=float)
    in_scope = np.asarray(dataset["is_motor_task"], dtype=bool)
    groups = {
        "all": np.ones(len(labels), dtype=bool),
        "motor_tasks": in_scope,
        "outside_motor": ~in_scope,
    }
    report = {}
    for name, mask in groups.items():
        y, s = labels[mask], scores[mask]
        report[name] = {
            "n": int(mask.sum()),
            "auroc": _auroc(y, s),
            "accuracy": float(((s > 0.5) == (y > 0.5)).mean()) if len(y) else None,
            "yes_fraction": float(y.mean()) if len(y) else None,
        }
    return report
