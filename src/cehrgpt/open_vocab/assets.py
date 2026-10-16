"""
What the patient timeline QA head needs besides the model: the roll-up of the concepts to the questions
and the frozen embeddings of the question names.

The universe (question_universe) and the embeddings (question_embeddings) are built from the OHDSI vocabulary
alone, before any tokenizer exists, and serve any tokenizer. Nothing is prepared from a tokenizer: when training
starts, OpenVocabAssets.build reads the vocabulary of the tokenizer the run already loaded and

* rolls the concepts of the vocabulary up to the questions of the universe (a concept counts for itself and for the
  questions above it in the hierarchy, within its own domain);
* keeps the questions that have a concept of the vocabulary at or below them, and, with a minimum patient count,
  at least that many patients in the data the tokenizer was trained on (from the concept statistics the tokenizer
  saved, which cover every concept seen in the training data, also those that did not make it into the
  vocabulary);
* keeps the embedding rows of those questions only, so the model holds the embeddings it can train on and no more.
"""

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set

import numpy as np
import polars as pl

from cehrgpt.open_vocab.labels import QuestionHierarchy

LOG = logging.getLogger(__name__)


@dataclass
class OpenVocabAssets:
    hierarchy: QuestionHierarchy
    question_concept_ids: np.ndarray  # the concept id of every row of the embedding matrix
    embeddings: np.ndarray  # (rows, dimension)

    def __post_init__(self):
        self.question_concept_ids = np.asarray(self.question_concept_ids, dtype=np.int64)
        if len(self.question_concept_ids) != len(self.embeddings):
            raise ValueError(f"{len(self.question_concept_ids)} questions for {len(self.embeddings)} embeddings")
        if len(np.unique(self.question_concept_ids)) != len(self.question_concept_ids):
            raise ValueError("a question has more than one embedding row")

    @property
    def num_rows(self) -> int:
        return len(self.question_concept_ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @classmethod
    def build(
        cls,
        universe_dir: str,
        vocab_path: str,
        embeddings_dir: str,
        token_ids: Iterable[int],
        concept_counts: Optional[Dict[int, int]] = None,
        min_patient_count: int = 0,
    ) -> "OpenVocabAssets":
        """
        Args:
            universe_dir: the folder with the <domain>_universe.parquet files of question_universe
            vocab_path: the OHDSI folder with concept/ and concept_ancestor/ (parquet)
            embeddings_dir: the folder of question_embeddings, built from the same universes
            token_ids: the concept ids of the vocabulary of the tokenizer
            concept_counts: patients per concept in the training data (read_concept_counts), only used
                with min_patient_count
            min_patient_count: keep the questions with at least this many patients; 0 keeps all
        """
        # the encoder libraries are only needed to build the embeddings, not to read them
        from cehrgpt.open_vocab.question_embeddings import QuestionEmbeddings

        files = sorted(glob.glob(os.path.join(os.path.expanduser(universe_dir), "*_universe.parquet")))
        if not files:
            raise FileNotFoundError(f"No *_universe.parquet files in {universe_dir}")
        hierarchy = build_hierarchy(files, os.path.expanduser(vocab_path), set(token_ids), concept_counts, min_patient_count)
        embeddings = QuestionEmbeddings.load(os.path.expanduser(embeddings_dir))
        concept_ids = embeddings.texts.sort("row")["concept_id"].to_numpy()
        # the embeddings cover the whole universe; the hierarchy only has the questions of this tokenizer
        keep = np.isin(concept_ids, hierarchy.question_ids())
        if not keep.any():
            raise ValueError(
                "None of the questions of the embeddings has a concept of the vocabulary at or below it; "
                "were the embeddings built from the universes in universe_dir?"
            )
        LOG.info("%s of the %s question embeddings have concepts of the vocabulary below them and are used", int(keep.sum()), len(keep))
        return cls(hierarchy, concept_ids[keep], embeddings.embeddings[keep])


def vocabulary_concept_ids(vocab: Iterable[str]) -> Set[int]:
    """The concept ids among the tokens of a vocabulary (the digit strings)."""
    return {int(token) for token in vocab if token.isdigit()}


def read_concept_counts(tokenizer_dirs: Sequence[str]) -> Dict[int, int]:
    """
    The number of patients with each concept (or a concept below it, with an ontology) in the training data of
    the tokenizers, from the concept statistics they saved. The statistics are the share of the patients, in
    multiples of 1 / the number of patients, so the smallest one is one patient. The maximum over the tokenizers.
    """
    counts: Dict[int, int] = {}
    for folder in tokenizer_dirs:
        with open(os.path.join(os.path.expanduser(folder), "cehrgpt_concept_stats.json")) as f:
            shares = {int(k): v for k, v in json.load(f).items() if k.isdigit() and v > 0}
        if not shares:
            continue
        patients = 1.0 / min(shares.values())
        for concept_id, share in shares.items():
            counts[concept_id] = max(counts.get(concept_id, 0), int(round(share * patients)))
    return counts


def build_hierarchy(
    universe_files: Sequence[str],
    vocab_path: str,
    token_ids: Set[int],
    concept_counts: Optional[Dict[int, int]] = None,
    min_patient_count: int = 0,
) -> QuestionHierarchy:
    """
    The roll-up of the concepts of a vocabulary to the questions of the universes (one parquet file per domain,
    named <domain>_universe.parquet), without the questions that have no concept of the vocabulary at or below
    them, or fewer than min_patient_count patients in concept_counts.
    """
    from cehrgpt.open_vocab.question_universe import CATEGORIES, _scan

    ancestor = _scan(vocab_path, "concept_ancestor").select(
        pl.col("ancestor_concept_id").cast(pl.Int64), pl.col("descendant_concept_id").cast(pl.Int64)
    ).collect()
    concept = _scan(vocab_path, "concept").select(pl.col("concept_id").cast(pl.Int64), "domain_id").collect()
    if min_patient_count > 0 and concept_counts is None:
        raise ValueError("min_patient_count needs concept_counts")
    questions: Dict[int, set] = {}
    for universe_file in universe_files:
        domain = os.path.basename(universe_file).split("_universe")[0].capitalize()
        if domain not in CATEGORIES:
            raise ValueError(f"{universe_file} is not a <domain>_universe.parquet file of {list(CATEGORIES)}")
        question_ids = pl.read_parquet(universe_file)["concept_id"].to_list()
        if min_patient_count > 0:
            question_ids = [q for q in question_ids if concept_counts.get(q, 0) >= min_patient_count]
        # a concept rolls up only to the questions of its own domain
        tokens = concept.filter((pl.col("domain_id") == domain) & pl.col("concept_id").is_in(list(token_ids)))["concept_id"].to_list()
        pairs = ancestor.filter(pl.col("ancestor_concept_id").is_in(question_ids) & pl.col("descendant_concept_id").is_in(tokens))
        for token, question in pairs.select("descendant_concept_id", "ancestor_concept_id").iter_rows():
            questions.setdefault(token, set()).add(question)
        asked = set(question_ids)
        for token in tokens:
            # a concept that is a question counts for itself, also if the ancestor table does not list it
            if token in asked:
                questions.setdefault(token, set()).add(token)
        kept = len({q for qs in questions.values() for q in qs} & asked)
        LOG.info("%s: %s of %s questions have a concept of the vocabulary below them", domain, kept, len(question_ids))
    return QuestionHierarchy(questions)
