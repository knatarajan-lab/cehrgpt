"""
What the patient timeline QA head needs besides the model: the roll-up of the concepts to the questions
and the frozen embeddings of the question names.

    python -m cehrgpt.open_vocab.assets --universe_dir UNIVERSES --vocab_path OHDSI_DIR --output HIERARCHY.parquet

writes the hierarchy of the universes of question_universe once, so that training needs neither the OHDSI
tables nor the universes, only HIERARCHY.parquet and the folder of question_embeddings.
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

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
    def load(cls, hierarchy_path: str, embeddings_dir: str) -> "OpenVocabAssets":
        # the encoder libraries are only needed to build the embeddings, not to read them
        from cehrgpt.open_vocab.question_embeddings import QuestionEmbeddings

        embeddings = QuestionEmbeddings.load(os.path.expanduser(embeddings_dir))
        return cls(
            QuestionHierarchy.load(os.path.expanduser(hierarchy_path)),
            embeddings.texts.sort("row")["concept_id"].to_numpy(),
            embeddings.embeddings,
        )


def build_hierarchy(universe_files: Sequence[str], vocab_path: str, domains: Optional[Sequence[str]] = None) -> QuestionHierarchy:
    """The roll-up of the tokens to the questions of the universes (one parquet file per domain)."""
    from cehrgpt.open_vocab.question_universe import _scan

    ancestor = _scan(vocab_path, "concept_ancestor").select(
        pl.col("ancestor_concept_id").cast(pl.Int64), pl.col("descendant_concept_id").cast(pl.Int64)
    ).collect()
    universes = {os.path.basename(f): pl.read_parquet(f) for f in universe_files}
    return QuestionHierarchy.from_universes(universes, ancestor)


def main(argv: Optional[Sequence[str]] = None) -> QuestionHierarchy:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe_dir", required=True, help="Folder with the *_universe.parquet files")
    parser.add_argument("--vocab_path", required=True, help="OHDSI folder with concept_ancestor/ (parquet)")
    parser.add_argument("--output", required=True, help="The hierarchy parquet file to write")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    files = sorted(glob.glob(os.path.join(args.universe_dir, "*_universe.parquet")))
    if not files:
        raise FileNotFoundError(f"No *_universe.parquet files in {args.universe_dir}")
    hierarchy = build_hierarchy(files, args.vocab_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    hierarchy.save(args.output)
    LOG.info("%s concepts roll up to the questions of %s universes, written to %s", len(hierarchy), len(files), args.output)
    return hierarchy


if __name__ == "__main__":
    main()
