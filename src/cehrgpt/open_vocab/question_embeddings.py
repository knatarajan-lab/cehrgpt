"""
Embeddings of the question names, computed once with a frozen encoder (BioLORD-2023 by default).

The open vocabulary head never runs the encoder while it trains: it looks up the rows of this
matrix, one per question. Synonyms are not embedded; the encoder already maps different wordings of a
concept close together. At inference the encoder runs once per free text query, with the same text
template, pooling and normalisation.

    python -m cehrgpt.open_vocab.question_embeddings \
        --universe_files OUT/condition_universe.parquet OUT/procedure_universe.parquet OUT/drug_universe.parquet \
        --output_dir EMBEDDINGS

Writes embeddings.npy (questions x dimension, float32), question_texts.parquet (row, concept_id,
text) and question_embedder.json (the encoder and the template, to be checked at inference).
"""

import argparse
import json
import logging
import os
from typing import Dict, Optional, Sequence

import numpy as np
import polars as pl

from cehrgpt.analysis.concept_name_embeddings import (
    DEFAULT_MODEL,
    HuggingFaceEncoder,
    embed_with_cache,
)

LOG = logging.getLogger(__name__)

DEFAULT_TEMPLATE = "{text}"
# embed_with_cache keeps its own description in embedder.json; this one is the encoder and the template
DESCRIPTION_FILE = "question_embedder.json"


def build_text_table(universe: pl.DataFrame) -> pl.DataFrame:
    """One row per question, in the order of the universe: row, concept_id, text (the concept name)."""
    return (
        universe.select("concept_id", pl.col("concept_name").alias("text"))
        .unique(subset="concept_id", keep="first", maintain_order=True)
        .with_row_index("row")
    )


class QuestionEmbeddings:
    """The embedding matrix and the lookup from a concept id to its row."""

    def __init__(self, embeddings: np.ndarray, texts: pl.DataFrame, description: Optional[Dict] = None):
        if embeddings.shape[0] != len(texts):
            raise ValueError(f"{embeddings.shape[0]} embeddings for {len(texts)} texts")
        self.embeddings = embeddings
        self.texts = texts
        self.description = description or {}
        self._row = dict(zip(texts["concept_id"].to_list(), texts["row"].to_list()))

    @classmethod
    def load(cls, folder: str) -> "QuestionEmbeddings":
        with open(os.path.join(folder, DESCRIPTION_FILE)) as f:
            description = json.load(f)
        return cls(
            np.load(os.path.join(folder, "embeddings.npy")),
            pl.read_parquet(os.path.join(folder, "question_texts.parquet")),
            description,
        )

    def __len__(self) -> int:
        return len(self._row)

    def __contains__(self, concept_id: int) -> bool:
        return concept_id in self._row

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def rows(self, concept_ids: Sequence[int]) -> np.ndarray:
        missing = [c for c in concept_ids if c not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} concepts have no embedding, e.g. {missing[:5]}")
        return np.asarray([self._row[c] for c in concept_ids], dtype=np.int64)

    def vectors(self, concept_ids: Sequence[int]) -> np.ndarray:
        return self.embeddings[self.rows(concept_ids)]


def embed_questions(
    universe: pl.DataFrame,
    encoder,
    output_dir: str,
    template: str = DEFAULT_TEMPLATE,
    overwrite: bool = False,
) -> QuestionEmbeddings:
    texts = build_text_table(universe)
    rendered = [template.format(text=text) for text in texts["text"].to_list()]
    embeddings = embed_with_cache(texts, rendered, encoder, output_dir, overwrite=overwrite)
    texts.write_parquet(os.path.join(output_dir, "question_texts.parquet"))
    description = {**encoder.describe(), "template": template, "dim": int(embeddings.shape[1])}
    with open(os.path.join(output_dir, DESCRIPTION_FILE), "w") as f:
        json.dump(description, f, indent=2)
    return QuestionEmbeddings(embeddings, texts, description)


def main(argv: Optional[Sequence[str]] = None) -> QuestionEmbeddings:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe_files", nargs="+", required=True, help="Parquet files of question_universe, united")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Applied to each name, with {text}")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    universe = pl.concat([pl.read_parquet(path).select("concept_id", "concept_name") for path in args.universe_files])
    encoder = HuggingFaceEncoder(args.model, batch_size=args.batch_size, device=args.device, fp16=args.fp16)
    result = embed_questions(universe, encoder, args.output_dir, args.template, args.overwrite)
    LOG.info("%s questions, dimension %s", len(result), result.dim)
    return result


if __name__ == "__main__":
    main()
