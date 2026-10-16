"""
Embed OMOP concept names with a text encoder and look at how they cluster.

The question this answers: do the vectors of the names alone separate the disease groups
(cardiovascular, respiratory, neoplastic, ...), and do they place a concept near its parent?
It is meant for choosing the text encoder whose vectors describe the tasks of a
MOTOR-style head.

Inputs are OMOP vocabulary tables (parquet file/folder or Athena's tab separated csv):
* ``concept`` for the names (SNOMED, RxNorm, ... are selected by vocabulary_id)
* ``concept_ancestor`` (optional) for the groups and the parent test
* ``concept_relationship`` (RxNorm only) for the RxNorm to ATC mapping

Two ways to colour the points:
* SNOMED: the disease group, i.e. the closest ancestor among the concepts a fixed number of
  levels below a root such as "Disease"
* RxNorm: the ATC class (--atc_level 1 is the anatomical main group, 2 the therapeutic
  subgroup), reached through the ATC links in concept_relationship and the ATC hierarchy

Outputs, in --output_dir: groups.csv, metrics.json, cluster_composition.csv, projection.csv and
projection.png (points coloured by group). The embeddings are cached in --cache_dir (default
--output_dir) as embeddings.npy / concepts.parquet / embedder.json; the cache is reused when the
model and the texts are the same, and a run that was interrupted resumes where it stopped.

Examples:
    python -m cehrgpt.analysis.concept_name_embeddings --concept_path CONCEPT.csv \
        --concept_ancestor_path CONCEPT_ANCESTOR.csv --output_dir out_snomed --domains Condition

    python -m cehrgpt.analysis.concept_name_embeddings --concept_path CONCEPT.csv \
        --concept_ancestor_path CONCEPT_ANCESTOR.csv \
        --concept_relationship_path CONCEPT_RELATIONSHIP.csv --vocabulary_id RxNorm \
        --concept_classes Ingredient --color_by atc_group --atc_level 1 --output_dir out_rxnorm
"""

import argparse
import json
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "FremyCompany/BioLORD-2023"
OTHER_GROUP = "Other"
MULTIPLE_GROUP = "Multiple groups"


# ---------------------------------------------------------------------------------------
# Loading the vocabulary
# ---------------------------------------------------------------------------------------
def read_table(path: str) -> pl.LazyFrame:
    """A parquet file or folder, or an Athena csv (tab separated, unquoted), as a lazy frame."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        frame = pl.scan_parquet(os.path.join(path, "*.parquet"))
    elif path.endswith((".csv", ".tsv", ".txt")):
        # Athena files are tab separated and do not quote their fields; read everything as
        # text and cast the ids afterwards, so one odd row cannot break the type inference
        frame = pl.scan_csv(
            path, separator="\t", quote_char=None, infer_schema_length=0
        )
    else:
        frame = pl.scan_parquet(path)
    return frame.rename({c: c.lower() for c in frame.collect_schema().names()})


def load_concepts(
    concept_path: str,
    vocabulary_id: str = "SNOMED",
    domains: Optional[Sequence[str]] = None,
    include_non_standard: bool = False,
    include_invalid: bool = False,
    concept_classes: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    The concepts to embed: concept_id, concept_name, domain_id, concept_class_id, concept_code.

    Args:
        concept_path: the OMOP concept table
        vocabulary_id: which vocabulary, e.g. SNOMED or RxNorm
        domains: keep only these domains (None keeps all)
        include_non_standard: also keep concepts that are not standard ("S")
        include_invalid: also keep deprecated concepts
        concept_classes: keep only these concept classes, e.g. Ingredient for RxNorm
    """
    frame = read_table(concept_path)
    names = set(frame.collect_schema().names())
    missing = {"concept_id", "concept_name", "vocabulary_id"} - names
    if missing:
        raise ValueError(f"{concept_path} has no column(s) {sorted(missing)}")
    frame = frame.with_columns(pl.col("concept_id").cast(pl.Int64))
    frame = frame.filter(pl.col("vocabulary_id") == vocabulary_id)
    if not include_non_standard and "standard_concept" in names:
        frame = frame.filter(pl.col("standard_concept") == "S")
    if not include_invalid and "invalid_reason" in names:
        frame = frame.filter(pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
    if domains:
        if "domain_id" not in names:
            raise ValueError("domains were given but the concept table has no domain_id")
        frame = frame.filter(pl.col("domain_id").is_in(list(domains)))
    if concept_classes:
        if "concept_class_id" not in names:
            raise ValueError("concept_classes were given but the table has no concept_class_id")
        frame = frame.filter(pl.col("concept_class_id").is_in(list(concept_classes)))
    for optional in ("domain_id", "concept_class_id", "concept_code"):
        if optional not in names:
            frame = frame.with_columns(pl.lit("").alias(optional))
    return (
        frame.select(
            "concept_id", "concept_name", "domain_id", "concept_class_id", "concept_code"
        )
        # A missing field is read as null from csv and as "" from parquet; make them the same
        .with_columns(pl.col("domain_id", "concept_class_id", "concept_code").fill_null(""))
        .filter(pl.col("concept_name").is_not_null() & (pl.col("concept_name") != ""))
        .unique(subset="concept_id", keep="first", maintain_order=True)
        .collect()
    )


def render_texts(concepts: pl.DataFrame, template: str) -> List[str]:
    """The text that is embedded for each concept, e.g. "{domain_id}: {concept_name}"."""
    return [template.format(**row) for row in concepts.iter_rows(named=True)]


# ---------------------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------------------
def mean_pool(hidden, attention_mask):
    """
    Average the token vectors of each text over its real tokens only.

    Padding is excluded through the attention mask, so a text gets the same vector whatever
    else is in its batch.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class HuggingFaceEncoder:
    """Mean pooled, L2 normalised sentence vectors from a transformers encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 128,
        max_length: int = 128,
        device: Optional[str] = None,
        fp16: bool = False,
        revision: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).eval().to(self.device)
        if fp16 and self.device.startswith("cuda"):
            self.model = self.model.half()

    def describe(self) -> Dict[str, object]:
        import transformers

        return {
            "model_name": self.model_name,
            # the commit of the model on the Hub (None for a local folder), so a model that changed under the
            # same name is noticed
            "revision": getattr(self.model.config, "_commit_hash", None),
            "pooling": "masked mean over tokens",
            "normalize": "l2",
            "max_length": self.max_length,
            "transformers_version": transformers.__version__,
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        # Longest texts together, so a batch wastes little on padding; restore the order after
        order = np.argsort([len(t) for t in texts], kind="stable")
        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [texts[i] for i in order[start : start + self.batch_size]]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                pooled = mean_pool(hidden, inputs["attention_mask"]).float()
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vectors.append(pooled.cpu().numpy())
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        result = np.empty((len(texts), vectors[0].shape[1]), dtype=np.float32)
        result[order] = np.concatenate(vectors)
        return result


def embed_with_cache(
    concepts: pl.DataFrame,
    texts: Sequence[str],
    encoder,
    output_dir: str,
    chunk_size: int = 50_000,
    overwrite: bool = False,
) -> np.ndarray:
    """
    Embed the texts, in chunks that are saved as they finish so a long run can be resumed.

    A cached chunk is reused only when the encoder description and the texts match, so
    changing the model or the template recomputes it.
    """
    os.makedirs(output_dir, exist_ok=True)
    description = {**encoder.describe(), "num_texts": len(texts), "chunk_size": chunk_size}
    description_path = os.path.join(output_dir, "embedder.json")
    parts_dir = os.path.join(output_dir, "embedding_parts")
    os.makedirs(parts_dir, exist_ok=True)
    texts_hash = _hash_texts(texts)
    description["texts_sha1"] = texts_hash
    previous = None
    if os.path.exists(description_path) and not overwrite:
        with open(description_path) as f:
            previous = json.load(f)
    if previous != description:
        for name in os.listdir(parts_dir):
            os.remove(os.path.join(parts_dir, name))
        with open(description_path, "w") as f:
            json.dump(description, f, indent=2)

    chunks = []
    for index, start in enumerate(range(0, len(texts), chunk_size)):
        path = os.path.join(parts_dir, f"part_{index:05d}.npy")
        if os.path.exists(path):
            chunks.append(np.load(path))
            continue
        LOG.info("Embedding concepts %s to %s of %s", start, start + chunk_size, len(texts))
        vectors = encoder.encode(list(texts[start : start + chunk_size]))
        np.save(path, vectors)
        chunks.append(vectors)
    embeddings = np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    concepts.write_parquet(os.path.join(output_dir, "concepts.parquet"))
    return embeddings


def _hash_texts(texts: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha1()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------
# Disease groups
# ---------------------------------------------------------------------------------------
def read_ancestors(path: str) -> pl.LazyFrame:
    frame = read_table(path)
    return frame.select(
        pl.col("ancestor_concept_id").cast(pl.Int64),
        pl.col("descendant_concept_id").cast(pl.Int64),
        pl.col("min_levels_of_separation").cast(pl.Int64),
    )


def find_concept_id(
    concept_path: str, name: str, vocabulary_id: str = "SNOMED"
) -> int:
    """The id of the standard concept with this exact name."""
    frame = read_table(concept_path)
    names = set(frame.collect_schema().names())
    frame = frame.filter((pl.col("concept_name") == name) & (pl.col("vocabulary_id") == vocabulary_id))
    if "standard_concept" in names:
        frame = frame.filter(pl.col("standard_concept") == "S")
    ids = frame.select(pl.col("concept_id").cast(pl.Int64)).collect()["concept_id"].to_list()
    if not ids:
        raise ValueError(
            f"No standard {vocabulary_id} concept is named {name!r}; check the spelling or pass "
            "--group_root_id"
        )
    if len(ids) > 1:
        LOG.warning(
            "%s concepts are named %r (%s); using the lowest id %s. Pass --group_root_id to choose",
            len(ids), name, sorted(ids), min(ids),
        )
    return min(ids)


@dataclass
class DiseaseGroups:
    names: Dict[int, str]  # concept_id -> the group it was assigned to
    anchors: Dict[int, str]  # anchor concept_id -> anchor name
    sizes: Dict[str, int]  # group name -> number of concepts


def discover_anchors(
    ancestors: pl.LazyFrame, root_id: int, depth: int
) -> List[int]:
    """The concepts exactly `depth` levels below the root, e.g. its children for depth 1."""
    frame = ancestors.filter(
        (pl.col("ancestor_concept_id") == root_id)
        & (pl.col("min_levels_of_separation") == depth)
    ).select("descendant_concept_id")
    return sorted(frame.collect()["descendant_concept_id"].to_list())


def assign_disease_groups(
    ancestors: pl.LazyFrame,
    concept_ids: Sequence[int],
    anchor_names: Dict[int, str],
    max_groups: int = 20,
    min_group_size: int = 1,
    tie_break: str = "multiple",
) -> DiseaseGroups:
    """
    Put every concept in the group of its closest anchor ancestor.

    A concept below several anchors goes to the nearest one. When two anchors are equally near,
    tie_break "multiple" labels it as belonging to several groups and "first" takes the anchor
    with the lowest id. Concepts below no anchor, and the groups beyond the max_groups largest
    or smaller than min_group_size, are "Other".
    """
    if tie_break not in ("multiple", "first"):
        raise ValueError("tie_break must be 'multiple' or 'first'")
    ids = list(concept_ids)
    pairs = (
        ancestors.filter(
            pl.col("ancestor_concept_id").is_in(list(anchor_names))
            & pl.col("descendant_concept_id").is_in(ids)
        )
        .collect()
        .with_columns(
            pl.col("min_levels_of_separation")
            .min()
            .over("descendant_concept_id")
            .alias("closest")
        )
        .filter(pl.col("min_levels_of_separation") == pl.col("closest"))
        .group_by("descendant_concept_id")
        .agg(pl.col("ancestor_concept_id").sort().alias("anchors"))
    )
    assignment: Dict[int, Tuple[int, ...]] = {
        row["descendant_concept_id"]: tuple(row["anchors"])
        for row in pairs.iter_rows(named=True)
    }

    def label(concept_id: int) -> str:
        anchors = assignment.get(concept_id)
        if not anchors:
            return OTHER_GROUP
        if len(anchors) > 1 and tie_break == "multiple":
            return MULTIPLE_GROUP
        return anchor_names[anchors[0]]

    names = {concept_id: label(concept_id) for concept_id in ids}
    names, sizes = _cap_groups(names, max_groups, min_group_size)
    return DiseaseGroups(names, {a: anchor_names[a] for a in anchor_names}, sizes)


def _cap_groups(
    names: Dict[int, str], max_groups: int, min_group_size: int
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Keep the max_groups largest groups that reach min_group_size; the rest become Other."""
    counts: Dict[str, int] = {}
    for group in names.values():
        counts[group] = counts.get(group, 0) + 1
    real = sorted(
        (g for g in counts if g not in (OTHER_GROUP, MULTIPLE_GROUP)),
        key=lambda g: (-counts[g], g),
    )
    kept = {g for g in real[:max_groups] if counts[g] >= min_group_size}
    capped = {
        c: (g if g in kept or g == MULTIPLE_GROUP else OTHER_GROUP) for c, g in names.items()
    }
    sizes: Dict[str, int] = {}
    for group in capped.values():
        sizes[group] = sizes.get(group, 0) + 1
    return capped, sizes


# ---------------------------------------------------------------------------------------
# ATC classes for RxNorm
# ---------------------------------------------------------------------------------------
ATC_LEVEL_CLASSES = {1: "ATC 1st", 2: "ATC 2nd", 3: "ATC 3rd", 4: "ATC 4th", 5: "ATC 5th"}


def load_atc_pairs(
    concept_path: str,
    relationship_path: str,
    drug_vocabularies: Sequence[str] = ("RxNorm",),
    relationship_ids: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    The links between a drug concept and an ATC concept, from concept_relationship.

    A link is a valid relationship row between a concept of one of drug_vocabularies and a
    concept of the ATC vocabulary, in either direction (OMOP stores both "ATC - RxNorm" and
    "RxNorm - ATC"). Pass relationship_ids to restrict it to particular relationship names.

    Returns a frame with drug_concept_id and atc_concept_id.
    """
    concepts = read_table(concept_path).select(
        pl.col("concept_id").cast(pl.Int64), "vocabulary_id"
    )
    relationship = read_table(relationship_path)
    names = set(relationship.collect_schema().names())
    relationship = relationship.select(
        pl.col("concept_id_1").cast(pl.Int64),
        pl.col("concept_id_2").cast(pl.Int64),
        *(["relationship_id"] if "relationship_id" in names else []),
        *(["invalid_reason"] if "invalid_reason" in names else []),
    )
    if "invalid_reason" in names:
        relationship = relationship.filter(
            pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == "")
        )
    if relationship_ids:
        relationship = relationship.filter(pl.col("relationship_id").is_in(list(relationship_ids)))
    drugs = concepts.filter(pl.col("vocabulary_id").is_in(list(drug_vocabularies))).select(
        pl.col("concept_id").alias("drug")
    )
    atc = concepts.filter(pl.col("vocabulary_id") == "ATC").select(pl.col("concept_id").alias("atc"))

    def oriented(drug_side: str, atc_side: str) -> pl.LazyFrame:
        return (
            relationship.join(drugs, left_on=drug_side, right_on="drug")
            .join(atc, left_on=atc_side, right_on="atc")
            .select(
                pl.col(drug_side).alias("drug_concept_id"),
                pl.col(atc_side).alias("atc_concept_id"),
            )
        )

    return (
        pl.concat([oriented("concept_id_1", "concept_id_2"), oriented("concept_id_2", "concept_id_1")])
        .unique()
        .collect()
    )


def atc_level_labels(concept_path: str, atc_level: int) -> Dict[int, str]:
    """The ATC concepts of one level, as concept_id -> "code: name"."""
    if atc_level not in ATC_LEVEL_CLASSES:
        raise ValueError(f"atc_level must be one of {sorted(ATC_LEVEL_CLASSES)}")
    frame = read_table(concept_path)
    names = set(frame.collect_schema().names())
    frame = frame.filter(
        (pl.col("vocabulary_id") == "ATC")
        & (pl.col("concept_class_id") == ATC_LEVEL_CLASSES[atc_level])
    ).collect()
    labels = {}
    for row in frame.iter_rows(named=True):
        code = row.get("concept_code") if "concept_code" in names else None
        labels[int(row["concept_id"])] = f"{code}: {row['concept_name']}" if code else str(row["concept_name"])
    if not labels:
        raise ValueError(
            f"No ATC concepts of class {ATC_LEVEL_CLASSES[atc_level]!r} in the concept table"
        )
    return labels


def assign_atc_groups(
    pairs: pl.DataFrame,
    ancestors: Optional[pl.LazyFrame],
    level_labels: Dict[int, str],
    drug_ids: Sequence[int],
    ingredient_ids: Sequence[int] = (),
    max_groups: int = 20,
    min_group_size: int = 1,
    tie_break: str = "multiple",
) -> DiseaseGroups:
    """
    Put every drug in the ATC class, at the chosen level, that its ATC links lead to.

    The links reach an ATC concept of any level (RxNorm ingredients are linked to the fifth),
    so each is climbed to its ancestor at the chosen level through concept_ancestor. A drug
    with links into several classes is "Multiple groups" with tie_break "multiple", and takes the
    class with the lowest concept id with "first". A drug without a link of its own (a clinical
    drug, say) takes the classes of the ingredients above it in concept_ancestor, when
    ingredient_ids are given. Whatever is left is "Other".
    """
    if tie_break not in ("multiple", "first"):
        raise ValueError("tie_break must be 'multiple' or 'first'")
    level_ids = list(level_labels)
    linked = set(pairs["atc_concept_id"].to_list())
    # ATC concept -> the class at the chosen level it belongs to; a concept of that level is its own
    climb: Dict[int, set] = {a: ({a} if a in level_labels else set()) for a in linked}
    if ancestors is not None and linked:
        rows = (
            ancestors.filter(
                pl.col("ancestor_concept_id").is_in(level_ids)
                & pl.col("descendant_concept_id").is_in(list(linked))
            )
            .collect()
        )
        for row in rows.iter_rows(named=True):
            climb[row["descendant_concept_id"]].add(row["ancestor_concept_id"])
    elif any(not v for v in climb.values()):
        raise ValueError("climbing the ATC hierarchy needs --concept_ancestor_path")

    classes: Dict[int, set] = {}
    for row in pairs.iter_rows(named=True):
        classes.setdefault(row["drug_concept_id"], set()).update(climb[row["atc_concept_id"]])

    drugs = list(drug_ids)
    missing = [d for d in drugs if not classes.get(d)]
    if missing and ingredient_ids and ancestors is not None:
        above = (
            ancestors.filter(
                pl.col("ancestor_concept_id").is_in(list(ingredient_ids))
                & pl.col("descendant_concept_id").is_in(missing)
                & (pl.col("min_levels_of_separation") >= 1)
            )
            .collect()
        )
        for row in above.iter_rows(named=True):
            inherited = classes.get(row["ancestor_concept_id"])
            if inherited:
                classes.setdefault(row["descendant_concept_id"], set()).update(inherited)

    def label(drug_id: int) -> str:
        found = classes.get(drug_id)
        if not found:
            return OTHER_GROUP
        if len(found) > 1 and tie_break == "multiple":
            return MULTIPLE_GROUP
        return level_labels[min(found)]

    names, sizes = _cap_groups({d: label(d) for d in drugs}, max_groups, min_group_size)
    return DiseaseGroups(names, dict(level_labels), sizes)


def ingredients_of(concept_path: str, vocabularies: Sequence[str] = ("RxNorm",)) -> List[int]:
    """The ingredient concepts of the drug vocabularies."""
    frame = read_table(concept_path).filter(
        pl.col("vocabulary_id").is_in(list(vocabularies))
        & (pl.col("concept_class_id") == "Ingredient")
    )
    return frame.select(pl.col("concept_id").cast(pl.Int64)).collect()["concept_id"].to_list()


def direct_parents(
    ancestors: pl.LazyFrame, concept_ids: Sequence[int]
) -> Dict[int, List[int]]:
    """child -> its direct parents, both limited to concept_ids."""
    ids = list(concept_ids)
    frame = (
        ancestors.filter(
            (pl.col("min_levels_of_separation") == 1)
            & pl.col("ancestor_concept_id").is_in(ids)
            & pl.col("descendant_concept_id").is_in(ids)
        )
        .collect()
        .group_by("descendant_concept_id")
        .agg(pl.col("ancestor_concept_id").alias("parents"))
    )
    return {r["descendant_concept_id"]: list(r["parents"]) for r in frame.iter_rows(named=True)}


# ---------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------
def top_k_cosine(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    exclude: Optional[np.ndarray] = None,
    chunk: int = 512,
) -> np.ndarray:
    """
    Indices of the k most similar corpus rows for each query, most similar first.

    Rows are assumed to be unit length, so the dot product is the cosine. `exclude[i]` is a
    corpus row to leave out for query i, typically the query itself.
    """
    k = min(k, corpus.shape[0] - (1 if exclude is not None else 0))
    result = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        scores = queries[start : start + chunk] @ corpus.T
        if exclude is not None:
            rows = np.arange(scores.shape[0])
            scores[rows, exclude[start : start + chunk]] = -np.inf
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        result[start : start + chunk] = np.take_along_axis(top, order, axis=1)
    return result


def cluster_agreement(
    embeddings: np.ndarray,
    groups: Sequence[str],
    n_clusters: int,
    seed: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """K-means on the vectors, compared with the disease groups."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    clusters = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=4096
    ).fit_predict(embeddings)
    metrics = {
        "n_clusters": int(n_clusters),
        "adjusted_rand_index": float(adjusted_rand_score(groups, clusters)),
        "normalized_mutual_information": float(normalized_mutual_info_score(groups, clusters)),
        "homogeneity": float(homogeneity_score(groups, clusters)),
        "completeness": float(completeness_score(groups, clusters)),
    }
    composition = []
    groups_array = np.asarray(groups)
    for cluster in range(n_clusters):
        members = groups_array[clusters == cluster]
        if len(members) == 0:
            continue
        values, counts = np.unique(members, return_counts=True)
        top = int(np.argmax(counts))
        composition.append(
            {
                "cluster": cluster,
                "size": int(len(members)),
                "top_group": str(values[top]),
                "top_group_share": float(counts[top] / len(members)),
            }
        )
    return metrics, composition


def neighbor_group_purity(
    embeddings: np.ndarray,
    groups: Sequence[str],
    k: int = 10,
    n_queries: int = 20_000,
    seed: int = 0,
) -> Dict[str, object]:
    """
    How often a concept's nearest neighbours are in its own disease group.

    The chance level is what you would get by picking neighbours at random, so a model whose
    vectors ignore the groups scores about that.
    """
    rng = np.random.RandomState(seed)
    groups_array = np.asarray(groups)
    n = len(groups_array)
    queries = np.sort(rng.choice(n, size=min(n_queries, n), replace=False))
    neighbours = top_k_cosine(embeddings[queries], embeddings, k, exclude=queries)
    same = groups_array[neighbours] == groups_array[queries][:, None]
    _, counts = np.unique(groups_array, return_counts=True)
    shares = counts / n
    per_group = {
        str(g): float(same[groups_array[queries] == g].mean())
        for g in np.unique(groups_array[queries])
    }
    return {
        "k": int(neighbours.shape[1]),
        "n_queries": int(len(queries)),
        "purity": float(same.mean()),
        "chance": float((shares**2).sum()),
        "purity_by_group": per_group,
    }


def parent_retrieval(
    embeddings: np.ndarray,
    concept_ids: Sequence[int],
    parents: Dict[int, List[int]],
    ks: Sequence[int] = (1, 5, 10),
    n_queries: int = 10_000,
    seed: int = 0,
) -> Dict[str, object]:
    """Whether a concept's parent is among its k nearest neighbours by name."""
    position = {int(c): i for i, c in enumerate(concept_ids)}
    children = sorted(c for c in parents if int(c) in position)
    if not children:
        return {"n_queries": 0}
    rng = np.random.RandomState(seed)
    if len(children) > n_queries:
        children = sorted(rng.choice(children, size=n_queries, replace=False).tolist())
    rows = np.array([position[int(c)] for c in children])
    neighbours = top_k_cosine(embeddings[rows], embeddings, max(ks), exclude=rows)
    parent_rows = [
        {position[int(p)] for p in parents[c] if int(p) in position} for c in children
    ]
    result: Dict[str, object] = {"n_queries": len(children)}
    for k in ks:
        hits = [
            bool(parent_rows[i] & set(neighbours[i, :k].tolist()))
            for i in range(len(children))
        ]
        result[f"accuracy_at_{k}"] = float(np.mean(hits))
    return result


# ---------------------------------------------------------------------------------------
# Projection and plot
# ---------------------------------------------------------------------------------------
def project_2d(
    embeddings: np.ndarray, method: str = "auto", seed: int = 0
) -> Tuple[np.ndarray, str]:
    """Two dimensional coordinates, and the method that was used."""
    from sklearn.decomposition import PCA

    if method == "auto":
        try:
            import umap  # noqa: F401

            method = "umap"
        except ImportError:
            method = "tsne"
    if method == "pca" or embeddings.shape[0] < 5:
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings), "pca"
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(embeddings), "umap"
    if method == "tsne":
        from sklearn.manifold import TSNE

        reduced = embeddings
        if embeddings.shape[1] > 50:
            reduced = PCA(n_components=50, random_state=seed).fit_transform(embeddings)
        perplexity = float(min(30, max(2, (embeddings.shape[0] - 1) // 3)))
        coordinates = TSNE(
            n_components=2, perplexity=perplexity, init="pca", random_state=seed
        ).fit_transform(reduced)
        return coordinates, "tsne"
    raise ValueError(f"Unknown projection method {method}")


def plot_projection(
    coordinates: np.ndarray,
    groups: Sequence[str],
    path: str,
    title: str,
    label_groups: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups_array = np.asarray(groups)
    unique = [g for g in sorted(set(groups_array)) if g not in (OTHER_GROUP, MULTIPLE_GROUP)]
    # tab20 lists each colour next to its pale twin; take the strong ones first so that
    # neighbouring groups are easy to tell apart
    palette = []
    for name in ("tab20", "tab20b", "tab20c"):
        colours_of = list(plt.get_cmap(name).colors)
        palette += colours_of[0::2] + colours_of[1::2]
    colours = {g: palette[i % len(palette)] for i, g in enumerate(unique)}
    # Few points get bigger markers, so a small plot is still readable
    marker_size = float(np.clip(600 / np.sqrt(max(len(groups_array), 1)), 3, 60))
    colours[OTHER_GROUP] = (0.78, 0.78, 0.78)
    colours[MULTIPLE_GROUP] = (0.55, 0.55, 0.55)
    figure, axis = plt.subplots(figsize=(13, 9))
    # The grey groups first, so the coloured ones sit on top
    for group in [OTHER_GROUP, MULTIPLE_GROUP] + unique:
        mask = groups_array == group
        if not mask.any():
            continue
        axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=marker_size, alpha=0.6,
            color=colours[group], label=f"{group} ({int(mask.sum())})", linewidths=0,
        )
        if label_groups and group in unique:
            centre = np.median(coordinates[mask], axis=0)
            axis.annotate(group, centre, fontsize=7, weight="bold", ha="center")
    axis.set_title(textwrap.fill(title, 110), fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    legend = axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    # The legend keys have a fixed size, whatever the size of the points in the plot
    for handle in getattr(legend, "legend_handles", None) or legend.legendHandles:
        handle.set_sizes([40])
        handle.set_alpha(1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------
def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept_path", required=True, help="OMOP concept table (parquet file/folder or Athena csv)")
    parser.add_argument("--concept_ancestor_path", default=None, help="concept_ancestor table; needed for the SNOMED groups, the ATC hierarchy and the parent test")
    parser.add_argument("--concept_relationship_path", default=None, help="concept_relationship table; needed to colour RxNorm by ATC class")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, help="Where the embeddings are cached (default: output_dir). Share one cache between runs on the same concepts and model, e.g. with different group depths, to embed only once")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face encoder (or a local folder)")
    parser.add_argument("--vocabulary_id", default="SNOMED", help="SNOMED, RxNorm, ...")
    parser.add_argument("--domains", nargs="+", default=None, help="Keep only these domains, e.g. Condition; default all")
    parser.add_argument("--concept_classes", nargs="+", default=None, help="Keep only these concept classes; for RxNorm Ingredient is the usual choice")
    parser.add_argument("--include_non_standard", action="store_true")
    parser.add_argument("--include_invalid", action="store_true")
    parser.add_argument("--max_concepts", type=int, default=0, help="Use a random sample of this many concepts (0 = all)")
    parser.add_argument("--text_template", default="{concept_name}", help='e.g. "{domain_id}: {concept_name}"')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=50_000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if a cache matches")
    parser.add_argument("--color_by", choices=["disease_group", "atc_group", "domain_id", "concept_class_id"], default=None, help="Default: atc_group for RxNorm vocabularies, disease_group otherwise")
    parser.add_argument("--group_root_name", default="Disease", help="SNOMED: exact name of the concept whose descendants define the disease groups")
    parser.add_argument("--group_root_id", type=int, default=None, help="SNOMED: use this concept id as the root instead of looking up the name")
    parser.add_argument("--group_depth", type=int, default=2, help="SNOMED: levels below the root at which the groups are taken (1 = its children)")
    parser.add_argument("--atc_level", type=int, default=1, choices=sorted(ATC_LEVEL_CLASSES), help="RxNorm: ATC level of the groups (1 = anatomical main group, 2 = therapeutic subgroup, ...)")
    parser.add_argument("--atc_relationship_ids", nargs="+", default=None, help="RxNorm: restrict the ATC links to these relationship_id values")
    parser.add_argument("--rxnorm_vocabularies", nargs="+", default=["RxNorm"], help="Vocabularies treated as the drug side of the ATC links")
    parser.add_argument("--max_groups", type=int, default=20)
    parser.add_argument("--min_group_size", type=int, default=20)
    parser.add_argument("--tie_break", choices=["multiple", "first"], default="multiple")
    parser.add_argument("--num_clusters", type=int, default=0, help="k-means clusters (0 = the number of groups)")
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--projection", choices=["auto", "tsne", "umap", "pca"], default="auto")
    parser.add_argument("--max_plot_points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def concept_names_by_id(concept_path: str, concept_ids: Sequence[int]) -> Dict[int, str]:
    frame = (
        read_table(concept_path)
        .select(pl.col("concept_id").cast(pl.Int64), "concept_name")
        .filter(pl.col("concept_id").is_in(list(concept_ids)))
        .collect()
    )
    return dict(zip(frame["concept_id"].to_list(), frame["concept_name"].to_list()))


def compute_groups(
    args: argparse.Namespace,
    concepts: pl.DataFrame,
    ancestors: Optional[pl.LazyFrame],
) -> Tuple[List[str], Optional[DiseaseGroups]]:
    """The group of every concept, in the order of the concepts frame."""
    concept_ids = concepts["concept_id"].to_list()
    color_by = args.color_by or ("atc_group" if args.vocabulary_id.lower().startswith("rxnorm") else "disease_group")
    if color_by in ("domain_id", "concept_class_id"):
        return concepts[color_by].to_list(), None
    if color_by == "atc_group":
        if not args.concept_relationship_path:
            raise ValueError("--color_by atc_group needs --concept_relationship_path")
        pairs = load_atc_pairs(
            args.concept_path, args.concept_relationship_path,
            args.rxnorm_vocabularies, args.atc_relationship_ids,
        )
        if len(pairs) == 0:
            raise ValueError("No RxNorm to ATC links found in concept_relationship; check --atc_relationship_ids")
        grouping = assign_atc_groups(
            pairs, ancestors, atc_level_labels(args.concept_path, args.atc_level), concept_ids,
            ingredients_of(args.concept_path, args.rxnorm_vocabularies),
            args.max_groups, args.min_group_size, args.tie_break,
        )
        return [grouping.names[c] for c in concept_ids], grouping
    if ancestors is None:
        raise ValueError("--color_by disease_group needs --concept_ancestor_path")
    root = args.group_root_id or find_concept_id(args.concept_path, args.group_root_name, args.vocabulary_id)
    anchor_ids = discover_anchors(ancestors, root, args.group_depth)
    if not anchor_ids:
        raise ValueError(
            f"No concept is {args.group_depth} level(s) below concept {root}; "
            "try another --group_depth or --group_root_name"
        )
    all_names = concept_names_by_id(args.concept_path, anchor_ids + [root])
    anchor_names = {a: str(all_names.get(a, a)) for a in anchor_ids}
    LOG.info(
        "Disease groups are the %s concepts %s level(s) below %s (%s): %s",
        len(anchor_ids), args.group_depth, all_names.get(root, root), root,
        sorted(anchor_names.values()),
    )
    grouping = assign_disease_groups(
        ancestors, concept_ids, anchor_names, args.max_groups, args.min_group_size, args.tie_break
    )
    return [grouping.names[c] for c in concept_ids], grouping


def _short_model_name(model: str) -> str:
    """A Hugging Face name as it is; a local folder as its last two components."""
    if os.path.isabs(model) or model.startswith((".", "~")):
        return "/".join(model.rstrip("/").split("/")[-2:])
    return model


def run(args: argparse.Namespace, encoder=None) -> Dict[str, object]:
    os.makedirs(args.output_dir, exist_ok=True)
    concepts = load_concepts(
        args.concept_path, args.vocabulary_id, args.domains,
        args.include_non_standard, args.include_invalid, args.concept_classes,
    )
    if args.max_concepts and len(concepts) > args.max_concepts:
        concepts = concepts.sample(n=args.max_concepts, seed=args.seed).sort("concept_id")
    if len(concepts) == 0:
        raise ValueError("No concepts left after filtering")
    LOG.info("%s concepts to embed", len(concepts))

    if encoder is None:
        encoder = HuggingFaceEncoder(args.model, args.batch_size, args.max_length, fp16=args.fp16)
    texts = render_texts(concepts, args.text_template)
    embeddings = embed_with_cache(
        concepts, texts, encoder, args.cache_dir or args.output_dir, args.chunk_size, args.overwrite
    )
    concept_ids = concepts["concept_id"].to_list()

    ancestors = read_ancestors(args.concept_ancestor_path) if args.concept_ancestor_path else None
    metrics: Dict[str, object] = {
        "vocabulary_id": args.vocabulary_id,
        "num_concepts": len(concepts),
        "embedding_dim": int(embeddings.shape[1]),
        "model": args.model,
    }
    groups, grouping = compute_groups(args, concepts, ancestors)
    if grouping is not None:
        sizes = sorted(grouping.sizes.items(), key=lambda kv: -kv[1])
        pl.DataFrame({"group": [g for g, _ in sizes], "num_concepts": [n for _, n in sizes]}).write_csv(
            os.path.join(args.output_dir, "groups.csv")
        )
        LOG.info("Groups: %s", sizes)
        metrics["group_sizes"] = dict(sizes)

    # Cluster analysis and neighbour tests only make sense with more than one real group
    labelled = [i for i, g in enumerate(groups) if g not in (OTHER_GROUP, MULTIPLE_GROUP)]
    group_count = len({groups[i] for i in labelled})
    if group_count > 1:
        subset = np.asarray(labelled)
        n_clusters = args.num_clusters or group_count
        agreement, composition = cluster_agreement(embeddings[subset], [groups[i] for i in labelled], n_clusters, args.seed)
        metrics["cluster_agreement"] = agreement
        metrics["neighbour_group_purity"] = neighbor_group_purity(
            embeddings[subset], [groups[i] for i in labelled], args.knn, seed=args.seed
        )
        pl.DataFrame(composition).write_csv(os.path.join(args.output_dir, "cluster_composition.csv"))
    # A concept's parent is only a meaningful neighbour inside one hierarchy (SNOMED)
    if ancestors is not None and not args.vocabulary_id.lower().startswith("rxnorm"):
        metrics["parent_retrieval"] = parent_retrieval(
            embeddings, concept_ids, direct_parents(ancestors, concept_ids), seed=args.seed
        )

    rng = np.random.RandomState(args.seed)
    plotted = np.sort(rng.choice(len(concepts), size=min(args.max_plot_points, len(concepts)), replace=False))
    coordinates, method = project_2d(embeddings[plotted], args.projection, args.seed)
    pl.DataFrame(
        {
            "concept_id": [concept_ids[i] for i in plotted],
            "concept_name": concepts["concept_name"].gather(plotted).to_list(),
            "group": [groups[i] for i in plotted],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    ).write_csv(os.path.join(args.output_dir, "projection.csv"))
    plot_projection(
        coordinates, [groups[i] for i in plotted],
        os.path.join(args.output_dir, "projection.png"),
        f"{args.vocabulary_id} concept names, {_short_model_name(args.model)} "
        f"({method}, {len(plotted)} of {len(concepts)} concepts)",
    )
    metrics["projection"] = method
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(create_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
