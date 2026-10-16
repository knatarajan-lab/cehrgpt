"""
Build the labelled queries for the patient query model.

Inputs are the outputs of the existing linear probing pipeline:

* the feature parquet files written by compute_cehrgpt_features (one row per patient and
  prediction time with the linear probing hidden state in ``features``). Extract them with
  --disable_combine_global_local: the model is conditioned on the linear probing state only,
  not on the base model's last-token state
* the tokenized full patient sequences, which supply the events after the prediction time
  that the labels are computed from
* the tokenizer folder: concept names, the ontology (for descendants and domains), the
  numeric lab statistics (to leave out measurements) and the MOTOR task set

A query is a concept (covering all its descendants) and a time window in days after the
prediction time; the label says whether the patient has the concept in that window.
"""

import argparse
import glob
import json
import logging
import os
import random
from collections import Counter
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk

from cehrgpt.qa.concept_filter import (
    DEFAULT_ALLOWED_DOMAINS,
    eligible_concepts,
    load_concept_names,
    load_motor_codes,
    load_value_concepts,
)
from cehrgpt.qa.concept_hierarchy import ConceptHierarchy, query_concepts
from cehrgpt.qa.query_generation import (
    DEFAULT_WINDOWS,
    PatientFuture,
    PatientTimeline,
    QueryConfig,
    QueryCoverage,
    generate_queries,
)

LOG = logging.getLogger(__name__)
QUERY_VOCABULARY_FILE = "query_vocabulary.json"


def coverage_report(
    dataset: Dataset, vocabulary: Dict[str, int], motor_codes: Set[str]
) -> Dict[str, float]:
    """How much of the query vocabulary the queries reach."""
    asked = Counter(dataset["concept_id"])
    with_yes = {c for c, y in zip(dataset["concept_id"], dataset["label"]) if y}
    outside_motor = [c for c in vocabulary if c not in motor_codes]
    n = len(vocabulary)
    labels = np.asarray(dataset["label"], dtype=float)
    return {
        "query_concepts": n,
        "concepts_queried": len(asked),
        "concepts_queried_fraction": len(asked) / n if n else 0.0,
        "concepts_with_a_yes": len(with_yes),
        "concepts_with_a_yes_fraction": len(with_yes) / n if n else 0.0,
        "concepts_queried_outside_motor": sum(1 for c in outside_motor if c in asked),
        "concepts_outside_motor": len(outside_motor),
        "least_queried_concept_count": min(asked.values()) if asked else 0,
        "median_queries_per_queried_concept": (
            float(np.median(list(asked.values()))) if asked else 0.0
        ),
        "yes_fraction": float(labels.mean()) if len(labels) else 0.0,
    }


def iter_query_rows(
    feature_files: List[str],
    tokenized_dataset: Dataset,
    hierarchy: ConceptHierarchy,
    eligible_tokens: Set[str],
    vocabulary: Dict[str, int],
    queries_per_patient: int,
    seed: int,
    query_config: Optional[Dict] = None,
    motor_codes: Optional[Set[str]] = None,
    coverage_candidates: int = 8,
    max_patients: int = 0,
) -> Iterator[Dict]:
    person_row = {
        int(person_id): row
        for row, person_id in enumerate(tokenized_dataset["person_id"])
    }
    rng = random.Random(seed)
    config = QueryConfig(**(query_config or {}))
    coverage = QueryCoverage(coverage_candidates)
    pool = sorted(vocabulary)
    concept_set = set(pool)
    patients = 0
    for path in feature_files:
        table = pq.read_table(
            path, columns=["subject_id", "prediction_time_posix", "features"]
        ).to_pylist()
        for record in table:
            if max_patients and patients >= max_patients:
                return
            row = person_row.get(int(record["subject_id"]))
            if row is None:
                continue
            sequence = tokenized_dataset[row]
            timeline = PatientTimeline(
                concept_ids=sequence["concept_ids"],
                epoch_times=list(sequence["epoch_times"]),
                index_time=float(record["prediction_time_posix"]),
            )
            future = PatientFuture(timeline, hierarchy, eligible_tokens, concept_set)
            queries = generate_queries(
                future, pool, rng, queries_per_patient, config, motor_codes or (), coverage
            )
            if not queries:
                continue
            patients += 1
            for query in queries:
                yield {
                    "patient_features": [float(x) for x in record["features"]],
                    "concept_id": query.concept_id,
                    "start_days": query.start_days,
                    "end_days": query.end_days,
                    "label": query.label,
                    "is_motor_task": query.concept_id in (motor_codes or ()),
                    "covered_tokens": vocabulary[query.concept_id],
                    "person_id": int(record["subject_id"]),
                    "prediction_time_posix": float(record["prediction_time_posix"]),
                }


def build_query_dataset(
    features_folder: str,
    tokenized_full_dataset_path: str,
    tokenizer_path: str,
    output_path: str,
    queries_per_patient: int = 4,
    allowed_domains: Tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
    exclude_categorical_concepts: bool = False,
    max_covered_tokens: Optional[int] = None,
    coverage_candidates: int = 8,
    max_patients: int = 0,
    seed: int = 42,
    query_config: Optional[Dict] = None,
    expected_feature_dim: int = 0,
) -> Dataset:
    feature_files = sorted(
        glob.glob(os.path.join(features_folder, "**", "*.parquet"), recursive=True)
    )
    if not feature_files:
        raise FileNotFoundError(f"No parquet files under {features_folder}")
    all_names = load_concept_names(tokenizer_path)
    hierarchy = ConceptHierarchy.from_tokenizer_folder(tokenizer_path)
    # Queries are about conditions, procedures and medications: no measurement or
    # observation records and none of the artificial tokens
    eligible = eligible_concepts(
        all_names,
        load_value_concepts(tokenizer_path, exclude_categorical_concepts),
        hierarchy.domains,
        allowed_domains,
    )
    vocabulary = query_concepts(
        eligible, hierarchy, allowed_domains, max_covered_tokens
    )
    LOG.info(
        "%s of %s tokens are eligible; %s concepts can be queried",
        len(eligible),
        len(all_names),
        len(vocabulary),
    )
    if not vocabulary:
        raise ValueError("No query concepts: check the tokenizer folder and the domains")
    motor_codes = load_motor_codes(tokenizer_path)

    feature_dim = len(
        pq.read_table(feature_files[0], columns=["features"]).slice(0, 1).to_pylist()[0][
            "features"
        ]
    )
    if expected_feature_dim and feature_dim != expected_feature_dim:
        raise ValueError(
            f"The features are {feature_dim}-dimensional but {expected_feature_dim} were "
            "expected. The model takes the linear probing hidden state only; extract the "
            "features with --disable_combine_global_local so the last-token state is not "
            "appended."
        )
    tokenized = load_from_disk(tokenized_full_dataset_path)
    # A DatasetDict holds splits; the labels only need every patient's sequence once
    if hasattr(tokenized, "values"):
        tokenized = concatenate_datasets(list(tokenized.values()))

    schema = Features(
        {
            "patient_features": Sequence(Value("float32"), length=feature_dim),
            "concept_id": Value("string"),
            "start_days": Value("int32"),
            "end_days": Value("int32"),
            "label": Value("bool"),
            "is_motor_task": Value("bool"),
            "covered_tokens": Value("int32"),
            "person_id": Value("int64"),
            "prediction_time_posix": Value("float64"),
        }
    )
    dataset = Dataset.from_generator(
        iter_query_rows,
        gen_kwargs={
            "feature_files": feature_files,
            "tokenized_dataset": tokenized,
            "hierarchy": hierarchy,
            "eligible_tokens": set(eligible),
            "vocabulary": vocabulary,
            "queries_per_patient": queries_per_patient,
            "seed": seed,
            "query_config": query_config,
            "motor_codes": motor_codes,
            "coverage_candidates": coverage_candidates,
            "max_patients": max_patients,
        },
        features=schema,
    )
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    # The model needs the vocabulary (and its order) to embed the query concepts
    with open(os.path.join(output_path, QUERY_VOCABULARY_FILE), "w") as f:
        json.dump(
            {"concept_ids": sorted(vocabulary), "covered_tokens": vocabulary}, f
        )
    report = coverage_report(dataset, vocabulary, motor_codes)
    with open(os.path.join(output_path, "coverage_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    LOG.info("Query coverage: %s", report)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_folder", required=True)
    parser.add_argument("--tokenized_full_dataset_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--queries_per_patient", type=int, default=4)
    parser.add_argument("--max_patients", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected_feature_dim",
        type=int,
        default=0,
        help="Fail if the features have another size, e.g. 768 for the linear probing state "
        "alone (the default extraction appends a second 768 unless combining is disabled)",
    )
    parser.add_argument(
        "--allowed_domains", nargs="+", default=list(DEFAULT_ALLOWED_DOMAINS)
    )
    parser.add_argument(
        "--exclude_categorical_concepts",
        action="store_true",
        help="Also leave out concepts with categorical results (observation-like records)",
    )
    parser.add_argument(
        "--max_covered_tokens",
        type=int,
        default=None,
        help="Drop broad concepts that cover more than this many tokens (a query like "
        "'any clinical finding' is true for almost everyone)",
    )
    parser.add_argument(
        "--coverage_candidates",
        type=int,
        default=8,
        help="Among this many random candidates query the least queried concept so far; "
        "1 follows the data and neglects rare concepts",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[v for w in DEFAULT_WINDOWS for v in w],
        help="Windows as flat pairs of days: start1 end1 start2 end2 ...",
    )
    parser.add_argument("--positive_fraction", type=float, default=0.5)
    parser.add_argument("--new_onset_only", action="store_true")
    parser.add_argument("--out_of_scope_fraction", type=float, default=None)
    args = vars(parser.parse_args())
    flat = args.pop("windows")
    if len(flat) % 2:
        parser.error("--windows needs an even number of values")
    query_config = {
        "windows": tuple(zip(flat[::2], flat[1::2])),
        "positive_fraction": args.pop("positive_fraction"),
        "new_onset_only": args.pop("new_onset_only"),
        "out_of_scope_fraction": args.pop("out_of_scope_fraction"),
    }
    QueryConfig(**query_config)  # fail on bad values before the long run starts
    args["allowed_domains"] = tuple(args["allowed_domains"])
    dataset = build_query_dataset(**args, query_config=query_config)
    print(f"Wrote {len(dataset)} queries to {args['output_path']}")


if __name__ == "__main__":
    main()
