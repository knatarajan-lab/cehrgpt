"""
Build the question/answer dataset for the patient language model.

Inputs are the outputs of the existing linear probing pipeline:

* the feature parquet files written by compute_cehrgpt_features (one row per
  patient and prediction time with the linear probing hidden state in ``features``).
  Extract them with --disable_combine_global_local: the language model is conditioned on the
  linear probing state only, not on the base model's last-token state
* the tokenized full patient sequences, which supply the events after the prediction time
  that the answers are computed from
* the tokenizer folder, for the readable concept names

Everything is generated locally from templates; no patient data is sent anywhere.
"""

import argparse
import glob
import json
import logging
import os
import pickle
import random
from collections import Counter
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, load_from_disk

from cehrgpt.tokenization.tokenization_constants import (
    MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME,
)
from cehrgpt.qa.concept_filter import (
    DEFAULT_ALLOWED_DOMAINS,
    eligible_concepts,
    load_concept_domains,
    load_value_concepts,
)
from cehrgpt.qa.question_templates import (
    ConceptCoverage,
    PatientTimeline,
    QAConfig,
    generate_qa_pairs,
)

LOG = logging.getLogger(__name__)
CONCEPT_NAME_FILE = "concept_name_mapping.json"


def load_concept_names(tokenizer_path: str) -> Dict[str, str]:
    with open(os.path.join(tokenizer_path, CONCEPT_NAME_FILE)) as f:
        return json.load(f)


def load_motor_codes(tokenizer_path: str) -> Set[str]:
    """The concepts that are MOTOR tasks; empty if the tokenizer has no task information."""
    path = os.path.join(tokenizer_path, MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME)
    if not os.path.exists(path):
        LOG.warning(
            "%s not found, so no question can be tagged as inside the MOTOR task set", path
        )
        return set()
    with open(path, "rb") as f:
        info = pickle.load(f) or {}
    return {str(code) for code in info.get("motor_time_to_event_codes", [])}


def coverage_report(
    dataset: Dataset, eligible: Dict[str, str], motor_codes: Set[str]
) -> Dict[str, float]:
    """How much of the eligible vocabulary the questions reach."""
    asked = Counter(dataset["concept_id"])
    happened = Counter(
        c for c, occurred in zip(dataset["concept_id"], dataset["event_occurred"]) if occurred
    )
    vocabulary = len(eligible)
    outside_motor = [c for c in eligible if c not in motor_codes]
    return {
        "eligible_concepts": vocabulary,
        "concepts_asked": len(asked),
        "concepts_asked_fraction": len(asked) / vocabulary if vocabulary else 0.0,
        "concepts_with_an_event_answer": len(happened),
        "concepts_with_an_event_answer_fraction": (
            len(happened) / vocabulary if vocabulary else 0.0
        ),
        "concepts_asked_outside_motor": sum(1 for c in outside_motor if c in asked),
        "concepts_outside_motor": len(outside_motor),
        "least_asked_concept_count": min(asked.values()) if asked else 0,
        "median_questions_per_asked_concept": (
            float(np.median(list(asked.values()))) if asked else 0.0
        ),
    }


def iter_qa_rows(
    feature_files: List[str],
    tokenized_dataset: Dataset,
    concept_names: Dict[str, str],
    negative_pool: List[str],
    questions_per_patient: int,
    seed: int,
    max_patients: int = 0,
    qa_config: Optional[Dict] = None,
    motor_codes: Optional[Set[str]] = None,
    coverage_candidates: int = 8,
) -> Iterator[Dict]:
    person_row = {
        int(person_id): row
        for row, person_id in enumerate(tokenized_dataset["person_id"])
    }
    rng = random.Random(seed)
    config = QAConfig(**(qa_config or {}))
    coverage = ConceptCoverage(coverage_candidates)
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
            pairs = generate_qa_pairs(
                timeline,
                concept_names,
                negative_pool,
                rng,
                questions_per_patient,
                config,
                motor_codes or (),
                coverage,
            )
            if not pairs:
                continue
            patients += 1
            for pair in pairs:
                yield {
                    "patient_features": [float(x) for x in record["features"]],
                    "question": pair.question,
                    "answer": pair.answer,
                    "qa_type": pair.qa_type,
                    "concept_id": pair.concept_id,
                    "is_motor_task": pair.concept_id in (motor_codes or ()),
                    "event_occurred": pair.event_occurred,
                    "person_id": int(record["subject_id"]),
                    "prediction_time_posix": float(record["prediction_time_posix"]),
                }


def build_qa_dataset(
    features_folder: str,
    tokenized_full_dataset_path: str,
    tokenizer_path: str,
    output_path: str,
    questions_per_patient: int = 4,
    concept_table: Optional[str] = None,
    allowed_domains: Tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
    exclude_categorical_concepts: bool = False,
    coverage_candidates: int = 8,
    max_patients: int = 0,
    seed: int = 42,
    qa_config: Optional[Dict] = None,
    expected_feature_dim: int = 0,
) -> Dataset:
    feature_files = sorted(
        glob.glob(os.path.join(features_folder, "**", "*.parquet"), recursive=True)
    )
    if not feature_files:
        raise FileNotFoundError(f"No parquet files under {features_folder}")
    all_names = load_concept_names(tokenizer_path)
    # Questions are about conditions, procedures and medications: no measurement or
    # observation records and none of the artificial tokens
    concept_names = eligible_concepts(
        all_names,
        load_value_concepts(tokenizer_path, exclude_categorical_concepts),
        load_concept_domains(concept_table) if concept_table else None,
        allowed_domains,
    )
    LOG.info(
        "%s of the %s vocabulary tokens are eligible for questions",
        len(concept_names),
        len(all_names),
    )
    if not concept_names:
        raise ValueError("No eligible concepts: check the concept table and the tokenizer")
    tokenized = load_from_disk(tokenized_full_dataset_path)
    # A DatasetDict holds splits; the answers only need every patient's sequence once
    if hasattr(tokenized, "values"):
        from datasets import concatenate_datasets

        tokenized = concatenate_datasets(list(tokenized.values()))
    negative_pool = sorted(concept_names)
    motor_codes = load_motor_codes(tokenizer_path)
    feature_dim = len(
        pq.read_table(feature_files[0], columns=["features"]).slice(0, 1).to_pylist()[0][
            "features"
        ]
    )
    if expected_feature_dim and feature_dim != expected_feature_dim:
        raise ValueError(
            f"The features are {feature_dim}-dimensional but {expected_feature_dim} were expected. "
            "The language model takes the linear probing hidden state only; extract the features "
            "with --disable_combine_global_local so the last-token state is not appended."
        )
    schema = Features(
        {
            "patient_features": Sequence(Value("float32"), length=feature_dim),
            "question": Value("string"),
            "answer": Value("string"),
            "qa_type": Value("string"),
            "concept_id": Value("string"),
            "is_motor_task": Value("bool"),
            "event_occurred": Value("bool"),
            "person_id": Value("int64"),
            "prediction_time_posix": Value("float64"),
        }
    )
    dataset = Dataset.from_generator(
        iter_qa_rows,
        gen_kwargs={
            "feature_files": feature_files,
            "tokenized_dataset": tokenized,
            "concept_names": concept_names,
            "negative_pool": negative_pool,
            "questions_per_patient": questions_per_patient,
            "seed": seed,
            "max_patients": max_patients,
            "qa_config": qa_config,
            "motor_codes": motor_codes,
            "coverage_candidates": coverage_candidates,
        },
        features=schema,
    )
    dataset.save_to_disk(output_path)
    report = coverage_report(dataset, concept_names, motor_codes)
    with open(os.path.join(output_path, "coverage_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    LOG.info("Concept coverage: %s", report)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_folder", required=True)
    parser.add_argument("--tokenized_full_dataset_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--questions_per_patient", type=int, default=4)
    parser.add_argument(
        "--concept_table",
        default=None,
        help="OMOP concept table (parquet or csv, file or folder) with concept_id and "
        "domain_id; keeps only --allowed_domains. Without it visit/observation/device "
        "concepts that carry no value cannot be told apart and stay eligible",
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
        "--coverage_candidates",
        type=int,
        default=8,
        help="Among this many random candidates ask about the least asked concept so far; "
        "1 follows the data and neglects rare concepts",
    )
    parser.add_argument(
        "--out_of_scope_fraction",
        type=float,
        default=None,
        help="Share of questions about concepts that are not MOTOR tasks",
    )
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
        "--horizons_in_days", type=int, nargs="+", default=list(QAConfig.horizons_in_days)
    )
    parser.add_argument("--will_occur_weight", type=float, default=1.0)
    parser.add_argument("--time_to_event_weight", type=float, default=1.0)
    parser.add_argument("--will_occur_positive_fraction", type=float, default=0.5)
    parser.add_argument("--time_to_event_positive_fraction", type=float, default=0.7)
    parser.add_argument("--no_paraphrase", action="store_true")
    parser.add_argument(
        "--new_onset_only",
        action="store_true",
        help="Only ask about concepts the patient has not had before the index date",
    )
    args = vars(parser.parse_args())
    qa_config = {
        "horizons_in_days": tuple(args.pop("horizons_in_days")),
        "will_occur_weight": args.pop("will_occur_weight"),
        "time_to_event_weight": args.pop("time_to_event_weight"),
        "will_occur_positive_fraction": args.pop("will_occur_positive_fraction"),
        "time_to_event_positive_fraction": args.pop("time_to_event_positive_fraction"),
        "paraphrase": not args.pop("no_paraphrase"),
        "new_onset_only": args.pop("new_onset_only"),
        "out_of_scope_fraction": args.pop("out_of_scope_fraction"),
    }
    QAConfig(**qa_config)  # fail on bad values before the long run starts
    args["allowed_domains"] = tuple(args["allowed_domains"])
    dataset = build_qa_dataset(**args, qa_config=qa_config)
    print(f"Wrote {len(dataset)} question/answer rows to {args['output_path']}")


if __name__ == "__main__":
    main()
