"""Evaluate patient-level time to the next diagnosis using model rollouts.

For each randomly sampled patient, this tool selects one eligible visit-end cutoff,
generates future trajectories, and reduces the rollouts to one patient-level risk
score: the negative restricted mean time to the first generated Condition token.
The observed outcome is the first ``condition_occurrence`` after the cutoff, with
administrative censoring at the requested horizon. Harrell's c-index is computed
across patients, along with a patient-level bootstrap confidence interval.

Example:
    python -m cehrgpt.tools.evaluate_diagnosis_rollouts \
        --model /path/to/model \
        --sequences /path/to/patient_sequence/test \
        --condition-occurrence /path/to/condition_occurrence \
        --concept /path/to/concept \
        --output /path/to/diagnosis_rollout_cindex.json \
        --device cuda:0
"""

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel


SECONDS_PER_DAY = 86400


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patient-level c-index for time to the next diagnosis"
    )
    parser.add_argument("--model", required=True, help="CEHR-GPT checkpoint")
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer path; defaults to --model",
    )
    parser.add_argument(
        "--sequences", required=True, help="Test patient_sequence parquet folder"
    )
    parser.add_argument(
        "--condition-occurrence",
        required=True,
        help="OMOP condition_occurrence parquet folder",
    )
    parser.add_argument(
        "--concept", required=True, help="OMOP concept parquet folder"
    )
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--rollout-batch-size", type=int)
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-n-trial",
        type=int,
        default=2,
        help="Maximum attempts to replace incomplete rollouts",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Patient-level bootstrap replicates; set to 0 to disable",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cuda:0 or cpu; auto chooses CUDA when available",
    )
    return parser


def parquet_files(path: str) -> List[str]:
    files = sorted(str(item) for item in Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


def select_cutoffs(
    sequence_path: str, sample_size: int, horizon_days: int, seed: int
) -> List[Dict[str, Any]]:
    """Select one random eligible visit-end cutoff per sampled patient."""
    files = parquet_files(sequence_path)
    dataset = ds.dataset(files, format="parquet")
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    rng = np.random.default_rng(seed)
    candidate_indices = rng.permutation(total_rows)
    selected = []
    offset = 0

    while len(selected) < sample_size and offset < total_rows:
        indices = candidate_indices[offset : offset + 2048]
        offset += len(indices)
        records = dataset.take(indices).to_pylist()
        for record in records:
            concepts = record["concept_ids"]
            epoch_times = record["epoch_times"]
            if len(concepts) < 2 or not epoch_times:
                continue

            sequence_end = max(epoch_times)
            prefix_max_time = -math.inf
            eligible = []
            for index, (token, epoch_time) in enumerate(
                zip(concepts, epoch_times)
            ):
                prefix_max_time = max(prefix_max_time, epoch_time)
                if (
                    is_visit_end(token)
                    and index >= 4
                    and sequence_end - prefix_max_time
                    >= horizon_days * SECONDS_PER_DAY
                ):
                    eligible.append((index, prefix_max_time))

            if not eligible:
                continue
            cutoff_index, cutoff_time = eligible[int(rng.integers(len(eligible)))]
            selected.append(
                {
                    "person_id": int(record["person_id"]),
                    "cutoff_index": int(cutoff_index),
                    "cutoff_timestamp": float(cutoff_time),
                    "prefix": concepts[: cutoff_index + 1],
                }
            )
            if len(selected) >= sample_size:
                break

    if len(selected) < sample_size:
        raise RuntimeError(
            f"Only found {len(selected)} patients with an eligible visit boundary"
        )
    return selected


def load_condition_tokens(
    concept_path: str, tokenizer: CehrGptTokenizer
) -> Set[str]:
    """Return Condition-domain concept IDs represented in the tokenizer."""
    table = ds.dataset(parquet_files(concept_path), format="parquet").to_table(
        columns=["concept_id", "domain_id"],
        filter=pc.field("domain_id") == "Condition",
    )
    vocab = tokenizer.get_vocab()
    return {
        str(concept_id)
        for concept_id in table["concept_id"].to_pylist()
        if str(concept_id) in vocab
    }


def load_observed_conditions(
    condition_path: str, person_ids: Sequence[int]
) -> Dict[int, List[Tuple[float, str]]]:
    """Read condition occurrences only for the sampled patients."""
    table = ds.dataset(parquet_files(condition_path), format="parquet").to_table(
        columns=["person_id", "condition_concept_id", "condition_start_datetime"],
        filter=pc.is_in(pc.field("person_id"), value_set=pa.array(person_ids)),
    )
    by_person = {person_id: [] for person_id in person_ids}
    for record in table.to_pylist():
        timestamp = record["condition_start_datetime"]
        if timestamp is None:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
        else:
            timestamp = timestamp.astimezone(dt.timezone.utc)
        by_person[int(record["person_id"])].append(
            (timestamp.timestamp(), str(record["condition_concept_id"]))
        )
    for events in by_person.values():
        events.sort()
    return by_person


def truncate_prefix(prefix: Sequence[str], max_length: int) -> List[str]:
    """Keep the newest complete-visit-aligned suffix that fits the model."""
    prefix = list(prefix)
    if len(prefix) <= max_length:
        return prefix
    start = len(prefix) - max_length
    for index in range(start, len(prefix)):
        if is_visit_start(prefix[index]):
            return prefix[index:]
    return prefix[-max_length:]


def harrell_c_index(rows: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    """Compute censored Harrell concordance across patient-level predictions."""
    score = 0.0
    comparable = 0
    for index, left in enumerate(rows):
        for right in rows[:index]:
            if left["observed_days"] == right["observed_days"]:
                continue
            if left["observed_days"] < right["observed_days"]:
                earlier, later = left, right
            else:
                earlier, later = right, left
            if not earlier["observed_event"]:
                continue
            comparable += 1
            if earlier["predicted_risk"] > later["predicted_risk"]:
                score += 1.0
            elif earlier["predicted_risk"] == later["predicted_risk"]:
                score += 0.5
    return score / comparable if comparable else math.nan, comparable


def bootstrap_c_index(
    rows: Sequence[Dict[str, Any]], n_boot: int, seed: int
) -> Tuple[float, float, int]:
    """Return a patient-level percentile interval for Harrell's c-index."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        value, comparable = harrell_c_index(sample)
        if comparable and not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, math.nan, 0
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def generate_diagnosis_rollouts(
    predictor: TimeToEventModel,
    prefix: Sequence[str],
    condition_tokens: Set[str],
    horizon_days: int,
    target_rollouts: int,
    max_n_trial: int,
) -> Tuple[List[float], List[bool], int]:
    """Generate valid rollouts, retrying trajectories cut off by token limits."""
    rollout_times = []
    rollout_events = []
    incomplete = 0
    original_num_return_sequences = predictor.generation_config.num_return_sequences
    try:
        for _ in range(max_n_trial):
            remaining = target_rollouts - len(rollout_times)
            if remaining <= 0:
                break
            predictor.generation_config.num_return_sequences = remaining
            for sequence in predictor.simulate(prefix):
                generated = sequence[len(prefix) :]
                elapsed_days = 0.0
                completed = False
                for token in generated:
                    if is_att_token(token):
                        elapsed_days += extract_time_interval_in_days(token)
                        if elapsed_days > horizon_days:
                            rollout_times.append(float(horizon_days))
                            rollout_events.append(False)
                            completed = True
                            break
                    elif token in condition_tokens:
                        rollout_times.append(float(min(elapsed_days, horizon_days)))
                        rollout_events.append(elapsed_days <= horizon_days)
                        completed = True
                        break
                    elif token == predictor.tokenizer.end_token:
                        rollout_times.append(float(horizon_days))
                        rollout_events.append(False)
                        completed = True
                        break
                if not completed:
                    incomplete += 1
    finally:
        predictor.generation_config.num_return_sequences = (
            original_num_return_sequences
        )
    return rollout_times, rollout_events, incomplete


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {requested}"
        )
    return device


def main(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.monotonic()
    torch.manual_seed(args.seed)
    if args.patients <= 1:
        raise ValueError("--patients must be greater than 1")
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")

    selected = select_cutoffs(
        args.sequences, args.patients, args.horizon_days, args.seed
    )
    person_ids = [record["person_id"] for record in selected]
    tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer or args.model)
    condition_tokens = load_condition_tokens(args.concept, tokenizer)
    if not condition_tokens:
        raise RuntimeError("No Condition-domain concepts occur in the tokenizer")
    observed_conditions = load_observed_conditions(
        args.condition_occurrence, person_ids
    )

    device = resolve_device(args.device)
    model = CEHRGPT2LMHeadModel.from_pretrained(args.model).eval().to(device)
    max_prompt_length = model.config.max_position_embeddings - args.max_new_tokens
    if max_prompt_length <= 0:
        raise ValueError(
            "--max-new-tokens must be smaller than max_position_embeddings "
            f"({model.config.max_position_embeddings})"
        )
    generation_config = TimeToEventModel.get_generation_config(
        tokenizer=tokenizer,
        max_length=model.config.max_position_embeddings,
        num_return_sequences=args.rollouts,
        max_new_tokens=args.max_new_tokens,
    )
    predictor = TimeToEventModel(
        tokenizer=tokenizer,
        model=model,
        outcome_events=list(condition_tokens),
        generation_config=generation_config,
        device=device,
        batch_size=args.rollout_batch_size or args.rollouts,
    )
    predictor.outcome_events = condition_tokens

    results = []
    vocab = tokenizer.get_vocab()
    for patient_index, record in enumerate(selected):
        cutoff = record["cutoff_timestamp"]
        horizon_end = cutoff + args.horizon_days * SECONDS_PER_DAY
        future = [
            event
            for event in observed_conditions[record["person_id"]]
            if cutoff < event[0] <= horizon_end
        ]
        if future:
            observed_days = (future[0][0] - cutoff) / SECONDS_PER_DAY
            observed_event = True
            observed_concept_id = future[0][1]
        else:
            observed_days = float(args.horizon_days)
            observed_event = False
            observed_concept_id = None

        prefix = [token for token in record["prefix"] if token in vocab]
        prefix = truncate_prefix(prefix, max_prompt_length)
        torch.manual_seed(args.seed + patient_index)
        rollout_times, rollout_events, incomplete_attempts = (
            generate_diagnosis_rollouts(
                predictor=predictor,
                prefix=prefix,
                condition_tokens=condition_tokens,
                horizon_days=args.horizon_days,
                target_rollouts=args.rollouts,
                max_n_trial=args.max_n_trial,
            )
        )
        valid_rollouts = len(rollout_times)
        missing = args.rollouts - valid_rollouts
        if missing:
            # Conservative fallback: an incomplete generated trajectory contributes
            # no predicted diagnosis before the administrative horizon.
            rollout_times.extend([float(args.horizon_days)] * missing)
            rollout_events.extend([False] * missing)

        restricted_mean = float(np.mean(rollout_times))
        results.append(
            {
                "person_id": record["person_id"],
                "cutoff_timestamp": cutoff,
                "observed_days": observed_days,
                "observed_event": observed_event,
                "observed_condition_concept_id": observed_concept_id,
                "predicted_event_probability": float(np.mean(rollout_events)),
                "predicted_restricted_mean_days": restricted_mean,
                "predicted_risk": -restricted_mean,
                "valid_rollouts": valid_rollouts,
                "imputed_incomplete_rollouts": missing,
                "discarded_incomplete_attempts": incomplete_attempts,
            }
        )
        print(
            f"patients={len(results)}/{args.patients} "
            f"observed_events={sum(item['observed_event'] for item in results)} "
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            flush=True,
        )

    c_index, comparable_pairs = harrell_c_index(results)
    ci_low, ci_high, bootstrap_resamples = (math.nan, math.nan, 0)
    if args.bootstrap_resamples > 0:
        ci_low, ci_high, bootstrap_resamples = bootstrap_c_index(
            results, args.bootstrap_resamples, args.seed
        )
    output = {
        "model": args.model,
        "device": str(device),
        "patients": len(results),
        "rollouts_per_patient": args.rollouts,
        "horizon_days": args.horizon_days,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "condition_token_count": len(condition_tokens),
        "observed_events": sum(item["observed_event"] for item in results),
        "comparable_pairs": comparable_pairs,
        "c_index": c_index,
        "c_index_bootstrap_95_ci": [ci_low, ci_high],
        "bootstrap_resamples": bootstrap_resamples,
        "valid_rollouts": sum(item["valid_rollouts"] for item in results),
        "imputed_incomplete_rollouts": sum(
            item["imputed_incomplete_rollouts"] for item in results
        ),
        "risk_score": "negative restricted mean generated time to diagnosis",
        "elapsed_seconds": time.monotonic() - started,
        "patients_detail": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    summary = {key: value for key, value in output.items() if key != "patients_detail"}
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
