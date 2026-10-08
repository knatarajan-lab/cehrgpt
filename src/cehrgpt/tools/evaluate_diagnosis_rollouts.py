"""Evaluate patient-level time to the next diagnosis using model rollouts.

For each randomly sampled patient, this tool selects one eligible visit-end cutoff,
generates future trajectories, and reduces the rollouts to one patient-level risk
score: the negative restricted mean time to the first generated Condition token.
The observed outcome is the first matching Condition-domain token in the patient
sequence after the cutoff, with administrative censoring at the requested horizon.
The outcome can be all diagnoses or one user-supplied OMOP diagnosis concept plus
its descendants. Harrell's c-index is computed across patients, along with a
patient-level bootstrap confidence interval.

Example:
    python -m cehrgpt.tools.evaluate_diagnosis_rollouts \
        --model /path/to/model \
        --sequences /path/to/patient_sequence/test \
        --concept /path/to/concept \
        --output /path/to/diagnosis_rollout_cindex.json \
        --gpu_ids all
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from lifelines.utils import concordance_index
from tqdm.auto import tqdm

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
        "--concept", required=True, help="OMOP concept parquet folder"
    )
    parser.add_argument(
        "--diagnosis_concept_id",
        "--diagnosis-concept-id",
        dest="diagnosis_concept_id",
        type=int,
        help=(
            "Optional ancestor Condition concept ID to evaluate instead of all "
            "diagnoses"
        ),
    )
    parser.add_argument(
        "--concept_ancestor",
        "--concept-ancestor",
        dest="concept_ancestor",
        help=(
            "OMOP concept_ancestor parquet folder; required with "
            "--diagnosis-concept-id"
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--rollout-batch-size", type=int)
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument(
        "--min_history_visits",
        "--min-history-visits",
        dest="min_history_visits",
        type=int,
        default=2,
        help="Minimum completed visits required before the prediction cutoff",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--top_p",
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability",
    )
    parser.add_argument(
        "--top_k",
        "--top-k",
        dest="top_k",
        type=int,
        default=300,
        help="Maximum number of tokens retained for sampling",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        dest="temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
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
    parser.add_argument(
        "--gpu_ids",
        "--gpu-ids",
        dest="gpu_ids",
        help="Run one worker per GPU: 'all' or comma-separated visible GPU indices",
    )
    parser.add_argument(
        "--partition_input",
        "--partition-input",
        dest="partition_input",
        help="Internal: JSON patient partition prepared by --gpu-ids",
    )
    parser.add_argument(
        "--trajectory_output",
        "--trajectory-output",
        dest="trajectory_output",
        help="Parquet output folder; defaults to <output>.trajectories",
    )
    parser.add_argument(
        "--trajectory_buffer_size",
        "--trajectory-buffer-size",
        dest="trajectory_buffer_size",
        type=int,
        default=100,
        help="Patients per trajectory Parquet part",
    )
    return parser


def parquet_files(path: str) -> List[str]:
    files = sorted(str(item) for item in Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


def find_observed_diagnosis(
    concepts: Sequence[str],
    epoch_times: Sequence[float],
    cutoff_index: int,
    cutoff_time: float,
    condition_tokens: Set[str],
    horizon_days: int,
    last_observed_time: float,
) -> Tuple[float, bool, str | None, float]:
    """Find the first Condition-domain sequence event after the cutoff."""
    horizon_end = min(
        cutoff_time + horizon_days * SECONDS_PER_DAY, last_observed_time
    )
    followup_days = max(0.0, (horizon_end - cutoff_time) / SECONDS_PER_DAY)
    future_conditions = [
        (epoch_time, token)
        for token, epoch_time in zip(
            concepts[cutoff_index + 1 :], epoch_times[cutoff_index + 1 :]
        )
        if token in condition_tokens and cutoff_time < epoch_time <= horizon_end
    ]
    if not future_conditions:
        return followup_days, False, None, followup_days
    event_time, concept_id = min(future_conditions)
    return (
        (event_time - cutoff_time) / SECONDS_PER_DAY,
        True,
        concept_id,
        followup_days,
    )


def eligible_visit_cutoffs(
    concepts: Sequence[str],
    epoch_times: Sequence[float],
    min_history_visits: int,
) -> List[Tuple[int, float, int]]:
    """Return visit-end cutoffs with enough history and positive follow-up."""
    if not epoch_times:
        return []
    last_observed_time = max(epoch_times)
    prefix_max_time = -math.inf
    completed_visits = 0
    eligible = []
    for index, (token, epoch_time) in enumerate(zip(concepts, epoch_times)):
        prefix_max_time = max(prefix_max_time, epoch_time)
        if is_visit_end(token):
            completed_visits += 1
            if (
                completed_visits >= min_history_visits
                and prefix_max_time < last_observed_time
            ):
                eligible.append((index, prefix_max_time, completed_visits))
    return eligible


def select_cutoffs(
    sequence_path: str,
    sample_size: int,
    horizon_days: int,
    seed: int,
    condition_tokens: Set[str],
    min_history_visits: int,
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

            eligible = eligible_visit_cutoffs(
                concepts,
                epoch_times,
                min_history_visits,
            )

            if not eligible:
                continue
            cutoff_index, cutoff_time, history_visit_count = eligible[
                int(rng.integers(len(eligible)))
            ]
            (
                observed_days,
                observed_event,
                observed_concept_id,
                followup_days,
            ) = find_observed_diagnosis(
                concepts,
                epoch_times,
                cutoff_index,
                cutoff_time,
                condition_tokens,
                horizon_days,
                max(epoch_times),
            )
            selected.append(
                {
                    "sample_index": len(selected),
                    "person_id": int(record["person_id"]),
                    "cutoff_index": int(cutoff_index),
                    "cutoff_timestamp": float(cutoff_time),
                    "history_visit_count": history_visit_count,
                    "followup_days": followup_days,
                    "prefix": concepts[: cutoff_index + 1],
                    "observed_days": observed_days,
                    "observed_event": observed_event,
                    "observed_condition_concept_id": observed_concept_id,
                }
            )
            if len(selected) >= sample_size:
                break

    if len(selected) < sample_size:
        raise RuntimeError(
            f"Only found {len(selected)} patients with at least "
            f"{min_history_visits} completed visits and positive follow-up"
        )
    return selected


def load_condition_tokens(
    concept_path: str,
    tokenizer: CehrGptTokenizer,
    diagnosis_concept_id: int | None = None,
    concept_ancestor_path: str | None = None,
) -> Set[str]:
    """Return tokenizer Condition IDs, optionally restricted to an ancestor."""
    vocab = tokenizer.get_vocab()
    allowed_concept_ids = None
    if diagnosis_concept_id is not None:
        diagnosis_token = str(diagnosis_concept_id)
        if not concept_ancestor_path:
            raise ValueError(
                "--concept-ancestor is required with --diagnosis-concept-id"
            )
        ancestor_table = ds.dataset(
            parquet_files(concept_ancestor_path), format="parquet"
        ).to_table(
            columns=["descendant_concept_id"],
            filter=pc.field("ancestor_concept_id") == diagnosis_concept_id,
        )
        allowed_concept_ids = {
            str(concept_id)
            for concept_id in ancestor_table["descendant_concept_id"].to_pylist()
        }
        # Do not depend on concept_ancestor containing the zero-level self row.
        allowed_concept_ids.add(diagnosis_token)

    table = ds.dataset(parquet_files(concept_path), format="parquet").to_table(
        columns=["concept_id", "domain_id"],
        filter=pc.field("domain_id") == "Condition",
    )
    condition_tokens = {
        str(concept_id)
        for concept_id in table["concept_id"].to_pylist()
        if str(concept_id) in vocab
        and (
            allowed_concept_ids is None
            or str(concept_id) in allowed_concept_ids
        )
    }
    if diagnosis_concept_id is not None and not condition_tokens:
        raise ValueError(
            f"Diagnosis concept {diagnosis_concept_id} and its Condition-domain "
            "descendants are not available in the tokenizer"
        )
    return condition_tokens


def load_concept_name(concept_path: str, concept_id: int) -> str:
    """Load one OMOP concept label using a predicate-pushed parquet scan."""
    table = ds.dataset(parquet_files(concept_path), format="parquet").to_table(
        columns=["concept_name"],
        filter=pc.field("concept_id") == concept_id,
    )
    if table.num_rows == 0:
        raise ValueError(f"Diagnosis concept {concept_id} is absent from --concept")
    return str(table["concept_name"][0].as_py())


def require_condition_tokens(
    condition_tokens: Set[str], diagnosis_concept_id: int | None
) -> None:
    """Raise an informative error when no requested outcome is tokenized."""
    if condition_tokens:
        return
    target = (
        f"diagnosis concept {diagnosis_concept_id} or its descendants"
        if diagnosis_concept_id is not None
        else "Condition-domain concepts"
    )
    raise RuntimeError(f"No {target} occur in the tokenizer")


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
    """Compute censored patient-level concordance with lifelines."""
    comparable = 0
    for index, left in enumerate(rows):
        for right in rows[:index]:
            if left["observed_days"] == right["observed_days"]:
                # At the same time, lifelines considers an observed event
                # comparable with a censored observation, but not two events or
                # two censored observations.
                comparable += int(
                    left["observed_event"] != right["observed_event"]
                )
                continue
            if left["observed_days"] < right["observed_days"]:
                earlier, later = left, right
            else:
                earlier, later = right, left
            if not earlier["observed_event"]:
                continue
            comparable += 1
    if not comparable:
        return math.nan, 0

    event_times = np.asarray([row["observed_days"] for row in rows], dtype=float)
    # lifelines expects a larger predicted score to mean longer survival, whereas
    # predicted_risk is larger for an earlier diagnosis.
    predicted_survival = -np.asarray(
        [row["predicted_risk"] for row in rows], dtype=float
    )
    event_observed = np.asarray(
        [row["observed_event"] for row in rows], dtype=bool
    )
    value = concordance_index(
        event_times=event_times,
        predicted_scores=predicted_survival,
        event_observed=event_observed,
    )
    return float(value), comparable


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
) -> Tuple[List[float], List[bool], int, List[Dict[str, Any]]]:
    """Generate valid rollouts, retrying trajectories cut off by token limits."""
    rollout_times = []
    rollout_events = []
    incomplete = 0
    trajectories = []
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
                completion_reason = "max_new_tokens"
                outcome_event = None
                time_to_event_days = None
                for token in generated:
                    if is_att_token(token):
                        elapsed_days += extract_time_interval_in_days(token)
                        if elapsed_days > horizon_days:
                            rollout_times.append(float(horizon_days))
                            rollout_events.append(False)
                            completed = True
                            completion_reason = "horizon"
                            time_to_event_days = float(horizon_days)
                            break
                    elif token in condition_tokens:
                        time_to_event_days = float(
                            min(elapsed_days, horizon_days)
                        )
                        rollout_times.append(time_to_event_days)
                        rollout_events.append(elapsed_days <= horizon_days)
                        completed = True
                        completion_reason = "condition"
                        outcome_event = token
                        break
                    elif token == predictor.tokenizer.end_token:
                        rollout_times.append(float(horizon_days))
                        rollout_events.append(False)
                        completed = True
                        completion_reason = "end_token"
                        time_to_event_days = float(horizon_days)
                        break
                trajectories.append(
                    {
                        "tokens": generated,
                        "completed": completed,
                        "completion_reason": completion_reason,
                        "outcome_event": outcome_event,
                        "time_to_event_days": time_to_event_days,
                        "generated_elapsed_days": float(elapsed_days),
                    }
                )
                if not completed:
                    incomplete += 1
    finally:
        predictor.generation_config.num_return_sequences = (
            original_num_return_sequences
        )
    return rollout_times, rollout_events, incomplete, trajectories


def resolve_gpu_ids(gpu_ids: str) -> List[str]:
    """Resolve GPU indices relative to the currently visible CUDA devices."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = (
        [item.strip() for item in visible.split(",") if item.strip()]
        if visible
        else None
    )
    num_visible = (
        len(visible_ids) if visible_ids is not None else torch.cuda.device_count()
    )
    if num_visible == 0:
        raise RuntimeError("--gpu-ids was provided but no CUDA device is visible")
    if gpu_ids.strip().lower() == "all":
        indices = list(range(num_visible))
    else:
        indices = [int(item) for item in gpu_ids.split(",") if item.strip()]
    if not indices:
        raise ValueError("--gpu-ids must contain at least one GPU")
    if len(set(indices)) != len(indices):
        raise ValueError(f"--gpu-ids contains duplicates: {gpu_ids}")
    out_of_range = [index for index in indices if not 0 <= index < num_visible]
    if out_of_range:
        raise ValueError(
            f"--gpu-ids {out_of_range} out of range; {num_visible} GPU(s) are visible"
        )
    return [
        visible_ids[index] if visible_ids is not None else str(index)
        for index in indices
    ]


def strip_cli_option(argv: Sequence[str], option: str) -> List[str]:
    """Remove a CLI option and its value in both supported spellings."""
    stripped = []
    skip_next = False
    for argument in argv:
        if skip_next:
            skip_next = False
        elif argument == option:
            skip_next = True
        elif not argument.startswith(option + "="):
            stripped.append(argument)
    return stripped


def write_json(path: Path, value: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    """Atomically write JSON so interrupted workers leave no valid-looking output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


TRAJECTORY_SCHEMA = pa.schema(
    [
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("person_id", pa.int64(), nullable=False),
        pa.field("cutoff_timestamp", pa.float64(), nullable=False),
        pa.field("history_visit_count", pa.int64(), nullable=False),
        pa.field("followup_days", pa.float64(), nullable=False),
        pa.field("prompt_tokens", pa.list_(pa.string()), nullable=False),
        pa.field(
            "generated_trajectories",
            pa.list_(
                pa.struct(
                    [
                        pa.field("tokens", pa.list_(pa.string()), nullable=False),
                        pa.field("completed", pa.bool_(), nullable=False),
                        pa.field("completion_reason", pa.string(), nullable=False),
                        pa.field("outcome_event", pa.string()),
                        pa.field("time_to_event_days", pa.float64()),
                        pa.field(
                            "generated_elapsed_days", pa.float64(), nullable=False
                        ),
                    ]
                )
            ),
            nullable=False,
        ),
    ]
)


class TrajectoryParquetWriter:
    """Incrementally write one nested trajectory record per patient."""

    def __init__(self, output_folder: Path, buffer_size: int):
        if buffer_size <= 0:
            raise ValueError("--trajectory-buffer-size must be positive")
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, Any]] = []
        self.part_index = 0

    def add(self, record: Dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=TRAJECTORY_SCHEMA)
        output_path = self.output_folder / f"part_{self.part_index:05d}.parquet"
        temporary = output_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(output_path)
        self.buffer.clear()
        self.part_index += 1


def trajectory_output_path(args: argparse.Namespace) -> Path:
    return Path(args.trajectory_output or (str(args.output) + ".trajectories"))


def build_output(
    args: argparse.Namespace,
    results: Sequence[Dict[str, Any]],
    condition_token_count: int,
    device: str,
    elapsed_seconds: float,
    trajectory_output: str,
) -> Dict[str, Any]:
    """Aggregate patient predictions into the final evaluation artifact."""
    c_index, comparable_pairs = harrell_c_index(results)
    ci_low, ci_high, bootstrap_resamples = (None, None, 0)
    if args.bootstrap_resamples > 0:
        ci_low, ci_high, bootstrap_resamples = bootstrap_c_index(
            results, args.bootstrap_resamples, args.seed
        )
    return {
        "model": args.model,
        "device": device,
        "patients": len(results),
        "rollouts_per_patient": args.rollouts,
        "horizon_days": args.horizon_days,
        "min_history_visits": args.min_history_visits,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "seed": args.seed,
        "diagnosis_concept_id": args.diagnosis_concept_id,
        "diagnosis_concept_name": getattr(
            args, "diagnosis_concept_name", None
        ),
        "condition_token_count": condition_token_count,
        "observed_events": sum(item["observed_event"] for item in results),
        "comparable_pairs": comparable_pairs,
        "c_index": None if math.isnan(c_index) else c_index,
        "c_index_bootstrap_95_ci": [ci_low, ci_high],
        "bootstrap_resamples": bootstrap_resamples,
        "valid_rollouts": sum(item["valid_rollouts"] for item in results),
        "imputed_incomplete_rollouts": sum(
            item["imputed_incomplete_rollouts"] for item in results
        ),
        "trajectory_output": trajectory_output,
        "risk_score": "negative restricted mean generated time to diagnosis",
        "elapsed_seconds": elapsed_seconds,
        "patients_detail": list(results),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {requested}"
        )
    return device


def run_on_multiple_gpus(
    args: argparse.Namespace, gpu_ids: Sequence[str], started: float
) -> Dict[str, Any]:
    """Partition selected patients and run one evaluator worker per GPU."""
    tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer or args.model)
    condition_tokens = load_condition_tokens(
        args.concept,
        tokenizer,
        args.diagnosis_concept_id,
        args.concept_ancestor,
    )
    require_condition_tokens(condition_tokens, args.diagnosis_concept_id)
    selected = select_cutoffs(
        args.sequences,
        args.patients,
        args.horizon_days,
        args.seed,
        condition_tokens,
        args.min_history_visits,
    )
    num_workers = min(len(gpu_ids), len(selected))
    partition_root = Path(str(args.output) + ".partitions")
    log_root = Path(str(args.output) + ".worker_logs")
    trajectory_root = trajectory_output_path(args)
    shutil.rmtree(partition_root, ignore_errors=True)
    shutil.rmtree(trajectory_root, ignore_errors=True)
    partition_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    trajectory_root.mkdir(parents=True)

    worker_argv = list(sys.argv[1:])
    for option in (
        "--gpu_ids",
        "--gpu-ids",
        "--device",
        "--partition_input",
        "--partition-input",
        "--output",
        "--patients",
        "--bootstrap-resamples",
        "--trajectory_output",
        "--trajectory-output",
    ):
        worker_argv = strip_cli_option(worker_argv, option)

    processes = []
    worker_outputs = []
    log_paths = []
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for worker_index in range(num_workers):
            partition = selected[worker_index::num_workers]
            partition_input = partition_root / f"partition_{worker_index}.json"
            worker_output = partition_root / f"result_{worker_index}.json"
            worker_trajectory_output = trajectory_root / f"worker_{worker_index}"
            log_name = f"worker_{worker_index}_gpu_{gpu_ids[worker_index]}.log"
            log_path = log_root / log_name
            write_json(
                partition_input,
                {
                    "condition_tokens": sorted(condition_tokens),
                    "patients": partition,
                },
            )
            worker_outputs.append(worker_output)
            log_paths.append(log_path)
            with log_path.open("w") as log_file:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "cehrgpt.tools.evaluate_diagnosis_rollouts",
                            *worker_argv,
                            "--partition-input",
                            str(partition_input),
                            "--output",
                            str(worker_output),
                            "--patients",
                            str(len(partition)),
                            "--bootstrap-resamples",
                            "0",
                            "--device",
                            "cuda:0",
                            "--trajectory-output",
                            str(worker_trajectory_output),
                        ],
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": gpu_ids[worker_index],
                        },
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                )
            print(
                f"Started worker {worker_index} on GPU {gpu_ids[worker_index]}: "
                f"{log_path}",
                flush=True,
            )
        exit_codes = []
        with tqdm(
            total=len(processes),
            desc="GPU workers",
            unit="worker",
            dynamic_ncols=True,
        ) as progress:
            for process in processes:
                exit_codes.append(process.wait())
                progress.update()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    failed = [index for index, code in enumerate(exit_codes) if code != 0]
    if failed:
        failures = ", ".join(
            f"worker {index} ({log_paths[index]})" for index in failed
        )
        raise RuntimeError(f"Diagnosis rollout workers failed: {failures}")

    results = []
    for worker_output in worker_outputs:
        worker_result = json.loads(worker_output.read_text())
        results.extend(worker_result["patients_detail"])
    results.sort(key=lambda item: item["sample_index"])
    output = build_output(
        args,
        results,
        len(condition_tokens),
        f"cuda:{','.join(gpu_ids[:num_workers])}",
        time.monotonic() - started,
        str(trajectory_root),
    )
    write_json(Path(args.output), output)
    shutil.rmtree(partition_root)
    return output


def main(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.monotonic()
    if args.patients <= 1 and not args.partition_input:
        raise ValueError("--patients must be greater than 1")
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.min_history_visits <= 0:
        raise ValueError("--min_history_visits must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top_k must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    args.diagnosis_concept_name = None
    if args.diagnosis_concept_id is not None and not args.partition_input:
        args.diagnosis_concept_name = load_concept_name(
            args.concept, args.diagnosis_concept_id
        )
        print(
            f"Diagnosis concept: {args.diagnosis_concept_name} "
            f"({args.diagnosis_concept_id})",
            flush=True,
        )

    if args.gpu_ids and not args.partition_input:
        gpu_ids = resolve_gpu_ids(args.gpu_ids)
        if len(gpu_ids) > 1:
            output = run_on_multiple_gpus(args, gpu_ids, started)
            summary = {
                key: value
                for key, value in output.items()
                if key != "patients_detail"
            }
            print(json.dumps(summary, indent=2))
            return output
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        args.device = "cuda:0"

    torch.manual_seed(args.seed)
    tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer or args.model)
    if args.partition_input:
        partition = json.loads(Path(args.partition_input).read_text())
        condition_tokens = set(partition["condition_tokens"])
        selected = partition["patients"]
    else:
        condition_tokens = load_condition_tokens(
            args.concept,
            tokenizer,
            args.diagnosis_concept_id,
            args.concept_ancestor,
        )
        selected = select_cutoffs(
            args.sequences,
            args.patients,
            args.horizon_days,
            args.seed,
            condition_tokens,
            args.min_history_visits,
        )
    require_condition_tokens(condition_tokens, args.diagnosis_concept_id)

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
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
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

    trajectory_folder = trajectory_output_path(args)
    shutil.rmtree(trajectory_folder, ignore_errors=True)
    trajectory_writer = TrajectoryParquetWriter(
        trajectory_folder, args.trajectory_buffer_size
    )
    results = []
    vocab = tokenizer.get_vocab()
    progress = tqdm(
        selected,
        total=len(selected),
        desc="Evaluating patients",
        unit="patient",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for record in progress:
        cutoff = record["cutoff_timestamp"]
        prefix = [token for token in record["prefix"] if token in vocab]
        prefix = truncate_prefix(prefix, max_prompt_length)
        torch.manual_seed(args.seed + record["sample_index"])
        (
            rollout_times,
            rollout_events,
            incomplete_attempts,
            generated_trajectories,
        ) = generate_diagnosis_rollouts(
            predictor=predictor,
            prefix=prefix,
            condition_tokens=condition_tokens,
            horizon_days=args.horizon_days,
            target_rollouts=args.rollouts,
            max_n_trial=args.max_n_trial,
        )
        valid_rollouts = len(rollout_times)
        missing = args.rollouts - valid_rollouts
        if missing:
            # Conservative fallback: an incomplete generated trajectory contributes
            # no predicted diagnosis before the administrative horizon.
            rollout_times.extend([float(args.horizon_days)] * missing)
            rollout_events.extend([False] * missing)

        restricted_mean = float(np.mean(rollout_times))
        trajectory_writer.add(
            {
                "sample_index": record["sample_index"],
                "person_id": record["person_id"],
                "cutoff_timestamp": cutoff,
                "history_visit_count": record["history_visit_count"],
                "followup_days": record["followup_days"],
                "prompt_tokens": prefix,
                "generated_trajectories": generated_trajectories,
            }
        )
        results.append(
            {
                "sample_index": record["sample_index"],
                "person_id": record["person_id"],
                "cutoff_timestamp": cutoff,
                "history_visit_count": record["history_visit_count"],
                "followup_days": record["followup_days"],
                "observed_days": record["observed_days"],
                "observed_event": record["observed_event"],
                "observed_condition_concept_id": record[
                    "observed_condition_concept_id"
                ],
                "predicted_event_probability": float(np.mean(rollout_events)),
                "predicted_restricted_mean_days": restricted_mean,
                "predicted_risk": -restricted_mean,
                "valid_rollouts": valid_rollouts,
                "imputed_incomplete_rollouts": missing,
                "discarded_incomplete_attempts": incomplete_attempts,
            }
        )
        progress.set_postfix(
            observed_events=sum(item["observed_event"] for item in results),
            valid_rollouts=sum(item["valid_rollouts"] for item in results),
        )

    trajectory_writer.flush()
    output = build_output(
        args,
        results,
        len(condition_tokens),
        str(device),
        time.monotonic() - started,
        str(trajectory_folder),
    )
    write_json(Path(args.output), output)
    summary = {key: value for key, value in output.items() if key != "patients_detail"}
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
