#!/usr/bin/env python3
"""
Steps 2 & 3: Generate counterfactual (non-treated) and treated trajectories.

Loads the two context parquets produced by extract_drug_initiation_sequences.py
and runs the CEHR-GPT model to extend each patient's history forward in time.

  non_treated_context.parquet  →  trajectories/{non_treated,treated}/<batch_i>.parquet

Each output parquet has one row per generated clinical event with columns:
  subject_id, arm (non_treated | treated), trajectory_id, prediction_time,
  window_last_observed_time, time, code, numeric_value, text_value, unit

Usage
-----
python generate_counterfactual_sequences.py \\
    --context_dir        /path/to/context_dir \\
    --model_name_or_path /path/to/cehrgpt_model \\
    --tokenizer_path     /path/to/tokenizer \\
    --output_dir         /path/to/trajectories \\
    --num_trajectories   10 \\
    --batch_size         8 \\
    --generation_input_length  1024 \\
    --generation_max_new_tokens 1024

Several GPUs
------------
Add ``--gpu_ids all`` (or e.g. ``--gpu_ids 0,1,3``) to run one worker process per GPU.
The work is cut into tasks of one batch of one trajectory of one arm.  The GPUs are divided into one
set per arm, in proportion to the batches each arm has left, so the arms are generated at the same
time; with fewer GPUs than arms all GPUs work on the arms one after the other.  Each worker loads
its own copy of the model and writes its batches into the same ``<arm>/<trajectory_id>`` folders.  A batch is finished once its
parquet exists, so rerunning the same command (also with a different number of GPUs)
continues where an interrupted run stopped.  Worker logs are in ``<output_dir>/worker_logs``.
"""

import argparse
import datetime
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import is_flash_attn_2_available

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.generation.generate_batch_hf_gpt_sequence import (
    generate_single_batch,
    normalize_value,
)
from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    extract_time_interval_in_hours,
    is_att_token,
    is_inpatient_hour_token,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.time_to_event.time_to_event_prediction import (
    resolve_gpu_ids,
    strip_cli_option,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NON_TREATED_ARM = "non_treated"
TREATED_ARM = "treated"


# ---------------------------------------------------------------------------
# Concept suppression during generation
# ---------------------------------------------------------------------------

def build_forbidden_token_ids(
    vocab_path: str,
    concept_ids: List[int],
    tokenizer: CehrGptTokenizer,
) -> List[int]:
    """
    Expand *concept_ids* to all OMOP descendants via concept_ancestor, then
    map each concept string to its tokenizer token ID.

    Only concepts present in the tokenizer vocabulary are included.
    Returns a deduplicated list of integer token IDs to suppress.
    """
    ancestor_glob = os.path.join(vocab_path, "concept_ancestor", "*.parquet")
    descendant_strs: List[str] = (
        pl.scan_parquet(ancestor_glob)
        .filter(pl.col("ancestor_concept_id").is_in(concept_ids))
        .select(pl.col("descendant_concept_id").cast(pl.String))
        .collect()["descendant_concept_id"]
        .to_list()
    )
    all_concepts: Set[str] = set(descendant_strs) | {str(c) for c in concept_ids}

    unk_id = tokenizer.unk_token_id
    forbidden: Set[int] = set()
    for concept_str in all_concepts:
        encoded = tokenizer.encode([concept_str])
        if encoded and encoded[0] != unk_id:
            forbidden.add(encoded[0])

    print(
        f"  Suppressing {len(forbidden):,} token IDs "
        f"({len(all_concepts):,} concept descendants of {len(concept_ids)} ingredient(s))"
    )
    return list(forbidden)


# Parallel array columns that must all be truncated with the same indices
_SEQUENCE_ARRAY_COLS = [
    "concept_ids", "input_ids", "value_indicators", "values",
    "visit_segments", "orders", "dates", "ages", "visit_concept_orders",
    "concept_value_masks", "concept_values", "mlm_skip_values",
    "priorities", "visit_concept_ids", "visit_rank_orders",
    "concept_orders", "record_ranks", "units", "event_group_ids",
    "epoch_times",
]


# ---------------------------------------------------------------------------
# Left-truncation with demographics preservation
# ---------------------------------------------------------------------------

def _update_year_age_tokens(
    concept_ids: List[str],
    input_ids: Optional[List[int]],
    first_kept_clinical_epoch: float,
    tokenizer: Optional["CehrGptTokenizer"],
) -> Tuple[List[str], Optional[List[int]]]:
    """
    Update the year and age tokens in the demographics header to reflect
    the patient's age/year at *first_kept_clinical_epoch*.

    Demographics layout (indices 0-3 by convention):
        concept_ids[0] = "year:{start_year}"
        concept_ids[1] = "age:{start_age}"
        concept_ids[2] = gender token
        concept_ids[3] = race token

    The birth year is inferred as ``start_year - start_age``.  From that
    we compute the new age at *first_kept_clinical_epoch* and overwrite
    both the concept_ids and (if tokenizer is provided) the input_ids.
    """
    if len(concept_ids) < 2:
        return concept_ids, input_ids

    year_tok = concept_ids[0]
    age_tok  = concept_ids[1]

    # Parse existing year and age tokens
    try:
        orig_year = int(year_tok.split(":")[1])
        orig_age  = int(age_tok.split(":")[1])
    except (IndexError, ValueError):
        return concept_ids, input_ids  # unexpected format — leave unchanged

    birth_year = orig_year - orig_age
    new_year   = datetime.datetime.utcfromtimestamp(first_kept_clinical_epoch).year
    new_age    = max(0, new_year - birth_year)

    new_year_tok = f"year:{new_year}"
    new_age_tok  = f"age:{new_age}"

    new_concept_ids = list(concept_ids)
    new_concept_ids[0] = new_year_tok
    new_concept_ids[1] = new_age_tok

    new_input_ids = list(input_ids) if input_ids is not None else None
    if new_input_ids is not None and tokenizer is not None:
        try:
            new_input_ids[0] = tokenizer.encode([new_year_tok])[0]
            new_input_ids[1] = tokenizer.encode([new_age_tok])[0]
        except Exception:
            pass  # leave unchanged if tokenizer doesn't know the token

    return new_concept_ids, new_input_ids


def _left_truncate_with_demographics(
    df: "pl.DataFrame",
    max_length: int,
    tokenizer: Optional["CehrGptTokenizer"] = None,
) -> "pl.DataFrame":
    """
    Truncate each patient's sequence to *max_length* tokens using a
    left-truncation strategy that preserves the demographics header and
    updates the year/age tokens to match the new temporal reference point.

    Strategy
    --------
    1. Find the demographics section = all tokens before the first ``[VS]``.
    2. Left-truncate the clinical tokens so that
       ``n_demo + n_clinical_kept == max_length``.
    3. Re-assemble: [demographics | most-recent clinical tokens].
    4. Update ``year:`` and ``age:`` tokens in the demographics header to
       reflect the patient's age/year at the start of the kept clinical
       tokens (derived from ``epoch_times``).

    All parallel array columns are truncated with the same index set.
    """
    array_cols = [c for c in _SEQUENCE_ARRAY_COLS if c in df.columns]

    rows_out = []
    for row in df.iter_rows(named=True):
        concept_ids = list(row.get("concept_ids") or [])
        n = len(concept_ids)

        if n <= max_length:
            rows_out.append(row)
            continue

        # Find end of demographics: index of the first [VS] token
        demo_end = 0
        for i, tok in enumerate(concept_ids):
            if tok == "[VS]":
                demo_end = i
                break

        n_clinical_keep = max_length - demo_end
        if n_clinical_keep <= 0:
            # Demographics alone exceed budget — fall back to right-truncation
            keep_indices = list(range(max_length))
        else:
            demo_indices     = list(range(demo_end))
            clinical_indices = list(range(demo_end, n))[-n_clinical_keep:]
            keep_indices     = demo_indices + clinical_indices

        new_row = dict(row)
        for col in array_cols:
            val = row.get(col)
            if val is not None and hasattr(val, "__len__") and len(val) == n:
                new_row[col] = [val[i] for i in keep_indices]

        # Update year/age tokens to reflect the new temporal reference point
        epoch_times = list(row.get("epoch_times") or [])
        first_clinical_orig_idx = keep_indices[demo_end] if len(keep_indices) > demo_end else None
        if first_clinical_orig_idx is not None and first_clinical_orig_idx < len(epoch_times):
            new_concept_ids, new_input_ids = _update_year_age_tokens(
                concept_ids=new_row["concept_ids"],
                input_ids=new_row.get("input_ids"),
                first_kept_clinical_epoch=float(epoch_times[first_clinical_orig_idx]),
                tokenizer=tokenizer,
            )
            new_row["concept_ids"] = new_concept_ids
            if new_input_ids is not None:
                new_row["input_ids"] = new_input_ids

        rows_out.append(new_row)

    return pl.DataFrame(rows_out, schema=df.schema)


# ---------------------------------------------------------------------------
# Data collator wrapper
# ---------------------------------------------------------------------------

class CounterfactualDataCollator:
    """
    Wraps CehrGptDataCollator to additionally include ``person_id``,
    ``index_date`` (= drug_epoch_time), and ``drug_concept_id`` in every
    batch so that downstream functions can track patient identity and
    exposure time.
    """

    def __init__(self, base_collator: CehrGptDataCollator):
        self.base_collator = base_collator

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        person_ids = [int(e["person_id"]) for e in examples]
        drug_epoch_times = [float(e["drug_epoch_time"]) for e in examples]
        drug_concept_ids = [str(e.get("drug_concept_id", "")) for e in examples]

        batch = self.base_collator(examples)
        batch["person_id"] = torch.tensor(person_ids, dtype=torch.int64)
        batch["index_date"] = torch.tensor(drug_epoch_times, dtype=torch.float64)
        batch["drug_concept_id"] = drug_concept_ids
        return batch


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectories_for_batch(
    batch: Dict[str, Any],
    cehrgpt_tokenizer: CehrGptTokenizer,
    cehrgpt_model: CEHRGPT2LMHeadModel,
    device: torch.device,
    arm: str,
    trajectory_id: int,
    max_length: int,
    forbidden_token_ids: Optional[List[int]] = None,
) -> pl.DataFrame:
    """
    Generate one set of trajectories for a single data batch.

    Returns a polars DataFrame with one row per generated clinical event.
    """
    subject_ids = batch["person_id"].squeeze().detach().cpu().tolist()
    # drug_epoch_time serves as the "prediction time" anchor
    prediction_times = batch["index_date"].squeeze().detach().cpu().tolist()
    batched_epoch_times = batch["epoch_times"].detach().cpu().tolist()
    batched_input_ids = batch["input_ids"]
    batched_ages = batch["ages"]
    batched_value_indicators = batch.get("value_indicators", None)
    batched_values = batch.get("values", None)

    batch_sequences = generate_single_batch(
        cehrgpt_model,
        cehrgpt_tokenizer,
        batched_input_ids,
        ages=batched_ages,
        values=batched_values,
        value_indicators=batched_value_indicators,
        max_length=max_length,
        top_p=1.0,
        top_k=cehrgpt_tokenizer.vocab_size,
        device=device,
        suppress_token_ids=forbidden_token_ids,
    )
    torch.cuda.empty_cache()

    trajectories: List[Dict[str, Any]] = []

    for sample_i, (concept_ids, value_indicators, values) in enumerate(
        zip(
            batch_sequences["sequences"],
            batch_sequences["value_indicators"],
            batch_sequences["values"],
        )
    ):
        (
            concept_ids,
            _is_numeric_types,
            number_as_values,
            concept_as_values,
            units,
        ) = normalize_value(concept_ids, values, cehrgpt_tokenizer)

        epoch_times = batched_epoch_times[sample_i]
        input_length = len(epoch_times)
        window_last_observed = epoch_times[input_length - 1]
        current_cursor = epoch_times[-1]

        generated_times: List[datetime.datetime] = []
        valid_indices: List[int] = []

        for i in range(input_length, len(concept_ids)):
            concept_id = concept_ids[i]
            if concept_id in (cehrgpt_tokenizer.pad_token, cehrgpt_tokenizer.end_token):
                continue
            if is_att_token(concept_id):
                current_cursor += extract_time_interval_in_days(concept_id) * 86_400
            elif is_inpatient_hour_token(concept_id):
                current_cursor += extract_time_interval_in_hours(concept_id) * 3_600
            elif is_visit_start(concept_id) or is_visit_end(concept_id):
                continue
            else:
                valid_indices.append(i)
                generated_times.append(
                    datetime.datetime.utcfromtimestamp(current_cursor).replace(tzinfo=None)
                )

        if not valid_indices:
            continue

        concept_ids_arr = np.asarray(concept_ids)
        trajectories.append(
            {
                "subject_id": int(subject_ids[sample_i]) if not isinstance(subject_ids, int) else subject_ids,
                "arm": arm,
                "trajectory_id": trajectory_id,
                "prediction_time": datetime.datetime.utcfromtimestamp(
                    prediction_times[sample_i]
                    if not isinstance(prediction_times, float)
                    else prediction_times
                ).replace(tzinfo=None),
                "window_last_observed_time": datetime.datetime.utcfromtimestamp(
                    window_last_observed
                ).replace(tzinfo=None),
                "time": generated_times,
                "code": concept_ids_arr[valid_indices].tolist(),
                "numeric_value": np.asarray(number_as_values)[valid_indices].tolist(),
                "text_value": np.asarray(concept_as_values)[valid_indices].tolist(),
                "unit": np.asarray(units)[valid_indices].tolist(),
            }
        )

    if not trajectories:
        return pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "arm": pl.String,
                "trajectory_id": pl.Int32,
                "prediction_time": pl.Datetime,
                "window_last_observed_time": pl.Datetime,
                "time": pl.Datetime,
                "code": pl.String,
                "numeric_value": pl.Float64,
                "text_value": pl.String,
                "unit": pl.String,
            }
        )

    return (
        pl.DataFrame(trajectories)
        .explode(["time", "code", "numeric_value", "text_value", "unit"])
    )


def context_num_rows(context_parquet: str) -> int:
    """Number of patient contexts in a parquet file or a directory of parquet shards."""
    path = str(context_parquet)
    pattern = os.path.join(path, "*.parquet") if os.path.isdir(path) else path
    return int(pl.scan_parquet(pattern).select(pl.len()).collect().item())


def batch_output_path(output_dir: Path, arm: str, traj_id: int, batch_i: int) -> Path:
    return output_dir / arm / str(traj_id) / f"batch_{batch_i:05d}.parquet"


def pending_tasks(
    arm: str,
    num_rows: int,
    batch_size: int,
    num_trajectories: int,
    output_dir: Path,
) -> List[Tuple[int, int]]:
    """
    ``(trajectory_id, batch_index)`` of every batch of *arm* whose parquet does not exist yet.

    Batch ``i`` always holds contexts ``[i * batch_size, (i + 1) * batch_size)`` of the arm,
    so a finished batch stays valid whatever else changes between runs (also the number of GPUs).
    """
    num_batches = math.ceil(num_rows / batch_size)
    return [
        (traj_id, batch_i)
        for traj_id in range(num_trajectories)
        for batch_i in range(num_batches)
        if not batch_output_path(output_dir, arm, traj_id, batch_i).exists()
    ]


def write_parquet_atomically(df: pl.DataFrame, path: Path) -> None:
    """A batch counts as finished once its parquet exists, so never leave a partial one."""
    temp_path = path.with_name(path.name + ".tmp")
    df.write_parquet(temp_path)
    os.replace(temp_path, path)


def run_generation(
    context_parquet: str,
    arm: str,
    cehrgpt_tokenizer: CehrGptTokenizer,
    cehrgpt_model: CEHRGPT2LMHeadModel,
    device: torch.device,
    output_dir: Path,
    num_trajectories: int,
    batch_size: int,
    context_length: int,
    max_length: int,
    include_values: bool,
    num_workers: int,
    forbidden_token_ids: Optional[List[int]] = None,
    tasks: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """
    Load a context parquet, run *num_trajectories* generation passes, and
    write output parquets under ``output_dir/<arm>/<trajectory_id>/``.

    Parameters
    ----------
    context_length
        Maximum number of input tokens fed to the model (= generation_input_length).
        The collator truncates each context to this length.
    max_length
        Total sequence length budget for model.generate()
        (= context_length + generation_max_new_tokens).
    tasks
        ``(trajectory_id, batch_index)`` pairs to generate, as handed out to a worker by
        ``run_on_multiple_gpus``.  Default: every batch whose parquet is missing.
    """
    output_dir = Path(output_dir)
    num_rows = context_num_rows(context_parquet)
    if tasks is None:
        tasks = pending_tasks(arm, num_rows, batch_size, num_trajectories, output_dir)
    else:
        tasks = [
            (traj_id, batch_i)
            for traj_id, batch_i in tasks
            if not batch_output_path(output_dir, arm, traj_id, batch_i).exists()
        ]
    if not tasks:
        print(f"[{arm}] nothing to generate ({num_rows:,} patient contexts)")
        return

    print(f"\n[{arm}] Loading context from {context_parquet} …")
    # Accept either a single .parquet file or a directory of parquet shards
    if os.path.isdir(context_parquet):
        df = pl.read_parquet(os.path.join(context_parquet, "*.parquet"))
    else:
        df = pl.read_parquet(context_parquet)
    print(f"[{arm}] {len(df):,} patient contexts, {len(tasks):,} batches to generate")

    # Left-truncate to context_length, preserving demographics header tokens
    # and updating the year/age tokens to the new temporal reference point.
    df = _left_truncate_with_demographics(df, max_length=context_length, tokenizer=cehrgpt_tokenizer)
    print(f"[{arm}] truncated to max {context_length} tokens (demographics year/age updated)")

    # Convert to HuggingFace Dataset (keeps all columns)
    dataset = Dataset.from_pandas(df.to_pandas())

    base_collator = CehrGptDataCollator(
        tokenizer=cehrgpt_tokenizer,
        max_length=context_length,  # truncate input to context window, not total budget
        include_values=include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=False,
        add_linear_prob_token=False,
    )
    collator = CounterfactualDataCollator(base_collator)

    batches_by_trajectory: Dict[int, List[int]] = {}
    for traj_id, batch_i in tasks:
        batches_by_trajectory.setdefault(traj_id, []).append(batch_i)

    for traj_id, batch_ids in sorted(batches_by_trajectory.items()):
        (output_dir / arm / str(traj_id)).mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=[
                list(range(b * batch_size, min((b + 1) * batch_size, len(dataset))))
                for b in batch_ids
            ],
            num_workers=num_workers,
            collate_fn=collator,
        )

        for batch_i, batch in zip(
            batch_ids,
            tqdm(dataloader, desc=f"[{arm}] trajectory {traj_id + 1}/{num_trajectories}"),
        ):
            df_out = generate_trajectories_for_batch(
                batch=batch,
                cehrgpt_tokenizer=cehrgpt_tokenizer,
                cehrgpt_model=cehrgpt_model,
                device=device,
                arm=arm,
                trajectory_id=traj_id,
                max_length=max_length,
                forbidden_token_ids=forbidden_token_ids,
            )
            write_parquet_atomically(df_out, batch_output_path(output_dir, arm, traj_id, batch_i))


def split_gpus_by_arm(gpu_ids: List[str], pending_by_arm: Dict[str, int]) -> Dict[str, List[str]]:
    """
    Divide the GPUs into one set per arm that has batches left, in proportion to the batches.

    Every arm gets a GPU of its own first; each further GPU goes to the arm with the most batches
    per GPU so far (the first arm on a tie), and an arm never gets more GPUs than it has batches.
    Returns ``{}`` when there are fewer GPUs than arms with work: the arms cannot run side by side.
    """
    arms = [arm for arm, pending in pending_by_arm.items() if pending > 0]
    if not arms or len(gpu_ids) < len(arms):
        return {}
    count = {arm: 1 for arm in arms}
    for _ in range(len(gpu_ids) - len(arms)):
        open_arms = [arm for arm in arms if count[arm] < pending_by_arm[arm]]
        if not open_arms:
            break
        count[max(open_arms, key=lambda arm: pending_by_arm[arm] / count[arm])] += 1
    groups: Dict[str, List[str]] = {}
    start = 0
    for arm in arms:
        groups[arm] = list(gpu_ids[start : start + count[arm]])
        start += count[arm]
    return groups


def run_on_multiple_gpus(args: argparse.Namespace, gpu_ids: List[str]) -> int:
    """
    Deal the pending batches out to one worker process per GPU.

    A task is one batch of one trajectory of one arm.  The GPUs are divided into one set per arm
    (``split_gpus_by_arm``), so the arms are generated at the same time; the batches of an arm go out
    in the order trajectory 0, 1, ... so its first trajectory is complete early.  With fewer GPUs
    than arms the batches of all arms are dealt out to all GPUs, arm after arm.  Every worker re-runs
    this module with ``CUDA_VISIBLE_DEVICES`` pinned to its GPU and a ``--task_file`` listing its
    tasks, loads its own copy of the model, and writes the batches into the usual
    ``<arm>/<trajectory_id>`` folders.  The pending tasks are recomputed from the parquet files that
    exist, so rerunning the command resumes an interrupted run.
    """
    arm_map = _resolve_arm_map(args)
    output_dir = Path(args.output_dir)
    tasks_by_arm: Dict[str, List[Tuple[int, int]]] = {}
    for arm, context_parquet in arm_map.items():
        num_rows = context_num_rows(str(context_parquet))
        tasks_by_arm[arm] = pending_tasks(
            arm, num_rows, args.batch_size, args.num_trajectories, output_dir
        )
    if not any(tasks_by_arm.values()):
        print("There is nothing left to generate.")
        return 0

    # (gpu, {arm: [[trajectory_id, batch_index], ...]}) for every worker
    workers: List[Tuple[str, Dict[str, List[List[int]]]]] = []
    groups = split_gpus_by_arm(gpu_ids, {arm: len(tasks) for arm, tasks in tasks_by_arm.items()})
    if groups:
        for arm, arm_gpus in groups.items():
            print(f"Arm {arm}: {len(tasks_by_arm[arm]):,} batches on GPU(s) {arm_gpus}")
            for worker_index, gpu in enumerate(arm_gpus):
                workers.append((gpu, {arm: [list(t) for t in tasks_by_arm[arm][worker_index :: len(arm_gpus)]]}))
    else:
        all_tasks = [(arm, t, b) for arm, tasks in tasks_by_arm.items() for t, b in tasks]
        num_workers = min(len(gpu_ids), len(all_tasks))
        print(
            f"Fewer GPUs than arms: {len(all_tasks):,} batches across GPU(s) {gpu_ids[:num_workers]}, "
            "arm after arm"
        )
        for worker_index in range(num_workers):
            worker_tasks: Dict[str, List[List[int]]] = {}
            for arm, traj_id, batch_i in all_tasks[worker_index::num_workers]:
                worker_tasks.setdefault(arm, []).append([traj_id, batch_i])
            workers.append((gpu_ids[worker_index], worker_tasks))

    task_root = output_dir / "temp" / "tasks"
    log_folder = output_dir / "worker_logs"
    shutil.rmtree(task_root, ignore_errors=True)
    task_root.mkdir(parents=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    worker_argv = strip_cli_option(sys.argv[1:], "--gpu_ids")
    processes: List[subprocess.Popen] = []
    log_paths: List[Path] = []
    # Make sure the workers don't outlive us when we are asked to stop
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for worker_index, (gpu, worker_tasks) in enumerate(workers):
            task_file = task_root / f"worker_{worker_index}.json"
            task_file.write_text(json.dumps(worker_tasks))
            log_path = log_folder / f"worker_{worker_index}_gpu_{gpu}.log"
            log_paths.append(log_path)
            with open(log_path, "w") as log_file:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "cehrgpt.analysis.counterfactual.generate_counterfactual_sequences",
                            *worker_argv,
                            "--task_file",
                            str(task_file),
                        ],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                )
            print(f"Started worker {worker_index} on GPU {gpu} for {list(worker_tasks)}, log: {log_path}")
        exit_codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    failed = [i for i, code in enumerate(exit_codes) if code != 0]
    if failed:
        for i in failed:
            print(f"Worker {i} exited with code {exit_codes[i]}, see {log_paths[i]}")
        return 1
    shutil.rmtree(task_root, ignore_errors=True)
    try:
        task_root.parent.rmdir()
    except OSError:
        pass
    print(f"All {len(workers)} workers finished")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate counterfactual trajectories for one or more named arms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Original drug-vs-no-drug pipeline (uses context_dir convention):
  generate_counterfactual_sequences.py \\
      --context_dir /data/contexts \\
      --model_name_or_path /models/cehrgpt \\
      --tokenizer_path /models/tokenizer \\
      --output_dir /data/trajectories

# Active-comparator pipeline (ACEi vs Thiazide); explicit arm paths:
  generate_counterfactual_sequences.py \\
      --arm_context acei:/data/ce/acei_ctx.parquet \\
      --arm_context thiazide:/data/ce/thiazide_ctx.parquet \\
      --model_name_or_path /models/cehrgpt \\
      --tokenizer_path /models/tokenizer \\
      --output_dir /data/ce/trajectories
""",
    )

    # --- arm specification (two mutually exclusive modes) -----------------
    arm_group = parser.add_mutually_exclusive_group(required=True)
    arm_group.add_argument(
        "--context_dir",
        default=None,
        help=(
            "Directory containing non_treated_context.parquet and "
            "treated_context.parquet (original pipeline convention)."
        ),
    )
    arm_group.add_argument(
        "--arm_context",
        dest="arm_contexts",
        action="append",
        metavar="NAME:PATH",
        default=None,
        help=(
            "Explicit arm specification as NAME:PATH.  Repeat for each arm. "
            "NAME becomes the sub-directory under --output_dir and the 'arm' "
            "label in generated trajectories. "
            "E.g. --arm_context acei:/data/acei_ctx.parquet "
            "     --arm_context thiazide:/data/thiazide_ctx.parquet"
        ),
    )

    # --- arms filter (only relevant when using --context_dir) -------------
    parser.add_argument(
        "--arms",
        default="non_treated,treated",
        help=(
            "Comma-separated arms to run when using --context_dir. "
            "Ignored when --arm_context is used. (default: non_treated,treated)"
        ),
    )

    # --- model / generation -----------------------------------------------
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to the pretrained CEHR-GPT model",
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        help="Path to the CehrGptTokenizer directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for generated trajectory parquets",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10,
        help="Number of independent trajectories to generate per patient per arm (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)",
    )
    parser.add_argument(
        "--generation_input_length",
        type=int,
        default=1024,
        help="Maximum context length fed to the model (default: 1024)",
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0)",
    )

    parser.add_argument(
        "--gpu_ids",
        default=None,
        help=(
            "Generate on several GPUs: 'all' or comma-separated indices of the visible GPUs, "
            "e.g. 0,1,3 (relative to CUDA_VISIBLE_DEVICES when it is set). One worker process "
            "per GPU, each with its own copy of the model, takes a share of the batches. "
            "Rerunning continues where an interrupted run stopped. Default: one process."
        ),
    )
    parser.add_argument(
        "--task_file",
        default=None,
        help="Internal, set by --gpu_ids: the batches this worker generates.",
    )

    # --- concept suppression (optional) ------------------------------------
    parser.add_argument(
        "--vocab_path",
        default=None,
        help=(
            "OMOP vocabulary root (must contain concept_ancestor/ parquets). "
            "Required when using --arm_suppress_concepts."
        ),
    )
    parser.add_argument(
        "--arm_suppress_concepts",
        dest="arm_suppress_concepts",
        action="append",
        metavar="NAME:CONCEPT_IDS",
        default=None,
        help=(
            "Suppress specific drug concepts during generation for a named arm. "
            "Format: NAME:ID1,ID2,... where NAME matches an --arm_context name and "
            "IDs are comma-separated OMOP ingredient concept_ids to suppress "
            "(expanded to all descendants via concept_ancestor). "
            "Repeat for each arm. Example: "
            "--arm_suppress_concepts tnfi:710062 "
            "--arm_suppress_concepts amitriptyline:1119119,4299871,937368,912263,19041065"
        ),
    )
    return parser.parse_args()


def _resolve_arm_map(args: argparse.Namespace) -> Dict[str, Path]:
    """
    Return an ordered dict of {arm_name: context_parquet_path} from CLI args.
    Supports both --context_dir (original convention) and --arm_context NAME:PATH.
    """
    if args.arm_contexts:
        arm_map: Dict[str, Path] = {}
        for spec in args.arm_contexts:
            if ":" not in spec:
                raise ValueError(
                    f"--arm_context must be NAME:PATH, got: '{spec}'"
                )
            name, path = spec.split(":", 1)
            name = name.strip()
            p = Path(path.strip())
            if not p.exists():
                raise FileNotFoundError(f"Context parquet not found: {p}")
            arm_map[name] = p
        return arm_map

    # Fallback: context_dir convention
    # extract_drug_initiation_sequences.py writes these as *directories* of
    # parquet shards, so accept both a single .parquet file and a directory.
    context_dir = Path(args.context_dir)

    def _resolve_context_path(name: str) -> Path:
        # Prefer directory (chunked shards), fall back to single file
        as_dir  = context_dir / name
        as_file = context_dir / f"{name}.parquet"
        if as_dir.is_dir():
            return as_dir
        if as_file.exists():
            return as_file
        raise FileNotFoundError(
            f"Context not found for arm '{name}': "
            f"tried {as_dir} (directory) and {as_file} (file)"
        )

    convention_names = {NON_TREATED_ARM: "non_treated_context", TREATED_ARM: "treated_context"}
    arms_to_run = [a.strip() for a in args.arms.split(",")]
    arm_map = {}
    for arm in arms_to_run:
        if arm not in convention_names:
            raise ValueError(
                f"Unknown arm '{arm}' for --context_dir mode. "
                f"Choose from: {list(convention_names)}"
            )
        arm_map[arm] = _resolve_context_path(convention_names[arm])
    return arm_map


def main() -> None:
    args = _parse_args()

    if args.gpu_ids and not args.task_file:
        gpu_ids = resolve_gpu_ids(args.gpu_ids)
        if len(gpu_ids) > 1:
            exit_code = run_on_multiple_gpus(args, gpu_ids)
            if exit_code:
                sys.exit(exit_code)
            return
        # CUDA has not been initialized yet, so this pins the single requested GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]

    worker_tasks: Optional[Dict[str, List[Tuple[int, int]]]] = None
    if args.task_file:
        with open(args.task_file) as task_file:
            worker_tasks = {
                arm: [(int(t), int(b)) for t, b in tasks]
                for arm, tasks in json.load(task_file).items()
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer …")
    tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_path)

    print("Loading model …")
    model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
        )
        .eval()
        .to(device)
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.end_token_id
    model.generation_config.bos_token_id = tokenizer.end_token_id

    include_values = model.config.include_values
    max_length = args.generation_input_length + args.generation_max_new_tokens

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_map = _resolve_arm_map(args)
    print(f"\nArms to generate: {list(arm_map)}")

    # Build per-arm forbidden token ID sets from --arm_suppress_concepts specs
    forbidden_map: Dict[str, Optional[List[int]]] = {arm: None for arm in arm_map}
    if args.arm_suppress_concepts and args.vocab_path:
        print("\nBuilding concept suppression token IDs …")
        for spec in args.arm_suppress_concepts:
            if ":" not in spec:
                raise ValueError(
                    f"--arm_suppress_concepts must be NAME:CONCEPT_IDS, got: '{spec}'"
                )
            arm_name, ids_str = spec.split(":", 1)
            arm_name = arm_name.strip()
            concept_ids = [int(x.strip()) for x in ids_str.split(",")]
            print(f"  [{arm_name}] suppressing {len(concept_ids)} ingredient(s) …")
            forbidden_map[arm_name] = build_forbidden_token_ids(
                args.vocab_path, concept_ids, tokenizer
            )

    for arm, context_parquet in arm_map.items():
        if worker_tasks is not None and arm not in worker_tasks:
            continue
        run_generation(
            context_parquet=str(context_parquet),
            arm=arm,
            cehrgpt_tokenizer=tokenizer,
            cehrgpt_model=model,
            device=device,
            output_dir=output_dir,
            num_trajectories=args.num_trajectories,
            batch_size=args.batch_size,
            context_length=args.generation_input_length,
            max_length=max_length,
            include_values=include_values,
            num_workers=args.num_workers,
            forbidden_token_ids=forbidden_map[arm],
            tasks=worker_tasks[arm] if worker_tasks is not None else None,
        )

    print("\nAll generation complete.")


if __name__ == "__main__":
    main()
