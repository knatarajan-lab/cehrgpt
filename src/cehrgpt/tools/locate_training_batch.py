"""
Identify which patients were in the batch at a given training step, without running the
model.

Reproduces the dataset construction and the sample-packing batch order of
`hf_cehrgpt_pretrain_runner` for the same YAML, then walks the batch sampler and reports
the person_ids and sequence lengths of the batch at a chosen step. Useful for tracing a
loss or gradient spike back to the data that produced it.

The step can be given directly with `--global_step`, or back-calculated from a logged
learning rate with `--learning_rate`, since the LR schedule is a deterministic function of
the step.

Fidelity notes - the batch order depends on all of these, so they must match the run:

  * The batch order is NOT the dataset order. `SamplePackingBatchSampler` shuffles with
    `torch.randperm` seeded by `seed + epoch`, so it is deterministic but permuted.
    `shuffle_records` is a different setting (it shuffles records *within* a patient
    sequence in the collator) and does not affect batch composition.
  * The dataset is filtered by `min_num_tokens` (and `drop_long_sequences`) after loading,
    which changes the indices the sampler draws from.
  * Under distributed training the sampler partitions indices by rank, so the batch at a
    given step differs per rank. Pass `--num_replicas`/`--rank` to match a multi-GPU run;
    the defaults assume a single process.

The prepared-dataset cache is named `<data_folder basename>_<md5>`, where the hash covers
data_folder, tokenizer_name_or_path, validation_split_percentage, test_eval_ratio,
split_by_patient and chronological_split. Editing any of those in the YAML after a run
makes that run's cache unreachable by recomputation, so `--dataset_path` can point at it
directly. The runner loads the `<basename>_<hash>` directory as a DatasetDict and uses its
`train` split; this tool accepts either that directory or its `train/` subdirectory.

Usage:
    python -m cehrgpt.tools.locate_training_batch \
        --yaml_file /path/to/comet_s_qwen2.yaml \
        --learning_rate 0.00019925420155579422 \
        --dataset_path /mnt/ssd1/.../train_4f96e9a851c44706919ddde8333c98f7
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    ModelArguments,
)
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import DatasetDict, load_from_disk
from transformers import HfArgumentParser, TrainingArguments

from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments


def parse_config(yaml_file: str):
    parser = HfArgumentParser(
        (CehrGPTArguments, DataTrainingArguments, ModelArguments, TrainingArguments)
    )
    return parser.parse_yaml_file(yaml_file=os.path.expanduser(yaml_file))


def _describe_cache_candidates(dataset_prepared_path: str) -> str:
    """List the cache directories that do exist, to make a hash mismatch diagnosable."""
    root = Path(os.path.expanduser(dataset_prepared_path))
    if not root.exists():
        return f"  (dataset_prepared_path does not exist: {root})"
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        return f"  (no cache directories under {root})"
    lines = [f"  Available caches under {root}:"]
    for candidate in candidates:
        shape = (
            "DatasetDict"
            if (candidate / "dataset_dict.json").exists()
            else "Dataset" if (candidate / "dataset_info.json").exists() else "unknown"
        )
        lines.append(f"    {candidate.name}  [{shape}]")
    return "\n".join(lines)


def _as_train_split(loaded, source: str):
    """
    Normalise whatever `load_from_disk` returned into the train split.

    The runner loads the parent directory, which is DatasetDict-shaped
    (`dataset_dict.json` plus `train/` and `validation/` subdirectories), and then uses
    `processed_dataset["train"]`. Pointing directly at the `train/` subdirectory yields a
    bare `Dataset` instead, which is equally usable here because this tool only needs the
    train split - the sample-packing batch order is built from
    `processed_dataset["train"]["num_of_concepts"]` alone.
    """
    if isinstance(loaded, (DatasetDict, dict)):
        if "train" not in loaded:
            raise RuntimeError(
                f"{source} is DatasetDict-shaped but has no 'train' split; found "
                f"{sorted(loaded.keys())}."
            )
        print(f"  loaded DatasetDict with splits {sorted(loaded.keys())}; using ['train']")
        return loaded["train"]
    print(f"  loaded a single Dataset; treating it as the train split")
    return loaded


def load_train_split(cehrgpt_args, data_args, model_args, dataset_path: Optional[str]):
    """Mirror the runner's dataset resolution, including the post-load filter."""
    if dataset_path:
        resolved = Path(os.path.expanduser(dataset_path))
        if not resolved.exists():
            raise RuntimeError(f"--dataset_path does not exist: {resolved}")
        print(f"Loading dataset from --dataset_path: {resolved}")
        dataset = _as_train_split(load_from_disk(str(resolved)), str(resolved))
    else:
        if cehrgpt_args.tokenized_dataset_name:
            prepared_ds_path = Path(
                os.path.join(
                    data_args.dataset_prepared_path, cehrgpt_args.tokenized_dataset_name
                )
            )
        else:
            prepared_ds_path = generate_prepared_ds_path(data_args, model_args)

        if os.path.exists(os.path.join(data_args.data_folder, "dataset_dict.json")):
            print(f"Loading dataset from data_folder: {data_args.data_folder}")
            dataset = _as_train_split(
                load_from_disk(os.path.expanduser(data_args.data_folder)),
                data_args.data_folder,
            )
        elif any(prepared_ds_path.glob("*")):
            print(f"Loading prepared dataset: {prepared_ds_path}")
            dataset = _as_train_split(
                load_from_disk(str(prepared_ds_path)), str(prepared_ds_path)
            )
        else:
            raise RuntimeError(
                "No prepared dataset found.\n"
                f"  data_folder     : {data_args.data_folder}\n"
                f"  computed cache  : {prepared_ds_path}\n"
                "The cache name is a hash of data_folder, tokenizer_name_or_path, "
                "validation_split_percentage, test_eval_ratio, split_by_patient and "
                "chronological_split, so editing any of those in the YAML after a run "
                "makes the old cache unreachable by recomputation.\n"
                f"{_describe_cache_candidates(data_args.dataset_prepared_path)}\n"
                "  Pass --dataset_path <cache dir> to use one directly."
            )

    before = len(dataset)

    # Same filter the runner applies before handing the dataset to the trainer.
    def filter_func(examples):
        if cehrgpt_args.drop_long_sequences:
            return [
                model_args.max_position_embeddings >= _ >= data_args.min_num_tokens
                for _ in examples["num_of_concepts"]
            ]
        return [_ >= data_args.min_num_tokens for _ in examples["num_of_concepts"]]

    filter_args = {"batched": True, "batch_size": data_args.preprocessing_batch_size}
    if not data_args.streaming:
        filter_args["num_proc"] = data_args.preprocessing_num_workers
    dataset = dataset.filter(filter_func, **filter_args)

    print(
        f"  train rows: {before} -> {len(dataset)} after the "
        f"min_num_tokens={data_args.min_num_tokens}"
        f"{', drop_long_sequences' if cehrgpt_args.drop_long_sequences else ''} filter"
    )
    return dataset


def invert_lr_schedule(
    learning_rate: float,
    base_lr: float,
    num_training_steps: int,
    num_warmup_steps: int,
    scheduler_type: str,
) -> Optional[int]:
    """
    Recover the global step from a logged learning rate.

    The trainer calls `scheduler.step()` once per optimizer step, so after global step N
    the LR equals `base_lr * lambda(N)`.
    """
    factor = learning_rate / base_lr
    scheduler_type = str(scheduler_type).lower().replace("schedulertype.", "")

    if factor > 1.0 + 1e-9:
        return None

    # Warmup branch is shared by the common schedulers: lambda(N) = N / warmup.
    warmup_candidate = factor * num_warmup_steps
    if num_warmup_steps > 0 and warmup_candidate <= num_warmup_steps:
        warmup_step = int(round(warmup_candidate))
        if warmup_step < num_warmup_steps:
            return warmup_step

    decay_span = max(1, num_training_steps - num_warmup_steps)
    if scheduler_type in ("linear", "schedulertype.linear"):
        # lambda(N) = (T - N) / (T - W)
        return int(round(num_training_steps - factor * decay_span))
    if scheduler_type == "cosine":
        # lambda(N) = 0.5 * (1 + cos(pi * progress))
        progress = math.acos(max(-1.0, min(1.0, 2.0 * factor - 1.0))) / math.pi
        return int(round(num_warmup_steps + progress * decay_span))
    if scheduler_type == "constant_with_warmup":
        return None
    raise ValueError(
        f"Cannot invert scheduler type {scheduler_type!r}; pass --global_step instead."
    )


def describe_batch(dataset, indices: List[int], lengths, max_position_embeddings):
    rows = []
    columns = dataset.column_names
    id_column = next(
        (c for c in ("person_id", "subject_id", "patient_id") if c in columns), None
    )
    subset = dataset.select(indices)
    ids = subset[id_column] if id_column else [None] * len(indices)
    for position, dataset_index in enumerate(indices):
        raw_length = lengths[dataset_index]
        rows.append(
            {
                "position_in_batch": position,
                "dataset_index": dataset_index,
                id_column or "row_id": ids[position],
                "num_of_concepts": raw_length,
                "packed_length": min(raw_length, max_position_embeddings) + 2,
                "truncated": raw_length > max_position_embeddings,
            }
        )
    return rows, (id_column or "row_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Load this prepared-dataset directory instead of recomputing the cache hash "
        "from the YAML. Accepts the DatasetDict parent (the <basename>_<hash> directory, "
        "which is what the runner loads) or its train/ subdirectory. Needed when the YAML "
        "has been edited since the run, because the hash would no longer match.",
    )
    parser.add_argument(
        "--global_step",
        type=int,
        default=None,
        help="Global step to inspect (1-based, as the trainer logs it).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Logged learning rate at the step of interest; used to derive --global_step.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Also summarise this many batches either side of the target.",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="World size of the original run (the sampler shards indices by rank).",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Write the target batch's rows to this CSV.",
    )
    args = parser.parse_args()

    if args.global_step is None and args.learning_rate is None:
        parser.error("Provide --global_step or --learning_rate")

    cehrgpt_args, data_args, model_args, training_args = parse_config(args.yaml_file)
    if not cehrgpt_args.sample_packing:
        print(
            "WARNING: sample_packing is false in this config, so the real run did not use "
            "SamplePackingBatchSampler. The batch order below will not match."
        )

    train_dataset = load_train_split(
        cehrgpt_args, data_args, model_args, args.dataset_path
    )
    lengths = train_dataset["num_of_concepts"]

    # Exactly the sampler SamplePackingTrainer.get_train_dataloader builds.
    sampler = SamplePackingBatchSampler(
        lengths=lengths,
        max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
        max_position_embeddings=model_args.max_position_embeddings,
        drop_last=training_args.dataloader_drop_last,
        seed=training_args.seed,
        num_replicas=args.num_replicas,
        rank=args.rank,
        negative_sampling_probability=None,
        labels=None,
    )

    # The trainer derives its step budget from len(dataloader), which for a batch_sampler
    # dataloader is len(sampler) - and that is an ESTIMATE in this implementation, not the
    # true batch count. The LR schedule is built on the estimate, so the back-calculation
    # has to use it too.
    estimated_batches = len(sampler)
    grad_accum = training_args.gradient_accumulation_steps
    steps_per_epoch = max(estimated_batches // grad_accum, 1)
    if training_args.max_steps > 0:
        max_steps = training_args.max_steps
    else:
        max_steps = math.ceil(training_args.num_train_epochs * steps_per_epoch)
    warmup_steps = training_args.get_warmup_steps(max_steps)

    print("\nSchedule reconstruction")
    print(f"  train rows                : {len(lengths)}")
    print(f"  len(sampler) [estimate]   : {estimated_batches}")
    print(f"  gradient_accumulation     : {grad_accum}")
    print(f"  steps/epoch [estimate]    : {steps_per_epoch}")
    print(f"  num_train_epochs          : {training_args.num_train_epochs}")
    print(f"  max_steps                 : {max_steps}")
    print(f"  warmup_steps              : {warmup_steps}")
    print(f"  lr_scheduler_type         : {training_args.lr_scheduler_type}")
    print(f"  base learning_rate        : {training_args.learning_rate}")

    global_step = args.global_step
    if global_step is None:
        global_step = invert_lr_schedule(
            args.learning_rate,
            training_args.learning_rate,
            max_steps,
            warmup_steps,
            training_args.lr_scheduler_type,
        )
        if global_step is None:
            print(
                "\nCould not invert the schedule for that learning rate (constant "
                "schedule, or lr above base). Pass --global_step."
            )
            return 2
        print(f"\n  learning_rate {args.learning_rate} -> global_step {global_step}")
        print(f"  implied epoch             : {global_step / steps_per_epoch:.4f}")

    epoch = global_step // steps_per_epoch
    print(f"\nTarget: global_step={global_step} (epoch {epoch}, "
          f"{global_step % steps_per_epoch} batches into it)")

    # The sampler reshuffles per epoch via set_epoch, exactly as accelerate does.
    sampler.sampler.set_epoch(epoch)
    batch_index_in_epoch = global_step % steps_per_epoch
    # global_step N is produced by the N-th batch, i.e. 0-based index N-1.
    target_index = max(batch_index_in_epoch - 1, 0)

    collected = {}
    actual_batches = 0
    low = max(0, target_index - args.window)
    high = target_index + args.window
    for position, batch in enumerate(sampler):
        actual_batches += 1
        if low <= position <= high:
            collected[position] = list(batch)

    print(f"  batches actually yielded in epoch {epoch}: {actual_batches}")
    if actual_batches != estimated_batches:
        print(
            f"  NOTE: differs from the estimate ({estimated_batches}); "
            f"len(SamplePackingBatchSampler) is heuristic, so the trainer's epoch/step "
            f"bookkeeping is offset from the true batch count by this much."
        )
    if target_index not in collected:
        print(
            f"\nERROR: batch index {target_index} is beyond the {actual_batches} batches "
            f"in epoch {epoch}."
        )
        return 2

    all_lengths = np.asarray(lengths)
    print("\nDataset-wide num_of_concepts: "
          f"p50={np.percentile(all_lengths, 50):.0f} "
          f"p90={np.percentile(all_lengths, 90):.0f} "
          f"p99={np.percentile(all_lengths, 99):.0f} "
          f"max={all_lengths.max():.0f}")

    for position in sorted(collected):
        indices = collected[position]
        rows, id_column = describe_batch(
            train_dataset, indices, lengths, model_args.max_position_embeddings
        )
        packed = sum(r["packed_length"] for r in rows)
        raw = [r["num_of_concepts"] for r in rows]
        marker = "  <-- TARGET" if position == target_index else ""
        print(
            f"\nbatch {position} (global_step {position + 1}): {len(rows)} patients, "
            f"packed_tokens={packed}/{cehrgpt_args.max_tokens_per_batch}, "
            f"max_len={max(raw)}, truncated={sum(r['truncated'] for r in rows)}{marker}"
        )
        if position == target_index:
            print(f"  {'pos':>4} {'idx':>9} {id_column:>14} {'len':>7} {'packed':>7} trunc")
            for row in rows:
                print(
                    f"  {row['position_in_batch']:>4} {row['dataset_index']:>9} "
                    f"{str(row[id_column]):>14} {row['num_of_concepts']:>7} "
                    f"{row['packed_length']:>7} {row['truncated']}"
                )

    if args.output_csv:
        rows, id_column = describe_batch(
            train_dataset,
            collected[target_index],
            lengths,
            model_args.max_position_embeddings,
        )
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
