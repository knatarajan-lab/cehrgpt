#!/usr/bin/env python
"""Benchmark time-to-event rollout batch sizes and same-GPU worker counts."""

import argparse
import glob
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers.utils import is_flash_attn_2_available

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import read_backbone, resolve_attn_implementation
from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel
from cehrgpt.time_to_event.time_to_event_prediction import load_task_config_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True)
    parser.add_argument("--prediction_folder", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_patients", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=64)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--barrier_dir")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--use_bfloat16", action="store_true")
    return parser.parse_args()


def load_prefixes(folder: str, count: int) -> list[list[str]]:
    prefixes = []
    files = sorted(glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True))
    for parquet_file in files:
        prefixes.extend(pd.read_parquet(parquet_file, columns=["prefix"])["prefix"])
        if len(prefixes) >= count:
            return prefixes[:count]
    raise RuntimeError(f"Found {len(prefixes)} prefixes under {folder}, need {count}")


def synchronize_workers(barrier_dir: str, worker_index: int, num_workers: int) -> None:
    barrier = Path(barrier_dir)
    barrier.mkdir(parents=True, exist_ok=True)
    (barrier / f"ready_{worker_index}").touch()
    start_file = barrier / "start"
    if worker_index == 0:
        ready_files = [barrier / f"ready_{index}" for index in range(num_workers)]
        while not all(path.exists() for path in ready_files):
            time.sleep(0.1)
        start_file.touch()
    else:
        while not start_file.exists():
            time.sleep(0.1)


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CehrGptTokenizer.from_pretrained(args.model_folder)
    model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_folder,
            attn_implementation=resolve_attn_implementation(
                backbone=read_backbone(args.model_folder)
            ),
            torch_dtype=(
                torch.bfloat16
                if is_flash_attn_2_available() and args.use_bfloat16
                else "auto"
            ),
        )
        .eval()
        .to(device)
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.end_token_id
    model.generation_config.bos_token_id = tokenizer.end_token_id
    task = load_task_config_from_yaml(args.task_config)
    generation_config = TimeToEventModel.get_generation_config(
        tokenizer=tokenizer,
        max_length=model.config.n_positions,
        num_return_sequences=args.num_return_sequences,
        top_p=1.0,
        top_k=tokenizer.vocab_size,
        max_new_tokens=task.max_new_tokens,
    )
    predictor = TimeToEventModel(
        tokenizer=tokenizer,
        model=model,
        outcome_events=task.outcome_events,
        generation_config=generation_config,
        device=device,
        batch_size=args.batch_size,
    )

    # Prefix zero is a common untimed warm-up. The remaining prefixes are the
    # benchmark cohort and are deterministically sharded across workers.
    prefixes = load_prefixes(args.prediction_folder, args.num_patients + 1)
    torch.manual_seed(args.seed)
    predictor.predict_time_to_events(
        prefixes[0],
        task.future_visit_start,
        task.future_visit_end,
        task.prediction_window_start,
        task.prediction_window_end,
    )
    torch.cuda.synchronize()
    if args.barrier_dir:
        synchronize_workers(args.barrier_dir, args.worker_index, args.num_workers)

    selected = [
        (index, prefix)
        for index, prefix in enumerate(prefixes[1:])
        if index % args.num_workers == args.worker_index
    ]
    start = time.perf_counter()
    for patient_index, prefix in selected:
        torch.manual_seed(args.seed + patient_index + 1)
        predictor.predict_time_to_events(
            prefix,
            task.future_visit_start,
            task.future_visit_end,
            task.prediction_window_start,
            task.prediction_window_end,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"RESULT worker={args.worker_index}/{args.num_workers} "
        f"batch_size={args.batch_size} patients={len(selected)} "
        f"elapsed_seconds={elapsed:.6f} "
        f"seconds_per_patient={elapsed / len(selected):.6f}"
    )


if __name__ == "__main__":
    main()
