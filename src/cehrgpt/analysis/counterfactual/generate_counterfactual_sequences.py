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
"""

import argparse
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NON_TREATED_ARM = "non_treated"
TREATED_ARM = "treated"


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
    batched_value_indicators = batch["value_indicators"]
    batched_values = batch["values"]

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


def run_generation(
    context_parquet: str,
    arm: str,
    cehrgpt_tokenizer: CehrGptTokenizer,
    cehrgpt_model: CEHRGPT2LMHeadModel,
    device: torch.device,
    output_dir: Path,
    num_trajectories: int,
    batch_size: int,
    max_length: int,
    include_values: bool,
    num_workers: int,
) -> None:
    """
    Load a context parquet, run *num_trajectories* generation passes, and
    write output parquets under ``output_dir/<arm>/<trajectory_id>/``.
    """
    print(f"\n[{arm}] Loading context from {context_parquet} …")
    df = pl.read_parquet(context_parquet)
    print(f"[{arm}] {len(df):,} patient contexts")

    # Convert to HuggingFace Dataset (keeps all columns)
    dataset = Dataset.from_pandas(df.to_pandas())

    base_collator = CehrGptDataCollator(
        tokenizer=cehrgpt_tokenizer,
        max_length=max_length,
        include_values=include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=False,
        add_linear_prob_token=False,
    )
    collator = CounterfactualDataCollator(base_collator)

    for traj_id in range(num_trajectories):
        traj_dir = output_dir / arm / str(traj_id)
        traj_dir.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

        for batch_i, batch in tqdm(
            enumerate(dataloader),
            desc=f"[{arm}] trajectory {traj_id + 1}/{num_trajectories}",
        ):
            out_path = traj_dir / f"batch_{batch_i:05d}.parquet"
            if out_path.exists():
                continue

            df_out = generate_trajectories_for_batch(
                batch=batch,
                cehrgpt_tokenizer=cehrgpt_tokenizer,
                cehrgpt_model=cehrgpt_model,
                device=device,
                arm=arm,
                trajectory_id=traj_id,
                max_length=max_length,
            )
            df_out.write_parquet(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate counterfactual (non-treated) and treated trajectories"
    )
    parser.add_argument(
        "--context_dir",
        required=True,
        help="Directory containing non_treated_context.parquet and treated_context.parquet "
             "(output of extract_drug_initiation_sequences.py)",
    )
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
        "--arms",
        default="non_treated,treated",
        help="Comma-separated list of arms to generate; choices: non_treated, treated "
             "(default: non_treated,treated)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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

    context_dir = Path(args.context_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arms_to_run = [a.strip() for a in args.arms.split(",")]

    arm_to_file = {
        NON_TREATED_ARM: context_dir / "non_treated_context.parquet",
        TREATED_ARM: context_dir / "treated_context.parquet",
    }

    for arm in arms_to_run:
        if arm not in arm_to_file:
            raise ValueError(f"Unknown arm '{arm}'. Choose from: {list(arm_to_file)}")
        run_generation(
            context_parquet=str(arm_to_file[arm]),
            arm=arm,
            cehrgpt_tokenizer=tokenizer,
            cehrgpt_model=model,
            device=device,
            output_dir=output_dir,
            num_trajectories=args.num_trajectories,
            batch_size=args.batch_size,
            max_length=max_length,
            include_values=include_values,
            num_workers=args.num_workers,
        )

    print("\nAll generation complete.")


if __name__ == "__main__":
    main()
