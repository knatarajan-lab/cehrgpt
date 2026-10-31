#!/usr/bin/env python
"""Differentially validate legacy and optimized time-to-event generation."""

import argparse
import glob
import os

import pandas as pd
import torch
from transformers.utils import is_flash_attn_2_available

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import read_backbone, resolve_attn_implementation
from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True)
    parser.add_argument("--prediction_folder", required=True)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--num_return_sequences", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--use_bfloat16", action="store_true")
    parser.add_argument("--logit_atol", type=float, default=2e-2)
    parser.add_argument("--probability_tv_atol", type=float, default=1e-3)
    return parser.parse_args()


def load_prefixes(folder: str, count: int) -> list[list[str]]:
    prefixes = []
    parquet_files = sorted(
        glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
    )
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=["prefix"])
        prefixes.extend(frame["prefix"].tolist())
        if len(prefixes) >= count:
            return prefixes[:count]
    raise RuntimeError(f"Found only {len(prefixes)} prefixes under {folder}, need {count}")


def generate(
    model: CEHRGPT2LMHeadModel,
    tokenizer: CehrGptTokenizer,
    prompt: torch.Tensor,
    generation_config,
    *,
    share_prompt_prefill: bool,
    track_ages: bool,
    seed: int,
):
    model._generation_share_prompt_prefill = share_prompt_prefill
    model._generation_track_ages_override = track_ages
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        return model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
        )


def main() -> None:
    args = parse_args()
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
    if model.config.apply_rotary:
        raise RuntimeError(
            "This legacy-age-vs-skipped-age comparison requires apply_rotary=false"
        )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.end_token_id
    model.generation_config.bos_token_id = tokenizer.end_token_id

    generation_config = TimeToEventModel.get_generation_config(
        tokenizer=tokenizer,
        max_length=model.config.n_positions,
        num_return_sequences=args.num_return_sequences,
        top_p=1.0,
        max_new_tokens=args.max_new_tokens,
    )
    generation_config.output_logits = True
    prefixes = load_prefixes(args.prediction_folder, args.num_prompts)
    max_logit_difference = 0.0
    max_probability_tv = 0.0
    total_sampled_tokens = 0
    mismatched_sampled_tokens = 0
    identical_trajectories = 0

    for prompt_index, prefix in enumerate(prefixes):
        token_ids = tokenizer.encode(prefix)
        max_prompt_length = model.config.n_positions - args.max_new_tokens
        if len(token_ids) > max_prompt_length:
            raise RuntimeError(
                f"Prompt {prompt_index} has {len(token_ids)} encoded tokens, "
                f"exceeding the {max_prompt_length} token budget"
            )
        prompt = torch.tensor([token_ids], dtype=torch.long, device=device)
        seed = args.seed + prompt_index
        legacy = generate(
            model,
            tokenizer,
            prompt,
            generation_config,
            share_prompt_prefill=False,
            track_ages=True,
            seed=seed,
        )
        optimized = generate(
            model,
            tokenizer,
            prompt,
            generation_config,
            share_prompt_prefill=True,
            track_ages=False,
            seed=seed,
        )

        if len(legacy.logits) != len(optimized.logits):
            raise AssertionError(f"Prompt {prompt_index} generated a different step count")
        identical_prefix = torch.ones(
            args.num_return_sequences, dtype=torch.bool, device=device
        )
        for step, (legacy_logits, optimized_logits) in enumerate(
            zip(legacy.logits, optimized.logits)
        ):
            if not identical_prefix.any():
                break
            comparable_legacy = legacy_logits[identical_prefix]
            comparable_optimized = optimized_logits[identical_prefix]
            difference = (
                (comparable_legacy - comparable_optimized).abs().max().item()
            )
            max_logit_difference = max(max_logit_difference, difference)
            torch.testing.assert_close(
                comparable_optimized,
                comparable_legacy,
                rtol=0,
                atol=args.logit_atol,
            )
            legacy_probabilities = torch.softmax(comparable_legacy.float(), dim=-1)
            optimized_probabilities = torch.softmax(comparable_optimized.float(), dim=-1)
            probability_tv = (
                0.5
                * (legacy_probabilities - optimized_probabilities)
                .abs()
                .sum(dim=-1)
                .max()
                .item()
            )
            max_probability_tv = max(max_probability_tv, probability_tv)
            if probability_tv > args.probability_tv_atol:
                raise AssertionError(
                    f"Prompt {prompt_index}, step {step} probability TV "
                    f"{probability_tv} exceeds {args.probability_tv_atol}"
                )

            position = len(token_ids) + step
            same_step = legacy.sequences[:, position] == optimized.sequences[:, position]
            if model.config.include_values:
                legacy_value_mask = legacy.sequence_val_masks[:, position]
                optimized_value_mask = optimized.sequence_val_masks[:, position]
                same_value = legacy_value_mask == optimized_value_mask
                same_value &= (
                    ~legacy_value_mask
                    | (
                        legacy.sequence_vals[:, position]
                        == optimized.sequence_vals[:, position]
                    )
                )
                same_step &= same_value
            identical_prefix &= same_step

        legacy_generated = legacy.sequences[:, len(token_ids) :]
        optimized_generated = optimized.sequences[:, len(token_ids) :]
        token_matches = legacy_generated == optimized_generated
        total_sampled_tokens += token_matches.numel()
        mismatched_sampled_tokens += (~token_matches).sum().item()
        identical_trajectories += token_matches.all(dim=1).sum().item()
        print(
            f"PASS prompt={prompt_index} encoded_tokens={len(token_ids)} "
            f"sequences={args.num_return_sequences} generated_tokens={args.max_new_tokens} "
            f"identical_trajectories={token_matches.all(dim=1).sum().item()}"
        )

    print(
        f"PASS all prompts={len(prefixes)}; "
        f"max_abs_logit_difference={max_logit_difference:.8g}; "
        f"logit_tolerance={args.logit_atol}; "
        f"max_probability_tv={max_probability_tv:.8g}; "
        f"probability_tv_tolerance={args.probability_tv_atol}; "
        f"identical_trajectories={identical_trajectories}/"
        f"{len(prefixes) * args.num_return_sequences}; "
        f"sampled_token_mismatches={mismatched_sampled_tokens}/{total_sampled_tokens}"
    )


if __name__ == "__main__":
    main()
