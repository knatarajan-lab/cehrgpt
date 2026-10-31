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

        if not torch.equal(legacy.sequences, optimized.sequences):
            mismatch = (legacy.sequences != optimized.sequences).nonzero()[0].tolist()
            raise AssertionError(
                f"Prompt {prompt_index} generated-token mismatch at {mismatch}"
            )
        if len(legacy.logits) != len(optimized.logits):
            raise AssertionError(f"Prompt {prompt_index} generated a different step count")
        for legacy_logits, optimized_logits in zip(legacy.logits, optimized.logits):
            difference = (legacy_logits - optimized_logits).abs().max().item()
            max_logit_difference = max(max_logit_difference, difference)
            torch.testing.assert_close(
                optimized_logits,
                legacy_logits,
                rtol=0,
                atol=args.logit_atol,
            )
        if model.config.include_values:
            if not torch.equal(legacy.sequence_val_masks, optimized.sequence_val_masks):
                raise AssertionError(f"Prompt {prompt_index} value masks differ")
            if not torch.equal(legacy.sequence_vals, optimized.sequence_vals):
                raise AssertionError(f"Prompt {prompt_index} sampled values differ")
        print(
            f"PASS prompt={prompt_index} encoded_tokens={len(token_ids)} "
            f"sequences={args.num_return_sequences} generated_tokens={args.max_new_tokens}"
        )

    print(
        f"PASS all prompts={len(prefixes)}; sampled sequences are identical; "
        f"max_abs_logit_difference={max_logit_difference:.8g}; "
        f"tolerance={args.logit_atol}"
    )


if __name__ == "__main__":
    main()
