#!/usr/bin/env python
"""Differentially validate legacy and optimized time-to-event generation."""

import argparse
import copy
import glob
import os
from typing import Optional

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
    parser.add_argument(
        "--comparison",
        choices=(
            "combined",
            "shared_prefill",
            "age_tracking",
            "prompt_cache",
            "unpadded_mask",
            "determinism",
        ),
        default="combined",
        help="prompt_cache compares a generation that encodes the prompt against one that "
        "starts from what the first computed from the prompt, like the next batches of samples "
        "of a patient do. "
        "unpadded_mask compares flash attention with the attention mask of a prompt without "
        "padding kept (un-padded keys and values) and dropped. "
        "determinism runs the model twice on every prompt without reseeding and fails when the "
        "logits differ, which they do when dropout is applied in evaluation.",
    )
    parser.add_argument(
        "--reuse_num_return_sequences",
        type=int,
        default=None,
        help="With --comparison prompt_cache: the number of sequences of the generation that "
        "starts from the prompt of the first one, when it is not --num_return_sequences. The "
        "last batch of samples of a patient, or a trial again, is smaller than the first batch "
        "whose prompt it reuses. Both sides then decode greedily, the samples of batches of "
        "different sizes are not comparable.",
    )
    args = parser.parse_args()
    if args.reuse_num_return_sequences is not None and args.comparison != "prompt_cache":
        parser.error("--reuse_num_return_sequences needs --comparison prompt_cache")
    return args


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
    share_prompt_prefill: Optional[bool],
    track_ages: Optional[bool],
    seed: int,
    prompt_prefill=None,
    return_prompt_prefill: bool = False,
    drop_unpadded_mask: bool = True,
    greedy_rows: Optional[int] = None,
):
    """
    Generate from a prompt along one of the paths to compare.

    greedy_rows generates that many sequences greedily instead of sampling
    num_return_sequences of them.
    """
    if greedy_rows is not None:
        # Greedy decoding does not take num_return_sequences, the prompt is repeated instead
        prompt = prompt.expand(greedy_rows, -1)
        generation_config = copy.deepcopy(generation_config)
        generation_config.do_sample = False
        generation_config.num_return_sequences = 1
    model.cehrgpt._drop_unpadded_flash_attention_mask = drop_unpadded_mask
    if share_prompt_prefill is None:
        # The production behavior: the model decides, it does not share under flash attention
        model._generation_share_prompt_prefill = True
        model.__dict__.pop("_can_share_generation_prompt_prefill", None)
    else:
        model._generation_share_prompt_prefill = share_prompt_prefill
        # The model never shares the prefill under flash attention, where both sides of a
        # comparison would then run the same path and trivially match. The comparison is
        # there to find out whether sharing is equivalent, so it decides, not the attention
        # implementation.
        model._can_share_generation_prompt_prefill = lambda: share_prompt_prefill
    model._generation_track_ages_override = track_ages
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        return model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            **({"prompt_prefill": prompt_prefill} if prompt_prefill is not None else {}),
            **({"return_prompt_prefill": True} if return_prompt_prefill else {}),
        )


def count_prefill_rows(model: CEHRGPT2LMHeadModel, run) -> tuple:
    """The result of run(), and the batch size of every forward pass over a whole prompt."""
    prefill_rows = []

    def record(_module, args, kwargs):
        input_ids = args[0] if args else kwargs["input_ids"]
        if kwargs.get("past_key_values") is None and input_ids.shape[1] > 1:
            prefill_rows.append(input_ids.shape[0])

    handle = model.cehrgpt.register_forward_pre_hook(record, with_kwargs=True)
    try:
        return run(), prefill_rows
    finally:
        handle.remove()


def check_determinism(model, tokenizer, prefixes, device) -> None:
    """Fail when two forward passes over the same prompt, without reseeding, differ."""
    max_logit_difference = 0.0
    for prompt_index, prefix in enumerate(prefixes):
        input_ids = torch.tensor(
            [tokenizer.encode(prefix)], dtype=torch.long, device=device
        )
        logits = []
        with torch.inference_mode():
            for _ in range(2):
                logits.append(
                    model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                        ages=torch.zeros_like(input_ids),
                        output_motor_features=False,
                        return_dict=True,
                    ).logits.float()
                )
        difference = (logits[0] - logits[1]).abs().max().item()
        max_logit_difference = max(max_logit_difference, difference)
        print(
            f"prompt={prompt_index} encoded_tokens={input_ids.shape[1]} "
            f"max_abs_logit_difference={difference:.8g}"
        )
    status = "PASS" if max_logit_difference == 0.0 else "FAIL"
    print(
        f"{status} comparison=determinism; prompts={len(prefixes)}; "
        f"attn_pdrop={model.config.attn_pdrop}; "
        f"max_abs_logit_difference={max_logit_difference:.8g}"
    )
    if max_logit_difference != 0.0:
        raise AssertionError(
            "Two forward passes over the same prompt differ in evaluation: something random "
            "such as dropout is applied"
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
    if model.config.apply_rotary and args.comparison not in (
        "prompt_cache",
        "determinism",
    ):
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
    print(
        f"attn_implementation={getattr(model.config, '_attn_implementation', 'eager')}; "
        f"dtype={model.dtype}; attn_pdrop={model.config.attn_pdrop}"
    )
    if args.comparison == "determinism":
        check_determinism(model, tokenizer, prefixes, device)
        return
    comparison_paths = {
        "combined": ((False, True), (True, False)),
        "shared_prefill": ((False, False), (True, False)),
        "age_tracking": ((False, True), (False, False)),
        # Both sides run the production paths, the optimized one from the prompt's KV cache
        "prompt_cache": ((None, None), (None, None)),
        # Both sides run the production paths, the legacy one keeps the attention mask
        "unpadded_mask": ((None, None), (None, None)),
    }
    if args.comparison == "unpadded_mask" and (
        getattr(model.config, "_attn_implementation", "eager") != "flash_attention_2"
    ):
        raise RuntimeError(
            "unpadded_mask compares two paths of flash attention, which this model does not use"
        )
    legacy_rows = optimized_rows = None
    if args.reuse_num_return_sequences is not None:
        legacy_rows = args.num_return_sequences
        optimized_rows = args.reuse_num_return_sequences
    compared_rows = min(
        args.num_return_sequences,
        args.reuse_num_return_sequences or args.num_return_sequences,
    )
    (legacy_share_prefill, legacy_track_ages), (
        optimized_share_prefill,
        optimized_track_ages,
    ) = comparison_paths[args.comparison]
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
        legacy, legacy_prefill_rows = count_prefill_rows(
            model,
            lambda: generate(
                model,
                tokenizer,
                prompt,
                generation_config,
                share_prompt_prefill=legacy_share_prefill,
                track_ages=legacy_track_ages,
                seed=seed,
                return_prompt_prefill=args.comparison == "prompt_cache",
                drop_unpadded_mask=args.comparison != "unpadded_mask",
                greedy_rows=legacy_rows,
            ),
        )
        # What TimeToEventModel keeps from the first batch of samples of a patient for the
        # next ones: the logits of the first new token and the KV cache of the prompt
        prompt_prefill = legacy.prompt_prefill if args.comparison == "prompt_cache" else None
        optimized, optimized_prefill_rows = count_prefill_rows(
            model,
            lambda: generate(
                model,
                tokenizer,
                prompt,
                generation_config,
                share_prompt_prefill=optimized_share_prefill,
                track_ages=optimized_track_ages,
                seed=seed,
                prompt_prefill=prompt_prefill,
                greedy_rows=optimized_rows,
            ),
        )
        if (
            args.comparison in ("combined", "shared_prefill", "prompt_cache")
            and legacy_prefill_rows == optimized_prefill_rows
        ):
            # Both sides encoded the prompt the same way: the comparison would pass without
            # having compared anything
            raise AssertionError(
                f"Prompt {prompt_index}: both sides encoded the prompt with batch sizes "
                f"{legacy_prefill_rows}, the two paths of {args.comparison} are the same path"
            )

        if len(legacy.logits) != len(optimized.logits):
            raise AssertionError(f"Prompt {prompt_index} generated a different step count")
        # The sequences both sides have: the reusing side can have fewer or more
        legacy_sequences = legacy.sequences[:compared_rows]
        optimized_sequences = optimized.sequences[:compared_rows]
        identical_prefix = torch.ones(compared_rows, dtype=torch.bool, device=device)
        for step, (legacy_logits, optimized_logits) in enumerate(
            zip(legacy.logits, optimized.logits)
        ):
            if not identical_prefix.any():
                break
            comparable_legacy = legacy_logits[:compared_rows][identical_prefix]
            comparable_optimized = optimized_logits[:compared_rows][identical_prefix]
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
            same_step = legacy_sequences[:, position] == optimized_sequences[:, position]
            if model.config.include_values:
                legacy_value_mask = legacy.sequence_val_masks[:compared_rows, position]
                optimized_value_mask = optimized.sequence_val_masks[
                    :compared_rows, position
                ]
                same_value = legacy_value_mask == optimized_value_mask
                same_value &= (
                    ~legacy_value_mask
                    | (
                        legacy.sequence_vals[:compared_rows, position]
                        == optimized.sequence_vals[:compared_rows, position]
                    )
                )
                same_step &= same_value
            identical_prefix &= same_step

        legacy_generated = legacy_sequences[:, len(token_ids) :]
        optimized_generated = optimized_sequences[:, len(token_ids) :]
        token_matches = legacy_generated == optimized_generated
        total_sampled_tokens += token_matches.numel()
        mismatched_sampled_tokens += (~token_matches).sum().item()
        identical_trajectories += token_matches.all(dim=1).sum().item()
        print(
            f"PASS prompt={prompt_index} encoded_tokens={len(token_ids)} "
            f"sequences={compared_rows} generated_tokens={args.max_new_tokens} "
            f"prefill_rows={legacy_prefill_rows}->{optimized_prefill_rows} "
            f"identical_trajectories={token_matches.all(dim=1).sum().item()}"
        )

    print(
        f"PASS comparison={args.comparison}; prompts={len(prefixes)}; "
        f"max_abs_logit_difference={max_logit_difference:.8g}; "
        f"logit_tolerance={args.logit_atol}; "
        f"max_probability_tv={max_probability_tv:.8g}; "
        f"probability_tv_tolerance={args.probability_tv_atol}; "
        f"identical_trajectories={identical_trajectories}/"
        f"{len(prefixes) * compared_rows}; "
        f"sampled_token_mismatches={mismatched_sampled_tokens}/{total_sampled_tokens}"
    )


if __name__ == "__main__":
    main()
