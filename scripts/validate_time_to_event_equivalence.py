#!/usr/bin/env python
"""Differentially validate legacy and optimized time-to-event generation."""

import argparse
import copy
import glob
import os
import time
from typing import Optional

import pandas as pd
import torch
from transformers import StoppingCriteriaList
from transformers.utils import is_flash_attn_2_available

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import read_backbone, resolve_attn_implementation
from cehrgpt.time_to_event.time_to_event_model import (
    TimeToEventModel,
    TimeToEventStoppingCriteria,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True)
    parser.add_argument(
        "--tokenizer_folder",
        default=None,
        help="The folder of the tokenizer, when it is not saved with the model.",
    )
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
            "static_cache",
            "fast_decoding",
            "determinism",
        ),
        default="combined",
        help="prompt_cache compares a generation that encodes the prompt against one that "
        "starts from what the first computed from the prompt, like the next batches of samples "
        "of a patient do. "
        "unpadded_mask compares flash attention with the attention mask of a prompt without "
        "padding kept (un-padded keys and values) and dropped. "
        "static_cache compares a KV cache that grows at every step with one appended to in "
        "place. fast_decoding compares the trajectories of a generation with the ones of the "
        "generation time-to-event prediction runs: the cache appended to in place, and the "
        "trajectories that passed --prediction_window_end dropped from the batch. "
        "determinism runs the model twice on every prompt without reseeding and fails when the "
        "logits differ, which they do when dropout is applied in evaluation.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "flash_attention_2"),
        default=None,
        help="The attention of the model, when it is not the one it runs with by default. "
        "eager without --use_bfloat16 computes in float32, where two paths that compute the "
        "same thing agree to about 1e-5: a difference that stays there is a bug, one that "
        "is only there with bfloat16 flash attention is its rounding.",
    )
    parser.add_argument(
        "--prediction_window_end",
        type=int,
        default=30,
        help="With --comparison fast_decoding: the days after which a trajectory is finished.",
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
    static_kv_cache: bool = False,
    drop_finished_sequences: bool = False,
    stopping_criteria=None,
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            **({"prompt_prefill": prompt_prefill} if prompt_prefill is not None else {}),
            **({"return_prompt_prefill": True} if return_prompt_prefill else {}),
            **({"static_kv_cache": True} if static_kv_cache else {}),
            **({"drop_finished_sequences": True} if drop_finished_sequences else {}),
            **(
                {"stopping_criteria": StoppingCriteriaList([stopping_criteria])}
                if stopping_criteria is not None
                else {}
            ),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Not a field of the output: how long this generation took, for the report
    outputs.__dict__["seconds"] = time.perf_counter() - start
    return outputs


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


def count_rows(model: CEHRGPT2LMHeadModel, run) -> tuple:
    """The result of run(), and the number of sequences of every forward pass."""
    rows = []
    handle = model.cehrgpt.register_forward_pre_hook(
        lambda _module, args, kwargs: rows.append(
            (args[0] if args else kwargs["input_ids"]).shape[0]
        ),
        with_kwargs=True,
    )
    try:
        return run(), rows
    finally:
        handle.remove()


def check_row_independent_sampling(device, vocab_size: int) -> bool:
    """
    Whether torch.multinomial gives a row the same sample whatever the other rows are.

    Dropping the finished trajectories relies on it: the tokens are sampled for the whole
    batch, with a dummy distribution for the dropped rows, so that the rows that are left draw
    the random numbers they would draw with the whole batch.
    """
    generator_seed = 20240
    torch.manual_seed(0)
    probabilities = torch.rand(16, vocab_size, device=device)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    kept = torch.tensor([1, 4, 5, 9, 15], device=device)
    dummy = torch.zeros_like(probabilities)
    dummy[:, 0] = 1.0
    dummy[kept] = probabilities[kept]
    mismatches = 0
    for trial in range(100):
        samples = []
        for rows in (probabilities, dummy):
            torch.manual_seed(generator_seed + trial)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generator_seed + trial)
            samples.append(torch.multinomial(rows, num_samples=1).squeeze(1)[kept])
        mismatches += (samples[0] != samples[1]).sum().item()
    print(
        f"sampling per row: {'independent' if mismatches == 0 else 'NOT independent'} of "
        f"the other rows ({mismatches}/500 samples differ)"
    )
    return mismatches == 0


class StepLogits:
    """The logits of every decoding step of a generation, and the rows they are of."""

    def __init__(self, model):
        self.model = model
        self.logits, self.rows = [], []

    def __enter__(self):
        self._handle = self.model.lm_head.register_forward_hook(
            lambda _module, _args, output: self.logits.append(output[:, -1].float().cpu())
        )
        scatter = CEHRGPT2LMHeadModel._scatter_active_rows
        vocab_size = self.model.config.vocab_size

        def record_rows(probabilities, active_rows, batch_size):
            # Called for the tokens, then for the values of a model with values
            if probabilities.shape[-1] == vocab_size:
                self.rows.append(None if active_rows is None else active_rows.cpu())
            return scatter(probabilities, active_rows, batch_size)

        CEHRGPT2LMHeadModel._scatter_active_rows = staticmethod(record_rows)
        self._scatter = scatter
        return self

    def __exit__(self, *args):
        CEHRGPT2LMHeadModel._scatter_active_rows = staticmethod(self._scatter)
        self._handle.remove()


def diagnose_fast_decoding(args, prompt_index, legacy, fast, legacy_steps, fast_steps, prompt_length):
    """
    Where the trajectories of the two generations part, and by how much the logits differ.

    A trajectory is compared for as long as it has the same tokens on both sides: after that
    the two sides continue different trajectories, which says nothing about the model. The
    logits of the rows the model ran on with fast decoding are compared with the logits of
    the same rows of the whole batch.
    """
    num_sequences = legacy.sequences.shape[0]
    same_prefix = torch.ones(num_sequences, dtype=torch.bool)
    legacy_tokens = legacy.sequences[:, prompt_length:].cpu()
    fast_tokens = fast.sequences[:, prompt_length:].cpu()
    # A model with values also reads the value sampled for every lab: two trajectories with
    # the same tokens and another value are other trajectories
    legacy_values = fast_values = None
    if legacy.sequence_vals is not None:
        legacy_values = legacy.sequence_vals[:, prompt_length:].cpu()
        fast_values = fast.sequence_vals[:, prompt_length:].cpu()
    max_difference, max_tv, compared_steps = 0.0, 0.0, 0
    first_difference = None
    for step in range(min(len(legacy_steps.logits), len(fast_steps.logits))):
        rows = fast_steps.rows[step]
        active = torch.arange(num_sequences) if rows is None else rows
        legacy_logits = legacy_steps.logits[step]
        fast_logits = fast_steps.logits[step]
        # The prefill of a shared prompt has one row for all the sequences
        if legacy_logits.shape[0] == 1:
            legacy_logits = legacy_logits.expand(num_sequences, -1)
        if fast_logits.shape[0] == 1 and len(active) > 1:
            fast_logits = fast_logits.expand(len(active), -1)
        comparable = same_prefix[active]
        if comparable.any():
            expected = legacy_logits[active][comparable]
            actual = fast_logits[comparable]
            difference = (expected - actual).abs().max().item()
            tv = (
                0.5
                * (torch.softmax(expected, -1) - torch.softmax(actual, -1))
                .abs()
                .sum(-1)
                .max()
                .item()
            )
            compared_steps += 1
            if difference > 0 and first_difference is None:
                first_difference = (step, len(active), difference, tv)
            max_difference, max_tv = max(max_difference, difference), max(max_tv, tv)
        if step < min(legacy_tokens.shape[1], fast_tokens.shape[1]):
            same_prefix &= legacy_tokens[:, step] == fast_tokens[:, step]
            if legacy_values is not None:
                # The value of a dropped row is not sampled, and that row is not compared
                still_active = torch.zeros(num_sequences, dtype=torch.bool)
                still_active[active] = True
                same_prefix &= ~still_active | (
                    legacy_values[:, step] == fast_values[:, step]
                )
    within = max_difference <= args.logit_atol and max_tv <= args.probability_tv_atol
    if first_difference is None:
        where = "the logits never differ on the same trajectory"
    else:
        step, active_rows, difference, tv = first_difference
        where = (
            f"the logits first differ at step {step}, with {active_rows} of "
            f"{num_sequences} trajectories still running (logit difference "
            f"{difference:.4g}, probability TV {tv:.4g})"
        )
    print(
        f"  prompt={prompt_index} diagnosis: {where}; over {compared_steps} steps "
        f"max_abs_logit_difference={max_difference:.4g} (tolerance {args.logit_atol}) "
        f"max_probability_tv={max_tv:.4g} (tolerance {args.probability_tv_atol}); "
        f"identical_trajectories={int(same_prefix.sum())}/{num_sequences}"
    )
    return within, max_difference, max_tv


def check_fast_decoding(args, model, tokenizer, prefixes, generation_config, device):
    """
    Compare fast decoding with the generation without it, sampled with the same seed.

    Both sides stop a trajectory after --prediction_window_end days. When the model computes
    the same numbers on a batch of fewer rows, the trajectories are the same token for token.
    It does not always: a batch of another size can round differently (bfloat16), and a logit
    one rounding step away now and then moves a sample, after which that trajectory is
    another one. So when the trajectories differ, both sides are generated again with their
    logits recorded, and the comparison passes if the logits of the same trajectory stay
    within the tolerances: the two sides then sample from the same distributions.
    """
    if not model._can_use_static_kv_cache(
        {
            "use_cache": True,
            "attention_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        },
        False,
    ):
        print(
            "This model does not use the static KV cache (it is for the gpt2 backbone with "
            "flash attention): only dropping the finished trajectories is compared"
        )
    check_row_independent_sampling(device, model.config.vocab_size)
    pad_token_id = generation_config.pad_token_id
    total_tokens = 0
    mismatched_tokens = 0
    max_logit_difference = 0.0
    max_probability_tv = 0.0
    failed = False

    def run(prompt, seed, fast):
        return count_rows(
            model,
            lambda: generate(
                model,
                tokenizer,
                prompt,
                generation_config,
                share_prompt_prefill=None,
                track_ages=None,
                seed=seed,
                static_kv_cache=fast,
                drop_finished_sequences=fast,
                stopping_criteria=TimeToEventStoppingCriteria(
                    tokenizer, [], 0, -1, 0, args.prediction_window_end
                ),
            ),
        )

    for prompt_index, prefix in enumerate(prefixes):
        token_ids = tokenizer.encode(prefix)
        prompt = torch.tensor([token_ids], dtype=torch.long, device=device)
        seed = args.seed + prompt_index
        (legacy, legacy_rows), (fast, fast_rows) = run(prompt, seed, False), run(
            prompt, seed, True
        )
        steps = min(legacy.sequences.shape[1], fast.sequences.shape[1]) - len(token_ids)
        generated = legacy.sequences[:, len(token_ids) : len(token_ids) + steps]
        fast_generated = fast.sequences[:, len(token_ids) : len(token_ids) + steps]
        mismatches = generated != fast_generated
        if model.config.include_values:
            # The padding of a finished trajectory has no value
            mismatches |= (
                legacy.sequence_vals[:, len(token_ids) : len(token_ids) + steps]
                != fast.sequence_vals[:, len(token_ids) : len(token_ids) + steps]
            ) & (generated != pad_token_id)
        same = not mismatches.any() and legacy.sequences.shape == fast.sequences.shape
        total_tokens += mismatches.numel()
        mismatched_tokens += mismatches.sum().item()
        print(
            f"prompt={prompt_index} encoded_tokens={len(token_ids)} "
            f"sequences={generated.shape[0]} "
            f"generated_tokens={legacy.sequences.shape[1] - len(token_ids)}->"
            f"{fast.sequences.shape[1] - len(token_ids)} "
            f"mean_rows_per_step={sum(legacy_rows[1:]) / max(len(legacy_rows) - 1, 1):.1f}->"
            f"{sum(fast_rows[1:]) / max(len(fast_rows) - 1, 1):.1f} "
            f"seconds={legacy.seconds:.3f}->{fast.seconds:.3f} "
            f"trajectories={'identical' if same else 'differ'} "
            f"mismatched_tokens={mismatches.sum().item()}"
        )
        if not same:
            # Again, with the logits: the same seeds give the same generations
            with StepLogits(model) as legacy_steps:
                legacy, _ = run(prompt, seed, False)
            with StepLogits(model) as fast_steps:
                fast, _ = run(prompt, seed, True)
            within, difference, tv = diagnose_fast_decoding(
                args, prompt_index, legacy, fast, legacy_steps, fast_steps, len(token_ids)
            )
            max_logit_difference = max(max_logit_difference, difference)
            max_probability_tv = max(max_probability_tv, tv)
            failed |= not within
    print(
        f"{'FAIL' if failed else 'PASS'} comparison=fast_decoding; prompts={len(prefixes)}; "
        f"prediction_window_end={args.prediction_window_end}; "
        f"mismatched_tokens={mismatched_tokens}/{total_tokens}; "
        f"max_abs_logit_difference={max_logit_difference:.8g}; "
        f"logit_tolerance={args.logit_atol}; "
        f"max_probability_tv={max_probability_tv:.8g}; "
        f"probability_tv_tolerance={args.probability_tv_atol}"
    )
    if failed:
        raise AssertionError(
            "Fast decoding computes other logits than the generation without it, beyond the "
            "tolerances"
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CehrGptTokenizer.from_pretrained(
        args.tokenizer_folder or args.model_folder
    )
    model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_folder,
            attn_implementation=resolve_attn_implementation(
                backbone=read_backbone(args.model_folder),
                requested=args.attn_implementation,
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
    # The comparisons that switch the age tracking off, which a rotary model needs
    if model.config.apply_rotary and args.comparison in (
        "combined",
        "shared_prefill",
        "age_tracking",
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
    if args.comparison == "fast_decoding":
        generation_config.output_logits = False
        check_fast_decoding(
            args, model, tokenizer, prefixes, generation_config, device
        )
        return
    comparison_paths = {
        "combined": ((False, True), (True, False)),
        "shared_prefill": ((False, False), (True, False)),
        "age_tracking": ((False, True), (False, False)),
        # Both sides run the production paths, the optimized one from the prompt's KV cache
        "prompt_cache": ((None, None), (None, None)),
        # Both sides run the production paths, the legacy one keeps the attention mask
        "unpadded_mask": ((None, None), (None, None)),
        # Both sides run the production paths, the optimized one with the static KV cache
        "static_cache": ((None, None), (None, None)),
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
                static_kv_cache=args.comparison == "static_cache",
            ),
        )
        if args.comparison == "static_cache" and optimized.past_key_values is not None:
            # A generation with the static cache does not return its cache
            raise RuntimeError(
                "This model does not use the static KV cache, which is for the gpt2 backbone "
                "with flash attention and prompts without padding"
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
            # Both are reported when either fails: a logit can be a rounding step of bfloat16
            # away (0.03 for a logit between 4 and 8) without the probabilities moving
            if difference > args.logit_atol or probability_tv > args.probability_tv_atol:
                mismatched = (
                    (comparable_legacy - comparable_optimized).abs() > args.logit_atol
                ).float().mean().item()
                raise AssertionError(
                    f"Prompt {prompt_index}, step {step}: max_abs_logit_difference="
                    f"{difference:.8g} (tolerance {args.logit_atol}, "
                    f"{100 * mismatched:.2f}% of the logits above it); "
                    f"max_probability_tv={probability_tv:.8g} "
                    f"(tolerance {args.probability_tv_atol})"
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
            f"seconds={legacy.seconds:.3f}->{optimized.seconds:.3f} "
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
