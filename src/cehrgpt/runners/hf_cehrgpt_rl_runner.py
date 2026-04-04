"""
Entry point for CEHR-GPT RL fine-tuning (REINFORCE + KL).

Usage
-----
python -m cehrgpt.runners.hf_cehrgpt_rl_runner \\
    --model_name_or_path /path/to/pretrained \\
    --tokenizer_name_or_path /path/to/tokenizer \\
    --tokenized_full_dataset_path /path/to/tokenized_dataset \\
    --output_dir /path/to/rl_output \\
    --num_rollouts 4 \\
    --kl_beta 0.05 \\
    [... standard TrainingArguments ...]
"""

import copy
import json
import os
from pathlib import Path
from typing import Optional, Set

import torch
from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
from datasets import load_from_disk
from transformers import TrainingArguments, set_seed
from transformers.utils import logging

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.rl.grpo_trainer import CehrGptGRPOTrainer
from cehrgpt.rl.reward import compute_prevalence_stats
from cehrgpt.rl.rl_data_collator import RLDataCollator
from cehrgpt.runners.hf_gpt_rl_runner_argument_dataclass import RLArguments
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

_CONDITION_DOMAIN = "Condition"

LOG = logging.get_logger("transformers")


def _load_target_concept_ids(
    cehrgpt_args: CehrGPTArguments,
    cehrgpt_tokenizer: CehrGptTokenizer,
) -> Set[str]:
    """
    Identify condition concept IDs for RL reward targets.

    If ``vocab_dir`` is provided and the Athena parquet files exist there, load
    the OMOP ontology and return all concept IDs in the tokenizer vocabulary whose
    domain is "Condition".  Otherwise fall back to the tokenizer's curated
    ``_motor_time_to_event_codes`` list.
    """
    vocab_dir = cehrgpt_args.vocab_dir
    if vocab_dir:
        vocab_dir = os.path.expanduser(vocab_dir)
    concept_parquet_dir = os.path.join(vocab_dir, "concept") if vocab_dir else None

    if concept_parquet_dir and os.path.isdir(concept_parquet_dir):
        try:
            from cehrgpt.omop.ontology import Ontology

            LOG.info("Loading OMOP ontology from %s", vocab_dir)
            ontology = Ontology(vocab_dir)
            # All token strings in the tokenizer vocabulary
            all_vocab_tokens: Set[str] = set(cehrgpt_tokenizer.get_vocab().keys())
            condition_ids = {
                tok for tok in all_vocab_tokens
                if tok.isnumeric() and ontology.get_domain(tok) == _CONDITION_DOMAIN
            }
            if condition_ids:
                LOG.info(
                    "Using %d ontology-derived Condition concept IDs as RL targets.",
                    len(condition_ids),
                )
                return condition_ids
            LOG.warning(
                "Ontology loaded but no Condition concepts found in tokenizer vocab; "
                "falling back to _motor_time_to_event_codes."
            )
        except Exception as exc:
            LOG.warning("Failed to load ontology from %s (%s); falling back.", vocab_dir, exc)

    fallback = set(cehrgpt_tokenizer._motor_time_to_event_codes)
    LOG.info(
        "Using %d _motor_time_to_event_codes as RL targets.", len(fallback)
    )
    return fallback


def _load_prevalence_stats(
    rl_args: RLArguments,
    train_dataset,
    target_concept_ids: Set[str],
) -> dict:
    """Load prevalence stats from disk or compute them from the training split."""
    if rl_args.prevalence_stats_path and os.path.exists(rl_args.prevalence_stats_path):
        LOG.info("Loading prevalence stats from %s", rl_args.prevalence_stats_path)
        with open(rl_args.prevalence_stats_path) as f:
            raw = json.load(f)
        # JSON keys are strings; convert back to (concept_id, window) tuples
        return {
            (entry["concept_id"], entry["window"]): entry["prevalence"]
            for entry in raw
        }

    LOG.info(
        "Computing prevalence stats over training split (%d examples)...",
        len(train_dataset),
    )
    stats = compute_prevalence_stats(
        dataset=train_dataset,
        target_concept_ids=target_concept_ids,
        windows=rl_args.prediction_windows,
        num_proc=rl_args.prevalence_num_proc,
        batch_size=rl_args.prevalence_batch_size,
    )
    # Optionally persist for reuse
    if rl_args.prevalence_stats_path:
        os.makedirs(os.path.dirname(rl_args.prevalence_stats_path), exist_ok=True)
        with open(rl_args.prevalence_stats_path, "w") as f:
            json.dump(
                [
                    {"concept_id": cid, "window": w, "prevalence": p}
                    for (cid, w), p in stats.items()
                ],
                f,
                indent=2,
            )
        LOG.info("Saved prevalence stats to %s", rl_args.prevalence_stats_path)

    return stats


def main():
    from cehrgpt.runners.gpt_runner_util import parse_dynamic_arguments

    (
        model_args,
        cehrgpt_args,
        rl_args,
        training_args,
    ) = parse_dynamic_arguments(
        (ModelArguments, CehrGPTArguments, RLArguments, TrainingArguments)
    )

    set_seed(training_args.seed)
    # The RL collator needs concept_ids / epoch_times / ages which are not in the
    # model's forward() signature.  Prevent the Trainer from stripping them.
    training_args.remove_unused_columns = False

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer_path = os.path.expanduser(model_args.tokenizer_name_or_path)
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_path)
    LOG.info("Loaded tokenizer from %s  (vocab_size=%d)", tokenizer_path, cehrgpt_tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Target condition IDs
    # Use ontology-derived Condition concepts when vocab_dir is available;
    # otherwise fall back to the tokenizer's _motor_time_to_event_codes.
    # ------------------------------------------------------------------
    target_concept_ids: Set[str] = _load_target_concept_ids(cehrgpt_args, cehrgpt_tokenizer)
    if not target_concept_ids:
        raise RuntimeError(
            "No target condition concept IDs found.  "
            "Provide --vocab_dir pointing to Athena parquet files, or ensure "
            "the tokenizer was built with --include_motor_time_to_event true."
        )
    LOG.info("Using %d target condition concepts for RL rewards.", len(target_concept_ids))

    # ------------------------------------------------------------------
    # Dataset  (must already be tokenized with concept_ids / epoch_times / ages)
    # ------------------------------------------------------------------
    dataset_path = os.path.expanduser(cehrgpt_args.tokenized_full_dataset_path)
    LOG.info("Loading tokenized dataset from %s", dataset_path)
    dataset = load_from_disk(dataset_path)

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", dataset.get("test"))

    # ------------------------------------------------------------------
    # Prevalence statistics
    # ------------------------------------------------------------------
    prevalence_stats = _load_prevalence_stats(rl_args, train_dataset, target_concept_ids)

    # ------------------------------------------------------------------
    # Policy model
    # ------------------------------------------------------------------
    model_path = os.path.expanduser(model_args.model_name_or_path)
    LOG.info("Loading policy model from %s", model_path)
    model = CEHRGPT2LMHeadModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    # ------------------------------------------------------------------
    # Reference model (frozen copy of the policy at RL start)
    # ------------------------------------------------------------------
    LOG.info("Creating frozen reference model.")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Move ref model to the same device(s) as the policy after Trainer sets it up.
    # Trainer handles device placement for `model`; we mirror it here.
    if training_args.no_cuda or not torch.cuda.is_available():
        ref_device = torch.device("cpu")
    else:
        ref_device = torch.device(f"cuda:{training_args.local_rank}" if training_args.local_rank >= 0 else "cuda")
    ref_model = ref_model.to(ref_device)

    # ------------------------------------------------------------------
    # Data collator
    # ------------------------------------------------------------------
    data_collator = RLDataCollator(
        tokenizer=cehrgpt_tokenizer,
        target_concept_ids=target_concept_ids,
        prediction_windows=rl_args.prediction_windows,
        max_prefix_length=rl_args.max_prefix_length,
        min_prefix_visits=rl_args.min_prefix_visits,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = CehrGptGRPOTrainer(
        ref_model=ref_model,
        rl_args=rl_args,
        prevalence_stats=prevalence_stats,
        target_concept_ids=target_concept_ids,
        cehrgpt_tokenizer=cehrgpt_tokenizer,
        # Standard Trainer kwargs
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    LOG.info("Starting RL fine-tuning.")
    checkpoint: Optional[str] = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif Path(training_args.output_dir).exists():
        # Auto-resume from last checkpoint if output dir already has one
        from cehrbert.runners.runner_util import get_last_hf_checkpoint
        checkpoint = get_last_hf_checkpoint(training_args)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
