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
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
from datasets import load_from_disk
from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import is_main_process
from transformers.utils import logging

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.rl.grpo_trainer import CehrGptGRPOTrainer
from cehrgpt.rl.ppo_trainer import CehrGptPPOTrainer
from cehrgpt.rl.rl_data_collator import RLDataCollator
from cehrgpt.runners.hf_gpt_rl_runner_argument_dataclass import RLArguments
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

LOG = logging.get_logger("transformers")


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
    # Dataset  (must already be tokenized with concept_ids / epoch_times / ages)
    # ------------------------------------------------------------------
    dataset_path = os.path.expanduser(cehrgpt_args.tokenized_full_dataset_path)
    LOG.info("Loading tokenized dataset from %s", dataset_path)
    dataset = load_from_disk(dataset_path)

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", dataset.get("test"))

    # Pre-filter sequences that have too few visits for the RL collator to
    # produce a valid prefix/future split, so the collator never returns an
    # empty batch and causes DDP deadlocks.
    min_vs = rl_args.min_prefix_visits + 1  # prefix visits + at least 1 future visit
    def _has_enough_visits(example):
        return sum(1 for t in example["concept_ids"] if t == "[VS]") >= min_vs

    before = len(train_dataset)
    train_dataset = train_dataset.filter(_has_enough_visits)
    if is_main_process(training_args.local_rank):
        LOG.info(
            "Filtered train dataset: %d → %d examples (kept %.1f%% with >= %d visits)",
            before, len(train_dataset), 100 * len(train_dataset) / before, min_vs,
        )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.filter(_has_enough_visits)

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
        max_prefix_length=rl_args.max_prefix_length,
        min_prefix_visits=rl_args.min_prefix_visits,
        max_future_length=rl_args.max_future_length,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer_cls = CehrGptPPOTrainer if rl_args.trainer_type == "ppo" else CehrGptGRPOTrainer
    LOG.info("Using trainer: %s", trainer_cls.__name__)
    trainer = trainer_cls(
        ref_model=ref_model,
        rl_args=rl_args,
        cehrgpt_tokenizer=cehrgpt_tokenizer,
        eval_sample_size=rl_args.eval_sample_size,
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
