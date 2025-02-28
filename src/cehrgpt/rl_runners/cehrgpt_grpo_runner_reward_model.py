import os
from functools import partial

import polars as pl
from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
from cehrbert.runners.runner_util import get_last_hf_checkpoint, load_parquet_as_dataset
from transformers.utils import logging
from trl import GRPOConfig, GRPOTrainer

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel, CehrGptForClassification
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.omop.vocab_utils import generate_concept_maps
from cehrgpt.rl_runners.grpo.rewards import reward_valid_sequences
from cehrgpt.runners.gpt_runner_util import parse_dynamic_arguments
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTGRPOArguments

logger = logging.get_logger("transformers")


def main():
    cehrgpt_grpo_args, model_args, training_args = parse_dynamic_arguments(
        (CehrGPTGRPOArguments, ModelArguments, GRPOConfig)
    )

    dataset = load_parquet_as_dataset(cehrgpt_grpo_args.demographics_prompt_dir)
    dataset = dataset.map(
        lambda batch: {
            "prompt": [concept_ids[:4] for concept_ids in batch["concept_ids"]]
        },
        batch_size=1024,
        batched=True,
        remove_columns=dataset.column_names,
    )
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    cehrgpt_model = CEHRGPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    reward_tokenizer = CehrGptTokenizer.from_pretrained(model_args.reward_model_path)
    reward_model = CehrGptForClassification.from_pretrained(
        model_args.reward_model_path
    )

    concept = pl.read_parquet(
        os.path.join(cehrgpt_grpo_args.vocabulary_dir, "concept", "*.parquet")
    )
    _, concept_domain_map = generate_concept_maps(concept)
    reward_valid_sequence_func = partial(
        reward_valid_sequences, concept_domain_map=concept_domain_map
    )
    reward_valid_sequence_func.__name__ = f"reward_valid_sequence"

    # Detecting last checkpoint.
    checkpoint = get_last_hf_checkpoint(training_args)
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    logger.info("The existing checkpoint is %s", checkpoint)

    trainer = GRPOTrainer(
        model=cehrgpt_model,
        processing_class=cehrgpt_tokenizer,
        reward_funcs=[reward_valid_sequence_func, reward_model],
        args=training_args,
        train_dataset=dataset,
        reward_processing_classes=[None, reward_tokenizer],
    )
    trainer.generation_config.return_dict_in_generate = False
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
