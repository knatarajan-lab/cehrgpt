from cehrbert.data_generators.hf_data_generator.hf_dataset import (
    apply_cehrbert_dataset_mapping,
)
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    load_parquet_as_dataset,
)
from datasets import DatasetDict, load_from_disk
from transformers import set_seed
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dpo_collator import CehrGptDPODataCollator
from cehrgpt.data.hf_cehrgpt_dpo_dataset_mapping import HFCehrGptDPOTokenizationMapping
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.rl_finetune.cehrgpt_dpo_trainer import CehrGptDPOTrainer
from cehrgpt.runners.gpt_runner_util import parse_dpo_runner_args
from cehrgpt.runners.hf_cehrgpt_finetune_runner import load_pretrained_tokenizer

LOG = logging.get_logger("transformers")


def main():
    cehrgpt_args, data_args, model_args, dpo_config = parse_dpo_runner_args()
    tokenizer = load_pretrained_tokenizer(model_args)
    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
    else:
        dataset = load_parquet_as_dataset(data_args.data_folder)
        # Random split
        train_val = dataset.train_test_split(
            test_size=data_args.validation_split_percentage, seed=dpo_config.seed
        )
        train_set, validation_set = train_val["train"]
        # Organize them into a single DatasetDict
        dataset = DatasetDict({"train": train_set, "validation": validation_set})
        if data_args.streaming:
            dataset = {
                k: v.to_iterable_dataset(num_shards=dpo_config.dataloader_num_workers)
                for k, v in dataset.items()
            }

        processed_dataset = apply_cehrbert_dataset_mapping(
            dataset,
            mapping_function=HFCehrGptDPOTokenizationMapping(tokenizer),
            batch_size=data_args.preprocessing_batch_size,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming,
        )

        processed_dataset = processed_dataset.filter(
            lambda batch: [
                len(chosen_concept_ids) < model_args.max_position_embeddings
                for chosen_concept_ids in batch["chosen_concept_ids"]
            ],
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            num_proc=data_args.preprocessing_num_workers,
        ).filter(
            lambda batch: [
                len(rejected_concept_ids) < model_args.max_position_embeddings
                for rejected_concept_ids in batch["rejected_concept_ids"]
            ],
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            num_proc=data_args.preprocessing_num_workers,
        )

    if not data_args.streaming:
        processed_dataset.save_to_disk(prepared_ds_path)

    # Set seed before initializing model.
    set_seed(dpo_config.seed)
    processed_dataset.set_format("pt")

    # A hacky way to prevent the training from removing unmatched inputs
    dpo_config.label_names = [
        "chosen_input_ids",
        "rejected_input_ids",
        "chosen_concept_values",
        "rejected_concept_values",
        "chosen_concept_value_masks",
        "rejected_concept_value_masks",
    ]

    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )
    model = CEHRGPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=attn_implementation,
    )
    ref_model = CEHRGPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=attn_implementation,
    )

    # Initialize Trainer for final training on the combined train+val set
    trainer = CehrGptDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=CehrGptDPODataCollator(
            tokenizer=tokenizer,
            max_length=model_args.max_position_embeddings,
            pretraining=False,
            include_ttv_prediction=False,
            use_sub_time_tokenization=False,
        ),
    )
    # Train the model on the combined train + val set
    checkpoint = get_last_hf_checkpoint(dpo_config)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
