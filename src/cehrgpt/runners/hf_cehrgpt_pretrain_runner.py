import os
from functools import partial
from typing import Optional, Union

import datasets
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    CacheFileCollector,
    create_dataset_from_meds_reader,
)
from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    ModelArguments,
)
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
)
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk
from transformers import EarlyStoppingCallback, Trainer, set_seed
from transformers.trainer_utils import is_main_process
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_pretraining_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)
from cehrgpt.data.hf_cehrgpt_dataset_mapping import MedToCehrGPTDatasetMapping
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.omop.ontology import Ontology
from cehrgpt.runners.data_utils import get_torch_dtype
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments
from cehrgpt.runners.sample_packing_trainer import SamplePackingTrainer

LOG = logging.get_logger("transformers")


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) / state.best_metric
            > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


def tokenizer_exists(tokenizer_name_or_path: str) -> bool:
    # Try to load the pretrained tokenizer
    try:
        CehrGptTokenizer.from_pretrained(os.path.abspath(tokenizer_name_or_path))
        return True
    except Exception:
        LOG.info(f"The tokenizer does not exist at {tokenizer_name_or_path}")
        return False


def load_and_create_tokenizer(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
    dataset: Union[Dataset, DatasetDict],
) -> CehrGptTokenizer:

    # Try to load the pretrained tokenizer
    tokenizer_abspath = os.path.expanduser(model_args.tokenizer_name_or_path)
    if not tokenizer_exists(tokenizer_abspath):
        if cehrgpt_args.include_motor_time_to_event and not cehrgpt_args.vocab_dir:
            raise RuntimeError(
                "motor_vocab_dir must be specified if include_motor_time_to_event is True"
            )
        ontology: Optional[Ontology] = None
        concept_name_mapping = {}
        if cehrgpt_args.vocab_dir:
            LOG.info("Loading concept data from disk at %s", cehrgpt_args.vocab_dir)
            concept_pd = pd.read_parquet(
                os.path.join(cehrgpt_args.vocab_dir, "concept")
            )
            for row in concept_pd.itertuples():
                concept_name_mapping[str(getattr(row, "concept_id"))] = getattr(
                    row, "concept_name"
                )

            if cehrgpt_args.motor_use_ontology:
                LOG.info("Creating ontology for MOTOR TTE predictions")
                ontology = Ontology(cehrgpt_args.vocab_dir)
                train_val_dataset = datasets.concatenate_datasets(
                    [dataset["train"], dataset["validation"]]
                )
                ontology.prune_to_dataset(
                    train_val_dataset,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_ontologies={"SPL", "HemOnc", "LOINC"},
                )

        LOG.info("Started training the tokenizer ...")
        train_val_dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"]]
        )
        tokenizer = CehrGptTokenizer.train_tokenizer(
            train_val_dataset,
            concept_name_mapping,
            data_args,
            PretrainedEmbeddings(cehrgpt_args.pretrained_embedding_path),
            num_motor_tasks=(
                cehrgpt_args.num_motor_tasks
                if cehrgpt_args.include_motor_time_to_event
                else None
            ),
            apply_entropy_filter=cehrgpt_args.apply_entropy_filter,
            min_prevalence=cehrgpt_args.min_prevalence,
            ontology=ontology,
        )
        LOG.info("Finished training the tokenizer ...")
        tokenizer.save_pretrained(tokenizer_abspath)
        LOG.info("Saved the tokenizer to %s", tokenizer_abspath)
    else:
        LOG.info("The tokenizer exists and will be loaded from %s", tokenizer_abspath)
        tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_abspath)
    return tokenizer


def load_and_create_model(
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
    tokenizer: CehrGptTokenizer,
) -> CEHRGPT2LMHeadModel:
    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )
    torch_dtype = get_torch_dtype(model_args.torch_dtype)
    model_abspath = os.path.expanduser(model_args.model_name_or_path)
    if cehrgpt_args.continue_pretrain:
        try:
            pretrained_model = CEHRGPT2LMHeadModel.from_pretrained(
                model_abspath,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
            )
            if (
                pretrained_model.config.max_position_embeddings
                < model_args.max_position_embeddings
            ):
                LOG.info(
                    f"Increase model.config.max_position_embeddings to {model_args.max_position_embeddings}"
                )
                pretrained_model.config.max_position_embeddings = (
                    model_args.max_position_embeddings
                )
                pretrained_model.resize_position_embeddings(
                    model_args.max_position_embeddings
                )
            return pretrained_model
        except Exception as e:
            LOG.error(
                f"When continue_pretrain is set to True, it assumes that CEHR-GPT has been trained "
                f"and will be used to pretrain on new datasets. The CEHR-GPT checkpoint must exist at {model_abspath}"
            )
            raise e
    try:
        model_config = CEHRGPTConfig.from_pretrained(
            model_abspath, attn_implementation=attn_implementation
        )
    except Exception as e:
        LOG.warning(e)
        if cehrgpt_args.causal_sfm:
            model_args.max_position_embeddings += 1
        if len(tokenizer.pretrained_token_ids) > 0:
            pretrained_embedding_dim = tokenizer.pretrained_embeddings.shape[1]
        else:
            pretrained_embedding_dim = model_args.hidden_size

        model_args_cehrgpt = model_args.as_dict()
        model_args_cehrgpt.pop("attn_implementation")
        # CEHR-GPT does not support this anymore
        model_args_cehrgpt.pop("exclude_position_ids")
        model_config = CEHRGPTConfig(
            activation_function=cehrgpt_args.activation_function,
            vocab_size=tokenizer.vocab_size,
            value_vocab_size=tokenizer.value_vocab_size,
            time_token_vocab_size=tokenizer.time_token_vocab_size,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            lab_token_ids=tokenizer.lab_token_ids,
            token_to_time_token_mapping=tokenizer.token_to_time_token_mapping,
            attn_implementation=attn_implementation,
            causal_sfm=cehrgpt_args.causal_sfm,
            demographics_size=cehrgpt_args.demographics_size,
            next_token_prediction_loss_weight=cehrgpt_args.next_token_prediction_loss_weight,
            lab_token_penalty=cehrgpt_args.lab_token_penalty,
            lab_token_loss_weight=cehrgpt_args.lab_token_loss_weight,
            value_prediction_loss_weight=cehrgpt_args.value_prediction_loss_weight,
            entropy_penalty=cehrgpt_args.entropy_penalty,
            entropy_penalty_alpha=cehrgpt_args.entropy_penalty_alpha,
            n_pretrained_embeddings_layers=cehrgpt_args.n_pretrained_embeddings_layers,
            use_pretrained_embeddings=len(tokenizer.pretrained_token_ids) > 0,
            pretrained_embedding_dim=pretrained_embedding_dim,
            apply_rotary=cehrgpt_args.apply_rotary,
            sample_packing_max_positions=(
                cehrgpt_args.max_tokens_per_batch
                if cehrgpt_args.sample_packing
                else model_args.max_position_embeddings
            ),
            include_motor_time_to_event=cehrgpt_args.include_motor_time_to_event,
            motor_tte_vocab_size=tokenizer.motor_tte_vocab_size,
            motor_time_to_event_weight=cehrgpt_args.motor_time_to_event_weight,
            motor_num_time_pieces=cehrgpt_args.motor_num_time_pieces,
            ve_token_id=tokenizer.ve_token_id,
            n_inner=cehrgpt_args.inner_dim,
            decoder_mlp=cehrgpt_args.decoder_mlp,
            **model_args_cehrgpt,
        )

    model = CEHRGPT2LMHeadModel(model_config)
    if tokenizer.pretrained_token_ids:
        model.cehrgpt.update_pretrained_embeddings(
            tokenizer.pretrained_token_ids,
            tokenizer.pretrained_embeddings,
        )
    if model.config.torch_dtype == torch.bfloat16:
        return model.bfloat16()
    elif model.config.torch_dtype == torch.float16:
        return model.half()
    return model


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    if cehrgpt_args.sample_packing and data_args.streaming:
        raise RuntimeError(
            f"sample_packing is not supported when streaming is enabled, please set streaming to False"
        )

    if data_args.streaming:
        # This is for disabling the warning message https://github.com/huggingface/transformers/issues/5486
        # This happens only when streaming is enabled
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to be set to 0
        # Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0
        training_args.dataloader_prefetch_factor = None

    processed_dataset: Optional[DatasetDict] = None
    cache_file_collector = CacheFileCollector()
    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)
    if os.path.exists(os.path.join(data_args.data_folder, "dataset_dict.json")):
        LOG.info(f"Loading prepared dataset from disk at {data_args.data_folder}...")
        processed_dataset = load_from_disk(data_args.data_folder)
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-GPT tokenizer from the output folder, otherwise an exception will be raised.
        tokenizer_name_or_path = os.path.expanduser(
            training_args.output_dir
            if cehrgpt_args.expand_tokenizer
            else model_args.tokenizer_name_or_path
        )
        if not tokenizer_exists(tokenizer_name_or_path):
            raise RuntimeError(
                f"The dataset has been tokenized but the corresponding tokenizer: "
                f"{model_args.tokenizer_name_or_path} does not exist"
            )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_name_or_path)
    elif any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-GPT tokenizer from the output folder, otherwise an exception will be raised.
        tokenizer_name_or_path = os.path.expanduser(
            training_args.output_dir
            if cehrgpt_args.expand_tokenizer
            else model_args.tokenizer_name_or_path
        )
        if not tokenizer_exists(tokenizer_name_or_path):
            raise RuntimeError(
                f"The dataset has been tokenized but the corresponding tokenizer: "
                f"{model_args.tokenizer_name_or_path} does not exist"
            )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_name_or_path)
    else:
        # Only run tokenization and data transformation in the main process in torch distributed training
        # otherwise the multiple processes will create tokenizers at the same time
        if is_main_process(training_args.local_rank):
            # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
            if data_args.is_data_in_meds:
                meds_extension_path = get_meds_extension_path(
                    data_folder=data_args.data_folder,
                    dataset_prepared_path=data_args.dataset_prepared_path,
                )
                try:
                    LOG.info(
                        "Trying to load the MEDS extension from disk at %s...",
                        meds_extension_path,
                    )
                    dataset = load_from_disk(meds_extension_path)
                    if data_args.streaming:
                        if isinstance(dataset, DatasetDict):
                            dataset = {
                                k: v.to_iterable_dataset(
                                    num_shards=training_args.dataloader_num_workers
                                )
                                for k, v in dataset.items()
                            }
                        else:
                            dataset = dataset.to_iterable_dataset(
                                num_shards=training_args.dataloader_num_workers
                            )
                except FileNotFoundError as e:
                    LOG.warning(e)
                    dataset = create_dataset_from_meds_reader(
                        data_args=data_args,
                        dataset_mappings=[
                            MedToCehrGPTDatasetMapping(
                                data_args=data_args,
                                include_inpatient_hour_token=cehrgpt_args.include_inpatient_hour_token,
                            )
                        ],
                        cache_file_collector=cache_file_collector,
                    )
                    if not data_args.streaming:
                        dataset.save_to_disk(str(meds_extension_path))
                        stats = dataset.cleanup_cache_files()
                        LOG.info(
                            "Clean up the cached files for the cehrgpt dataset transformed from the MEDS: %s",
                            stats,
                        )
                        # Clean up the files created from the data generator
                        cache_file_collector.remove_cache_files()
                        dataset = load_from_disk(str(meds_extension_path))
            else:
                # Load the dataset from the parquet files
                dataset = load_parquet_as_dataset(
                    os.path.expanduser(data_args.data_folder),
                    split="train",
                    streaming=data_args.streaming,
                )
                # If streaming is enabled, we need to manually split the data into train/val
                if data_args.streaming and data_args.validation_split_num:
                    dataset = dataset.shuffle(
                        buffer_size=10_000, seed=training_args.seed
                    )
                    train_set = dataset.skip(data_args.validation_split_num)
                    val_set = dataset.take(data_args.validation_split_num)
                    dataset = DatasetDict({"train": train_set, "validation": val_set})
                elif data_args.validation_split_percentage:
                    dataset = dataset.train_test_split(
                        test_size=data_args.validation_split_percentage,
                        seed=training_args.seed,
                    )
                    dataset = DatasetDict(
                        {"train": dataset["train"], "validation": dataset["test"]}
                    )
                else:
                    raise RuntimeError(
                        f"Can not split the data. If streaming is enabled, validation_split_num needs to be "
                        f"defined, otherwise validation_split_percentage needs to be provided. "
                        f"The current values are:\n"
                        f"validation_split_percentage: {data_args.validation_split_percentage}\n"
                        f"validation_split_num: {data_args.validation_split_num}\n"
                        f"streaming: {data_args.streaming}"
                    )

            # Create the CEHR-GPT tokenizer if it's not available in the output folder
            cehrgpt_tokenizer = load_and_create_tokenizer(
                data_args=data_args,
                model_args=model_args,
                cehrgpt_args=cehrgpt_args,
                dataset=dataset,
            )

            # Retrain the tokenizer in case we want to pretrain the model further using different datasets
            if cehrgpt_args.expand_tokenizer:
                new_tokenizer_path = os.path.expanduser(training_args.output_dir)
                try:
                    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
                        new_tokenizer_path
                    )
                except Exception:
                    cehrgpt_tokenizer = CehrGptTokenizer.expand_trained_tokenizer(
                        cehrgpt_tokenizer=cehrgpt_tokenizer,
                        dataset=dataset["train"],
                        data_args=data_args,
                        concept_name_mapping={},
                        pretrained_concept_embedding_model=PretrainedEmbeddings(
                            cehrgpt_args.pretrained_embedding_path
                        ),
                        apply_entropy_filter=cehrgpt_args.apply_entropy_filter,
                        min_prevalence=cehrgpt_args.min_prevalence,
                    )
                    cehrgpt_tokenizer.save_pretrained(
                        os.path.expanduser(training_args.output_dir)
                    )

            # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
            if not data_args.streaming:
                all_columns = dataset["train"].column_names
                if "visit_concept_ids" in all_columns:
                    dataset = dataset.remove_columns(["visit_concept_ids"])

            # sort the patient features chronologically and tokenize the data
            processed_dataset = create_cehrgpt_pretraining_dataset(
                dataset=dataset,
                cehrgpt_tokenizer=cehrgpt_tokenizer,
                data_args=data_args,
                cache_file_collector=cache_file_collector,
            )
            # only save the data to the disk if it is not streaming
            if not data_args.streaming:
                processed_dataset.save_to_disk(str(prepared_ds_path))
                stats = processed_dataset.cleanup_cache_files()
                LOG.info(
                    "Clean up the cached files for the cehrgpt pretraining dataset: %s",
                    stats,
                )
            cache_file_collector.remove_cache_files()

        # After main-process-only operations, synchronize all processes to ensure consistency
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Loading tokenizer in all processes in torch distributed training
        tokenizer_name_or_path = os.path.expanduser(
            training_args.output_dir
            if cehrgpt_args.expand_tokenizer
            else model_args.tokenizer_name_or_path
        )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_name_or_path)
        # Load the dataset from disk again to in torch distributed training
        if not data_args.streaming:
            processed_dataset = load_from_disk(str(prepared_ds_path))

    if processed_dataset is None:
        raise RuntimeError("The processed dataset cannot be None")

    def filter_func(examples):
        if cehrgpt_args.drop_long_sequences:
            return [
                model_args.max_position_embeddings >= _ >= data_args.min_num_tokens
                for _ in examples["num_of_concepts"]
            ]
        else:
            return [_ >= data_args.min_num_tokens for _ in examples["num_of_concepts"]]

    # Create the args for batched filtering
    filter_args = {"batched": True, "batch_size": data_args.preprocessing_batch_size}
    # If the dataset is not in a streaming mode, we could add num_proc to enable parallelization
    if not data_args.streaming:
        filter_args["num_proc"] = data_args.preprocessing_num_workers

    # The filter can't be applied to a DatasetDict of IterableDataset (in case of streaming)
    # we need to iterate through all the datasets and apply the filter separately
    if isinstance(processed_dataset, DatasetDict) or isinstance(
        processed_dataset, IterableDatasetDict
    ):
        for key in processed_dataset.keys():
            processed_dataset[key] = processed_dataset[key].filter(
                filter_func, **filter_args
            )
    else:
        processed_dataset = processed_dataset.filter(filter_func, **filter_args)

    model = load_and_create_model(model_args, cehrgpt_args, cehrgpt_tokenizer)

    # Try to update motor tte vocab size if the new configuration is different from the existing one
    if cehrgpt_args.include_motor_time_to_event:
        model.update_motor_tte_vocab_size(cehrgpt_tokenizer.motor_tte_vocab_size)

    # Expand tokenizer to adapt to the new pretraining dataset
    if model.config.vocab_size < cehrgpt_tokenizer.vocab_size:
        model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)
        # Update the pretrained embedding weights if they are available
        if model.config.use_pretrained_embeddings:
            model.cehrgpt.update_pretrained_embeddings(
                cehrgpt_tokenizer.pretrained_token_ids,
                cehrgpt_tokenizer.pretrained_embeddings,
            )
        elif cehrgpt_tokenizer.pretrained_token_ids:
            model.config.pretrained_embedding_dim = (
                cehrgpt_tokenizer.pretrained_embeddings.shape[1]
            )
            model.config.use_pretrained_embeddings = True
            model.cehrgpt.initialize_pretrained_embeddings()
            model.cehrgpt.update_pretrained_embeddings(
                cehrgpt_tokenizer.pretrained_token_ids,
                cehrgpt_tokenizer.pretrained_embeddings,
            )

    # Detecting last checkpoint.
    last_checkpoint = get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming and not cehrgpt_args.sample_packing:
        processed_dataset.set_format("pt")

    callbacks = []
    if cehrgpt_args.use_early_stopping:
        callbacks.append(
            CustomEarlyStoppingCallback(
                model_args.early_stopping_patience,
                cehrgpt_args.early_stopping_threshold,
            )
        )

    if cehrgpt_args.sample_packing:
        trainer_class = partial(
            SamplePackingTrainer,
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=model_args.max_position_embeddings,
            train_lengths=processed_dataset["train"]["num_of_concepts"],
            validation_lengths=(
                processed_dataset["validation"]
                if "validation" in processed_dataset
                else processed_dataset["test"]
            )["num_of_concepts"],
        )
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        data_collator_fn = partial(
            SamplePackingCehrGptDataCollator,
            cehrgpt_args.max_tokens_per_batch,
            model_args.max_position_embeddings,
        )
    else:
        trainer_class = Trainer
        data_collator_fn = CehrGptDataCollator

    trainer = trainer_class(
        model=model,
        data_collator=data_collator_fn(
            tokenizer=cehrgpt_tokenizer,
            max_length=(
                cehrgpt_args.max_tokens_per_batch
                if cehrgpt_args.sample_packing
                else model_args.max_position_embeddings
            ),
            shuffle_records=data_args.shuffle_records,
            include_ttv_prediction=model_args.include_ttv_prediction,
            use_sub_time_tokenization=model_args.use_sub_time_tokenization,
            include_values=model_args.include_values,
            include_motor_time_to_event=cehrgpt_args.include_motor_time_to_event,
            motor_tte_vocab_size=model.config.motor_tte_vocab_size,
            motor_num_time_pieces=cehrgpt_args.motor_num_time_pieces,
            motor_sampling_probability=cehrgpt_args.motor_sampling_probability,
        ),
        train_dataset=processed_dataset["train"],
        eval_dataset=(
            processed_dataset["validation"]
            if "validation" in processed_dataset
            else processed_dataset["test"]
        ),
        args=training_args,
        callbacks=callbacks,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
