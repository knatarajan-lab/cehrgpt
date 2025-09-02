import glob
import json
import os
import random
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.hf_cehrbert_finetune_runner import compute_metrics
from cehrbert.runners.hf_runner_argument_dataclass import (
    FineTuneModelType,
    ModelArguments,
)
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
)
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import expit as sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)
from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPTConfig,
    CehrGptForClassification,
    CEHRGPTPreTrainedModel,
)
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import (
    extract_cohort_sequences,
    get_torch_dtype,
    prepare_finetune_dataset,
)
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import tokenizer_exists
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments
from cehrgpt.runners.hyperparameter_search_util import perform_hyperparameter_search
from cehrgpt.runners.sample_packing_trainer import SamplePackingTrainer

LOG = logging.get_logger("transformers")


class UpdateNumEpochsBeforeEarlyStoppingCallback(TrainerCallback):
    """
    Callback to update metrics with the number of epochs completed before early stopping.

    based on the best evaluation metric (e.g., eval_loss).
    """

    def __init__(self, model_folder: str):
        self._model_folder = model_folder
        self._metrics_path = os.path.join(
            model_folder, "num_epochs_trained_before_early_stopping.json"
        )
        self._num_epochs_before_early_stopping = 0
        self._best_val_loss = float("inf")

    @property
    def num_epochs_before_early_stopping(self):
        return self._num_epochs_before_early_stopping

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if os.path.exists(self._metrics_path):
            with open(self._metrics_path, "r") as f:
                metrics = json.load(f)
            self._num_epochs_before_early_stopping = metrics[
                "num_epochs_before_early_stopping"
            ]
            self._best_val_loss = metrics["best_val_loss"]

    def on_evaluate(self, args, state, control, **kwargs):
        # Ensure metrics is available in kwargs
        metrics = kwargs.get("metrics")
        if metrics is not None and "eval_loss" in metrics:
            # Check and update if a new best metric is achieved
            if metrics["eval_loss"] < self._best_val_loss:
                self._num_epochs_before_early_stopping = round(state.epoch)
                self._best_val_loss = metrics["eval_loss"]

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        with open(self._metrics_path, "w") as f:
            json.dump(
                {
                    "num_epochs_before_early_stopping": self._num_epochs_before_early_stopping,
                    "best_val_loss": self._best_val_loss,
                },
                f,
            )


def load_pretrained_tokenizer(
    model_args,
) -> CehrGptTokenizer:
    try:
        return CehrGptTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    except Exception:
        raise ValueError(
            f"Can not load the pretrained tokenizer from {model_args.tokenizer_name_or_path}"
        )


def load_finetuned_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    model_name_or_path: str,
) -> CEHRGPTPreTrainedModel:
    if model_args.finetune_model_type == FineTuneModelType.POOLING.value:
        finetune_model_cls = CehrGptForClassification
    else:
        raise ValueError(
            f"finetune_model_type can be one of the following types {FineTuneModelType.POOLING.value}"
        )
    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )
    torch_dtype = get_torch_dtype(model_args.torch_dtype)
    # Try to create a new model based on the base model
    try:
        return finetune_model_cls.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
    except ValueError:
        raise ValueError(f"Can not load the finetuned model from {model_name_or_path}")


def model_init(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    tokenizer: CehrGptTokenizer,
):
    model = load_finetuned_model(
        model_args, training_args, model_args.model_name_or_path
    )

    if cehrgpt_args.class_weights:
        model.config.class_weights = cehrgpt_args.class_weights
        LOG.info(f"Setting class_weights to {model.config.class_weights}")

    if model.config.max_position_embeddings < model_args.max_position_embeddings:
        LOG.info(
            f"Increase model.config.max_position_embeddings to {model_args.max_position_embeddings}"
        )
        model.config.max_position_embeddings = model_args.max_position_embeddings
        model.resize_position_embeddings(model_args.max_position_embeddings)
    # Enable include_values when include_values is set to be False during pre-training
    if model_args.include_values and not model.cehrgpt.include_values:
        model.cehrgpt.include_values = True
    # Expand tokenizer to adapt to the finetuning dataset
    if model.config.vocab_size < tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
        # Update the pretrained embedding weights if they are available
        if model.config.use_pretrained_embeddings:
            model.cehrgpt.update_pretrained_embeddings(
                tokenizer.pretrained_token_ids, tokenizer.pretrained_embeddings
            )
        elif tokenizer.pretrained_token_ids:
            model.config.pretrained_embedding_dim = (
                tokenizer.pretrained_embeddings.shape[1]
            )
            model.config.use_pretrained_embeddings = True
            model.cehrgpt.initialize_pretrained_embeddings()
            model.cehrgpt.update_pretrained_embeddings(
                tokenizer.pretrained_token_ids, tokenizer.pretrained_embeddings
            )

    # Expand value tokenizer to adapt to the fine-tuning dataset
    if model.config.include_values:
        if model.config.value_vocab_size < tokenizer.value_vocab_size:
            model.resize_value_embeddings(tokenizer.value_vocab_size)
    # If lora is enabled, we add LORA adapters to the model
    if model_args.use_lora:
        # When LORA is used, the trainer could not automatically find this label,
        # therefore we need to manually set label_names to "classifier_label" so the model
        # can compute the loss during the evaluation
        if training_args.label_names:
            training_args.label_names.append("classifier_label")
        else:
            training_args.label_names = ["classifier_label"]

        if model_args.finetune_model_type == FineTuneModelType.POOLING.value:
            config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                modules_to_save=["classifier", "age_batch_norm", "dense_layer"],
            )
            model = get_peft_model(model, config)
        else:
            raise ValueError(
                f"The LORA adapter is not supported for {model_args.finetune_model_type}"
            )
    return model


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    tokenizer = load_pretrained_tokenizer(model_args)
    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )
    cache_file_collector = CacheFileCollector()
    processed_dataset = None
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        if cehrgpt_args.expand_tokenizer:
            if tokenizer_exists(training_args.output_dir):
                tokenizer = CehrGptTokenizer.from_pretrained(training_args.output_dir)
            else:
                LOG.warning(
                    f"CehrGptTokenizer must exist in {training_args.output_dir} "
                    f"when the dataset has been processed and expand_tokenizer is set to True. "
                    f"Please delete the processed dataset at {prepared_ds_path}."
                )
                processed_dataset = None
                shutil.rmtree(prepared_ds_path)

    if processed_dataset is None:
        if is_main_process(training_args.local_rank):
            # If the full dataset has been tokenized, we don't want to tokenize the cohort containing
            # the subset of the data. We should slice out the portion of the tokenized sequences for each sample
            if cehrgpt_args.tokenized_full_dataset_path is not None:
                processed_dataset = extract_cohort_sequences(
                    data_args, cehrgpt_args, cache_file_collector
                )
            else:
                final_splits = prepare_finetune_dataset(
                    data_args, training_args, cehrgpt_args, cache_file_collector
                )
                if cehrgpt_args.expand_tokenizer:
                    new_tokenizer_path = os.path.expanduser(training_args.output_dir)
                    if tokenizer_exists(new_tokenizer_path):
                        tokenizer = CehrGptTokenizer.from_pretrained(new_tokenizer_path)
                    else:
                        # Try to use the defined pretrained embeddings if exists, Otherwise we default to the pretrained model
                        # embedded in the pretrained model
                        pretrained_concept_embedding_model = PretrainedEmbeddings(
                            cehrgpt_args.pretrained_embedding_path
                        )
                        if not pretrained_concept_embedding_model.exists:
                            pretrained_concept_embedding_model = (
                                tokenizer.pretrained_concept_embedding_model
                            )
                        tokenizer = CehrGptTokenizer.expand_trained_tokenizer(
                            cehrgpt_tokenizer=tokenizer,
                            dataset=final_splits["train"],
                            data_args=data_args,
                            concept_name_mapping={},
                            pretrained_concept_embedding_model=pretrained_concept_embedding_model,
                        )
                        tokenizer.save_pretrained(
                            os.path.expanduser(training_args.output_dir)
                        )

                # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
                if not data_args.streaming:
                    all_columns = final_splits["train"].column_names
                    if "visit_concept_ids" in all_columns:
                        final_splits = final_splits.remove_columns(
                            ["visit_concept_ids"]
                        )

                processed_dataset = create_cehrgpt_finetuning_dataset(
                    dataset=final_splits,
                    cehrgpt_tokenizer=tokenizer,
                    data_args=data_args,
                    cache_file_collector=cache_file_collector,
                )
            if not data_args.streaming:
                processed_dataset.save_to_disk(str(prepared_ds_path))
                stats = processed_dataset.cleanup_cache_files()
                LOG.info(
                    "Clean up the cached files for the  cehrgpt finetuning dataset : %s",
                    stats,
                )

            # Remove any cached files if there are any
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
        tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_name_or_path)
        # Load the dataset from disk again to in torch distributed training
        processed_dataset = load_from_disk(str(prepared_ds_path))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming and not cehrgpt_args.sample_packing:
        processed_dataset.set_format("pt")

    if cehrgpt_args.few_shot_predict:
        # At least we need two examples to have a validation set for early stopping
        num_shots = max(cehrgpt_args.n_shots, 2)
        random_train_indices = random.sample(
            range(len(processed_dataset["train"])), k=num_shots
        )
        test_size = max(int(num_shots * data_args.validation_split_percentage), 1)
        few_shot_train_val_set = processed_dataset["train"].select(random_train_indices)
        train_val = few_shot_train_val_set.train_test_split(
            test_size=test_size, seed=training_args.seed
        )
        few_shot_train_set, few_shot_val_set = train_val["train"], train_val["test"]
        processed_dataset["train"] = few_shot_train_set
        processed_dataset["validation"] = few_shot_val_set

    config = CEHRGPTConfig.from_pretrained(model_args.model_name_or_path)
    if config.max_position_embeddings < model_args.max_position_embeddings:
        config.max_position_embeddings = model_args.max_position_embeddings

    # persist this parameter in case this is overwritten by sample packing
    per_device_eval_batch_size = training_args.per_device_eval_batch_size

    if cehrgpt_args.sample_packing:
        trainer_class = partial(
            SamplePackingTrainer,
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=config.max_position_embeddings,
            negative_sampling_probability=cehrgpt_args.negative_sampling_probability,
        )
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        data_collator_fn = partial(
            SamplePackingCehrGptDataCollator,
            cehrgpt_args.max_tokens_per_batch,
            config.max_position_embeddings,
        )
    else:
        trainer_class = Trainer
        data_collator_fn = CehrGptDataCollator

    # We suppress the additional learning objectives in fine-tuning
    data_collator = data_collator_fn(
        tokenizer=tokenizer,
        max_length=(
            cehrgpt_args.max_tokens_per_batch
            if cehrgpt_args.sample_packing
            else (
                config.max_position_embeddings - 1
                if config.causal_sfm
                else config.max_position_embeddings
            )
        ),
        include_values=model_args.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
        add_linear_prob_token=True,
    )

    if training_args.do_train:
        output_dir = training_args.output_dir
        if cehrgpt_args.hyperparameter_tuning:
            training_args, run_id = perform_hyperparameter_search(
                trainer_class,
                partial(model_init, model_args, training_args, cehrgpt_args, tokenizer),
                processed_dataset,
                data_collator,
                training_args,
                model_args,
                cehrgpt_args,
            )
            # We enforce retraining if cehrgpt_args.hyperparameter_tuning_percentage < 1.0
            cehrgpt_args.retrain_with_full |= (
                cehrgpt_args.hyperparameter_tuning_percentage < 1.0
            )
            output_dir = os.path.join(training_args.output_dir, f"run-{run_id}")

        if cehrgpt_args.hyperparameter_tuning and not cehrgpt_args.retrain_with_full:
            folders = glob.glob(os.path.join(output_dir, "checkpoint-*"))
            if len(folders) == 0:
                raise RuntimeError(
                    f"There must be a checkpoint folder under {output_dir}"
                )
            checkpoint_dir = folders[0]
            LOG.info("Best trial checkpoint folder: %s", checkpoint_dir)
            for file_name in os.listdir(checkpoint_dir):
                try:
                    full_file_name = os.path.join(checkpoint_dir, file_name)
                    destination = os.path.join(training_args.output_dir, file_name)
                    if os.path.isfile(full_file_name):
                        shutil.copy2(full_file_name, destination)
                except Exception as e:
                    LOG.error("Failed to copy %s: %s", file_name, str(e))
        else:
            # Initialize Trainer for final training on the combined train+val set
            trainer = trainer_class(
                model=model_init(model_args, training_args, cehrgpt_args, tokenizer),
                data_collator=data_collator,
                args=training_args,
                train_dataset=processed_dataset["train"],
                eval_dataset=processed_dataset["validation"],
                callbacks=[
                    EarlyStoppingCallback(model_args.early_stopping_patience),
                    UpdateNumEpochsBeforeEarlyStoppingCallback(
                        training_args.output_dir
                    ),
                ],
                tokenizer=tokenizer,
            )
            # Train the model on the combined train + val set
            checkpoint = get_last_hf_checkpoint(training_args)
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    if training_args.do_predict:
        if cehrgpt_args.sample_packing:
            batch_sampler = SamplePackingBatchSampler(
                lengths=processed_dataset["test"]["num_of_concepts"],
                max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
                max_position_embeddings=config.max_position_embeddings,
                drop_last=training_args.dataloader_drop_last,
                seed=training_args.seed,
            )
            per_device_eval_batch_size = 1
        else:
            batch_sampler = None
        test_dataloader = DataLoader(
            dataset=processed_dataset["test"],
            batch_size=per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=data_collator,
            pin_memory=training_args.dataloader_pin_memory,
            batch_sampler=batch_sampler,
        )
        do_predict(test_dataloader, model_args, training_args, cehrgpt_args)


def do_predict(
    test_dataloader: DataLoader,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    cehrgpt_args: CehrGPTArguments,
):
    """
    Performs inference on the test dataset using a fine-tuned model, saves predictions and evaluation metrics.

    The reason we created this custom do_predict is that there is a memory leakage for transformers trainer.predict(),
    for large test sets, it will throw the CPU OOM error

    Args:
        test_dataloader (DataLoader): DataLoader containing the test dataset, with batches of input features and labels.
        model_args (ModelArguments): Arguments for configuring and loading the fine-tuned model.
        training_args (TrainingArguments): Arguments related to training, evaluation, and output directories.
        cehrgpt_args (CehrGPTArguments):
    Returns:
        None. Results are saved to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and LoRA adapters if applicable
    model = (
        load_finetuned_model(model_args, training_args, training_args.output_dir)
        if not model_args.use_lora
        else load_lora_model(model_args, training_args, cehrgpt_args)
    )

    model = model.to(device).eval()

    # Ensure prediction folder exists
    test_prediction_folder = Path(training_args.output_dir) / "test_predictions"
    test_prediction_folder.mkdir(parents=True, exist_ok=True)

    LOG.info("Generating predictions for test set at %s", test_prediction_folder)

    test_losses = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            person_ids = batch.pop("person_id").numpy().astype(int).squeeze()
            if person_ids.ndim == 0:
                person_ids = np.asarray([person_ids])

            index_dates = batch.pop("index_date").numpy().squeeze()
            if index_dates.ndim == 0:
                index_dates = np.asarray([index_dates])

            index_dates = list(
                map(
                    lambda posix_time: datetime.utcfromtimestamp(posix_time).replace(
                        tzinfo=None
                    ),
                    index_dates.tolist(),
                )
            )

            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            output = model(**batch, output_attentions=False, output_hidden_states=False)
            test_losses.append(output.loss.item())

            # Collect logits and labels for prediction
            logits = output.logits.float().cpu().numpy().squeeze()
            if logits.ndim == 0:
                logits = np.asarray([logits])
            probabilities = sigmoid(logits)

            labels = (
                batch["classifier_label"].float().cpu().numpy().astype(bool).squeeze()
            )
            if labels.ndim == 0:
                labels = np.asarray([labels])

            # Save predictions to parquet file
            test_prediction_pd = pd.DataFrame(
                {
                    "subject_id": person_ids,
                    "prediction_time": index_dates,
                    "predicted_boolean_probability": probabilities,
                    "predicted_boolean_value": pd.Series(
                        [None] * len(person_ids), dtype=bool
                    ),
                    "boolean_value": labels,
                }
            )
            test_prediction_pd.to_parquet(test_prediction_folder / f"{index}.parquet")

    LOG.info(
        "Computing metrics using the test set predictions at %s", test_prediction_folder
    )
    # Load all predictions
    test_prediction_pd = pd.read_parquet(test_prediction_folder)
    # Compute metrics and save results
    metrics = compute_metrics(
        references=test_prediction_pd.boolean_value,
        probs=test_prediction_pd.predicted_boolean_probability,
    )
    metrics["test_loss"] = np.mean(test_losses)

    test_results_path = Path(training_args.output_dir) / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    LOG.info("Test results: %s", metrics)


def load_lora_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    cehrgpt_args: CehrGPTArguments,
) -> PeftModel:
    LOG.info("Loading base model from %s", model_args.model_name_or_path)
    model = load_finetuned_model(
        model_args, training_args, model_args.model_name_or_path
    )
    # Enable include_values when include_values is set to be False during pre-training
    if model_args.include_values and not model.cehrgpt.include_values:
        model.cehrgpt.include_values = True
    if cehrgpt_args.expand_tokenizer:
        tokenizer = CehrGptTokenizer.from_pretrained(training_args.output_dir)
        # Expand tokenizer to adapt to the finetuning dataset
        if model.config.vocab_size < tokenizer.vocab_size:
            model.resize_token_embeddings(tokenizer.vocab_size)
        if (
            model.config.include_values
            and model.config.value_vocab_size < tokenizer.value_vocab_size
        ):
            model.resize_value_embeddings(tokenizer.value_vocab_size)
    LOG.info("Loading LoRA adapter from %s", training_args.output_dir)
    return PeftModel.from_pretrained(model, model_id=training_args.output_dir)


if __name__ == "__main__":
    main()
