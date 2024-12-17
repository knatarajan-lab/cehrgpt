import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    create_dataset_from_meds_reader,
)
from cehrbert.runners.hf_cehrbert_finetune_runner import compute_metrics
from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    FineTuneModelType,
    ModelArguments,
)
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
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
from transformers.tokenization_utils_base import LARGE_INTEGER
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPTConfig,
    CehrGptForClassification,
    CEHRGPTPreTrainedModel,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments
from cehrgpt.runners.hyperparameter_search_util import perform_hyperparameter_search

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
    model_args: ModelArguments, model_name_or_path: str
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
    # Try to create a new model based on the base model
    try:
        return finetune_model_cls.from_pretrained(
            model_name_or_path, attn_implementation=attn_implementation
        )
    except ValueError:
        raise ValueError(f"Can not load the finetuned model from {model_name_or_path}")


def create_dataset_splits(data_args: DataTrainingArguments, seed: int):
    """
    Creates training, validation, and testing dataset splits based on specified splitting strategies.

    This function splits a dataset into training, validation, and test sets, using either chronological,
    patient-based, or random splitting strategies, depending on the parameters provided in `data_args`.

    - **Chronological split**: Sorts by a specified date and splits based on historical and future data.
    - **Patient-based split**: Splits by unique patient IDs to ensure that patients in each split are distinct.
    - **Random split**: Performs a straightforward random split of the dataset.

    If `data_args.test_data_folder` is provided, a test set is loaded directly from it. Otherwise,
    the test set is created by further splitting the validation set based on `test_eval_ratio`.

    Parameters:
        data_args (DataTrainingArguments): A configuration object containing data-related arguments, including:
            - `data_folder` (str): Path to the main dataset.
            - `test_data_folder` (str, optional): Path to an optional test dataset.
            - `chronological_split` (bool): Whether to split chronologically.
            - `split_by_patient` (bool): Whether to split by unique patient IDs.
            - `validation_split_percentage` (float): Percentage of data to use for validation.
            - `test_eval_ratio` (float): Ratio of test to validation data when creating a test set from validation.
            - `preprocessing_num_workers` (int): Number of processes for parallel data filtering.
            - `preprocessing_batch_size` (int): Batch size for batched operations.
        seed (int): Random seed for reproducibility of splits.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing:
            - `train_set` (Dataset): Training split of the dataset.
            - `validation_set` (Dataset): Validation split of the dataset.
            - `test_set` (Dataset): Test split of the dataset.

    Raises:
        FileNotFoundError: If `data_args.data_folder` or `data_args.test_data_folder` does not exist.
        ValueError: If incompatible arguments are passed for splitting strategies.

    Example Usage:
        data_args = DataTrainingArguments(
            data_folder="data/",
            validation_split_percentage=0.1,
            test_eval_ratio=0.2,
            chronological_split=True
        )
        train_set, validation_set, test_set = create_dataset_splits(data_args, seed=42)
    """
    dataset = load_parquet_as_dataset(data_args.data_folder)
    test_set = (
        None
        if not data_args.test_data_folder
        else load_parquet_as_dataset(data_args.test_data_folder)
    )

    if data_args.chronological_split:
        # Chronological split by sorting on `index_date`
        dataset = dataset.sort("index_date")
        total_size = len(dataset)
        train_end = int((1 - data_args.validation_split_percentage) * total_size)

        # Perform the split
        train_set = dataset.select(range(0, train_end))
        validation_set = dataset.select(range(train_end, total_size))

        if test_set is None:
            test_valid_split = validation_set.train_test_split(
                test_size=data_args.test_eval_ratio, seed=seed
            )
            validation_set, test_set = (
                test_valid_split["train"],
                test_valid_split["test"],
            )

    elif data_args.split_by_patient:
        # Patient-based split
        LOG.info("Using the split_by_patient strategy")
        unique_patient_ids = dataset.unique("person_id")
        LOG.info(f"There are {len(unique_patient_ids)} patients in total")

        np.random.seed(seed)
        np.random.shuffle(unique_patient_ids)

        train_end = int(
            len(unique_patient_ids) * (1 - data_args.validation_split_percentage)
        )
        train_patient_ids = set(unique_patient_ids[:train_end])

        if test_set is None:
            validation_end = int(
                train_end
                + len(unique_patient_ids)
                * data_args.validation_split_percentage
                * data_args.test_eval_ratio
            )
            val_patient_ids = set(unique_patient_ids[train_end:validation_end])
            test_patient_ids = set(unique_patient_ids[validation_end:])
        else:
            val_patient_ids, test_patient_ids = (
                set(unique_patient_ids[train_end:]),
                None,
            )

        # Helper function to apply patient-based filtering
        def filter_by_patient_ids(patient_ids):
            return dataset.filter(
                lambda batch: [pid in patient_ids for pid in batch["person_id"]],
                num_proc=data_args.preprocessing_num_workers,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
            )

        # Generate splits
        train_set = filter_by_patient_ids(train_patient_ids)
        validation_set = filter_by_patient_ids(val_patient_ids)
        if test_set is None:
            test_set = filter_by_patient_ids(test_patient_ids)

    else:
        # Random split
        train_val = dataset.train_test_split(
            test_size=data_args.validation_split_percentage, seed=seed
        )
        train_set, validation_set = train_val["train"], train_val["test"]

        if test_set is None:
            test_valid_split = validation_set.train_test_split(
                test_size=data_args.test_eval_ratio, seed=seed
            )
            validation_set, test_set = (
                test_valid_split["train"],
                test_valid_split["test"],
            )

    return train_set, validation_set, test_set


def model_init(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: CehrGptTokenizer,
):
    model = load_finetuned_model(model_args, model_args.model_name_or_path)
    # Enable include_values when include_values is set to be False during pre-training
    if model_args.include_values and not model.cehrgpt.include_values:
        model.cehrgpt.include_values = True
    # Enable position embeddings when position embeddings are disabled in pre-training
    if not model_args.exclude_position_ids and model.cehrgpt.exclude_position_ids:
        model.cehrgpt.exclude_position_ids = False
    # Expand tokenizer to adapt to the finetuning dataset
    if model.config.vocab_size < tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
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
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        if cehrgpt_args.expand_tokenizer:
            try:
                tokenizer = CehrGptTokenizer.from_pretrained(training_args.output_dir)
            except Exception:
                raise RuntimeError(
                    f"CehrGptTokenizer must exist in {training_args.output_dir} "
                    f"when the dataset has been processed and expand_tokenizer is set to True"
                )
    else:
        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_meds:
            meds_extension_path = get_meds_extension_path(
                data_folder=data_args.cohort_folder,
                dataset_prepared_path=data_args.dataset_prepared_path,
            )
            try:
                LOG.info(
                    f"Trying to load the MEDS extension from disk at {meds_extension_path}..."
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
            except Exception as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(
                    data_args, is_pretraining=False
                )
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
            train_set = dataset["train"]
            validation_set = dataset["validation"]
            test_set = dataset["test"]
        else:
            train_set, validation_set, test_set = create_dataset_splits(
                data_args=data_args, seed=training_args.seed
            )
        # Organize them into a single DatasetDict
        final_splits = DatasetDict(
            {"train": train_set, "validation": validation_set, "test": test_set}
        )

        if cehrgpt_args.expand_tokenizer:
            new_tokenizer_path = os.path.expanduser(training_args.output_dir)
            try:
                tokenizer = CehrGptTokenizer.from_pretrained(new_tokenizer_path)
            except Exception:
                tokenizer = CehrGptTokenizer.expand_trained_tokenizer(
                    cehrgpt_tokenizer=tokenizer,
                    dataset=final_splits["train"],
                    data_args=data_args,
                    concept_name_mapping={},
                )
                tokenizer.save_pretrained(os.path.expanduser(training_args.output_dir))

        processed_dataset = create_cehrgpt_finetuning_dataset(
            dataset=final_splits, cehrgpt_tokenizer=tokenizer, data_args=data_args
        )
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    processed_dataset.set_format("pt")

    config = CEHRGPTConfig.from_pretrained(model_args.model_name_or_path)
    # We suppress the additional learning objectives in fine-tuning
    data_collator = CehrGptDataCollator(
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings,
        include_values=model_args.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
    )

    if training_args.do_train:
        if cehrgpt_args.hyperparameter_tuning:
            model_args.early_stopping_patience = LARGE_INTEGER
            training_args = perform_hyperparameter_search(
                partial(model_init, model_args, training_args, tokenizer),
                processed_dataset,
                data_collator,
                training_args,
                model_args,
                cehrgpt_args,
            )
            # Always retrain with the full set when hyperparameter tuning is set to true
            retrain_with_full_set(
                model_args, training_args, tokenizer, processed_dataset, data_collator
            )
        else:
            # Initialize Trainer for final training on the combined train+val set
            trainer = Trainer(
                model=model_init(model_args, training_args, tokenizer),
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

            # Retrain the model with full set using the num of epoches before earlying stopping
            if cehrgpt_args.retrain_with_full:
                update_num_epoch_before_early_stopping_callback = None
                for callback in trainer.callback_handler.callbacks:
                    if isinstance(callback, UpdateNumEpochsBeforeEarlyStoppingCallback):
                        update_num_epoch_before_early_stopping_callback = callback

                if update_num_epoch_before_early_stopping_callback is None:
                    raise RuntimeError(
                        f"{UpdateNumEpochsBeforeEarlyStoppingCallback} must be included as a callback!"
                    )
                final_num_epochs = (
                    update_num_epoch_before_early_stopping_callback.num_epochs_before_early_stopping
                )
                training_args.num_train_epochs = final_num_epochs
                LOG.info(
                    "Num Epochs before early stopping: %s",
                    training_args.num_train_epochs,
                )
                retrain_with_full_set(
                    model_args,
                    training_args,
                    tokenizer,
                    processed_dataset,
                    data_collator,
                )

    if training_args.do_predict:
        test_dataloader = DataLoader(
            dataset=processed_dataset["test"],
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=data_collator,
            pin_memory=training_args.dataloader_pin_memory,
        )
        do_predict(test_dataloader, model_args, training_args, cehrgpt_args)


def retrain_with_full_set(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: CehrGptTokenizer,
    dataset: DatasetDict,
    data_collator: CehrGptDataCollator,
) -> None:
    """
    Retrains a model on the full training and validation dataset for final performance evaluation.

    This function consolidates the training and validation datasets into a single
    dataset for final model training, updates the output directory for the final model,
    and disables evaluation during training. It resumes from the latest checkpoint if available,
    trains the model on the combined dataset, and saves the model along with training metrics
    and state information.

    Args:
        model_args (ModelArguments): Model configuration and hyperparameters.
        training_args (TrainingArguments): Training configuration, including output directory,
                                           evaluation strategy, and other training parameters.
        tokenizer (CehrGptTokenizer): Tokenizer instance specific to CEHR-GPT.
        dataset (DatasetDict): A dictionary containing the 'train' and 'validation' datasets.
        data_collator (CehrGptDataCollator): Data collator for handling data batching and tokenization.

    Returns:
        None
    """
    # Initialize Trainer for final training on the combined train+val set
    full_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    training_args.output_dir = os.path.join(training_args.output_dir, "full")
    LOG.info(
        "Final output_dir for final_training_args.output_dir %s",
        training_args.output_dir,
    )
    Path(training_args.output_dir).mkdir(exist_ok=True)
    # Disable evaluation
    training_args.evaluation_strategy = "no"
    checkpoint = get_last_hf_checkpoint(training_args)
    final_trainer = Trainer(
        model=model_init(model_args, training_args, tokenizer),
        data_collator=data_collator,
        args=training_args,
        train_dataset=full_dataset,
        tokenizer=tokenizer,
    )
    final_train_result = final_trainer.train(resume_from_checkpoint=checkpoint)
    final_trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = final_train_result.metrics
    final_trainer.log_metrics("train", metrics)
    final_trainer.save_metrics("train", metrics)
    final_trainer.save_state()


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
        load_finetuned_model(model_args, training_args.output_dir)
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
            person_ids = batch.pop("person_id").numpy().squeeze().astype(int)
            index_dates = (
                map(
                    datetime.fromtimestamp,
                    batch.pop("index_date").numpy().squeeze().tolist(),
                )
                if "index_date" in batch
                else None
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            output = model(**batch, output_attentions=False, output_hidden_states=False)
            test_losses.append(output.loss.item())

            # Collect logits and labels for prediction
            logits = output.logits.cpu().numpy().squeeze()
            labels = batch["classifier_label"].cpu().numpy().squeeze().astype(bool)
            probabilities = sigmoid(logits)
            # Save predictions to parquet file
            test_prediction_pd = pd.DataFrame(
                {
                    "subject_id": person_ids,
                    "prediction_time": index_dates,
                    "boolean_prediction_probability": probabilities,
                    "boolean_prediction": logits,
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
        probs=test_prediction_pd.boolean_prediction_probability,
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
    model = load_finetuned_model(model_args, model_args.model_name_or_path)
    # Enable include_values when include_values is set to be False during pre-training
    if model_args.include_values and not model.cehrgpt.include_values:
        model.cehrgpt.include_values = True
    # Enable position embeddings when position embeddings are disabled in pre-training
    if not model_args.exclude_position_ids and model.cehrgpt.exclude_position_ids:
        model.cehrgpt.exclude_position_ids = False
    if cehrgpt_args.expand_tokenizer:
        tokenizer = CehrGptTokenizer.from_pretrained(training_args.output_dir)
        # Expand tokenizer to adapt to the finetuning dataset
        if model.config.vocab_size < tokenizer.vocab_size:
            model.resize_token_embeddings(tokenizer.vocab_size)
        if model.config.value_vocab_size < tokenizer.value_vocab_size:
            model.resize_value_embeddings(tokenizer.value_vocab_size)
    LOG.info("Loading LoRA adapter from %s", training_args.output_dir)
    return PeftModel.from_pretrained(model, model_id=training_args.output_dir)


if __name__ == "__main__":
    main()
