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
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
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
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    # Try to create a new model based on the base model
    try:
        return finetune_model_cls.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
    except ValueError:
        raise ValueError(f"Can not load the finetuned model from {model_name_or_path}")


def create_dataset_splits(data_args: DataTrainingArguments, seed: int):
    """
    Splits a dataset into training, validation, and testing subsets using specified parameters from `data_args`.

    This function accommodates both streaming and batch data processing scenarios. In streaming mode,
    the function performs stateful shuffling and manual splitting based on predefined record counts.
    In batch mode, it leverages built-in functions to split the dataset randomly according to specified percentages.

    Parameters:
        data_args (DataTrainingArguments): Configuration object containing data-related parameters, which includes:
            - data_folder (str): Path to the dataset directory.
            - test_data_folder (str, optional): Path to the directory containing separate test data.
            - streaming (bool): Flag to indicate if the dataset should be processed as a stream.
            - validation_split_num (int, optional): Number of records to use for validation when streaming.
            - validation_split_percentage (float): Percentage of data to allocate to validation in batch mode.
            - test_eval_ratio (float, optional): Ratio of validation set size to allocate to test set creation if no separate test dataset is provided.
            - chronological_split (bool, optional): Flag to indicate if splits should be made based on chronological order (not implemented in the current function).
            - split_by_patient (bool, optional): Flag to indicate if splits should ensure no patient overlap (not implemented in the current function).
        seed (int): Seed for random number generation to ensure reproducibility of splits.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing training, validation, and testing datasets in that order.

    Raises:
        ValueError: If `validation_split_num` is not defined or is zero when required in streaming mode.

    Examples:
        >>> data_args = DataTrainingArguments(
                data_folder="data/",
                validation_split_percentage=0.1,
                test_eval_ratio=0.2,
                streaming=True,
                validation_split_num=500
            )
        >>> train_set, validation_set, test_set = create_dataset_splits(data_args, seed=42)
    """
    dataset = load_parquet_as_dataset(
        data_args.data_folder, streaming=data_args.streaming
    )
    test_set = (
        None
        if not data_args.test_data_folder
        else load_parquet_as_dataset(data_args.test_data_folder)
    )

    if data_args.streaming:
        # When streaming, we apply a stateful shuffle if possible, or a basic shuffle if not.
        dataset = dataset.shuffle(
            buffer_size=10000, seed=seed
        )  # Buffer size may need to be adjusted based on memory.
        # Assume validation_split_num indicates how many records to use for validation.
        # The ratio of test to validation needs to be handled manually.
        if (
            hasattr(data_args, "validation_split_num")
            and data_args.validation_split_num > 0
        ):
            # Assume validation_take_size and test_take_size are calculated based on total expected records
            validation_take_size = data_args.validation_split_num
            test_take_size = int(validation_take_size * data_args.test_eval_ratio)

            # Stream through the dataset, taking validation and test subsets sequentially
            combined_validation_test_set = dataset.take(validation_take_size)
            # The remainder is the training set, but to correctly separate validation and test sets:
            # We need to ensure that we take and skip the right amounts in one sequence without revisiting
            if test_take_size > 0 and test_set is None:
                test_set = combined_validation_test_set.take(test_take_size)
                validation_set = combined_validation_test_set.skip(test_take_size)
            else:
                validation_set = combined_validation_test_set

            # For the training set, you'd normally continue streaming the rest after handling validation and test
            train_set = dataset.skip(
                validation_take_size
            )  # This assumes you can continue streaming after the take
        else:
            raise ValueError(
                "validation_split_num must be defined and greater than 0 for streaming datasets."
            )
    else:
        # Perform a random split for training, validation, and possibly test datasets in batch context.
        train_val_split = dataset.train_test_split(
            test_size=data_args.validation_split_percentage, seed=seed
        )
        train_set = train_val_split["train"]
        validation_set = train_val_split["test"]

        if (
            test_set is None
            and hasattr(data_args, "test_eval_ratio")
            and data_args.test_eval_ratio > 0
        ):
            test_valid_split = validation_set.train_test_split(
                test_size=data_args.test_eval_ratio, seed=seed
            )
            validation_set = test_valid_split["train"]
            test_set = test_valid_split["test"]

    return train_set, validation_set, test_set


def model_init(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: CehrGptTokenizer,
):
    model = load_finetuned_model(
        model_args, training_args, model_args.model_name_or_path
    )
    if model.config.max_position_embeddings < model_args.max_position_embeddings:
        LOG.info(
            f"Increase model.config.max_position_embeddings to {model_args.max_position_embeddings}"
        )
        model.config.max_position_embeddings = model_args.max_position_embeddings
        model.resize_position_embeddings(model_args.max_position_embeddings)
    # Enable include_values when include_values is set to be False during pre-training
    if model_args.include_values and not model.cehrgpt.include_values:
        model.cehrgpt.include_values = True
    # Enable position embeddings when position embeddings are disabled in pre-training
    if not model_args.exclude_position_ids and model.cehrgpt.exclude_position_ids:
        model.cehrgpt.exclude_position_ids = False
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

    if data_args.streaming:
        # This is for disabling the warning message https://github.com/huggingface/transformers/issues/5486
        # This happens only when streaming is enabled
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to be set to 0
        # Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0
        training_args.dataloader_prefetch_factor = None

    tokenizer = load_pretrained_tokenizer(model_args)
    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )

    processed_dataset = None
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        if cehrgpt_args.expand_tokenizer:
            try:
                tokenizer = CehrGptTokenizer.from_pretrained(training_args.output_dir)
            except Exception:
                LOG.warning(
                    f"CehrGptTokenizer must exist in {training_args.output_dir} "
                    f"when the dataset has been processed and expand_tokenizer is set to True. "
                    f"Please delete the processed dataset at {prepared_ds_path}."
                )
                processed_dataset = None
                shutil.rmtree(prepared_ds_path)

    if processed_dataset is None:
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
                    dataset.save_to_disk(str(meds_extension_path))
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
                # Try to use the defined pretrained embeddings if exists,
                # Otherwise we default to the pretrained model embedded in the pretrained model
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
                tokenizer.save_pretrained(os.path.expanduser(training_args.output_dir))

        processed_dataset = create_cehrgpt_finetuning_dataset(
            dataset=final_splits, cehrgpt_tokenizer=tokenizer, data_args=data_args
        )
        if not data_args.streaming:
            processed_dataset.save_to_disk(str(prepared_ds_path))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
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
    # We suppress the additional learning objectives in fine-tuning
    data_collator = CehrGptDataCollator(
        tokenizer=tokenizer,
        max_length=(
            config.max_position_embeddings - 1
            if config.causal_sfm
            else config.max_position_embeddings
        ),
        include_values=model_args.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
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
            person_ids = (
                batch.pop("person_id").numpy().squeeze().astype(int)
                if "person_id" in batch
                else None
            )
            index_dates = (
                map(
                    datetime.fromtimestamp,
                    batch.pop("index_date").numpy().squeeze(axis=-1).tolist(),
                )
                if "index_date" in batch
                else None
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            output = model(**batch, output_attentions=False, output_hidden_states=False)
            test_losses.append(output.loss.item())

            # Collect logits and labels for prediction
            logits = output.logits.float().cpu().numpy().squeeze()
            labels = (
                batch["classifier_label"].float().cpu().numpy().squeeze().astype(bool)
            )
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
    model = load_finetuned_model(
        model_args, training_args, model_args.model_name_or_path
    )
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
        if (
            model.config.include_values
            and model.config.value_vocab_size < tokenizer.value_vocab_size
        ):
            model.resize_value_embeddings(tokenizer.value_vocab_size)
    LOG.info("Loading LoRA adapter from %s", training_args.output_dir)
    return PeftModel.from_pretrained(model, model_id=training_args.output_dir)


if __name__ == "__main__":
    main()
