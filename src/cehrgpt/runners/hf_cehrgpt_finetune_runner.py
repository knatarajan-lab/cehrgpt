import json
import os.path
from datetime import datetime
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
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import expit as sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
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

LOG = logging.get_logger("transformers")


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
    else:

        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_med:
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
                    feature_names=["concept_ids"],
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
    collator = CehrGptDataCollator(
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
    )

    if training_args.do_train:
        model = load_finetuned_model(model_args, model_args.model_name_or_path)
        # Enable include_values when include_values is set to be False during pre-training
        if model_args.include_values and not model.cehrgpt.include_values:
            model.cehrgpt.include_values = True
        # Enable position embeddings when position embeddings are disabled in pre-training
        if not model_args.exclude_position_ids and model.cehrgpt.exclude_position_ids:
            model.cehrgpt.exclude_position_ids = False
        if cehrgpt_args.expand_tokenizer:
            model.resize_token_embeddings(tokenizer.vocab_size)
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

        trainer = Trainer(
            model=model,
            data_collator=collator,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            callbacks=[EarlyStoppingCallback()],
            args=training_args,
        )

        checkpoint = get_last_hf_checkpoint(training_args)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        test_dataloader = DataLoader(
            dataset=processed_dataset["test"],
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collator,
            pin_memory=training_args.dataloader_pin_memory,
        )
        do_predict(test_dataloader, model_args, training_args)


def do_predict(
    test_dataloader: DataLoader,
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    """
    Performs inference on the test dataset using a fine-tuned model, saves predictions and evaluation metrics.

    The reason we created this custom do_predict is that there is a memory leakage for transformers trainer.predict(),
    for large test sets, it will throw the CPU OOM error

    Args:
        test_dataloader (DataLoader): DataLoader containing the test dataset, with batches of input features and labels.
        model_args (ModelArguments): Arguments for configuring and loading the fine-tuned model.
        training_args (TrainingArguments): Arguments related to training, evaluation, and output directories.

    Returns:
        None. Results are saved to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and LoRA adapters if applicable
    model = (
        load_finetuned_model(model_args, training_args.output_dir)
        if not model_args.use_lora
        else load_lora_model(model_args, training_args)
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


def load_lora_model(model_args, training_args) -> PeftModel:
    LOG.info("Loading base model from %s", model_args.model_name_or_path)
    base_model = load_finetuned_model(model_args, model_args.model_name_or_path)
    LOG.info("Loading LoRA adapter from %s", training_args.output_dir)
    return PeftModel.from_pretrained(base_model, model_id=training_args.output_dir)


if __name__ == "__main__":
    main()
