from typing import Optional, Union

import numpy as np
import torch
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    create_dataset_from_meds_reader,
)
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert.runners.runner_util import (
    get_meds_extension_path,
    load_parquet_as_dataset,
)
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    concatenate_datasets,
    load_from_disk,
)
from transformers import TrainingArguments
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset_mapping import MedToCehrGPTDatasetMapping
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

LOG = logging.get_logger("transformers")


def get_torch_dtype(torch_dtype: Optional[str] = None) -> Union[torch.dtype, str]:
    if torch_dtype and hasattr(torch, torch_dtype):
        return getattr(torch, torch_dtype)
    return torch.float


def data_collate_fn(features, model_type: torch.dtype, collator):
    batch = collator(features)
    if model_type != torch.float32:
        for key, value in batch.items():
            # Only convert float32 tensors to bfloat16
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(model_type)
    return batch


def prepare_finetune_dataset(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    cache_file_collector: CacheFileCollector,
) -> DatasetDict:
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

        train_set = dataset["train"]
        validation_set = dataset["validation"]
        test_set = dataset["test"]

        if cehrgpt_args.meds_repartition:
            train_val_set = concatenate_datasets([train_set, validation_set])
            if data_args.streaming and data_args.validation_split_num:
                train_val_set = train_val_set.shuffle(
                    buffer_size=10_000, seed=training_args.seed
                )
                train_set = train_val_set.skip(data_args.validation_split_num)
                validation_set = train_val_set.take(data_args.validation_split_num)
            elif data_args.validation_split_percentage:
                dataset = train_val_set.train_test_split(
                    test_size=data_args.validation_split_percentage,
                    seed=training_args.seed,
                )
                train_set = dataset["train"]
                validation_set = dataset["test"]
            else:
                raise RuntimeError(
                    f"Can not split the data. If streaming is enabled, validation_split_num needs to be "
                    f"defined, otherwise validation_split_percentage needs to be provided. "
                    f"The current values are:\n"
                    f"validation_split_percentage: {data_args.validation_split_percentage}\n"
                    f"validation_split_num: {data_args.validation_split_num}\n"
                    f"streaming: {data_args.streaming}"
                )
    else:
        train_set, validation_set, test_set = create_dataset_splits(
            data_args=data_args, seed=training_args.seed
        )
    # Organize them into a single DatasetDict
    final_splits = DatasetDict(
        {"train": train_set, "validation": validation_set, "test": test_set}
    )
    return final_splits


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
