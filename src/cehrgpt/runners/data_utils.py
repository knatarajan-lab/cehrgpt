import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import polars as pl
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
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from meds import held_out_split, subject_id_field, train_split, tuning_split
from transformers import TrainingArguments
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset_mapping import (
    ExtractTokenizedSequenceDataMapping,
    MedToCehrGPTDatasetMapping,
)
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
    cache_file_collector: Optional[CacheFileCollector] = None,
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
                if cache_file_collector:
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
            data_args=data_args,
            cehrgpt_args=cehrgpt_args,
            seed=training_args.seed,
        )
    # Organize them into a single DatasetDict
    final_splits = DatasetDict(
        {"train": train_set, "validation": validation_set, "test": test_set}
    )
    return final_splits


# Helper function to apply patient-based filtering
def filter_by_patient_ids(
        dataset: Dataset,
        patient_ids: List[Union[str, int]],
        data_args: DataTrainingArguments,
):
    unique_ids = set(patient_ids)
    return dataset.filter(
        lambda batch: [pid in unique_ids for pid in batch["person_id"]],
        num_proc=data_args.preprocessing_num_workers,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
    )


def load_patient_splits(
        patient_splits_path: str
) -> Tuple[List[Union[str, int]], List[Union[str, int]], List[Union[str, int]]]:
    # If the patient_splits path is a folder (cehrgpt patient splits), then we assume that it contains a list of parquet files.
    if os.path.isdir(patient_splits_path):
        patient_splits = pl.read_parquet(
            os.path.join(patient_splits_path, "*.parquet")
        )
    else:
        # If the patient_splits path is a file, it must be a parquet file
        file_parts = os.path.splitext(patient_splits_path)
        if not file_parts or not file_parts[-1].endswith("parquet"):
            raise ValueError(
                f"{patient_splits_path} is not a valid patient splits path."
            )
        patient_splits = pl.read_parquet(patient_splits_path)

    if len(patient_splits) == 0:
        raise RuntimeError(
            f"The patient_splits at {patient_splits_path} contains empty rows"
        )
    split_values = patient_splits.select("split").unique()["split"].to_list()
    is_split_meds_schema = np.all(
        [
            split in split_values
            for split in [train_split, tuning_split, held_out_split]
        ]
    )
    # Auto-detect whether or not the patient splits follow the MEDS schema or not.
    is_data_in_meds = (subject_id_field in patient_splits.columns) and is_split_meds_schema
    if is_data_in_meds:
        train_patient_ids = patient_splits.filter(pl.col("split") == train_split)[
            subject_id_field
        ].to_list()
        val_patient_ids = patient_splits.filter(pl.col("split") == tuning_split)[
            subject_id_field
        ].to_list()
        test_patient_ids = patient_splits.filter(pl.col("split") == held_out_split)[
            subject_id_field
        ].to_list()
    else:
        train_patient_ids = patient_splits.filter(pl.col("split") == "train")[
            "person_id"
        ].to_list()
        val_patient_ids = patient_splits.filter(pl.col("split") == "validation")[
            "person_id"
        ].to_list()
        test_patient_ids = patient_splits.filter(pl.col("split") == "test")[
            "person_id"
        ].to_list()

    return train_patient_ids, val_patient_ids, test_patient_ids


def create_dataset_splits(
    data_args: DataTrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    seed: int,
):
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
        cehrgpt_args (CehrGPTArguments): A configuration object containing CehrGPTArguments.
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

    # Patient-based split
    LOG.info("Using the split_by_patient strategy")
    unique_patient_ids = dataset.unique("person_id")
    np.random.seed(seed)
    np.random.shuffle(unique_patient_ids)
    train_end = int(
        len(unique_patient_ids) * (1 - data_args.validation_split_percentage)
    )
    LOG.info(f"There are {len(unique_patient_ids)} patients in total")

    train_patient_ids, val_patient_ids, test_patient_ids = None, None, None

    if cehrgpt_args.patient_splits_path:
        train_patient_ids, val_patient_ids, test_patient_ids = load_patient_splits(
            cehrgpt_args.patient_splits_path
        )

    # In case that we could not retrieve the corresponding patient_ids
    if not train_patient_ids:
        train_patient_ids = unique_patient_ids[:train_end]

    if not val_patient_ids:
        val_patient_ids = unique_patient_ids[train_end:]

    if not test_set and not test_patient_ids:
        validation_end = int(len(val_patient_ids) * data_args.test_eval_ratio)
        val_patient_ids = val_patient_ids[:validation_end]
        test_patient_ids = val_patient_ids[validation_end:]

    # Generate splits
    train_set = filter_by_patient_ids(
        dataset=dataset,
        patient_ids=train_patient_ids,
        data_args=data_args
    ).shuffle(
        seed=seed
    )
    validation_set = filter_by_patient_ids(
        dataset=dataset,
        patient_ids=val_patient_ids,
        data_args=data_args,
    )
    if not test_set:
        test_set = filter_by_patient_ids(
            dataset=dataset,
            patient_ids=test_patient_ids,
            data_args=data_args,
        )

    LOG.info("Train size: %s", len(train_set))
    LOG.info("Validation size: %s", len(validation_set))
    LOG.info("Test size: %s", len(test_set))

    return train_set, validation_set, test_set


def extract_cohort_sequences(
    data_args: DataTrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> DatasetDict:
    """
    Extracts and processes cohort-specific tokenized sequences from a pre-tokenized dataset,.

    based on the provided cohort Parquet files and observation window constraints.

    This function performs the following steps:
    1. Loads cohort definitions from Parquet files located in `data_args.cohort_folder`.
    2. Renames relevant columns if the data originates from a Meds format.
    3. Filters a pre-tokenized dataset (loaded from `cehrgpt_args.tokenized_full_dataset_path`)
       to include only patients present in the cohort.
    4. Aggregates each person's index date and label into a mapping.
    5. Checks for consistency to ensure all cohort person_ids are present in the tokenized dataset.
    6. Applies a transformation (`ExtractTokenizedSequenceDataMapping`) to generate
       observation-window-constrained patient sequences.
    7. Caches both the filtered and processed datasets using the provided `cache_file_collector`.

    Args:
        data_args (DataTrainingArguments): Configuration parameters for data processing,
            including cohort folder, observation window, batch size, and parallelism.
        cehrgpt_args (CehrGPTArguments): Contains paths to pre-tokenized datasets and CEHR-GPT-specific arguments.
        cache_file_collector (CacheFileCollector): Utility to register and manage dataset cache files.

    Returns:
        DatasetDict: A Hugging Face `DatasetDict` containing the processed datasets (e.g., train/validation/test),
                     where each entry includes sequences filtered and truncated by the observation window.

    Raises:
        RuntimeError: If any `person_id` in the cohort is missing from the tokenized dataset.
    """

    cohort = pl.read_parquet(os.path.join(data_args.cohort_folder, "*.parquet"))
    if data_args.is_data_in_meds:
        cohort = cohort.rename(
            mapping={
                "prediction_time": "index_date",
                "subject_id": "person_id",
                "boolean_value": "label",
            }
        )
    all_person_ids = cohort["person_id"].unique().to_list()
    # In case the label column does not exist, we add a fake column to the dataframe so subsequent process can work
    if "label" not in cohort.columns:
        cohort = cohort.with_columns(
            pl.Series(
                name="label", values=np.zeros_like(cohort["person_id"].to_numpy())
            )
        )

    # data_args.observation_window
    tokenized_dataset = load_from_disk(cehrgpt_args.tokenized_full_dataset_path)
    filtered_tokenized_dataset = tokenized_dataset.filter(
        lambda batch: [person_id in all_person_ids for person_id in batch["person_id"]],
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        num_proc=data_args.preprocessing_num_workers,
    )
    person_index_date_agg = cohort.group_by("person_id").agg(
        pl.struct("index_date", "label").alias("index_date_label")
    )
    # Convert to dictionary
    person_index_date_map: Dict[int, List[datetime]] = dict(
        zip(
            person_index_date_agg["person_id"].to_list(),
            person_index_date_agg["index_date_label"].to_list(),
        )
    )
    LOG.info(f"person_index_date_agg: {person_index_date_agg}")
    tokenized_person_ids = []
    for _, dataset in filtered_tokenized_dataset.items():
        tokenized_person_ids.extend(dataset["person_id"])
    missing_person_ids = [
        person_id
        for person_id in person_index_date_map.keys()
        if person_id not in tokenized_person_ids
    ]
    if missing_person_ids:
        raise RuntimeError(
            f"There are {len(missing_person_ids)} missing in the tokenized dataset. "
            f"The list contains: {missing_person_ids}"
        )
    processed_dataset = filtered_tokenized_dataset.map(
        ExtractTokenizedSequenceDataMapping(
            person_index_date_map, data_args.observation_window
        ).batch_transform,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=filtered_tokenized_dataset["train"].column_names,
    )
    return processed_dataset
