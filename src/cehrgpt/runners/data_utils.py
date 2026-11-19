import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import polars as pl
import torch
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_meds_extension_path,
    load_parquet_as_dataset,
)
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from meds import held_out_split, subject_id_field, train_split, tuning_split
from transformers import TrainingArguments
from transformers.utils import logging

from cehrgpt.data.cache_cleanup import clean_up_dataset_caches, remove_dataset_cache_files
from cehrgpt.data.hf_cehrgpt_dataset_mapping import (
    ExtractTokenizedSequenceDataMapping,
    MedToCehrGPTDatasetMapping,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.data.meds_dataset import create_meds_dataset
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
            dataset = create_meds_dataset(
                data_args=data_args,
                rules_file=cehrgpt_args.meds_visit_rules_file,
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
                # The files created from the data generator are removed with the cache files of the dataset
                stats = clean_up_dataset_caches(dataset, cache_file_collector)
                LOG.info(
                    "Clean up the cached files for the cehrgpt dataset transformed from the MEDS: %s",
                    stats,
                )
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


def _clamp_num_proc(dataset_len: int, num_proc: Optional[int], batch_size: int) -> Optional[int]:
    """
    Clamp num_proc so we don't spawn more workers than there are rows.

    datasets splits the rows evenly across the workers, whatever the batch size is, so the
    clamp must not depend on it: capping at ceil(rows / batch_size) leaves a dataset smaller
    than the batch size on a single process, however many workers were requested. batch_size
    is only kept for the callers.
    """
    if num_proc is None:
        return None
    return min(num_proc, max(1, dataset_len))


# Helper function to apply patient-based filtering
def filter_by_patient_ids(
        dataset: Dataset,
        patient_ids: List[Union[str, int]],
        data_args: DataTrainingArguments,
):
    unique_ids = set(patient_ids)
    return dataset.filter(
        lambda batch: [pid in unique_ids for pid in batch["person_id"]],
        num_proc=_clamp_num_proc(len(dataset), data_args.preprocessing_num_workers, data_args.preprocessing_batch_size),
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
    )


def load_patient_splits(
        patient_splits_path: str,
        unique_patient_ids: List[Union[str, int]],
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

    patient_splits = patient_splits.filter(
        pl.col(subject_id_field if is_data_in_meds else "person_id").is_in(
            unique_patient_ids
        )
    )

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
            cehrgpt_args.patient_splits_path,
            unique_patient_ids
        )

    # In case that we could not retrieve the corresponding patient_ids
    if not train_patient_ids:
        train_patient_ids = unique_patient_ids[:train_end]

    if not val_patient_ids:
        val_patient_ids = unique_patient_ids[train_end:]

    if not test_set and not test_patient_ids:
        validation_end = int(len(val_patient_ids) * data_args.test_eval_ratio)
        test_patient_ids = val_patient_ids[validation_end:]
        val_patient_ids = val_patient_ids[:validation_end]

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


def generate_cohort_prepared_ds_path(
    data_args: DataTrainingArguments,
    model_args,
    cehrgpt_args: CehrGPTArguments,
) -> Path:
    """
    The folder a prepared cohort dataset is cached in.

    generate_prepared_ds_path names it after the cohort folder, the tokenizer and the split
    settings. A cohort sliced out of the full tokenized dataset (extract_cohort_sequences) is
    also cut to the observation window, so the window and the dataset it was sliced from are
    added to the name. Without them a run with another window loads the sequences of the first.
    """
    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )
    if cehrgpt_args.tokenized_full_dataset_path is None:
        return prepared_ds_path
    # None, 0 and a negative window all keep the whole history (ExtractTokenizedSequenceDataMapping)
    window = data_args.observation_window
    window_name = str(window) if window and window > 0 else "all"
    full_dataset_path = os.path.abspath(
        os.path.expanduser(cehrgpt_args.tokenized_full_dataset_path)
    )
    full_dataset_hash = hashlib.md5(full_dataset_path.encode("utf-8")).hexdigest()[:8]
    return prepared_ds_path.with_name(
        f"{prepared_ds_path.name}_obs{window_name}_{full_dataset_hash}"
    )


def extract_cohort_sequences(
    data_args: DataTrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    tokenizer: Optional[CehrGptTokenizer] = None,
) -> DatasetDict:
    """
    The cohort sequences sliced out of the tokenized dataset, as cache files next to that dataset.

    See _cohort_persons_and_transform for the steps. save_cohort_sequences_in_chunks writes the same
    sequences to a folder a chunk of persons at a time, without the cache of the whole cohort.
    """
    filtered_tokenized_dataset, transform = _cohort_persons_and_transform(
        data_args, cehrgpt_args, tokenizer
    )
    return DatasetDict({
        split: _slice_cohort_sequences(ds, transform, data_args)
        for split, ds in filtered_tokenized_dataset.items()
    })


def _cohort_persons_and_transform(
    data_args: DataTrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    tokenizer: Optional[CehrGptTokenizer] = None,
) -> Tuple[DatasetDict, ExtractTokenizedSequenceDataMapping]:
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
    6. Builds the transformation (`ExtractTokenizedSequenceDataMapping`) that _slice_cohort_sequences
       applies to generate observation-window-constrained patient sequences, with recalculated
       demographic tokens prepended to each extracted slice.

    Args:
        data_args (DataTrainingArguments): Configuration parameters for data processing,
            including cohort folder, observation window, batch size, and parallelism.
        cehrgpt_args (CehrGPTArguments): Contains paths to pre-tokenized datasets and CEHR-GPT-specific arguments.
        tokenizer (Optional[CehrGptTokenizer]): Tokenizer used to encode the recalculated demographic
            tokens (year, age) for the `input_ids` column. If ``None``, the original demographic
            token ids are reused as a fallback.
    Returns:
        The tokenized dataset filtered to the persons of the cohort (per split), and the transformation
        that slices their sequences at the index dates.

    Raises:
        RuntimeError: If any `person_id` in the cohort is missing from the tokenized dataset.
    """

    lf = pl.scan_parquet(os.path.join(data_args.cohort_folder, "**", "*.parquet"))
    # Auto-detect MEDS format from the actual schema, regardless of the is_data_in_meds flag
    is_meds = data_args.is_data_in_meds or "subject_id" in set(lf.schema.names())
    if is_meds:
        id_col, time_col, label_col = "subject_id", "prediction_time", "boolean_value"
    else:
        id_col, time_col, label_col = "person_id", "index_date", "label"

    cols_to_load = [id_col, time_col]
    if label_col in lf.schema:
        cols_to_load.append(label_col)
    cohort = lf.select(cols_to_load).collect()
    LOG.info("Loaded cohort with %s rows from %s", len(cohort), data_args.cohort_folder)

    if is_meds:
        rename_map = {"subject_id": "person_id", "prediction_time": "index_date"}
        if "boolean_value" in cohort.columns:
            rename_map["boolean_value"] = "label"
        cohort = cohort.rename(rename_map)

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

    all_person_ids_set = set(all_person_ids)
    filtered_tokenized_dataset = DatasetDict({
        split: ds.filter(
            lambda batch: [pid in all_person_ids_set for pid in batch["person_id"]],
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            num_proc=_clamp_num_proc(len(ds), data_args.preprocessing_num_workers, data_args.preprocessing_batch_size),
        )
        for split, ds in tokenized_dataset.items()
    })

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
    tokenized_person_ids = set()
    for _, dataset in filtered_tokenized_dataset.items():
        tokenized_person_ids.update(dataset["person_id"])
    missing_person_ids = [
        person_id
        for person_id in person_index_date_map.keys()
        if person_id not in tokenized_person_ids
    ]
    if missing_person_ids:
        if not cehrgpt_args.allow_missing_tokenized_persons:
            raise RuntimeError(
                f"There are {len(missing_person_ids)} missing in the tokenized dataset. "
                f"The list contains: {missing_person_ids}"
            )
        # Only persons present in the tokenized dataset are sliced below, so the missing
        # cohort members are simply left out of the result.
        LOG.warning(
            "%s of %s cohort persons are missing in the tokenized dataset and will be "
            "skipped. First missing person_ids: %s",
            len(missing_person_ids),
            len(person_index_date_map),
            missing_person_ids[:20],
        )
        if len(missing_person_ids) == len(person_index_date_map):
            raise RuntimeError(
                "None of the cohort persons exist in the tokenized dataset, "
                "so there is nothing to extract."
            )
    if tokenizer is not None and not hasattr(tokenizer, "encode"):
        # The third argument used to be a CacheFileCollector; fail here with a clear message
        # instead of an AttributeError inside a datasets.map worker process.
        raise TypeError(
            "extract_cohort_sequences expects a CehrGptTokenizer as its third argument, "
            f"got {type(tokenizer).__name__}"
        )
    transform = ExtractTokenizedSequenceDataMapping(
        person_index_date_map, data_args.observation_window, tokenizer
    )
    return filtered_tokenized_dataset, transform


def _slice_cohort_sequences(
    ds: Dataset, transform: ExtractTokenizedSequenceDataMapping, data_args: DataTrainingArguments
) -> Dataset:
    """Slice the sequences of the persons of ds at their index dates."""
    return ds.map(
        transform.batch_transform,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        # Cap writer_batch_size independently: clinical sequences can be very
        # long, and matching it to preprocessing_batch_size can exceed the
        # 2 GB Arrow write limit for large cohorts.
        writer_batch_size=min(data_args.preprocessing_batch_size, 500),
        num_proc=_clamp_num_proc(len(ds), data_args.preprocessing_num_workers, data_args.preprocessing_batch_size),
        remove_columns=ds.column_names,
    )


# Written last by save_cohort_sequences_in_chunks: the chunks of every split, in order
COHORT_CHUNKS_FILE = "cohort_chunks.json"


def save_cohort_sequences_in_chunks(
    data_args: DataTrainingArguments,
    cehrgpt_args: CehrGPTArguments,
    tokenizer: Optional[CehrGptTokenizer],
    output_path: Union[str, Path],
    num_persons_per_chunk: int,
) -> Dict[str, List[str]]:
    """
    Slice the cohort sequences out of the tokenized dataset and save them a chunk of persons at a time.

    One person gives a sequence per index date, so a slice is several times the size of the persons'
    sequences: for MIMIC 80 to 130 GB. Sliced at once, the cache files of the slice and the saved copy of
    it were on disk together. Here every chunk is saved to <output_path>/<split>/chunk_<n> and its cache
    files removed before the next one is sliced, so the disk holds the saved slice and one chunk's cache.
    load_cohort_sequence_chunks joins the chunks in order, which gives the rows of extract_cohort_sequences.

    The file COHORT_CHUNKS_FILE is written last; a folder without it is an interrupted run.
    """
    if num_persons_per_chunk < 1:
        raise ValueError(f"num_persons_per_chunk must be >= 1, got {num_persons_per_chunk}")
    filtered_tokenized_dataset, transform = _cohort_persons_and_transform(
        data_args, cehrgpt_args, tokenizer
    )
    output_path = Path(output_path)
    chunks: Dict[str, List[str]] = {}
    for split, ds in filtered_tokenized_dataset.items():
        split_chunks = []
        for start in range(0, len(ds), num_persons_per_chunk):
            persons = ds.select(range(start, min(start + num_persons_per_chunk, len(ds))))
            sliced = _slice_cohort_sequences(persons, transform, data_args)
            if len(sliced) > 0:
                # save_to_disk raises on a dataset without rows
                name = f"chunk_{len(split_chunks):05d}"
                sliced.save_to_disk(str(output_path / split / name))
                split_chunks.append(name)
            removed = remove_dataset_cache_files(sliced) + remove_dataset_cache_files(persons)
            LOG.info(
                "Sliced the %s persons %s to %s of %s into %s sequences, removed %s cache files",
                split, start, start + len(persons), len(ds), len(sliced), removed,
            )
        if split_chunks:
            chunks[split] = split_chunks
        # the indices of the persons of the cohort, written by the filter next to the tokenized dataset
        remove_dataset_cache_files(ds)
    with open(output_path / COHORT_CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)
    return chunks


def is_cohort_sequence_chunks(path: Union[str, Path]) -> bool:
    return (Path(path) / COHORT_CHUNKS_FILE).exists()


def load_cohort_sequence_chunks(path: Union[str, Path]) -> DatasetDict:
    """The cohort sequences that save_cohort_sequences_in_chunks saved, a split per list of chunks."""
    path = Path(path)
    with open(path / COHORT_CHUNKS_FILE) as f:
        chunks = json.load(f)
    return DatasetDict({
        split: concatenate_datasets([load_from_disk(str(path / split / name)) for name in names])
        for split, names in chunks.items()
    })
