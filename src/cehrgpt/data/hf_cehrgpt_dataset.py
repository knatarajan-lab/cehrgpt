from typing import Optional, Union

from cehrbert.data_generators.hf_data_generator.hf_dataset import (
    FINETUNING_COLUMNS,
    apply_cehrbert_dataset_mapping,
)
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict

from cehrgpt.data.hf_cehrgpt_dataset_mapping import (
    HFCehrGptTokenizationMapping,
    HFFineTuningMapping,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

CEHRGPT_COLUMNS = [
    "person_id",
    "concept_ids",
    "concept_values",
    "concept_value_masks",
    "num_of_concepts",
    "num_of_visits",
    "values",
    "value_indicators",
    "epoch_times",
    "ages",
    "genders",
    "races",
    "position_ids",
]

TRANSFORMER_COLUMNS = ["input_ids"]


def _apply_cehrgpt_dataset_mapping(
    dataset,
    mapping_function,
    data_args,
    cache_file_collector=None,
):
    if data_args.streaming:
        return apply_cehrbert_dataset_mapping(
            dataset,
            mapping_function,
            batch_size=data_args.preprocessing_batch_size,
            streaming=True,
            cache_file_collector=cache_file_collector,
        )

    def map_dataset(split):
        return split.map(
            mapping_function.batch_transform,
            num_proc=data_args.preprocessing_num_workers,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            new_fingerprint=mapping_function.cache_fingerprint(split._fingerprint),
        )

    if isinstance(dataset, DatasetDict):
        for name in dataset:
            dataset[name] = map_dataset(dataset[name])
    else:
        dataset = map_dataset(dataset)

    if mapping_function.remove_columns():
        dataset = dataset.remove_columns(mapping_function.remove_columns())
    if cache_file_collector:
        cache_file_collector.add_cache_files(dataset)
    return dataset


def create_cehrgpt_pretraining_dataset(
    dataset: Union[Dataset, DatasetDict],
    cehrgpt_tokenizer: CehrGptTokenizer,
    data_args: DataTrainingArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> Union[Dataset, DatasetDict]:
    required_columns = TRANSFORMER_COLUMNS + CEHRGPT_COLUMNS
    # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset["train"].column_names
        else:
            all_columns = dataset.column_names
        if "visit_concept_ids" in all_columns:
            dataset.remove_columns(["visit_concept_ids"])
    dataset = _apply_cehrgpt_dataset_mapping(
        dataset,
        HFCehrGptTokenizationMapping(cehrgpt_tokenizer),
        data_args,
        cache_file_collector=cache_file_collector,
    )
    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset["train"].column_names
        else:
            all_columns = dataset.column_names
        columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
        dataset = dataset.remove_columns(columns_to_remove)

    return dataset


def create_cehrgpt_finetuning_dataset(
    dataset: Union[Dataset, DatasetDict],
    cehrgpt_tokenizer: CehrGptTokenizer,
    data_args: DataTrainingArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> Union[Dataset, DatasetDict]:
    required_columns = TRANSFORMER_COLUMNS + CEHRGPT_COLUMNS + FINETUNING_COLUMNS
    # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset["train"].column_names
        else:
            all_columns = dataset.column_names
        if "visit_concept_ids" in all_columns:
            dataset.remove_columns(["visit_concept_ids"])
    mapping_functions = [
        HFFineTuningMapping(cehrgpt_tokenizer),
    ]
    for mapping_function in mapping_functions:
        dataset = _apply_cehrgpt_dataset_mapping(
            dataset,
            mapping_function,
            data_args,
            cache_file_collector=cache_file_collector,
        )

    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset["train"].column_names
        else:
            all_columns = dataset.column_names
        columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset
