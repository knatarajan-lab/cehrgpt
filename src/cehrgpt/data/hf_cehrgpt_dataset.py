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
    dataset = apply_cehrbert_dataset_mapping(
        dataset,
        HFCehrGptTokenizationMapping(cehrgpt_tokenizer),
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.preprocessing_batch_size,
        streaming=data_args.streaming,
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
        dataset = apply_cehrbert_dataset_mapping(
            dataset,
            mapping_function,
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            streaming=data_args.streaming,
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
