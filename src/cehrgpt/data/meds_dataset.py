# Adapted from cehrbert.data_generators.hf_data_generator.meds_utils (MIT License, same copyright holder as cehrgpt).
"""
Read a MEDS dataset (a meds_reader database) as patients with visits, by the visit rules of the dataset.

cehrbert chooses the rules of the conversion by the name of a class that is in its code. These functions are the same as
the ones of cehrbert with the rules (cehrgpt.data.meds_visit_rules) in the place of that name, so they can come from the
metadata of the dataset or from a yaml file. Datasets without rules are read with the packaged rules of the conversion that the arguments name.
"""

import functools
import os
from typing import List, Optional, Tuple

import meds_reader
import numpy as np
import pandas as pd
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset import apply_cehrbert_dataset_mapping
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    convert_one_patient,
    get_subject_split,
)
from cehrbert.med_extension.schema_extension import CehrBertPatient
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers.utils import logging

from cehrgpt.data.meds_visit_rules import (
    VisitRules,
    build_conversion,
    packaged_rules_for_conversion,
    resolve_visit_rules,
)

LOG = logging.get_logger("transformers")


def _meds_to_cehrbert_generator_with_rules(
    shards: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_to_db: str,
    default_visit_id: int,
    rules: VisitRules,
    meds_exclude_tables: Optional[List[str]] = None,
) -> CehrBertPatient:
    conversion = build_conversion(
        rules,
        default_visit_id=default_visit_id,
        meds_exclude_tables=meds_exclude_tables,
    )
    with meds_reader.SubjectDatabase(path_to_db) as patient_database:
        for shard in shards:
            for patient_id, prediction_time, observation_window, label in shard:
                patient = patient_database[patient_id]
                converted_patient = convert_one_patient(patient, conversion, prediction_time, observation_window, label)
                # there are patients whose birthdate is none
                visits = converted_patient["visits"]
                if converted_patient["birth_datetime"] is None:
                    LOG.warning("patient_id: %s does not have a valid birth_datetime, therefore skipped", patient_id)
                elif visits is None or len(visits) == 0:
                    LOG.debug("patient_id: %s does not have visits, therefore skipped", patient_id)
                else:
                    yield converted_patient


def _create_cehrbert_data_from_meds_with_rules(
    data_args: DataTrainingArguments,
    rules: VisitRules,
    split: str,
    default_visit_id: int = 1,
    dataset_mappings: Optional[List[DatasetMapping]] = None,
    cache_file_collector: Optional[CacheFileCollector] = None,
):
    assert split in ["held_out", "train", "tuning"]
    batches = []
    if data_args.cohort_folder:
        # Load the entire cohort
        cohort = pd.read_parquet(os.path.expanduser(data_args.cohort_folder))
        patient_split = get_subject_split(os.path.expanduser(data_args.data_folder))
        subject_ids = patient_split[split]
        cohort_split = cohort[cohort.subject_id.isin(subject_ids)]
        for cohort_row in cohort_split.itertuples():
            subject_id = cohort_row.subject_id
            prediction_time = cohort_row.prediction_time
            label = int(cohort_row.boolean_value)
            batches.append((subject_id, prediction_time, data_args.observation_window, label))
    else:
        patient_split = get_subject_split(os.path.expanduser(data_args.data_folder))
        for subject_id in patient_split[split]:
            batches.append((subject_id, None, None, None))

    features = Features(
        {
            "patient_id": Value(dtype="int32"),
            "birth_datetime": Value(dtype="timestamp[us]"),  # Use timestamp with microsecond precision
            "gender": Value(dtype="string"),
            "race": Value(dtype="string"),
            "ethnicity": Value(dtype="string"),
            "index_date": Value(dtype="timestamp[us]"),
            "age_at_index": Value(dtype="int16"),
            "label": Value(dtype="float32"),  # Using float to accommodate both int and float labels
            "visits": Sequence(
                feature={
                    "visit_type": Value(dtype="string"),
                    "visit_start_datetime": Value(dtype="timestamp[us]"),
                    "visit_end_datetime": Value(dtype="timestamp[us]"),
                    "discharge_facility": Value(dtype="string"),
                    "events": Sequence(
                        feature={
                            "time": Value(dtype="timestamp[us]"),
                            "code": Value(dtype="string"),
                            "text_value": Value(dtype="string"),
                            "numeric_value": Value(dtype="float32"),
                            "unit": Value(dtype="string"),
                            "properties": {
                                "visit_id": Value(dtype="int32"),
                                "table": Value(dtype="string"),
                            },
                        }
                    ),
                }
            ),
        }
    )

    split_batches = np.array_split(np.asarray(batches), data_args.preprocessing_num_workers)
    batch_func = functools.partial(
        _meds_to_cehrbert_generator_with_rules,
        path_to_db=os.path.expanduser(data_args.data_folder),
        default_visit_id=default_visit_id,
        rules=rules,
        meds_exclude_tables=data_args.meds_exclude_tables,
    )
    dataset = Dataset.from_generator(
        batch_func,
        gen_kwargs={
            "shards": split_batches,
        },
        num_proc=(data_args.preprocessing_num_workers if not data_args.streaming else None),
        writer_batch_size=data_args.preprocessing_batch_size,
        streaming=data_args.streaming,
        features=features,
    )
    # These cached files need to be deleted manually
    if cache_file_collector:
        cache_file_collector.add_cache_files(dataset)

    if dataset_mappings:
        for dataset_mapping in dataset_mappings:
            # Convert the CehrBertPatient to CehrBert data inputs
            dataset = apply_cehrbert_dataset_mapping(
                dataset,
                dataset_mapping,
                num_proc=data_args.preprocessing_num_workers,
                batch_size=data_args.preprocessing_batch_size,
                streaming=data_args.streaming,
                cache_file_collector=cache_file_collector,
            )
    return dataset


def create_dataset_from_meds_reader_with_rules(
    data_args: DataTrainingArguments,
    rules: VisitRules,
    default_visit_id: int = 1,
    dataset_mappings: Optional[List[DatasetMapping]] = None,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> DatasetDict:

    LOG.info("The visit rules: %s", rules.to_dict())
    LOG.info("The att_function_type: %s", data_args.att_function_type)
    LOG.info("The inpatient_att_function_type: %s", data_args.inpatient_att_function_type)
    LOG.info("The include_auxiliary_token: %s", data_args.include_auxiliary_token)
    LOG.info("The include_demographic_prompt: %s", data_args.include_demographic_prompt)
    LOG.info("The meds_exclude_tables: %s", "\n".join(data_args.meds_exclude_tables))
    train_dataset = _create_cehrbert_data_from_meds_with_rules(
        data_args=data_args,
        rules=rules,
        split="train",
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
    tuning_dataset = _create_cehrbert_data_from_meds_with_rules(
        data_args=data_args,
        rules=rules,
        split="tuning",
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
    held_out_dataset = _create_cehrbert_data_from_meds_with_rules(
        data_args=data_args,
        rules=rules,
        split="held_out",
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
    return DatasetDict({"train": train_dataset, "validation": tuning_dataset, "test": held_out_dataset})


def create_meds_dataset(
    data_args: DataTrainingArguments,
    rules_file: Optional[str] = None,
    default_visit_id: int = 1,
    dataset_mappings: Optional[List[DatasetMapping]] = None,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> DatasetDict:
    """
    The train, validation and test patients of the MEDS dataset in data_args.data_folder, with the mappings applied.

    The visits are made by the rules that the dataset declares in its metadata, with the keys of rules_file on top. A
    dataset that has none and no rules_file is read with the packaged rules of the conversion that data_args names
    (meds_to_cehrbert_conversion_type: omop or mimic4), with a warning.
    """
    rules = resolve_visit_rules(data_args.data_folder, rules_file)
    if rules is None:
        rules = packaged_rules_for_conversion(data_args.meds_to_cehrbert_conversion_type)
        LOG.warning(
            "The dataset in %s declares no visit rules (%s) and none was given with --meds_visit_rules_file: using the "
            "packaged rules %s for %s. Declare the rules of the dataset in its metadata.",
            data_args.data_folder,
            os.path.join("metadata", "visit_rules.yaml"),
            rules.source,
            data_args.meds_to_cehrbert_conversion_type,
        )
    return create_dataset_from_meds_reader_with_rules(
        data_args=data_args,
        rules=rules,
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
