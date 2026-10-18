# Adapted from cehrbert.data_generators.hf_data_generator.meds_utils (MIT License, same copyright holder as cehrgpt).
"""
Read a MEDS dataset (a meds_reader database) as patients with visits, by the visit rules of the dataset.

CEHR-GPT uses one generic visit builder whose matching rules come from cehrgpt.data.meds_visit_rules: the annotated OMOP
and MIMIC defaults, and a YAML file of the user for a dataset that codes its visits differently.
"""

import collections
import functools
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import meds_reader
import numpy as np
import pandas as pd
from cehrbert.data_generators.hf_data_generator import DEFAULT_INPATIENT_CONCEPT_ID, UNKNOWN_VALUE
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset import apply_cehrbert_dataset_mapping
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping
from cehrbert.data_generators.hf_data_generator.meds_utils import get_subject_split
from cehrbert.data_generators.hf_data_generator.patient_block import (
    PatientBlock,
    PatientDemographics,
    merge_patient_blocks,
)
from cehrbert.med_extension.schema_extension import CehrBertPatient, Visit
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from meds.schema import birth_code
from transformers.utils import logging

from cehrgpt.data.meds_visit_rules import (
    VisitRules,
    YamlDrivenConversion,
    build_conversion,
    resolve_visit_rules,
)

LOG = logging.get_logger("transformers")


def generate_demographics_and_patient_blocks_from_rules(
    patient: meds_reader.Subject,
    conversion: YamlDrivenConversion,
    prediction_time: datetime = None,
    observation_window: int = None,
):
    """Build visit boundaries directly from a patient's events and the YAML matching rules."""
    birth_datetime = None
    race = None
    gender = None
    ethnicity = None
    clinical_events = []
    observation_start = (
        prediction_time - timedelta(days=observation_window) if prediction_time and observation_window else None
    )

    for event in patient.events:
        event_code = event.code.upper()
        if event_code.startswith(birth_code):
            birth_datetime = event.time
        elif event_code.startswith("RACE"):
            race = event.code
        elif event_code.startswith("GENDER"):
            gender = event.code
        elif event_code.startswith("ETHNICITY"):
            ethnicity = event.code
        elif event.time is not None:
            if prediction_time is not None and event.time > prediction_time:
                break
            if observation_start is None or event.time >= observation_start:
                clinical_events.append(event)

    demographics = PatientDemographics(
        birth_datetime=birth_datetime,
        race=race,
        gender=gender,
        ethnicity=ethnicity,
    )
    patient_blocks = _build_patient_blocks(patient, clinical_events, conversion, prediction_time)
    return demographics, patient_blocks


def _split_days_at_discharges(days_of_events, conversion):
    """
    Cut a day where a stay ends and another begins, so that the discharge is not in the block of the next stay.

    A block is one day, but a patient can be discharged and be back (or be moved from observation to an inpatient stay,
    with the discharge and the admission at the same time) within that day. The events from the ED or admission event
    on, and those after the discharge, go in a block of their own.
    """
    discharge = [re.compile(rule) for rule in conversion.get_discharge_matching_rules()]
    starts = [
        re.compile(rule)
        for rule in conversion.get_ed_admission_matching_rules() + conversion.get_admission_matching_rules()
    ]

    def is_discharge(event):
        return any(rule.match(event.code) for rule in discharge)

    def is_start(event):
        return any(rule.match(event.code) for rule in starts) and not is_discharge(event)

    for events in days_of_events:
        # at the same time the discharge comes first: it is the end of the stay that the admission follows
        events = sorted(events, key=lambda event: (event.time, not is_discharge(event), event.code))
        part = []
        for event in events:
            if part and any(is_discharge(e) for e in part) and is_start(event):
                yield part
                part = []
            part.append(event)
        yield part


def _build_patient_blocks(patient, events, conversion, prediction_time):
    """
    Build the visits of a patient: the visit ids that the events carry, then the rules for the events that have none.

    1. Visit ids. The events with a visit id make one block per id. An event more than a day before the start or after
       the end of its block is taken out of it: the time of an event is the ground truth.
    2. Events without a visit id (and the ones taken out) join the block whose dates contain their date.
    3. Rules. The events that are still alone are put in one block per day, and the blocks are joined into stays by the
       ED, admission and discharge patterns: an ED block followed by an admission within 24 hours begins a stay, a
       discharge closes it, and a block that starts within 12 hours after a discharge stays with it. A stay that
       never gets a discharge (before the next admission, or before the record ends) ends where its events stop:
       after ``max_stay_gap_days`` without one. It stays open only if the prediction time is that close to its last
       event. A day without such events is a visit of its own.

    A dataset with visit ids is built by 1 and 2. A dataset without any, such as MIMIC-IV, is built by 3. A dataset
    with some of both gets both. The blocks of a visit that contain each other are merged when there are visit ids.
    """
    events_by_visit = collections.defaultdict(list)
    unlinked_events = []
    for event in events:
        visit_id = getattr(event, "visit_id", None)
        if visit_id:
            events_by_visit[visit_id].append(event)
        else:
            unlinked_events.append(event)

    # 1. the blocks of the visit ids
    blocks_by_visit = {}
    for visit_id, visit_events in events_by_visit.items():
        block = PatientBlock(visit_events, visit_id, conversion)
        events_inside_boundary = []
        for event in block.events:
            before_start = (block.block_start_time - event.time).days > 1
            after_end = block.block_end_time and (event.time - block.block_end_time).days > 1
            if before_start or after_end:
                unlinked_events.append(event)
            else:
                events_inside_boundary.append(event)
        block.events = events_inside_boundary
        blocks_by_visit[visit_id] = block

    # 2. the events without a visit join the block that their date is in
    unlinked_by_date = collections.defaultdict(list)
    for event in sorted(unlinked_events, key=lambda event: event.time):
        unlinked_by_date[event.time.date()].append(event)
    for event_date in list(unlinked_by_date):
        for block in blocks_by_visit.values():
            if block.get_visit_start_datetime().date() <= event_date <= block.get_visit_end_datetime().date():
                block.events.extend(unlinked_by_date.pop(event_date))
                block.events.sort(key=lambda event: (event.time, event.code))
                break

    # 3. what is left is one block per day, joined into stays by the rules; a new block never takes the id of a visit
    next_visit_id = max([*blocks_by_visit, conversion.default_visit_id - 1]) + 1
    daily_blocks = [
        PatientBlock(daily_events, next_visit_id + offset, conversion)
        for offset, daily_events in enumerate(_split_days_at_discharges(unlinked_by_date.values(), conversion))
    ]
    admission_discharge_pairs = []
    incomplete_pairs = set()
    max_gap_hours = conversion.rules.max_stay_gap_days * 24

    def hours_between(earlier, later):
        return (daily_blocks[later].min_time - daily_blocks[earlier].max_time).total_seconds() / 3600

    def end_of_events(start, last):
        """The last block, up to last, that follows the start without a gap of more than max_stay_gap_days."""
        end = start
        while end < last and hours_between(end, end + 1) <= max_gap_hours:
            end += 1
        return end

    active_ed_index = None
    active_admission_index = None
    for index, block in enumerate(daily_blocks):
        if block.has_ed_admission and active_admission_index is None:
            active_ed_index = index
        if block.has_admission:
            if active_ed_index is not None and (
                active_ed_index == index or hours_between(active_ed_index, index) <= 24
            ):
                # the ED visit leads to this admission and the stay starts with it
                active_admission_index = active_ed_index
            else:
                # an ED visit more than 24 hours before stays a visit of its own; the stay starts at the admission
                if active_admission_index is not None:
                    # the stay before had no discharge: it ends where its events stop
                    admission_discharge_pairs.append(
                        (active_admission_index, end_of_events(active_admission_index, index - 1))
                    )
                active_admission_index = index
            active_ed_index = None

        if block.has_discharge:
            if active_admission_index is not None:
                admission_discharge_pairs.append((active_admission_index, index))
            active_admission_index = None
            active_ed_index = None

    if active_admission_index is not None:
        # no discharge until the end of the record
        last_index = len(daily_blocks) - 1
        pair = (active_admission_index, end_of_events(active_admission_index, last_index))
        admission_discharge_pairs.append(pair)
        hours_to_prediction = (
            (prediction_time - daily_blocks[last_index].max_time).total_seconds() / 3600
            if prediction_time is not None
            else None
        )
        if pair[1] == last_index and hours_to_prediction is not None and hours_to_prediction <= max_gap_hours:
            incomplete_pairs.add(pair)

    for admission_index, discharge_index in admission_discharge_pairs:
        visit_id = daily_blocks[admission_index].visit_id
        for index in range(admission_index, discharge_index + 1):
            daily_blocks[index].visit_id = visit_id
            daily_blocks[index].visit_type = DEFAULT_INPATIENT_CONCEPT_ID
            daily_blocks[index].is_incomplete = (admission_index, discharge_index) in incomplete_pairs

        next_index = discharge_index + 1
        if next_index < len(daily_blocks):
            next_block = daily_blocks[next_index]
            discharge_block = daily_blocks[discharge_index]
            hours_after_discharge = (next_block.min_time - discharge_block.max_time).total_seconds() / 3600
            if hours_after_discharge < 0:
                raise ValueError(
                    f"next block starts at {next_block.min_time}, before discharge at {discharge_block.max_time}"
                )
            # a block that starts a stay of its own (an ED visit or an admission) is not the tail of this one
            if hours_after_discharge <= 12 and not next_block.has_ed_or_hospital_admission:
                next_block.visit_id = visit_id
                next_block.visit_type = DEFAULT_INPATIENT_CONCEPT_ID

    # the visits of the visit ids and the ones of the rules, in time; contained visits are merged
    blocks = list(blocks_by_visit.values())
    if not blocks:
        return daily_blocks
    blocks.extend(daily_blocks)
    if daily_blocks:
        blocks.sort(key=lambda block: block.block_start_time)
    return merge_patient_blocks(patient, blocks)


def convert_one_patient_with_rules(
    patient: meds_reader.Subject,
    conversion: YamlDrivenConversion,
    prediction_time: datetime = None,
    observation_window: int = None,
    label=None,
) -> CehrBertPatient:
    """Convert one patient with the visit ids of the events and the event rules declared in YAML."""
    demographics, patient_blocks = generate_demographics_and_patient_blocks_from_rules(
        patient=patient,
        conversion=conversion,
        prediction_time=prediction_time,
        observation_window=observation_window,
    )

    blocks_by_visit = collections.defaultdict(list)
    for patient_block in patient_blocks:
        blocks_by_visit[patient_block.visit_id].append(patient_block)

    visits = []
    for blocks in blocks_by_visit.values():
        is_inpatient_or_ed = any(block.has_ed_or_hospital_admission for block in blocks)
        is_incomplete = any(getattr(block, "is_incomplete", False) for block in blocks)
        discharge_facility = (
            next(filter(None, (block.discharged_to for block in blocks)), None) if is_inpatient_or_ed else None
        )
        events = []
        for block in blocks:
            events.extend(block.get_meds_events())
        visits.append(
            Visit(
                visit_type=blocks[0].visit_type,
                visit_start_datetime=min(block.get_visit_start_datetime() for block in blocks),
                visit_end_datetime=(
                    None if is_incomplete else max(block.get_visit_end_datetime() for block in blocks)
                ),
                discharge_facility=discharge_facility or UNKNOWN_VALUE,
                events=events,
                is_inpatient_or_ed=is_inpatient_or_ed,
            )
        )

    age_at_index = -1
    if prediction_time is not None and demographics.birth_datetime is not None:
        age_at_index = prediction_time.year - demographics.birth_datetime.year
        if (prediction_time.month, prediction_time.day) < (
            demographics.birth_datetime.month,
            demographics.birth_datetime.day,
        ):
            age_at_index -= 1

    return CehrBertPatient(
        patient_id=patient.subject_id,
        birth_datetime=demographics.birth_datetime,
        visits=visits,
        race=demographics.race or UNKNOWN_VALUE,
        gender=demographics.gender or UNKNOWN_VALUE,
        ethnicity=demographics.ethnicity or UNKNOWN_VALUE,
        index_date=prediction_time,
        age_at_index=age_at_index,
        label=label,
    )


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
                converted_patient = convert_one_patient_with_rules(
                    patient, conversion, prediction_time, observation_window, label
                )
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
                    "is_inpatient_or_ed": Value(dtype="bool"),
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

    The visits are made from the standard OMOP and MIMIC defaults, with the yaml file of the user (rules_file) applied
    on top. Nothing is read from the metadata of the dataset.
    """
    rules = resolve_visit_rules(rules_file)
    return create_dataset_from_meds_reader_with_rules(
        data_args=data_args,
        rules=rules,
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
