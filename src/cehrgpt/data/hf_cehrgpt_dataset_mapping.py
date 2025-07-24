import datetime
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import (
    ED_VISIT_TYPE_CODES,
    INPATIENT_VISIT_TYPE_CODES,
    INPATIENT_VISIT_TYPES,
    DatasetMapping,
    VisitObject,
    get_value,
    has_events_and_get_events,
    replace_escape_chars,
)
from cehrbert.med_extension.schema_extension import Event
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert_data.const.common import NA
from cehrbert_data.decorators.patient_event_decorator_base import get_att_function
from datasets.formatting.formatting import LazyBatch
from dateutil.relativedelta import relativedelta
from pandas import Series
from transformers.utils import logging

from cehrgpt.gpt_utils import (
    construct_age_sequence,
    encode_demographics,
    extract_time_interval_in_days,
    is_att_token,
    is_clinical_event,
    multiple_of_10,
)
from cehrgpt.models.tokenization_hf_cehrgpt import (
    NONE_BIN,
    UNKNOWN_BIN,
    CehrGptTokenizer,
)

CEHRGPT_COLUMNS = [
    "concept_ids",
    "concept_value_masks",
    "number_as_values",
    "concept_as_values",
    "is_numeric_types",
    "concept_values",
    "units",
    "position_ids",
    "ages",
    "epoch_times",
]

LOG = logging.get_logger(__name__)


def convert_date_to_posix_time(index_date: datetime.date) -> float:
    return datetime.datetime.combine(
        index_date, datetime.datetime.min.time()
    ).timestamp()


class DatasetMappingDecorator(DatasetMapping):

    def batch_transform(
        self, records: Union[LazyBatch, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Drop index_date if it contains None.

        :param records:
        :return:
        """
        if isinstance(records, LazyBatch):
            table = records.pa_table

            if "index_date" in table.column_names:
                index_col = table.column("index_date")
                if index_col.null_count > 0:
                    table = table.drop(["index_date"])
            records = LazyBatch(pa_table=table, formatter=records.formatter)
        else:
            if "index_date" in records:
                if pd.isna(records["index_date"][0]):
                    del records["index_date"]
        return super().batch_transform(records=records)

    def transform(self, record: Dict[str, Any]) -> Union[Dict[str, Any], Series]:
        raise NotImplemented("Must be implemented")


class MedToCehrGPTDatasetMapping(DatasetMappingDecorator):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        include_inpatient_hour_token: bool = True,
    ):
        self._time_token_function = get_att_function(data_args.att_function_type)
        self._include_auxiliary_token = data_args.include_auxiliary_token
        self._inpatient_time_token_function = get_att_function(
            data_args.inpatient_att_function_type
        )
        self._include_demographic_prompt = data_args.include_demographic_prompt
        self._include_inpatient_hour_token = include_inpatient_hour_token

    """
    This mapping function converts the MED (https://github.com/Medical-Event-Data-Standard/meds/tree/main) extension
    to the CehrGPT format. We make several assumptions
    - The first event contains the demographic information
    - From the second event onward
        - the time of the event is visit_start_datetime.
        - the first measurement contains the code indicating a standard OMOP Visit concept_id (e.g. 9201, 9202)
        - in case of inpatient visits, the last measurement is assumed to
            contain the standard OMOP concept id for discharge facilities (e.g 8536)
        - in case of inpatient visits, datetime_value of the last measurement stores visit_end_datetime
    """

    def remove_columns(self):
        return ["patient_id", "visits", "birth_datetime"]

    @staticmethod
    def _update_cehrgpt_record(
        cehrgpt_record: Dict[str, Any],
        code: str,
        concept_value_mask: int = 0,
        number_as_value: float = 0.0,
        concept_as_value: str = "0",
        is_numeric_type: int = 0,
        unit: str = NA,
    ) -> None:
        cehrgpt_record["concept_ids"].append(replace_escape_chars(code))
        cehrgpt_record["concept_value_masks"].append(concept_value_mask)
        cehrgpt_record["number_as_values"].append(number_as_value)
        cehrgpt_record["concept_as_values"].append(concept_as_value)
        cehrgpt_record["units"].append(unit)
        cehrgpt_record["is_numeric_types"].append(is_numeric_type)

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        cehrgpt_record = {
            "person_id": record["patient_id"],
            "concept_ids": [],
            "concept_value_masks": [],
            "number_as_values": [],
            "concept_as_values": [],
            "units": [],
            "is_numeric_types": [],
        }
        # Extract the demographic information
        birth_datetime = record["birth_datetime"]
        if isinstance(birth_datetime, pd.Timestamp):
            birth_datetime = birth_datetime.to_pydatetime()
        gender = record["gender"]
        race = record["race"]
        visits = record["visits"]
        # This indicates this is columnar format
        if isinstance(visits, dict):
            visits = sorted(self.convert_visit_columnar_to_python(visits))
        else:
            visits = sorted(visits, key=lambda _: get_value(_, "visit_start_datetime"))

        # Add the demographic tokens
        first_visit = visits[0]
        first_visit_start_datetime: datetime.datetime = get_value(
            first_visit, "visit_start_datetime"
        )
        year_str = f"year:{str(first_visit_start_datetime.year)}"
        age_str = f"age:{str(relativedelta(first_visit_start_datetime, birth_datetime).years)}"
        self._update_cehrgpt_record(cehrgpt_record, year_str)
        self._update_cehrgpt_record(cehrgpt_record, age_str)
        self._update_cehrgpt_record(cehrgpt_record, gender)
        self._update_cehrgpt_record(cehrgpt_record, race)

        # Use a data cursor to keep track of time
        datetime_cursor: Optional[datetime.datetime] = None
        visit: VisitObject
        # Loop through all the visits
        for i, visit in enumerate(visits):
            events: Generator[Event, None, None] = get_value(visit, "events")
            has_events, events = has_events_and_get_events(events)
            if not has_events:
                continue

            visit_start_datetime: datetime.datetime = get_value(
                visit, "visit_start_datetime"
            )
            # If visit_end_datetime is populated for the inpatient visit, we update the datetime_cursor
            visit_end_datetime: Optional[datetime.datetime] = get_value(
                visit, "visit_end_datetime"
            )

            # We assume the first measurement to be the visit type of the current visit
            visit_type = get_value(visit, "visit_type")
            is_er_or_inpatient = (
                visit_type in INPATIENT_VISIT_TYPES
                or visit_type in INPATIENT_VISIT_TYPE_CODES
                or visit_type in ED_VISIT_TYPE_CODES
            )

            # Add artificial time tokens to the patient timeline if timedelta exists
            if datetime_cursor is not None:
                time_delta = max((visit_start_datetime - datetime_cursor).days, 0)
                # This generates an artificial time token depending on the choice of the time token functions
                self._update_cehrgpt_record(
                    cehrgpt_record,
                    code=self._time_token_function(time_delta),
                )

            datetime_cursor = visit_start_datetime
            # Add a [VS] token
            self._update_cehrgpt_record(
                cehrgpt_record,
                code="[VS]",
            )
            # Add a visit type token
            self._update_cehrgpt_record(
                cehrgpt_record,
                code=visit_type,
            )
            # We need to insert an inpatient hour token right after the visit type, we calculate the hour interval
            # with respect to the midnight of the day
            if is_er_or_inpatient and self._include_inpatient_hour_token:
                if datetime_cursor.hour > 0:
                    # This generates an artificial time token depending on the choice of the time token functions
                    self._update_cehrgpt_record(
                        cehrgpt_record,
                        code=f"i-H{datetime_cursor.hour}",
                    )

            # Keep track of the existing outpatient events, we don't want to add them again
            existing_duplicate_events = list()
            for e in events:
                # If the event doesn't have a time stamp, we skip it
                event_time: datetime.datetime = e["time"]
                if not event_time:
                    continue

                # If numeric_value exists, this is a concept/value tuple, we indicate this using a concept_value_mask
                numeric_value = e.get("numeric_value", None)
                text_value = e.get("text_value", None)
                # The unit might be populated with a None value
                unit = e.get("unit", NA) if e.get("unit", NA) else NA
                concept_value_mask = int(
                    numeric_value is not None or text_value is not None
                )
                is_numeric_type = int(numeric_value is not None)
                code = replace_escape_chars(e["code"])

                # Create the event identity
                event_identity = (
                    (event_time, code, text_value, unit)
                    if is_er_or_inpatient
                    else (event_time.date(), code, text_value, unit)
                )

                # Add a medical token to the patient timeline
                # If this is an inpatient visit, we use the event time stamps to calculate age and date
                # because the patient can stay in the hospital for a period of time.
                if is_er_or_inpatient:
                    # Calculate the time diff in days w.r.t the previous measurement
                    time_diff_days = (event_time - datetime_cursor).days
                    # Update the datetime_cursor if the time diff between two neighboring measurements is greater than and
                    # equal to 1 day
                    if self._inpatient_time_token_function and time_diff_days > 0:
                        # This generates an artificial time token depending on the choice of the time token functions
                        self._update_cehrgpt_record(
                            cehrgpt_record,
                            code=f"i-{self._inpatient_time_token_function(time_diff_days)}",
                        )

                    if self._include_inpatient_hour_token:
                        # if the time difference in days is greater than 0, we calculate the hour interval
                        # with respect to the midnight of the day
                        time_diff_hours = (
                            event_time.hour
                            if time_diff_days > 0
                            else int(
                                (event_time - datetime_cursor).total_seconds() // 3600
                            )
                        )

                        if time_diff_hours > 0:
                            # This generates an artificial time token depending on the choice of the time token functions
                            self._update_cehrgpt_record(
                                cehrgpt_record,
                                code=f"i-H{time_diff_hours}",
                            )

                if event_identity in existing_duplicate_events:
                    continue

                self._update_cehrgpt_record(
                    cehrgpt_record,
                    code=code,
                    concept_value_mask=concept_value_mask,
                    unit=unit,
                    number_as_value=numeric_value if numeric_value else 0.0,
                    concept_as_value=(
                        replace_escape_chars(text_value) if text_value else "0"
                    ),
                    is_numeric_type=is_numeric_type,
                )
                existing_duplicate_events.append(event_identity)
                # we only want to update the time stamp when data_cursor is less than the event time
                if datetime_cursor < event_time or datetime_cursor is None:
                    datetime_cursor = event_time
                    # We need to bound the datetime_cursor if the current visit is an admission type of visit
                    # as the associated events could be generated after the visits are complete
                    if is_er_or_inpatient and visit_end_datetime is not None:
                        datetime_cursor = min(datetime_cursor, visit_end_datetime)

            # For inpatient or ER visits, we want to discharge_facility to the end of the visit
            if is_er_or_inpatient:
                # If visit_end_datetime is populated for the inpatient visit, we update the datetime_cursor
                if visit_end_datetime is not None:
                    datetime_cursor = visit_end_datetime

                if self._include_auxiliary_token:
                    # Reuse the age and date calculated for the last event in the patient timeline for the discharge
                    # facility event
                    discharge_facility = get_value(visit, "discharge_facility")
                    if not discharge_facility:
                        discharge_facility = "0"

                    self._update_cehrgpt_record(
                        cehrgpt_record,
                        code=discharge_facility,
                    )

            # Reuse the age and date calculated for the last event in the patient timeline
            self._update_cehrgpt_record(
                cehrgpt_record,
                code="[VE]",
            )

        # Generate the orders of the concepts that the cehrbert dataset mapping function expects
        cehrgpt_record["orders"] = list(
            range(1, len(cehrgpt_record["concept_ids"]) + 1)
        )

        # Add some count information for this sequence
        cehrgpt_record["num_of_concepts"] = len(cehrgpt_record["concept_ids"])
        cehrgpt_record["num_of_visits"] = len(visits)

        if record.get("index_date", None) is not None:
            cehrgpt_record["index_date"] = record["index_date"]
        if record.get("label", None) is not None:
            cehrgpt_record["label"] = record["label"]
        if record.get("age_at_index", None) is not None:
            cehrgpt_record["age_at_index"] = record["age_at_index"]

        return cehrgpt_record


class HFCehrGptTokenizationMapping(DatasetMappingDecorator):
    def __init__(
        self,
        concept_tokenizer: CehrGptTokenizer,
    ):
        self._concept_tokenizer = concept_tokenizer
        self._lab_token_ids = self._concept_tokenizer.lab_token_ids

    def remove_columns(self):
        return [
            "concept_value_masks",
            "is_numeric_types",
        ]

    def filter_out_invalid_tokens(self, record: Dict[str, Any]) -> Dict[str, Any]:
        column_names = []
        seq_length = len(record["concept_ids"])

        # We can't have "0" as a token in the tokenizer because it would break tokenization for "Race/0", "Visit/0"
        # This is a pre-caution
        if "0" in record["concept_ids"]:
            if isinstance(record["concept_ids"], np.ndarray):
                record["concept_ids"][record["concept_ids"] == "0"] = "Unknown"
            else:
                record["concept_ids"] = [
                    "Unknown" if x == "0" else x for x in record["concept_ids"]
                ]

        for k, v in record.items():
            if k not in CEHRGPT_COLUMNS:
                continue
            if isinstance(v, (list, np.ndarray)) and len(v) == seq_length:
                column_names.append(k)
        valid_concept_ids = self._concept_tokenizer.get_vocab().keys()
        valid_indices = [
            idx
            for idx, concept_id in enumerate(record["concept_ids"])
            if concept_id in valid_concept_ids
        ]
        if len(valid_indices) != len(record["concept_ids"]):
            for column in column_names:
                values = record[column]
                record[column] = [values[idx] for idx in valid_indices]
        return record

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:

        # Getting gender and race to the record
        gender, race = record["concept_ids"][2:4]
        # Reconstruct the ages input before the filter is applied
        record["ages"] = construct_age_sequence(
            record["concept_ids"], record.get("ages", None)
        )
        # Remove the tokens from patient sequences that do not exist in the tokenizer
        record = self.filter_out_invalid_tokens(record)

        # If any concept has a value associated with it, we normalize the value
        record["input_ids"] = self._concept_tokenizer.encode(record["concept_ids"])
        gender_id = self._concept_tokenizer.encode_gender(gender)
        record["gender"] = gender_id
        race_id = self._concept_tokenizer.encode_race(race)
        record["racer"] = race_id
        record["position_ids"] = np.clip(record["ages"], a_min=0, a_max=120)
        # record["position_ids"] = [
        #     encode_demographics(
        #         age=age,
        #         race=race_id,
        #         gender=gender_id,
        #         max_age=200,
        #         max_race=multiple_of_10(self._concept_tokenizer.race_size),
        #         max_gender=multiple_of_10(self._concept_tokenizer.gender_size),
        #     )
        #     for age in np.clip(record["ages"], a_min=0, a_max=120)
        # ]
        assert len(record["input_ids"]) == len(record["concept_ids"]), (
            "The number of tokens must equal to the number of concepts\n"
            f"decoded concept_ids: {self._concept_tokenizer.decode(record['input_ids'], skip_special_tokens=False)}"
        )
        assert len(record["input_ids"]) == len(
            record["position_ids"]
        ), "The number of tokens must equal to the number of positions\n"
        record["value_indicators"] = record["concept_value_masks"]
        if "number_as_values" not in record or "concept_as_values" not in record:
            record["number_as_values"] = [
                float(value) if isinstance(value, float) else None
                for value in record["concept_values"]
            ]
            record["is_numeric_types"] = [
                int(isinstance(value, float)) for value in record["concept_values"]
            ]
            record["concept_as_values"] = [
                value if isinstance(value, str) else None
                for value in record["concept_values"]
            ]
        if np.any(np.asarray(record["concept_value_masks"]) > 0):
            values = []
            for i, (
                concept_id,
                unit,
                concept_value_mask,
                number_as_value,
                concept_as_value,
                is_numeric_type,
            ) in enumerate(
                zip(
                    record["concept_ids"],
                    record["units"],
                    record["concept_value_masks"],
                    record["number_as_values"],
                    record["concept_as_values"],
                    record["is_numeric_types"],
                )
            ):
                if concept_value_mask == 1:
                    value = UNKNOWN_BIN
                    if is_numeric_type == 1:
                        if concept_id in self._concept_tokenizer.numeric_concept_ids:
                            value = self._concept_tokenizer.normalize(
                                concept_id, unit, number_as_value
                            )
                    elif isinstance(concept_as_value, str):
                        value = concept_as_value
                    values.append(value)
                else:
                    values.append(NONE_BIN)
            assert len(values) == len(record["input_ids"])
            record["values"] = self._concept_tokenizer.encode_value(values)
        else:
            record["values"] = self._concept_tokenizer.encode_value(
                [NONE_BIN for _ in range(len(record["concept_value_masks"]))]
            )
        # Delete these features because they contain null values and pyarrow cannot concatenate multiple records
        del record["number_as_values"]
        del record["concept_as_values"]
        return record


class HFFineTuningMapping(HFCehrGptTokenizationMapping):
    """Consider removing this transformation in the future."""

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = super().transform(record)
        record.update(
            {
                "age_at_index": (
                    record["age"] if "age" in record else record["age_at_index"]
                ),
                "classifier_label": int(record["label"] > 0),
                "index_date": (
                    convert_date_to_posix_time(record["index_date"])
                    if "index_date" in record
                    else None
                ),
            }
        )
        return record

    def remove_columns(self):
        columns = super().remove_columns()
        columns.append("label")
        return columns


class OptimizedMotorTTEDatasetMapping(DatasetMappingDecorator):
    """Optimized dataset mapping for Motor Time-to-Event label preprocessing."""

    def __init__(self, tokenizer: CehrGptTokenizer):
        self.tokenizer = tokenizer
        self.motor_code_cache: Dict[str, List[str]] = {}

        # Pre-compute token type lookups for faster processing
        # Cache for token type checks
        self.att_token_cache = {}
        self.clinical_event_cache = {}
        self.time_interval_cache = {}

    def _is_att_token_cached(self, concept_id: str) -> bool:
        """Cached version of is_att_token check."""
        if concept_id not in self.att_token_cache:
            self.att_token_cache[concept_id] = is_att_token(concept_id)
        return self.att_token_cache[concept_id]

    def _is_clinical_event_cached(self, concept_id: str) -> bool:
        """Cached version of is_clinical_event check."""
        if concept_id not in self.clinical_event_cache:
            self.clinical_event_cache[concept_id] = is_clinical_event(concept_id)
        return self.clinical_event_cache[concept_id]

    def _get_time_interval_cached(self, concept_id: str) -> int:
        """Cached version of extract_time_interval_in_days."""
        if concept_id not in self.time_interval_cache:
            self.time_interval_cache[concept_id] = extract_time_interval_in_days(
                concept_id
            )
        return self.time_interval_cache[concept_id]

    def transform(self, record: Dict[str, Any]) -> Union[Dict[str, Any], Series]:
        """Optimized transformation function that adds motor TTE labels to a record."""
        motor_labels = self._create_motor_tte_labels_optimized(record)
        record.update(motor_labels)
        return record

    def _create_motor_tte_labels_optimized(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimized version of motor TTE label creation with several performance improvements:

        1. Pre-computed token type checks with caching
        2. Vectorized operations where possible
        3. Reduced memory allocations
        4. Early termination conditions
        """
        concept_ids = record["concept_ids"]
        event_times = record["epoch_times"]

        # Pre-allocate lists with estimated sizes to reduce memory allocations
        n_concepts = len(concept_ids)
        motor_censor_times = []
        motor_tte_tasks = []
        motor_tte_times = []
        motor_tte_task_indicators = [False] * n_concepts
        motor_tte_label_offsets = []

        # Use dict for O(1) lookups instead of repeated list operations
        time_to_event_dict: Dict[str, int] = {}
        before_time_token = False

        # Reverse iteration with enumeration for better performance
        for i, (concept_id, event_time) in enumerate(
            zip(reversed(concept_ids), reversed(event_times))
        ):
            reverse_idx = n_concepts - 1 - i
            is_included = False

            if before_time_token and time_to_event_dict:
                # Batch process all motor codes for this time point
                current_tasks = []
                current_times = []

                for motor_code, motor_time in time_to_event_dict.items():
                    motor_token_id = self.tokenizer.get_motor_token_id(motor_code)
                    current_tasks.append(motor_token_id)
                    current_times.append(motor_time - event_time)

                motor_tte_tasks.extend(current_tasks)
                motor_tte_times.extend(current_times)
                motor_tte_label_offsets.append(len(time_to_event_dict))
                motor_censor_times.append(event_times[-1] - event_time)
                before_time_token = False
                is_included = True

            # Use cached token type checks
            if self._is_att_token_cached(concept_id):
                time_interval = self._get_time_interval_cached(concept_id)
                if time_interval > 0:
                    before_time_token = True
            elif self._is_clinical_event_cached(concept_id):
                # Use cached motor codes
                if concept_id in self.motor_code_cache:
                    motor_codes = self.motor_code_cache[concept_id]
                else:
                    motor_codes = self.tokenizer.get_motor_parents(concept_id)
                    self.motor_code_cache[concept_id] = motor_codes

                # Batch update time_to_event_dict
                for motor_code in motor_codes:
                    time_to_event_dict[motor_code] = event_time

            motor_tte_task_indicators[reverse_idx] = is_included

        # Early return if no motor tasks found
        if not motor_tte_times:
            LOG.debug(
                "No MOTOR tasks detected for this sample. "
                "Length: %s, last 10 concepts: %s",
                len(concept_ids),
                concept_ids[-10:] if len(concept_ids) >= 10 else concept_ids,
            )

        # Reverse lists back to chronological order
        motor_tte_times.reverse()
        motor_tte_tasks.reverse()
        motor_tte_label_offsets.reverse()
        motor_censor_times.reverse()

        # Use numpy for cumsum operation (faster than pure Python)
        motor_tte_label_offsets = np.cumsum(motor_tte_label_offsets).tolist()
        motor_tte_label_offsets = [0] + motor_tte_label_offsets

        # Pad motor_censor_times
        motor_censor_times = motor_censor_times + [-100]

        # Shift task indicators
        motor_tte_task_indicators = motor_tte_task_indicators[1:] + [False]

        return {
            "motor_censor_times": motor_censor_times,
            "motor_tte_tasks": motor_tte_tasks,
            "motor_tte_times": motor_tte_times,
            "motor_tte_label_offsets": motor_tte_label_offsets,
            "motor_tte_task_indicators": motor_tte_task_indicators,
        }


class VectorizedMotorTTEDatasetMapping(DatasetMappingDecorator):
    """
    Further optimized version using vectorized operations where possible.

    This version pre-computes token type arrays for even faster processing.
    """

    def __init__(self, tokenizer: CehrGptTokenizer):
        self.tokenizer = tokenizer
        self.motor_code_cache: Dict[str, List[str]] = {}

        # Pre-compute vocab-wide token type mappings
        self._precompute_vocab_mappings()

    def _precompute_vocab_mappings(self):
        """Pre-compute token type mappings for entire vocabulary."""
        LOG.info("Pre-computing vocabulary-wide token mappings...")

        vocab = self.tokenizer.get_vocab()
        self.vocab_to_idx = {token: idx for idx, token in enumerate(vocab.keys())}
        self.vocab_tokens = list(vocab.keys())

        # Pre-compute boolean arrays for token types
        n_vocab = len(self.vocab_tokens)
        self.is_att_token_array = np.zeros(n_vocab, dtype=bool)
        self.is_clinical_event_array = np.zeros(n_vocab, dtype=bool)
        self.time_intervals_array = np.full(n_vocab, -1, dtype=int)

        for i, token in enumerate(self.vocab_tokens):
            if is_att_token(token):
                self.is_att_token_array[i] = True
                try:
                    self.time_intervals_array[i] = extract_time_interval_in_days(token)
                except (ValueError, AttributeError):
                    self.time_intervals_array[i] = -1

            if is_clinical_event(token):
                self.is_clinical_event_array[i] = True

        LOG.info(f"Processed {n_vocab} vocabulary tokens")

    def transform(self, record: Dict[str, Any]) -> Union[Dict[str, Any], Series]:
        """Vectorized transformation using pre-computed token type arrays."""
        motor_labels = self._create_motor_tte_labels_vectorized(record)
        record.update(motor_labels)
        return record

    def _create_motor_tte_labels_vectorized(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Highly optimized vectorized version using pre-computed token type arrays."""
        concept_ids = record["concept_ids"]
        event_times = np.asarray(record["epoch_times"])

        # Convert concept_ids to indices for vectorized operations
        try:
            concept_indices = np.array([self.vocab_to_idx[cid] for cid in concept_ids])
        except KeyError as e:
            LOG.warning(f"Unknown concept ID found: {e}")
            # Fallback to non-vectorized version for records with unknown tokens
            return self._create_motor_tte_labels_fallback(record)

        # Vectorized token type detection
        is_att_tokens = self.is_att_token_array[concept_indices]
        is_clinical_events = self.is_clinical_event_array[concept_indices]
        time_intervals = self.time_intervals_array[concept_indices]

        # Find valid time tokens (att tokens with positive intervals)
        valid_time_tokens = is_att_tokens & (time_intervals > 0)

        # Process in reverse order but use vectorized operations where possible
        n_concepts = len(concept_ids)
        motor_tte_task_indicators = np.zeros(n_concepts, dtype=bool)

        # Compute the vectorized censor times
        motor_censor_times = (
            event_times[-1] - event_times[np.roll(valid_time_tokens, -1)]
        ).tolist()
        motor_tte_tasks = []
        motor_tte_times = []
        motor_tte_label_offsets = []

        time_to_event_dict: Dict[str, Any] = {}
        before_time_token_indices = np.where(np.roll(valid_time_tokens, -1))[0].tolist()

        for start_index, end_index in zip(
            reversed(before_time_token_indices),
            reversed(before_time_token_indices[1:] + [n_concepts]),
        ):
            motor_tte_task_indicators[start_index] = True
            current_event_time = event_times[start_index]
            # Slice out all the tokens between two time intervals
            # start_index + 1 excludes the prediction token
            # end_index + 1 includes the last token right before the time token due to exclusive right indexing
            section_concept_indices = concept_indices[start_index + 1 : end_index + 1]
            section_event_times = event_times[start_index + 1 : end_index + 1]
            section_is_clinical_events = is_clinical_events[
                start_index + 1 : end_index + 1
            ]
            section_clinical_concept_indices = section_concept_indices[
                section_is_clinical_events
            ]
            section_event_times = section_event_times[section_is_clinical_events]

            for i in range(len(section_clinical_concept_indices) - 1, -1, -1):
                concept_index = section_clinical_concept_indices[i]
                concept_event_time = section_event_times[i]
                concept_id = self.vocab_tokens[concept_index]
                # Use cached motor codes
                if concept_id in self.motor_code_cache:
                    motor_codes = self.motor_code_cache[concept_id]
                else:
                    motor_codes = self.tokenizer.get_motor_parents(concept_id)
                    self.motor_code_cache[concept_id] = motor_codes

                for motor_code in motor_codes:
                    time_to_event_dict[motor_code] = concept_event_time

            current_tasks = []
            current_times = []
            for motor_code, motor_time in time_to_event_dict.items():
                motor_token_id = self.tokenizer.get_motor_token_id(motor_code)
                current_tasks.append(motor_token_id)
                current_times.append(motor_time - current_event_time)

            motor_tte_tasks.extend(current_tasks)
            motor_tte_times.extend(current_times)
            motor_tte_label_offsets.append(len(time_to_event_dict))

        # Early return if no motor tasks found
        if not motor_tte_times:
            LOG.debug(
                "No MOTOR tasks detected for this sample. "
                "Length: %s, last 10 concepts: %s",
                len(concept_ids),
                concept_ids[-10:] if len(concept_ids) >= 10 else concept_ids,
            )

        # Reverse and finalize
        motor_tte_times.reverse()
        motor_tte_tasks.reverse()
        motor_tte_label_offsets.reverse()
        motor_censor_times.reverse()

        motor_tte_label_offsets = np.cumsum(motor_tte_label_offsets).tolist()
        motor_tte_label_offsets = [0] + motor_tte_label_offsets
        motor_censor_times = motor_censor_times + [-100]

        return {
            "motor_censor_times": motor_censor_times,
            "motor_tte_tasks": motor_tte_tasks,
            "motor_tte_times": motor_tte_times,
            "motor_tte_label_offsets": motor_tte_label_offsets,
            "motor_tte_task_indicators": motor_tte_task_indicators.tolist(),
        }

    def _create_motor_tte_labels_fallback(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback method for records with unknown tokens."""
        # Use the original optimized method as fallback
        optimized_mapper = OptimizedMotorTTEDatasetMapping(self.tokenizer)
        return optimized_mapper._create_motor_tte_labels_optimized(record)
