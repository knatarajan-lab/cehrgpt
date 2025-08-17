import datetime
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd
from cehrbert.data_generators.hf_data_generator import UNKNOWN_VALUE
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
from cehrbert_data.const.artificial_tokens import (
    DISCHARGE_UNKNOWN_TOKEN,
    GENDER_UNKNOWN_TOKEN,
    RACE_UNKNOWN_TOKEN,
)
from cehrbert_data.const.common import NA
from cehrbert_data.decorators.patient_event_decorator_base import get_att_function
from datasets.formatting.formatting import LazyBatch
from dateutil.relativedelta import relativedelta
from pandas import Series

from cehrgpt.gpt_utils import (
    construct_age_sequence,
    construct_time_sequence,
    encode_demographics,
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
    "epoch_times",
    "ages",
]


def convert_date_to_posix_time(index_date: Union[datetime.date, int, float]) -> float:
    if isinstance(index_date, datetime.date):
        return (
            datetime.datetime.combine(index_date, datetime.datetime.min.time())
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
    elif isinstance(index_date, datetime.datetime):
        return index_date.replace(tzinfo=datetime.timezone.utc).timestamp()
    return index_date


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
        time: datetime.datetime,
        age: int,
        concept_value_mask: int = 0,
        number_as_value: float = 0.0,
        concept_as_value: str = "0",
        is_numeric_type: int = 0,
        unit: str = NA,
    ) -> None:
        cehrgpt_record["concept_ids"].append(replace_escape_chars(code))
        cehrgpt_record["ages"].append(age)
        cehrgpt_record["concept_value_masks"].append(concept_value_mask)
        cehrgpt_record["number_as_values"].append(number_as_value)
        cehrgpt_record["concept_as_values"].append(concept_as_value)
        cehrgpt_record["units"].append(unit)
        cehrgpt_record["is_numeric_types"].append(is_numeric_type)
        cehrgpt_record["epoch_times"].append(
            time.replace(tzinfo=datetime.timezone.utc).timestamp()
        )

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        cehrgpt_record = {
            "person_id": record["patient_id"],
            "concept_ids": [],
            "ages": [],
            "concept_value_masks": [],
            "number_as_values": [],
            "concept_as_values": [],
            "units": [],
            "is_numeric_types": [],
            "epoch_times": [],
        }
        # Extract the demographic information
        birth_datetime = record["birth_datetime"]
        if isinstance(birth_datetime, pd.Timestamp):
            birth_datetime = birth_datetime.to_pydatetime()
        gender = record["gender"]
        gender = GENDER_UNKNOWN_TOKEN if gender == UNKNOWN_VALUE else gender
        race = record["race"]
        race = RACE_UNKNOWN_TOKEN if race == UNKNOWN_VALUE else race
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
        starting_age = relativedelta(first_visit_start_datetime, birth_datetime).years
        year_str = f"year:{str(first_visit_start_datetime.year)}"
        age_str = f"age:{starting_age}"
        self._update_cehrgpt_record(
            cehrgpt_record, year_str, first_visit_start_datetime, starting_age
        )
        self._update_cehrgpt_record(
            cehrgpt_record, age_str, first_visit_start_datetime, starting_age
        )
        self._update_cehrgpt_record(
            cehrgpt_record, gender, first_visit_start_datetime, starting_age
        )
        self._update_cehrgpt_record(
            cehrgpt_record, race, first_visit_start_datetime, starting_age
        )

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
                    time=visit_start_datetime,
                    age=relativedelta(datetime_cursor, birth_datetime).years,
                )

            datetime_cursor = visit_start_datetime
            # Add a [VS] token
            self._update_cehrgpt_record(
                cehrgpt_record,
                code="[VS]",
                time=datetime_cursor,
                age=relativedelta(datetime_cursor, birth_datetime).years,
            )
            # Add a visit type token
            self._update_cehrgpt_record(
                cehrgpt_record,
                code=visit_type,
                time=datetime_cursor,
                age=relativedelta(datetime_cursor, birth_datetime).years,
            )
            # We need to insert an inpatient hour token right after the visit type, we calculate the hour interval
            # with respect to the midnight of the day
            if is_er_or_inpatient and self._include_inpatient_hour_token:
                if datetime_cursor.hour > 0:
                    # This generates an artificial time token depending on the choice of the time token functions
                    self._update_cehrgpt_record(
                        cehrgpt_record,
                        code=f"i-H{datetime_cursor.hour}",
                        time=datetime_cursor,
                        age=relativedelta(datetime_cursor, birth_datetime).years,
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
                if numeric_value is None and text_value is not None:
                    if text_value.isnumeric():
                        numeric_value = float(text_value)

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
                            time=event_time,
                            age=relativedelta(event_time, birth_datetime).years,
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
                                time=event_time,
                                age=relativedelta(event_time, birth_datetime).years,
                            )

                if event_identity in existing_duplicate_events:
                    continue

                self._update_cehrgpt_record(
                    cehrgpt_record,
                    code=code,
                    time=event_time,
                    age=relativedelta(event_time, birth_datetime).years,
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
                        discharge_facility = DISCHARGE_UNKNOWN_TOKEN
                    else:
                        discharge_facility = (
                            DISCHARGE_UNKNOWN_TOKEN
                            if discharge_facility == UNKNOWN_VALUE
                            else discharge_facility
                        )
                    self._update_cehrgpt_record(
                        cehrgpt_record,
                        code=discharge_facility,
                        time=datetime_cursor,
                        age=relativedelta(datetime_cursor, birth_datetime).years,
                    )

            # Reuse the age and date calculated for the last event in the patient timeline
            self._update_cehrgpt_record(
                cehrgpt_record,
                code="[VE]",
                time=datetime_cursor,
                age=relativedelta(datetime_cursor, birth_datetime).years,
            )

        # Generate the orders of the concepts that the cehrbert dataset mapping function expects
        cehrgpt_record["orders"] = list(
            range(1, len(cehrgpt_record["concept_ids"]) + 1)
        )

        # Add some count information for this sequence
        cehrgpt_record["num_of_concepts"] = len(cehrgpt_record["concept_ids"])
        cehrgpt_record["num_of_visits"] = len(visits)

        if record.get("index_date", None) is not None:
            cehrgpt_record["index_date"] = (
                record["index_date"].replace(tzinfo=datetime.timezone.utc).timestamp()
            )
        if record.get("label", None) is not None:
            cehrgpt_record["label"] = record["label"]
        if record.get("age_at_index", None) is not None:
            cehrgpt_record["age_at_index"] = record["age_at_index"]

        assert len(cehrgpt_record["epoch_times"]) == len(
            cehrgpt_record["concept_ids"]
        ), "The number of time stamps must match with the number of concepts in the sequence"

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
        # Reconstruct the ages input before the filter is applied
        record["ages"] = construct_age_sequence(
            record["concept_ids"], record.get("ages", None)
        )
        record["epoch_times"] = construct_time_sequence(
            record["concept_ids"], record.get("epoch_times", None)
        )
        # Remove the tokens from patient sequences that do not exist in the tokenizer
        record = self.filter_out_invalid_tokens(record)
        # If any concept has a value associated with it, we normalize the value
        record["input_ids"] = self._concept_tokenizer.encode(record["concept_ids"])
        assert len(record["input_ids"]) == len(record["concept_ids"]), (
            "The number of tokens must equal to the number of concepts\n"
            f"decoded concept_ids: {self._concept_tokenizer.decode(record['input_ids'], skip_special_tokens=False)}"
        )
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


class ExtractTokenizedSequenceDataMapping:
    def __init__(
        self,
        person_index_date_map: Dict[int, List[Dict[str, Any]]],
        observation_window: int = 0,
    ):
        self.person_index_date_map = person_index_date_map
        self.observation_window = observation_window

    def _calculate_prediction_start_time(self, prediction_time: float):
        if self.observation_window and self.observation_window > 0:
            return max(prediction_time - self.observation_window * 24 * 3600, 0)
        return 0

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        person_id = record["person_id"]
        prediction_times = self.person_index_date_map[person_id]
        prediction_start_end_times = [
            (
                self._calculate_prediction_start_time(
                    prediction_time_label_map["index_date"]
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                ),
                prediction_time_label_map["index_date"]
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp(),
                prediction_time_label_map["label"],
            )
            for prediction_time_label_map in prediction_times
        ]
        observation_window_indices = np.zeros(
            (len(prediction_times), len(record["epoch_times"])), dtype=bool
        )
        for i, epoch_time in enumerate(record["epoch_times"]):
            for sample_n, (
                feature_extraction_time_start,
                feature_extraction_end_end,
                _,
            ) in enumerate(prediction_start_end_times):
                if (
                    feature_extraction_time_start
                    <= epoch_time
                    <= feature_extraction_end_end
                ):
                    observation_window_indices[sample_n][i] = True

        seq_length = len(record["epoch_times"])
        time_series_columns = ["concept_ids", "input_ids"]
        static_inputs = dict()
        for k, v in record.items():
            if k in ["concept_ids", "input_ids"]:
                continue
            if isinstance(v, (list, np.ndarray)) and len(v) == seq_length:
                time_series_columns.append(k)
            else:
                static_inputs[k] = v

        batched_samples = defaultdict(list)
        for (_, index_date, label), observation_window_index in zip(
            prediction_start_end_times, observation_window_indices
        ):
            for k, v in static_inputs.items():
                batched_samples[k].append(v)
            batched_samples["classifier_label"].append(label)
            batched_samples["index_date"].append(index_date)
            try:
                start_age = int(record["concept_ids"][1].split(":")[1])
            except Exception:
                start_age = -1
            batched_samples["age_at_index"].append(start_age)
            for time_series_column in time_series_columns:
                batched_samples[time_series_column].append(
                    np.asarray(record[time_series_column])[observation_window_index]
                )
        return batched_samples

    def batch_transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        all_batched_record = defaultdict(list)
        all_columns = record.keys()
        for i in range(len(record["concept_ids"])):
            one_record = {}
            for column in all_columns:
                one_record[column] = record[column][i]
            new_batched_record = self.transform(one_record)
            for k, v in new_batched_record.items():
                all_batched_record[k].extend(v)
        return all_batched_record
