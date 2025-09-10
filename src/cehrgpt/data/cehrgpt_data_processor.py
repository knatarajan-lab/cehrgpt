import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping
from transformers.utils import logging

from cehrgpt.gpt_utils import (
    DEMOGRAPHIC_PROMPT_SIZE,
    collect_demographic_prompts_at_visits,
    construct_age_sequence,
    construct_time_sequence,
    extract_time_interval_in_days,
    extract_time_interval_in_hours,
    is_att_token,
    is_clinical_event,
    is_inpatient_att_token,
    is_inpatient_hour_token,
    random_slice_gpt_sequence,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

TIME_TO_EVENT_MAX_TIME = 3650
INPATIENT_STAY_DURATION_LIMIT = 30
LOG = logging.get_logger("transformers")


class CehrGptDataProcessor(DatasetMapping):
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        max_length: int,
        shuffle_records: bool = False,
        include_values: bool = False,
        include_ttv_prediction: bool = False,
        include_motor_time_to_event: bool = False,
        motor_sampling_probability: float = 0.5,
        pretraining: bool = True,
        include_demographics: bool = False,
        add_linear_prob_token: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.vs_token_id = tokenizer.vs_token_id
        self.ve_token_id = tokenizer.ve_token_id

        self.shuffle_records = shuffle_records
        self.include_values = include_values
        self.include_ttv_prediction = include_ttv_prediction
        self.pretraining = pretraining
        self.include_demographics = include_demographics
        self.add_linear_prob_token = add_linear_prob_token
        self.empty_array = np.asarray([])

        if self.pretraining and self.add_linear_prob_token:
            raise ValueError(
                "pretraining and add_linear_prob_token cannot be specify at the same time"
            )

        # Motor related codes
        self.include_motor_time_to_event = include_motor_time_to_event
        self.motor_sampling_probability = motor_sampling_probability
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

    @staticmethod
    def _convert_time_to_event(concept_ids):
        def default_value(c):
            try:
                if is_att_token(c):
                    time_to_visit = extract_time_interval_in_days(c)
                    if (
                        is_inpatient_att_token(c)
                        and time_to_visit > INPATIENT_STAY_DURATION_LIMIT
                    ):
                        return -100
                    return time_to_visit
                elif is_inpatient_hour_token(c):
                    return extract_time_interval_in_hours(c) / 24
                return -100
            except ValueError:
                return -100

        return [float(default_value(_)) for _ in concept_ids]

    def random_sort(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if "record_ranks" not in record:
            return record

        sorting_column = record["record_ranks"]
        random_order = np.random.rand(len(sorting_column))

        if self.include_values:
            iterator = zip(
                sorting_column,
                random_order,
                record["input_ids"],
                record["value_indicators"],
                record["values"],
            )
            sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1], tup2[2]))
            _, _, sorted_input_ids, sorted_value_indicators, sorted_values = zip(
                *list(sorted_list)
            )
            record["input_ids"] = sorted_input_ids
            record["value_indicators"] = sorted_value_indicators
            record["values"] = sorted_values
        else:
            iterator = zip(sorting_column, random_order, record["input_ids"])
            sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1], tup2[2]))
            _, _, sorted_input_ids = zip(*list(sorted_list))
            record["input_ids"] = sorted_input_ids
        return record

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:

        if self.shuffle_records:
            example = self.random_sort(example)

        if "concept_ids" not in example:
            input_ids = example["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.detach().tolist()
            example["concept_ids"] = self.tokenizer.decode(
                input_ids, skip_special_tokens=False
            )
        example["ages"] = pd.Series(example["ages"]).ffill().tolist()
        example = self.slice_out_input_sequence(example)
        # Add the motor labels
        if self.include_motor_time_to_event:
            motor_inputs = self.create_time_to_event_labels(example)
            example.update(motor_inputs)
        del example["concept_ids"]
        return example

    def update_inputs_based_on_indexes(
        self,
        record: Dict[str, Any],
        start_index,
        end_index,
        add_end_token: bool = False,
        demographic_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        last_token_id = (
            self.tokenizer.linear_token_id
            if self.add_linear_prob_token
            else self.tokenizer.end_token_id
        )

        add_last_token = self.add_linear_prob_token | add_end_token

        # Slice out the concept ids
        record["concept_ids"] = (
            (demographic_tokens if demographic_tokens is not None else [])
            + (record["concept_ids"][start_index:end_index])
            + (
                self.tokenizer.decode([last_token_id], skip_special_tokens=False)
                if add_last_token
                else []
            )
        )

        record["input_ids"] = np.concatenate(
            [
                (
                    np.asarray(self.tokenizer.encode(demographic_tokens))
                    if demographic_tokens is not None
                    else self.empty_array
                ),
                np.asarray(record["input_ids"][start_index:end_index]),
                (np.asarray([last_token_id]) if add_last_token else self.empty_array),
            ]
        ).astype(np.int32)

        record["ages"] = np.concatenate(
            [
                (
                    np.full([DEMOGRAPHIC_PROMPT_SIZE], record["ages"][0])
                    if demographic_tokens is not None
                    else self.empty_array
                ),
                np.asarray(record["ages"][start_index:end_index]),
                (
                    np.asarray([record["ages"][-1]])
                    if add_last_token
                    else self.empty_array
                ),
            ]
        ).astype(np.int32)

        # For the new datasets, they contain the column "epoch_times"
        record["epoch_times"] = np.concatenate(
            [
                (
                    np.zeros([DEMOGRAPHIC_PROMPT_SIZE])
                    if demographic_tokens is not None
                    else self.empty_array
                ),
                np.asarray(record["epoch_times"][start_index:end_index]),
                (
                    np.asarray([record["epoch_times"][-1]])
                    if add_last_token
                    else self.empty_array
                ),
            ]
        ).astype(np.float32)

        if self.include_values:
            record["value_indicators"] = np.concatenate(
                [
                    (
                        np.zeros([DEMOGRAPHIC_PROMPT_SIZE])
                        if demographic_tokens is not None
                        else self.empty_array
                    ),
                    np.asarray(record["value_indicators"][start_index:end_index]),
                    np.asarray([False]) if add_last_token else self.empty_array,
                ]
            ).astype(np.bool_)
            record["values"] = np.concatenate(
                [
                    (
                        np.full(
                            [DEMOGRAPHIC_PROMPT_SIZE], self.tokenizer.pad_value_token_id
                        )
                        if demographic_tokens is not None
                        else self.empty_array
                    ),
                    np.asarray(record["values"][start_index:end_index]),
                    (
                        np.asarray([self.tokenizer.pad_value_token_id])
                        if add_last_token
                        else self.empty_array
                    ),
                ]
            ).astype(np.int32)

        if self.include_ttv_prediction:
            record["time_to_visits"] = np.concatenate(
                [
                    (
                        np.full([DEMOGRAPHIC_PROMPT_SIZE], -100.0)
                        if demographic_tokens is not None
                        else self.empty_array
                    ),
                    np.asarray(record["time_to_visits"][start_index:end_index]),
                    np.asarray([-100.0]) if add_last_token else self.empty_array,
                ]
            ).astype(np.float32)

        return record

    def slice_out_input_sequence(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Adding the start and end indices to extract a portion of the patient sequence."""
        # Subtract one for the [END] or [LINEAR_PROB] token when sample_packing is not enabled
        new_max_length = (
            self.max_length - 1
            if self.add_linear_prob_token or self.pretraining
            else self.max_length
        )
        concept_ids = record["concept_ids"]
        seq_length = len(record["input_ids"])

        # For backward compatibility, in case these two columns do not already exist
        record["ages"] = construct_age_sequence(record["concept_ids"], record["ages"])
        record["epoch_times"] = construct_time_sequence(
            record["concept_ids"], record["epoch_times"]
        )

        if self.include_ttv_prediction:
            record["time_to_visits"] = np.asarray(
                self._convert_time_to_event(record["concept_ids"])
            )

        # Return the record directly if the actual sequence length is less than the max sequence
        if seq_length <= new_max_length:
            # We only add [END] to the end of the sequence in pre-training
            record = self.update_inputs_based_on_indexes(
                record, 0, seq_length, add_end_token=self.pretraining
            )
            return record

        if self.pretraining:
            end_index = new_max_length
            # There is a 50% chance we randomly slice out a portion of the patient history and update the demographic
            # prompt depending on the new starting point
            if random.random() < 0.5:
                start_index, end_index, demographic_tokens = random_slice_gpt_sequence(
                    concept_ids, new_max_length
                )
                if start_index != end_index:
                    record = self.update_inputs_based_on_indexes(
                        record, start_index, end_index + 1, add_end_token=False
                    )
                    return record

            # The default employs a right truncation strategy, where the demographic prompt is reserved
            for i in reversed(list(range(0, end_index))):
                current_token = record["input_ids"][i]
                if current_token == self.ve_token_id:
                    # Plus one because slicing is right exclusive
                    end_index = i + 1
                    break

            record = self.update_inputs_based_on_indexes(
                record=record, start_index=0, end_index=end_index, add_end_token=False
            )
            return record
        else:
            if self.include_demographics:
                # We employ a left truncation strategy, where the most recent patient history is reserved for fine-tuning
                demographic_prompts_at_visits = collect_demographic_prompts_at_visits(
                    concept_ids
                )
                for token_index, demographic_prompt in demographic_prompts_at_visits:
                    if (
                        seq_length - token_index
                        <= new_max_length - DEMOGRAPHIC_PROMPT_SIZE
                    ):
                        return self.update_inputs_based_on_indexes(
                            record=record,
                            start_index=token_index,
                            end_index=seq_length,
                            add_end_token=False,
                            demographic_tokens=demographic_prompt,
                        )
            else:
                start_index = seq_length - new_max_length
                end_index = seq_length
                for i in range(start_index, end_index):
                    current_token = record["input_ids"][i]
                    if current_token == self.vs_token_id:
                        return self.update_inputs_based_on_indexes(
                            record=record,
                            start_index=i,
                            end_index=end_index,
                            add_end_token=False,
                        )

            # This could happen when the last visit contains more than new_max_length number of tokens
            # We simply take the last new_max_length number of tokens from the patient sequence
            if len(record["input_ids"]) > new_max_length:
                record = self.update_inputs_based_on_indexes(
                    record=record,
                    start_index=-new_max_length,
                    end_index=seq_length,
                    add_end_token=False,
                )
            return record

    def create_time_to_event_labels(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates time-to-event (TTE) labels and censoring indicators for each visit in a patient's timeline.

        Processes the input sequence in reverse to compute the number of days from each visit (marked by [VE])
        to the occurrence of future motor-related events.

        Args:
            record (Dict[str, Any]): A dictionary containing the encoded patient sequence with the key "input_ids".
                This sequence includes [VS], [VE], time delta tokens, and motor TTE concept codes.

        Returns:
            Dict[str, Any]: The updated input record with added keys:
                - "time_to_event_vectors": np.ndarray of shape [num_visits, motor_vocab_size], containing time-to-event values
                - "event_indicators": np.ndarray of shape [num_visits, motor_vocab_size], where 0 = event occurred, 1 = censored
        """

        """Highly optimized vectorized version using pre-computed token type arrays."""
        concept_ids = record["concept_ids"]
        # Convert concept_ids to indices for vectorized operations
        concept_indices = np.array([self.vocab_to_idx[cid] for cid in concept_ids])
        # Vectorized token type detection
        is_att_tokens = self.is_att_token_array[concept_indices]
        is_clinical_events = self.is_clinical_event_array[concept_indices]
        time_intervals = self.time_intervals_array[concept_indices]

        # Find valid time tokens (att tokens with positive intervals)
        valid_time_tokens = is_att_tokens & (time_intervals > 0)
        n_concepts = len(concept_ids)

        # We need to make sure event_times is monotonic
        event_times = np.zeros(n_concepts, dtype=float)
        previous_time_stamp = record["epoch_times"][0]
        for i, time_stamp in enumerate(record["epoch_times"]):
            if time_stamp < previous_time_stamp:
                time_stamp = previous_time_stamp
            else:
                previous_time_stamp = time_stamp
            event_times[i] = time_stamp

        # Determine prediction positions
        before_valid_time_tokens = np.roll(valid_time_tokens, -1)
        # We randomly make predictions at 50% of the sequence positions
        prediction_positions = (
            np.random.rand(n_concepts) < self.motor_sampling_probability
        )
        # We don't predict at the att time tokens
        prediction_positions &= ~is_att_tokens
        # We disable TTE predictions using the demographics alone
        prediction_positions[:4] = False
        # We take the union of the random prediction positions and the positions right before time token
        prediction_positions = prediction_positions | before_valid_time_tokens
        # We exclude the events that occur at the last time stamp
        prediction_positions &= event_times != event_times[-1]

        prediction_indices = np.where(prediction_positions)[0]
        if len(prediction_indices) == 0:
            return {
                "motor_censor_times": [],
                "motor_row_indices": [],
                "motor_col_indices": [],
                "motor_values": [],
                "motor_tte_task_indicators": [False] * n_concepts,
            }

        # Pre-compute all motor codes for clinical events to avoid repeated lookups
        clinical_positions = np.where(is_clinical_events)[0]
        motor_codes_cache = {}  # position -> list of (motor_code, motor_token_id)

        for pos in clinical_positions:
            concept_id = concept_ids[pos]
            if concept_id in self.motor_code_cache:
                motor_codes = self.motor_code_cache[concept_id]
            else:
                motor_codes = self.tokenizer.get_motor_parents(concept_id)
                self.motor_code_cache[concept_id] = motor_codes

            if motor_codes:
                motor_codes_cache[pos] = [
                    (motor_code, self.tokenizer.get_motor_token_id(motor_code))
                    for motor_code in motor_codes
                ]

        # Process sections in REVERSE order but build results in FORWARD order
        section_boundaries = np.concatenate([prediction_indices, [n_concepts]])
        last_event_time = event_times[-1]

        # Pre-allocate arrays with exact size needed
        num_prediction_positions = len(prediction_indices)
        motor_censor_times = np.zeros(num_prediction_positions, dtype=float)
        motor_tte_task_indicators = np.zeros(n_concepts, dtype=bool)

        # Store sparse matrix data grouped by row for efficient construction
        sparse_data_by_row = {}  # row_idx -> [(col_idx, value), ...]

        # Global motor event state that accumulates as we go backwards
        global_motor_events = {}  # motor_code -> earliest_future_time

        # Process in reverse order but assign to forward row indices
        for i in range(len(prediction_indices) - 1, -1, -1):
            start_index = prediction_indices[i]
            end_index = section_boundaries[i + 1]
            current_event_time = event_times[start_index]

            # Add new motor events from this section to global state
            section_start = start_index + 1
            section_end = end_index + 1 if end_index < n_concepts else n_concepts

            # Process clinical events in this section (in reverse order within section)
            section_clinical_positions = clinical_positions[
                (clinical_positions >= section_start)
                & (clinical_positions < section_end)
            ]

            for pos in reversed(section_clinical_positions):
                if pos in motor_codes_cache:
                    concept_time = event_times[pos]
                    if concept_time > current_event_time:
                        for motor_code, motor_token_id in motor_codes_cache[pos]:
                            global_motor_events[motor_code] = (
                                concept_time,
                                motor_token_id,
                            )

            # Store sparse matrix data for current prediction position
            # Even if global_motor_events is empty, we still need to record this position
            # because it indicates all motor tasks are censored at this time point
            sparse_data_by_row[i] = [
                (motor_token_id, motor_time - current_event_time)
                for motor_code, (
                    motor_time,
                    motor_token_id,
                ) in global_motor_events.items()
            ]
            motor_tte_task_indicators[start_index] = True
            motor_censor_times[i] = last_event_time - current_event_time

        # Build final sparse matrix lists in forward order (no reversals needed)
        motor_row_indices = []
        motor_col_indices = []
        motor_values = []

        for row_idx in sorted(sparse_data_by_row.keys()):
            for col_idx, value in sparse_data_by_row[row_idx]:
                motor_row_indices.append(row_idx)
                motor_col_indices.append(col_idx)
                motor_values.append(value)

        # Filter out unused positions from motor_censor_times
        motor_censor_times = [
            motor_censor_times[i] for i in sorted(sparse_data_by_row.keys())
        ]

        if len(motor_row_indices) == 0:
            LOG.debug(
                "No MOTOR tasks detected for this sample. "
                "Length: %s, last 10 concepts: %s",
                len(concept_ids),
                concept_ids[-10:] if len(concept_ids) >= 10 else concept_ids,
            )

        return {
            "motor_censor_times": motor_censor_times,
            "motor_row_indices": motor_row_indices,
            "motor_col_indices": motor_col_indices,
            "motor_values": motor_values,
            "motor_tte_task_indicators": motor_tte_task_indicators.tolist(),
        }
