import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping
from numba import jit, types
from numba.typed import Dict as NumbaDict
from transformers.utils import logging

from cehrgpt.gpt_utils import (
    DEMOGRAPHIC_PROMPT_SIZE,
    collect_demographic_prompts_at_visits,
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


@jit(nopython=True, cache=True)
def _process_time_intervals_jit(
    concept_indices: np.ndarray,
    is_att_tokens: np.ndarray,
    is_clinical_events: np.ndarray,
    time_intervals: np.ndarray,
    event_times: np.ndarray,
    motor_sampling_probability: float,
    motor_concept_to_parents: NumbaDict,
    motor_parent_to_token_id: NumbaDict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled core logic for processing time-to-event labels.

    Returns:
        - motor_tte_tasks: motor token IDs
        - motor_tte_times: time to event values
        - motor_censor_times: censoring times
        - motor_tte_label_offsets: cumulative offsets for grouping
        - motor_tte_task_indicators: boolean array indicating prediction positions
    """
    n_concepts = len(concept_indices)

    # Find valid time tokens (att tokens with positive intervals)
    valid_time_tokens = is_att_tokens & (time_intervals > 0)
    before_valid_time_tokens = np.zeros(n_concepts, dtype=types.boolean)
    before_valid_time_tokens[:-1] = valid_time_tokens[1:]

    # Random prediction positions
    prediction_positions = np.random.random(n_concepts) < motor_sampling_probability
    # Don't predict at att time tokens
    prediction_positions &= ~is_att_tokens
    # Disable TTE predictions using demographics alone (first 4 positions)
    prediction_positions[:4] = False
    # Take union with positions right before time tokens
    prediction_positions |= before_valid_time_tokens
    # Exclude events at last timestamp
    prediction_positions &= event_times != event_times[-1]

    # Get prediction indices
    prediction_indices = np.where(prediction_positions)[0]
    n_predictions = len(prediction_indices)

    # Pre-allocate result arrays (we'll trim them later)
    max_possible_tasks = n_predictions * 50  # Conservative estimate
    motor_tte_tasks = np.full(max_possible_tasks, -1, dtype=np.int32)
    motor_tte_times = np.full(max_possible_tasks, -1.0, dtype=np.float32)
    motor_censor_times = np.full(n_predictions + 1, -100.0, dtype=np.float32)
    motor_tte_label_offsets = np.zeros(n_predictions + 1, dtype=np.int32)
    motor_tte_task_indicators = np.zeros(n_concepts, dtype=types.boolean)

    task_count = 0

    # Process in reverse order
    for pred_idx in range(n_predictions - 1, -1, -1):
        start_index = prediction_indices[pred_idx]
        if pred_idx == n_predictions - 1:
            end_index = n_concepts - 1
        else:
            end_index = prediction_indices[pred_idx + 1]

        current_event_time = event_times[start_index]

        # Track unique motor codes and their earliest event times
        motor_code_times = NumbaDict.empty(
            key_type=types.int32, value_type=types.float32
        )

        # Process section from start_index+1 to end_index (inclusive)
        for i in range(end_index, start_index, -1):  # Reverse order
            if i >= n_concepts:
                continue

            concept_index = concept_indices[i]
            concept_event_time = event_times[i]

            # Only process clinical events
            if not is_clinical_events[i]:
                continue

            # Get motor parents for this concept
            if concept_index in motor_concept_to_parents:
                motor_parents = motor_concept_to_parents[concept_index]

                for motor_parent in motor_parents:
                    if motor_parent in motor_parent_to_token_id:
                        motor_token_id = motor_parent_to_token_id[motor_parent]

                        # Only update if this is the earliest occurrence (we're going in reverse)
                        if motor_token_id not in motor_code_times:
                            motor_code_times[motor_token_id] = concept_event_time

        # Add tasks for this prediction position
        if len(motor_code_times) > 0:
            start_task_idx = task_count

            for motor_token_id, motor_time in motor_code_times.items():
                if task_count < max_possible_tasks:
                    motor_tte_tasks[task_count] = motor_token_id
                    motor_tte_times[task_count] = motor_time - current_event_time
                    task_count += 1

            motor_censor_times[pred_idx] = event_times[-1] - current_event_time
            motor_tte_label_offsets[pred_idx] = task_count - start_task_idx
            motor_tte_task_indicators[start_index] = True

    # Trim arrays to actual size
    motor_tte_tasks = motor_tte_tasks[:task_count]
    motor_tte_times = motor_tte_times[:task_count]

    # Convert offsets to cumulative
    cumsum = 0
    for i in range(len(motor_tte_label_offsets)):
        temp = motor_tte_label_offsets[i]
        motor_tte_label_offsets[i] = cumsum
        cumsum += temp

    # Add final censor time
    motor_censor_times[-1] = -100.0

    return (
        motor_tte_tasks,
        motor_tte_times,
        motor_censor_times,
        motor_tte_label_offsets,
        motor_tte_task_indicators,
    )


class CehrGptLabelTransformation(DatasetMapping):
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

        # Motor related codes
        self.include_motor_time_to_event = include_motor_time_to_event
        self.motor_sampling_probability = motor_sampling_probability
        self.motor_code_cache: Dict[str, List[str]] = {}
        # Pre-compute vocab-wide token type mappings
        self._precompute_vocab_mappings()

        # Pre-compute motor code mappings for JIT
        if self.include_motor_time_to_event:
            self._precompute_motor_mappings()

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

    def _precompute_motor_mappings(self):
        """Pre-compute motor code mappings for JIT compatibility."""
        LOG.info("Pre-computing motor code mappings for JIT...")

        # Create numba-compatible dictionaries
        self.motor_concept_to_parents = NumbaDict.empty(
            key_type=types.int32, value_type=types.int32[:]
        )
        self.motor_parent_to_token_id = NumbaDict.empty(
            key_type=types.int32, value_type=types.int32
        )

        # Build mapping from concept indices to motor parent indices
        motor_parent_to_idx = {}
        next_motor_idx = 0

        for concept_idx, token in enumerate(self.vocab_tokens):
            if is_clinical_event(token):
                motor_codes = self.tokenizer.get_motor_parents(token)
                if motor_codes:
                    motor_parent_indices = []
                    for motor_code in motor_codes:
                        if motor_code not in motor_parent_to_idx:
                            motor_parent_to_idx[motor_code] = next_motor_idx
                            next_motor_idx += 1
                        motor_parent_indices.append(motor_parent_to_idx[motor_code])

                    self.motor_concept_to_parents[concept_idx] = np.array(
                        motor_parent_indices, dtype=np.int32
                    )

        # Build mapping from motor parent indices to token IDs
        for motor_code, motor_idx in motor_parent_to_idx.items():
            motor_token_id = self.tokenizer.get_motor_token_id(motor_code)
            self.motor_parent_to_token_id[motor_idx] = motor_token_id

        LOG.info(f"Pre-computed {len(motor_parent_to_idx)} motor code mappings")

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

        # There might be nan position_ids in-between, let's use the forward fill method to fill the nan values
        example["position_ids"] = pd.Series(example["position_ids"]).ffill().tolist()
        # start = time.time()
        example = self.slice_out_input_sequence(example)
        # print(f"slice_out_input_sequence.call: {time.time() - start}")

        # Add the motor labels
        if self.include_motor_time_to_event:
            # start = time.time()
            motor_inputs = self.create_time_to_event_labels(example)
            example.update(motor_inputs)
            # print(f"create_time_to_event_labels.call: {time.time() - start}")

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

        linear_token_id = (
            self.tokenizer.linear_token_id
            if self.tokenizer.linear_token_id
            else self.tokenizer.oov_token_id
        )
        eos_token_id = (
            linear_token_id
            if self.add_linear_prob_token
            else self.tokenizer.end_token_id
        )

        # Slice out the concept ids
        record["concept_ids"] = (
            (demographic_tokens if demographic_tokens is not None else [])
            + (record["concept_ids"][start_index:end_index])
            + (
                self.tokenizer.decode([eos_token_id], skip_special_tokens=False)
                if add_end_token
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
                (np.asarray([eos_token_id]) if add_end_token else self.empty_array),
            ]
        ).astype(np.int32)

        record["position_ids"] = np.concatenate(
            [
                (
                    np.full([DEMOGRAPHIC_PROMPT_SIZE], record["position_ids"][0])
                    if demographic_tokens is not None
                    else self.empty_array
                ),
                np.asarray(record["position_ids"][start_index:end_index]),
                (
                    np.asarray([record["position_ids"][-1]])
                    if add_end_token
                    else self.empty_array
                ),
            ]
        ).astype(np.int32)

        if self.include_values:
            record["value_indicators"] = np.concatenate(
                [
                    (
                        np.zeros([DEMOGRAPHIC_PROMPT_SIZE])
                        if demographic_tokens is not None
                        else self.empty_array
                    ),
                    np.asarray(record["value_indicators"][start_index:end_index]),
                    np.asarray([False]) if add_end_token else self.empty_array,
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
                        if add_end_token
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
                    np.asarray(
                        self._convert_time_to_event(
                            record["concept_ids"][start_index:end_index]
                        )
                    ),
                    np.asarray([-100.0]) if add_end_token else self.empty_array,
                ]
            ).astype(np.float32)

        # For the new datasets, they contain the column "epoch_times"
        if "epoch_times" in record:
            epoch_times = record["epoch_times"][start_index:end_index]
            record["epoch_times"] = np.concatenate(
                [
                    (
                        np.zeros([DEMOGRAPHIC_PROMPT_SIZE])
                        if demographic_tokens is not None
                        else self.empty_array
                    ),
                    np.asarray(epoch_times[start_index:end_index]),
                    (
                        np.asarray([epoch_times[-1]])
                        if add_end_token
                        else self.empty_array
                    ),
                ]
            ).astype(np.float32)

        return record

    def slice_out_input_sequence(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Adding the start and end indices to extract a portion of the patient sequence."""
        # Subtract one for the [END] or [LINEAR_PROB] token when sample_packing is not enabled
        new_max_length = (
            self.max_length - 1 if self.add_linear_prob_token else self.max_length
        )
        concept_ids = record["concept_ids"]
        seq_length = len(record["input_ids"])
        # Return the record directly if the actual sequence length is less than the max sequence
        if seq_length <= new_max_length:
            record = self.update_inputs_based_on_indexes(
                record, 0, seq_length, add_end_token=True
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
                            add_end_token=True,
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
                            add_end_token=True,
                        )

            # This could happen when the last visit contains more than new_max_length number of tokens
            # We simply take the last new_max_length number of tokens from the patient sequence
            if len(record["input_ids"]) > new_max_length:
                record = self.update_inputs_based_on_indexes(
                    record=record,
                    start_index=-new_max_length,
                    end_index=seq_length,
                    add_end_token=True,
                )

            return record

    def create_time_to_event_labels(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates time-to-event (TTE) labels and censoring indicators for each visit in a patient's timeline.

        Optimized version using Numba JIT compilation.
        """
        concept_ids = record["concept_ids"]
        concept_indices = np.array([self.vocab_to_idx[cid] for cid in concept_ids])

        # Vectorized token type detection using pre-computed arrays
        is_att_tokens = self.is_att_token_array[concept_indices]
        is_clinical_events = self.is_clinical_event_array[concept_indices]
        time_intervals = self.time_intervals_array[concept_indices]

        # Compute event times
        n_concepts = len(concept_ids)
        if "epoch_times" in record:
            event_times = np.array(record["epoch_times"], dtype=np.float32)
            # Ensure monotonicity
            for i in range(1, len(event_times)):
                if event_times[i] < event_times[i - 1]:
                    event_times[i] = event_times[i - 1]
        else:
            event_times = np.zeros(n_concepts, dtype=np.float32)
            valid_time_tokens = is_att_tokens & (time_intervals > 0)
            time_token_indices = np.where(valid_time_tokens)[0]
            time_token_event_times = np.cumsum(
                np.concatenate([np.zeros(1), time_intervals[valid_time_tokens]])
            )

            for i, (start, end) in enumerate(
                zip(
                    [0] + time_token_indices.tolist(),
                    time_token_indices.tolist() + [n_concepts],
                )
            ):
                event_times[start:end] = time_token_event_times[i]

        # Call JIT-compiled function
        (
            motor_tte_tasks,
            motor_tte_times,
            motor_censor_times,
            motor_tte_label_offsets,
            motor_tte_task_indicators,
        ) = _process_time_intervals_jit(
            concept_indices.astype(np.int32),
            is_att_tokens,
            is_clinical_events,
            time_intervals.astype(np.int32),
            event_times,
            self.motor_sampling_probability,
            self.motor_concept_to_parents,
            self.motor_parent_to_token_id,
        )

        if len(motor_tte_times) == 0:
            LOG.debug(
                "No MOTOR tasks detected for this sample. "
                "Length: %s, last 10 concepts: %s",
                len(concept_ids),
                concept_ids[-10:] if len(concept_ids) >= 10 else concept_ids,
            )

        return {
            "motor_censor_times": motor_censor_times.tolist(),
            "motor_tte_tasks": motor_tte_tasks.tolist(),
            "motor_tte_times": motor_tte_times.tolist(),
            "motor_tte_label_offsets": motor_tte_label_offsets.tolist(),
            "motor_tte_task_indicators": motor_tte_task_indicators.tolist(),
        }
