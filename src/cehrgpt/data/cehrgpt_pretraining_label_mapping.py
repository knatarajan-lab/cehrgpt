import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping
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


class CehrGptLabelTransformation(DatasetMapping):
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        max_length: int,
        include_values: bool = False,
        include_ttv_prediction: bool = False,
        include_motor_time_to_event: bool = False,
        motor_tte_vocab_size: int = 0,
        motor_num_time_pieces: int = 8,
        pretraining: bool = True,
        include_demographics: bool = False,
        add_linear_prob_token: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.vs_token_id = tokenizer.vs_token_id
        self.ve_token_id = tokenizer.ve_token_id

        self.include_values = include_values
        self.include_ttv_prediction = include_ttv_prediction
        self.pretraining = pretraining
        self.include_demographics = include_demographics
        self.add_linear_prob_token = add_linear_prob_token

        self.empty_tensor = self._convert_to_tensor([])
        # MOTOR TTE configuration
        if include_motor_time_to_event:
            assert motor_tte_vocab_size > 0, (
                f"motor_tte_vocab_size must be greater than 0 "
                f"when include_motor_time_to_event is set to True. "
                f"But motor_tte_vocab_size: {motor_tte_vocab_size} is provided"
            )

        self.include_motor_time_to_event = include_motor_time_to_event
        self.motor_tte_vocab_size = motor_tte_vocab_size
        self.motor_num_time_pieces = motor_num_time_pieces
        self.motor_time_interval = TIME_TO_EVENT_MAX_TIME // motor_num_time_pieces

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

    def _try_reverse_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.pretraining:
            return torch.flip(tensor, dims=[-1])
        return tensor

    @staticmethod
    def _convert_to_tensor(features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        else:
            return torch.tensor(features)

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

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:

        if "input_ids" not in example:
            return example

        example = self.slice_out_input_sequence(example)
        assert example["input_ids"].shape[0] <= self.max_length
        # assert example["attention_mask"].shape[0] <= self.max_length
        # assert example["attention_mask"].shape[0] == example["input_ids"].shape[0], (
        #     f'batch["attention_mask"].shape[0]: {example["attention_mask"].shape[0]}, '
        #     f'batch["input_ids"].shape[0]: {example["input_ids"].shape[0]}'
        # )
        assert example["input_ids"].max() < self.tokenizer.vocab_size, (
            f"batch['input_ids'].max(): {example['input_ids'].max()} must be smaller than "
            f"self.tokenizer.vocab_size: {self.tokenizer.vocab_size}. "
            f"batch['input_ids']: {example['input_ids']} "
        )
        del example["concept_ids"]
        return example

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
        event_times = np.asarray(record["epoch_times"])

        # Convert concept_ids to indices for vectorized operations
        concept_indices = np.array([self.vocab_to_idx[cid] for cid in concept_ids])

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
                    motor_codes = [concept_id]
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
        record["input_ids"] = torch.concat(
            [
                (
                    self._convert_to_tensor(self.tokenizer.encode(demographic_tokens))
                    if demographic_tokens is not None
                    else self.empty_tensor
                ),
                self._convert_to_tensor(record["input_ids"][start_index:end_index]),
                (
                    self._convert_to_tensor([eos_token_id])
                    if add_end_token
                    else self.empty_tensor
                ),
            ]
        ).to(torch.int32)
        record["position_ids"] = torch.concat(
            [
                (
                    torch.zeros([DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32)
                    .to(torch.float32)
                    .fill_(record["position_ids"][0])
                    if demographic_tokens is not None
                    else self.empty_tensor
                ),
                self._convert_to_tensor(record["position_ids"][start_index:end_index]),
                (
                    self._convert_to_tensor([record["position_ids"][-1]])
                    if add_end_token
                    else self.empty_tensor
                ),
            ]
        )
        if self.include_values:
            record["value_indicators"] = torch.concat(
                [
                    (
                        torch.zeros([DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32).to(
                            torch.bool
                        )
                        if demographic_tokens is not None
                        else self.empty_tensor
                    ),
                    self._convert_to_tensor(
                        record["value_indicators"][start_index:end_index]
                    ),
                    (
                        self._convert_to_tensor([False])
                        if add_end_token
                        else self.empty_tensor
                    ),
                ]
            ).to(torch.bool)

            record["values"] = torch.concat(
                [
                    (
                        torch.zeros([DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32)
                        .to(torch.int32)
                        .fill_(self.tokenizer.pad_value_token_id)
                        if demographic_tokens is not None
                        else self.empty_tensor
                    ),
                    self._convert_to_tensor(record["values"][start_index:end_index]),
                    (
                        self._convert_to_tensor([self.tokenizer.pad_value_token_id])
                        if add_end_token
                        else self.empty_tensor
                    ),
                ]
            )

        if self.include_ttv_prediction:
            record["time_to_visits"] = torch.concat(
                [
                    (
                        torch.zeros([DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32)
                        .to(torch.float32)
                        .fill_(-100.0)
                        if demographic_tokens is not None
                        else self.empty_tensor
                    ),
                    self._convert_to_tensor(
                        self._convert_time_to_event(
                            record["concept_ids"][start_index:end_index]
                        )
                    ),
                    (
                        self._convert_to_tensor([-100.0])
                        if add_end_token
                        else self.empty_tensor
                    ),
                ]
            )
        return record

    def slice_out_input_sequence(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Adding the start and end indices to extract a portion of the patient sequence."""
        # Subtract one for the [END] or [LINEAR_PROB] token when sample_packing is not enabled
        new_max_length = (
            self.max_length - 1 if self.add_linear_prob_token else self.max_length
        )
        input_ids = record["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().tolist()
        concept_ids = self.tokenizer.decode(input_ids)
        record["concept_ids"] = concept_ids
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
