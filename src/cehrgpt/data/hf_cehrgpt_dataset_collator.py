import copy
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.utils import logging

from cehrgpt.gpt_utils import (
    DEMOGRAPHIC_PROMPT_SIZE,
    collect_demographic_prompts_at_visits,
    extract_time_interval_in_days,
    is_att_token,
    is_inpatient_att_token,
    is_visit_end,
    random_slice_gpt_sequence,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

TIME_TO_EVENT_MAX_TIME = 3650
INPATIENT_STAY_DURATION_LIMIT = 30
LOG = logging.get_logger("transformers")


class CehrGptDataCollator:
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        max_length: int,
        shuffle_records: bool = False,
        include_values: bool = False,
        include_ttv_prediction: bool = False,
        use_sub_time_tokenization: bool = False,
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

        self.shuffle_records = shuffle_records
        self.include_values = include_values
        self.include_ttv_prediction = include_ttv_prediction
        self.use_sub_time_tokenization = use_sub_time_tokenization
        self.pretraining = pretraining
        self.include_demographics = include_demographics
        self.add_linear_prob_token = add_linear_prob_token

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

        if self.use_sub_time_tokenization:
            token_to_time_token_mapping = tokenizer.token_to_time_token_mapping
            if not token_to_time_token_mapping:
                raise ValueError(
                    "The token_to_time_token_mapping in CehrGptTokenizer cannot be None "
                    "when use_sub_time_tokenization is enabled"
                )
            # Create the tensors for converting time tokens to the sub time tokens
            self.time_tokens = torch.tensor(
                list(tokenizer.token_to_time_token_mapping.keys()), dtype=torch.int64
            )
            self.mapped_sub_time_tokens = torch.tensor(
                list(token_to_time_token_mapping.values()), dtype=torch.int64
            )

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
                return -100
            except ValueError:
                return -100

        return [float(default_value(_)) for _ in concept_ids]

    def __call__(self, examples):
        sample_packing = getattr(self, "sample_packing", False)
        examples = [self.generate_start_end_index(_, sample_packing) for _ in examples]
        examples = [self.random_sort(_) for _ in examples]
        batch = {}

        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [
            self._try_reverse_tensor(self._convert_to_tensor(example["input_ids"]))
            for example in examples
        ]

        batch_attention_mask = [
            self._try_reverse_tensor(
                self._convert_to_tensor(example["attention_mask"]).to(torch.float)
                if "attention_mask" in example
                else torch.ones_like(
                    self._convert_to_tensor(example["input_ids"]), dtype=torch.float
                )
            )
            for example in examples
        ]

        # Pad sequences to the max length in the batch
        batch["input_ids"] = self._try_reverse_tensor(
            pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).to(torch.int64)
        )

        batch["attention_mask"] = self._try_reverse_tensor(
            pad_sequence(batch_attention_mask, batch_first=True, padding_value=0.0)
        )
        assert batch["input_ids"].shape[1] <= self.max_length
        assert batch["attention_mask"].shape[1] <= self.max_length
        assert batch["attention_mask"].shape[1] == batch["input_ids"].shape[1], (
            f'batch["attention_mask"].shape[1]: {batch["attention_mask"].shape[1]}, '
            f'batch["input_ids"].shape[1]: {batch["input_ids"].shape[1]}'
        )
        assert batch["input_ids"].max() < self.tokenizer.vocab_size, (
            f"batch['input_ids'].max(): {batch['input_ids'].max()} must be smaller than "
            f"self.tokenizer.vocab_size: {self.tokenizer.vocab_size}. "
            f"batch['input_ids']: {batch['input_ids']} "
        )

        if "position_ids" in examples[0]:
            batch_position_ids = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["position_ids"])
                )
                for example in examples
            ]
            # Pad sequences to the max length in the batch
            batch["position_ids"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_position_ids,
                    batch_first=True,
                    padding_value=0,
                ).to(torch.int64)
            )

        if self.pretraining:
            batch["labels"] = torch.where(
                (batch["input_ids"] != self.tokenizer.pad_token_id)
                & batch["attention_mask"].to(torch.bool),
                batch["input_ids"],
                -100,
            )

        if self.use_sub_time_tokenization:
            time_token_indicators = torch.isin(batch["input_ids"], self.time_tokens)
            masked_tokens = batch["input_ids"].clone()
            masked_tokens[~time_token_indicators] = -1
            # Get the index of the sub_time_tokens from the time_tokens tensor
            sub_time_token_indices = torch.argmax(
                (
                    masked_tokens.unsqueeze(-1)
                    == self.time_tokens.unsqueeze(0).unsqueeze(0)
                ).to(torch.int32),
                dim=-1,
            )
            sub_time_tokens = self.mapped_sub_time_tokens[sub_time_token_indices]
            batch["time_token_indicators"] = time_token_indicators
            batch["sub_time_tokens"] = sub_time_tokens

        if self.include_ttv_prediction:
            batch_time_to_visits = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["time_to_visits"])
                )
                for example in examples
            ]
            batch["time_to_visits"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_time_to_visits, batch_first=True, padding_value=-100.0
                )
            )

        if self.include_motor_time_to_event:
            examples_with_motor_tte = [
                self.create_time_to_event_labels(_) for _ in examples
            ]
            batch_motor_time_to_event_vectors = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["time_to_event_vectors"])
                )
                for example in examples_with_motor_tte
            ]
            batch_motor_event_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["event_indicators"])
                )
                for example in examples_with_motor_tte
            ]
            batch_motor_time_to_event_to_include = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["time_to_event_to_include"])
                )
                for example in examples_with_motor_tte
            ]
            batch_motor_time_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["time_indicators"])
                )
                for example in examples_with_motor_tte
            ]

            batch_motor_time_to_event_vectors = torch.concat(
                batch_motor_time_to_event_vectors, dim=0
            ).to(torch.float32)

            # If every example in the batch only contains one visit, there would be no labels generated for MOTOR TTE
            # we only create the labels when any example has more than one visit
            if batch_motor_time_to_event_vectors.dim() <= 1:
                LOG.warning(
                    "There are no MOTOR TTE labels generated for this batch "
                    "because every example in this batch only contains one visit."
                )
            else:
                batch_size = len(examples)
                length, num_time_pieces, motor_tte_vocab_size = (
                    batch_motor_time_to_event_vectors.shape
                )
                padded_length = batch_size - length % batch_size
                batch["motor_time_to_event_vectors"] = (
                    torch.concat(
                        [
                            batch_motor_time_to_event_vectors,
                            torch.full(
                                (padded_length, num_time_pieces, motor_tte_vocab_size),
                                0.0,
                            ),
                        ],
                        dim=0,
                    )
                    .reshape((batch_size, -1, num_time_pieces, motor_tte_vocab_size))
                    .to(torch.float32)
                )

                # Motor event indicators that indicate there is an event occurred in this time interval
                batch_motor_event_indicators = torch.concat(
                    batch_motor_event_indicators, dim=0
                ).to(torch.bool)
                batch["motor_event_indicators"] = (
                    torch.concat(
                        [
                            batch_motor_event_indicators,
                            torch.full(
                                (padded_length, num_time_pieces, motor_tte_vocab_size),
                                False,
                            ),
                        ],
                        dim=0,
                    )
                    .reshape((batch_size, -1, num_time_pieces, motor_tte_vocab_size))
                    .to(torch.bool)
                )

                # Input to indicate whether the visit should be included for TTE predictions
                batch_motor_time_to_event_to_include = torch.concat(
                    batch_motor_time_to_event_to_include, dim=0
                ).to(torch.bool)
                batch["motor_time_to_event_to_include"] = (
                    torch.concat(
                        [
                            batch_motor_time_to_event_to_include,
                            torch.full((padded_length,), False),
                        ],
                        dim=0,
                    ).to(torch.bool)
                ).reshape((batch_size, -1))

                # Motor time indicators that indicate whether there are neither clinical events nor censor events
                batch_motor_time_indicators = torch.concat(
                    batch_motor_time_indicators, dim=0
                ).to(torch.bool)
                batch["motor_time_indicators"] = (
                    torch.concat(
                        [
                            batch_motor_time_indicators,
                            torch.full(
                                (padded_length, num_time_pieces, motor_tte_vocab_size),
                                False,
                            ),
                        ],
                        dim=0,
                    )
                    .reshape((batch_size, -1, num_time_pieces, motor_tte_vocab_size))
                    .to(torch.bool)
                )

                batch["motor_end_index"] = torch.concat(
                    [
                        torch.full((length, 1), 1, dtype=torch.int32),
                        torch.full((padded_length, 1), 0, dtype=torch.int32),
                    ]
                ).reshape((batch_size, -1))

        if self.include_values:
            batch_value_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["value_indicators"]).to(torch.bool)
                )
                for example in examples
            ]
            batch_values = [
                self._try_reverse_tensor(self._convert_to_tensor(example["values"]))
                for example in examples
            ]
            batch["value_indicators"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_value_indicators, batch_first=True, padding_value=False
                )
            )
            batch["values"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_values,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_value_token_id,
                ).to(torch.int64)
            )
            assert batch["value_indicators"].shape[1] <= self.max_length
            assert batch["values"].shape[1] <= self.max_length

            if self.pretraining:
                batch["true_value_indicators"] = batch["value_indicators"].clone()
                batch["true_values"] = torch.where(
                    batch["value_indicators"], batch["values"].clone(), -100
                )

        bz = len(examples)
        if "person_id" in examples[0]:
            batch["person_id"] = (
                torch.cat(
                    [
                        self._convert_to_tensor(example["person_id"]).reshape(-1, 1)
                        for example in examples
                    ],
                    dim=0,
                )
                .to(torch.int32)
                .reshape(bz, -1)
            )

        if "index_date" in examples[0]:
            batch["index_date"] = torch.cat(
                [
                    torch.tensor(example["index_date"], dtype=torch.float64).reshape(
                        -1, 1
                    )
                    for example in examples
                ],
                dim=0,
            ).reshape(bz, -1)

        if "age_at_index" in examples[0]:
            batch["age_at_index"] = (
                torch.cat(
                    [
                        self._convert_to_tensor(example["age_at_index"]).reshape(-1, 1)
                        for example in examples
                    ],
                    dim=0,
                )
                .to(torch.float32)
                .reshape(bz, -1)
            )

        if "classifier_label" in examples[0]:
            batch["classifier_label"] = (
                torch.cat(
                    [
                        self._convert_to_tensor(example["classifier_label"]).reshape(
                            -1, 1
                        )
                        for example in examples
                    ],
                    dim=0,
                )
                .to(torch.float32)
                .reshape(bz, -1)
            )

        return batch

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
        input_ids = record["input_ids"]
        sample_packing = getattr(self, "sample_packing", False)

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().tolist()

        # This potentially contains packed samples, we need to handle that
        packed_concept_ids = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        pad_indices = []
        if sample_packing:
            # We start from the first index
            for i in range(len(packed_concept_ids)):
                if packed_concept_ids[i] == self.tokenizer.pad_token:
                    # If we encounter consecutive pads, we should break out of the loop
                    if pad_indices and pad_indices[-1] == self.tokenizer.pad_token:
                        break
                    pad_indices.append(i)

        # If we did not find a pad, that means the whole sequence belongs to one sample
        if len(pad_indices) == 0:
            pad_indices.append(len(packed_concept_ids))

        timepiece_time_to_event_vectors = []
        timepiece_event_indicators = []
        timepiece_indicators = []
        time_to_event_to_includes = []

        for start_index, end_index in zip([0] + pad_indices[:-1], pad_indices):
            concept_ids = packed_concept_ids[start_index:end_index]
            if concept_ids[0] == self.tokenizer.pad_token:
                concept_ids.pop(0)
            time_to_event_vectors = []
            global_event_indicators = []

            # First collect TTE data in reverse chronological order
            censor_times = []
            time_to_event_data: List[Dict[str, int]] = []
            time_to_event_dict: Dict[str, int] = {}
            time_to_event_to_include: List[bool] = []
            next_future_visit_concepts = set()
            time_interval = 0

            # Reverse walk through concept_ids to calculate TTE from each [VE] point
            for concept_id in reversed(concept_ids):
                if is_visit_end(concept_id):
                    # Update TTE for existing concepts, or add new ones seen in this visit
                    for existing_concept_id in list(time_to_event_dict.keys()):
                        if existing_concept_id in next_future_visit_concepts:
                            time_to_event_dict[existing_concept_id] = time_interval
                        else:
                            time_to_event_dict[existing_concept_id] += time_interval

                    for next_concept_id in next_future_visit_concepts:
                        if next_concept_id not in time_to_event_dict:
                            time_to_event_dict[next_concept_id] = time_interval

                    # If the next visit occurs on the same day as the previous one, we don't want to do TTE for the
                    # previous visit
                    time_to_event_to_include.append(time_interval > 0)
                    time_to_event_data.append(copy.deepcopy(time_to_event_dict))
                    # Record the censor time at the end of the visit
                    if censor_times:
                        censor_times.append(censor_times[-1] + time_interval)
                    else:
                        censor_times.append(time_interval)
                    time_interval = 0
                    next_future_visit_concepts.clear()

                elif is_att_token(concept_id):
                    time_interval += extract_time_interval_in_days(concept_id)

                elif self.tokenizer.is_motor_time_to_event_code(concept_id):
                    next_future_visit_concepts.add(concept_id)

            if len(time_to_event_data) == 0:
                LOG.info(
                    "Vist end event is not detected for this sample, and is skipped for MOTOR tasks."
                    "It's likely this sample contains a long admission. length: %s, concept_ids[-10:] %s",
                    len(concept_ids),
                    concept_ids[-10:],
                )
                continue

            # Reverse back to chronological order for final labels
            time_to_event_data.reverse()
            censor_times.reverse()
            time_to_event_to_include.reverse()

            for censor_time, visit_tte_data in zip(censor_times, time_to_event_data):
                time_to_event_vector = np.full(
                    self.tokenizer.motor_tte_vocab_size,
                    fill_value=censor_time,
                    dtype=np.int32,
                )
                event_indicator = np.zeros(
                    self.tokenizer.motor_tte_vocab_size,
                    dtype=np.int32,
                )
                visit_token_ids = [
                    self.tokenizer.get_motor_token_id(concept_id)
                    for concept_id in visit_tte_data.keys()
                ]
                visit_tte_values = list(visit_tte_data.values())

                time_to_event_vector[visit_token_ids] = visit_tte_values
                event_indicator[visit_token_ids] = 1  # not censored (event occurred)

                time_to_event_vectors.append(time_to_event_vector)
                global_event_indicators.append(event_indicator)

            time_to_event_vectors = np.asarray(time_to_event_vectors)
            global_event_indicators = np.asarray(global_event_indicators).astype(bool)
            n_visits = len(time_to_event_vectors)

            timepiece_time_to_event_vector = np.full(
                (
                    self.motor_num_time_pieces,
                    n_visits,
                    self.tokenizer.motor_tte_vocab_size,
                ),
                fill_value=0,
                dtype=np.int32,
            )
            timepiece_event_indicator = np.zeros(
                (
                    self.motor_num_time_pieces,
                    n_visits,
                    self.tokenizer.motor_tte_vocab_size,
                ),
                dtype=bool,
            )
            timepiece_indicator = np.zeros(
                (
                    self.motor_num_time_pieces,
                    n_visits,
                    self.tokenizer.motor_tte_vocab_size,
                ),
                dtype=bool,
            )

            # Putting the event time and censor time into the corresponding time bins
            for bin_num in range(self.motor_num_time_pieces):
                start = self.motor_time_interval * bin_num
                end = self.motor_time_interval * (bin_num + 1)
                time_in_bin = np.clip(time_to_event_vectors - start, 0, end - start)
                timepiece_time_to_event_vector[bin_num] = time_in_bin
                event_indicator = (
                    global_event_indicators
                    & (start <= time_to_event_vectors)
                    & (time_to_event_vectors < end)
                )
                timepiece_event_indicator[bin_num] = event_indicator
                timepiece_indicator[bin_num] = time_in_bin > 0 | event_indicator

            timepiece_time_to_event_vectors.append(
                timepiece_time_to_event_vector.swapaxes(0, 1)
            )
            timepiece_event_indicators.append(timepiece_event_indicator.swapaxes(0, 1))
            timepiece_indicators.append(timepiece_indicator.swapaxes(0, 1))
            time_to_event_to_includes.append(np.asarray(time_to_event_to_include))

        record["time_to_event_vectors"] = np.concatenate(
            timepiece_time_to_event_vectors, axis=0
        )
        record["event_indicators"] = np.concatenate(timepiece_event_indicators, axis=0)
        record["time_indicators"] = np.concatenate(timepiece_indicators, axis=0)
        record["time_to_event_to_include"] = np.concatenate(
            time_to_event_to_includes, axis=0
        )
        return record

    def random_sort(self, record: Dict[str, Any]) -> Dict[str, Any]:

        if not self.shuffle_records:
            return record

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
            record["input_ids"] = self._convert_to_tensor(sorted_input_ids)
            record["value_indicators"] = self._convert_to_tensor(
                sorted_value_indicators
            )
            record["values"] = self._convert_to_tensor(sorted_values)
        else:
            iterator = zip(sorting_column, random_order, record["input_ids"])
            sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1], tup2[2]))
            _, _, sorted_input_ids = zip(*list(sorted_list))
            record["input_ids"] = self._convert_to_tensor(sorted_input_ids)
        return record

    def generate_start_end_index(
        self,
        record: Dict[str, Any],
        sample_packing: bool,
        max_length_allowed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Adding the start and end indices to extract a portion of the patient sequence."""
        # concept_ids will be used to for time to event predictions and identifying the visit starts
        max_length_allowed = (
            self.max_length if max_length_allowed is None else max_length_allowed
        )
        input_ids = record["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().tolist()
        concept_ids = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        seq_length = len(record["input_ids"])

        # Subtract one for the [END] token when sample_packing is not enabled
        new_max_length = (
            max_length_allowed if sample_packing else max_length_allowed - 1
        )

        if self.include_ttv_prediction:
            record["time_to_visits"] = torch.concat(
                [self._convert_to_tensor(self._convert_time_to_event(concept_ids))]
            )

        # If linear token exists, we will use it, otherwise we default to the OOV token
        linear_token_id = (
            self.tokenizer.linear_token_id
            if self.tokenizer.linear_token_id
            else self.tokenizer.oov_token_id
        )
        eos_token = (
            linear_token_id
            if self.add_linear_prob_token
            else self.tokenizer.end_token_id
        )

        # Return the record directly if the actual sequence length is less than the max sequence
        if seq_length <= new_max_length:
            if not sample_packing:
                record["input_ids"] = torch.concat(
                    [
                        self._convert_to_tensor(record["input_ids"]),
                        self._convert_to_tensor([eos_token]),
                    ]
                )
                if self.include_values:
                    record["value_indicators"] = torch.concat(
                        [
                            self._convert_to_tensor(record["value_indicators"]),
                            self._convert_to_tensor([False]),
                        ]
                    ).to(torch.bool)
                    record["values"] = torch.concat(
                        [
                            self._convert_to_tensor(record["values"]),
                            self._convert_to_tensor(
                                [self.tokenizer.pad_value_token_id]
                            ),
                        ]
                    )
                if self.include_ttv_prediction:
                    record["time_to_visits"] = torch.concat(
                        [
                            record["time_to_visits"],
                            self._convert_to_tensor([-100.0]),
                        ]
                    )
            return record

        if self.pretraining:
            # There is a 50% chance we randomly slice out a portion of the patient history and update the demographic
            # prompt depending on the new starting point
            if random.random() < 0.5 and not sample_packing:
                start_index, end_index, demographic_tokens = random_slice_gpt_sequence(
                    concept_ids, new_max_length
                )
                if start_index != end_index:
                    record["input_ids"] = self._convert_to_tensor(
                        record["input_ids"][start_index : end_index + 1]
                    )
                    if self.include_values:
                        record["value_indicators"] = self._convert_to_tensor(
                            record["value_indicators"][start_index : end_index + 1]
                        ).to(torch.bool)
                        record["values"] = self._convert_to_tensor(
                            record["values"][start_index : end_index + 1]
                        )
                    if self.include_ttv_prediction:
                        record["time_to_visits"] = self._convert_to_tensor(
                            self._convert_time_to_event(
                                concept_ids[start_index : end_index + 1]
                            )
                        )
                    return record

            # The default employs a right truncation strategy, where the demographic prompt is reserved
            end_index = new_max_length
            for i in reversed(list(range(0, end_index))):
                current_token = record["input_ids"][i]
                if current_token == self.ve_token_id:
                    # Plus one because slicing is right exclusive
                    end_index = i + 1
                    break

            record["input_ids"] = record["input_ids"][0:end_index]

            # We want to make sure we take the subset of attention_mask in sample packing if this field is available
            if sample_packing and "attention_mask" in record:
                record["attention_mask"] = record["attention_mask"][0:end_index]

            if sample_packing and "position_ids" in record:
                record["position_ids"] = record["position_ids"][0:end_index]

            if self.include_values:
                record["value_indicators"] = self._convert_to_tensor(
                    record["value_indicators"][0:end_index]
                ).to(torch.bool)
                record["values"] = self._convert_to_tensor(
                    record["values"][0:end_index]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = self._convert_to_tensor(
                    self._convert_time_to_event(concept_ids[0:end_index])
                )
            return record
        else:
            if self.include_demographics and not sample_packing:
                # We employ a left truncation strategy, where the most recent patient history is reserved for fine-tuning
                demographic_prompts_at_visits = collect_demographic_prompts_at_visits(
                    concept_ids
                )
                for token_index, demographic_prompt in demographic_prompts_at_visits:
                    if (
                        seq_length - token_index
                        <= new_max_length - DEMOGRAPHIC_PROMPT_SIZE
                    ):
                        demographic_tokens = self.tokenizer.encode(demographic_prompt)
                        record["input_ids"] = torch.concat(
                            [
                                self._convert_to_tensor(demographic_tokens),
                                self._convert_to_tensor(
                                    record["input_ids"][token_index:seq_length]
                                ),
                            ]
                        )
                        if self.include_values:
                            record["value_indicators"] = torch.concat(
                                [
                                    torch.zeros(
                                        [DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32
                                    ).to(torch.bool),
                                    self._convert_to_tensor(
                                        record["value_indicators"][
                                            token_index:seq_length
                                        ]
                                    ),
                                ]
                            )
                            record["values"] = torch.concat(
                                [
                                    torch.zeros(
                                        [DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32
                                    )
                                    .to(torch.int32)
                                    .fill_(self.tokenizer.pad_value_token_id),
                                    self._convert_to_tensor(
                                        record["values"][token_index:seq_length]
                                    ),
                                ]
                            )
                        if self.include_ttv_prediction:
                            record["time_to_visits"] = torch.concat(
                                [
                                    torch.zeros(
                                        [DEMOGRAPHIC_PROMPT_SIZE], dtype=torch.int32
                                    )
                                    .to(torch.float32)
                                    .fill_(-100.0),
                                    record["time_to_visits"][token_index:seq_length],
                                ]
                            )
                        break
            else:
                start_index = seq_length - new_max_length
                end_index = seq_length
                for i in range(start_index, end_index):
                    current_token = record["input_ids"][i]
                    if current_token == self.vs_token_id:
                        record["input_ids"] = record["input_ids"][i:end_index]
                        if sample_packing and "attention_mask" in record:
                            record["attention_mask"] = record["attention_mask"][
                                i:end_index
                            ]
                        if sample_packing and "position_ids" in record:
                            record["position_ids"] = record["position_ids"][i:end_index]
                        if self.include_values:
                            record["value_indicators"] = record["value_indicators"][
                                i:end_index
                            ]
                            record["values"] = record["values"][i:end_index]
                        if self.include_ttv_prediction:
                            record["time_to_visits"] = record["time_to_visits"][
                                i:end_index
                            ]
                        break

            # This could happen when the last visit contains more than new_max_length number of tokens
            # We simply take the last new_max_length number of tokens from the patient sequence
            if len(record["input_ids"]) > new_max_length:
                record["input_ids"] = record["input_ids"][-new_max_length:]
                if sample_packing and "attention_mask" in record:
                    record["attention_mask"] = record["attention_mask"][
                        -new_max_length:
                    ]
                if sample_packing and "position_ids" in record:
                    record["position_ids"] = record["position_ids"][-new_max_length:]
                if self.include_values:
                    record["value_indicators"] = record["value_indicators"][
                        -new_max_length:
                    ]
                    record["values"] = record["values"][-new_max_length:]
                if self.include_ttv_prediction:
                    record["time_to_visits"] = record["time_to_visits"][
                        -new_max_length:
                    ]

            if not sample_packing:
                # Finally we add the end token to the end of the sequence
                record["input_ids"] = torch.concat(
                    [
                        self._convert_to_tensor(record["input_ids"]),
                        self._convert_to_tensor([eos_token]),
                    ]
                )
                if self.include_values:
                    record["value_indicators"] = torch.concat(
                        [
                            self._convert_to_tensor(record["value_indicators"]),
                            self._convert_to_tensor([False]),
                        ]
                    ).to(torch.bool)
                    record["values"] = torch.concat(
                        [
                            self._convert_to_tensor(record["values"]),
                            self._convert_to_tensor(
                                [self.tokenizer.pad_value_token_id]
                            ),
                        ]
                    )
                if self.include_ttv_prediction:
                    record["time_to_visits"] = torch.concat(
                        [
                            record["time_to_visits"],
                            self._convert_to_tensor([-100.0]),
                        ]
                    )
            return record


class SamplePackingCehrGptDataCollator(CehrGptDataCollator):
    def __init__(self, max_tokens, max_position_embeddings, *args, **kwargs):
        self.max_tokens_per_batch = max_tokens
        self.max_position_embeddings = max_position_embeddings
        self.sample_packing = True
        self.add_end_token_in_sample_packing = kwargs.pop(
            "add_end_token_in_sample_packing", False
        )
        super(SamplePackingCehrGptDataCollator, self).__init__(*args, **kwargs)

    def __call__(self, examples):
        current_input_ids = []
        current_attention_mask = []
        current_position_ids = []
        current_value_indicators = []
        current_values = []

        # Demographics
        current_person_ids = []
        current_index_dates = []

        # Binary classification inputs
        current_ages = []
        current_labels = []

        for idx, example in enumerate(examples):

            # We only add an end token if the patient sequence could fit in the entire context window
            add_end_token = (
                len(example["input_ids"]) <= self.max_position_embeddings
                and self.add_end_token_in_sample_packing
            )
            # If the sample length exceeds the model's capacity, truncate this example
            if len(example["input_ids"]) > self.max_position_embeddings:
                example = self.generate_start_end_index(
                    example, False, self.max_position_embeddings
                )

            add_eos_token = add_end_token | self.add_linear_prob_token
            additional_tokens = []
            if add_end_token:
                additional_tokens.append(self.tokenizer.end_token_id)
            elif self.add_linear_prob_token:
                # Backward compatible
                linear_prob_token_id = (
                    self.tokenizer.linear_token_id
                    if self.tokenizer.linear_token_id is not None
                    else self.tokenizer.oov_token_id
                )
                additional_tokens.append(linear_prob_token_id)
            additional_tokens.append(self.tokenizer.pad_token_id)
            input_ids = example["input_ids"]
            # We add [END] [PAD], we want to attend to [END], adding [END] is important for sequence generation.
            # If the sequence length of the sequence is less than the context window, we add both [END][PAD], otherwise
            # we only add [PAD] token to the end of the sequence because it's not finished
            current_input_ids.extend(list(input_ids) + additional_tokens)
            current_attention_mask.extend(
                np.ones_like(input_ids).tolist() + ([1, 0] if add_eos_token else [0])
            )
            num_tokens_to_pad = 1 + int(add_eos_token)
            current_position_ids.extend(
                np.clip(
                    list(range(len(input_ids) + num_tokens_to_pad)),
                    0,
                    self.max_position_embeddings - 1,
                )
            )
            if self.include_values:
                current_value_indicators.extend(
                    list(example["value_indicators"]) + [False] * num_tokens_to_pad
                )
                current_values.extend(
                    list(example["values"])
                    + [self.tokenizer.pad_value_token_id] * num_tokens_to_pad
                )

            if "person_id" in example:
                current_person_ids.append(example["person_id"])

            if "index_date" in example:
                current_index_dates.append(example["index_date"])

            if "age_at_index" in example:
                current_ages.append(example["age_at_index"])

            if "classifier_label" in example:
                current_labels.append(example["classifier_label"])

        assert len(current_input_ids) <= self.max_tokens_per_batch, (
            f"The total number of tokens in the packed sequence should be less than {self.max_tokens_per_batch}\n"
            f"But the total number of tokens is: {len(current_input_ids)}"
        )
        packed_example = {
            "input_ids": current_input_ids,
            "attention_mask": current_attention_mask,
            "position_ids": current_position_ids,
        }
        if self.include_values:
            packed_example.update({"value_indicators": current_value_indicators})
            packed_example.update({"values": current_values})

        if current_labels:
            packed_example.update(
                {
                    "person_id": current_person_ids,
                    "index_date": current_index_dates,
                    "age_at_index": current_ages,
                    "classifier_label": current_labels,
                }
            )

        return super().__call__([packed_example])
