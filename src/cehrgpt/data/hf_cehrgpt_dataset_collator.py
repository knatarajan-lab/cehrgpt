from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.utils import logging

from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    extract_time_interval_in_hours,
    is_att_token,
    is_clinical_event,
    is_inpatient_att_token,
    is_inpatient_hour_token,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

INPATIENT_STAY_DURATION_LIMIT = 30
LOG = logging.get_logger("transformers")


class CehrGptDataCollator:
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        max_length: int,
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

        self.include_values = include_values
        self.include_ttv_prediction = include_ttv_prediction
        self.use_sub_time_tokenization = use_sub_time_tokenization
        self.pretraining = pretraining
        self.include_demographics = include_demographics
        self.add_linear_prob_token = add_linear_prob_token
        self.motor_code_cache: Dict[str, List[str]] = dict()

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
        self.motor_time_bins = (
            self.tokenizer.get_motor_time_bins(motor_num_time_pieces)
            if self.include_motor_time_to_event
            else []
        )

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

        self._precompute_motor_token_mappings()

    def _precompute_motor_token_mappings(self):
        if self.include_motor_time_to_event:
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
                        self.time_intervals_array[i] = extract_time_interval_in_days(
                            token
                        )
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

    def __call__(self, examples):
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

        batch_position_ids = [
            self._try_reverse_tensor(self._convert_to_tensor(example["position_ids"]))
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
            motor_tte_times = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_tte_times"])
                )
                for example in examples_with_motor_tte
            ]
            motor_tte_event_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_tte_event_indicators"])
                )
                for example in examples_with_motor_tte
            ]
            motor_tte_task_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_tte_task_indicators"])
                )
                for example in examples_with_motor_tte
            ]
            motor_tte_masks = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_tte_masks"])
                )
                for example in examples_with_motor_tte
            ]

            motor_tte_times = torch.concat(motor_tte_times, dim=0).to(torch.float32)

            # If every example in the batch only contains one visit, there would be no labels generated for MOTOR TTE
            # we only create the labels when any example has more than one visit
            if motor_tte_times.dim() <= 1:
                LOG.warning(
                    "There are no MOTOR TTE labels generated for this batch "
                    "because every example in this batch only contains one visit."
                )
            else:
                batch_size = len(examples)
                length, num_time_pieces, motor_tte_vocab_size = motor_tte_times.shape
                padded_length = batch_size - length % batch_size
                batch["motor_tte_times"] = (
                    torch.concat(
                        [
                            motor_tte_times,
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
                batch["motor_tte_event_indicators"] = (
                    torch.concat(
                        [
                            torch.concat(motor_tte_event_indicators, dim=0).to(
                                torch.bool
                            ),
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
                batch["motor_tte_task_indicators"] = pad_sequence(
                    motor_tte_task_indicators,
                    batch_first=True,
                    padding_value=False,
                ).to(torch.bool)

                # Motor time indicators that indicate whether there are neither clinical events nor censor events
                batch["motor_tte_masks"] = (
                    torch.concat(
                        [
                            torch.concat(motor_tte_masks, dim=0).to(torch.bool),
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

        time_vectors = []
        global_event_indicators = []

        motor_tte_event_indicators = []
        motor_tte_masks = []
        motor_tte_times = []

        motor_tte_label_offsets = record["motor_tte_label_offsets"]
        motor_censor_times = record["motor_censor_times"]

        for i, (start_index, end_index) in enumerate(
            zip(motor_tte_label_offsets, motor_tte_label_offsets[1:])
        ):
            censor_time = motor_censor_times[i]
            # This represents a pad between two samples
            if censor_time == -100:
                continue

            tte_tasks = record["motor_tte_tasks"][start_index:end_index]
            tte_times = record["motor_tte_times"][start_index:end_index]

            time_vector = np.full(
                self.tokenizer.motor_tte_vocab_size,
                fill_value=censor_time,
                dtype=np.int32,
            )
            event_indicator = np.zeros(
                self.tokenizer.motor_tte_vocab_size,
                dtype=np.int32,
            )

            time_vector[tte_tasks] = tte_times
            event_indicator[tte_tasks] = 1  # not censored (event occurred)

            time_vectors.append(time_vector)
            global_event_indicators.append(event_indicator)

        time_vectors = np.asarray(time_vectors, dtype=np.float32)
        global_event_indicators = np.asarray(global_event_indicators).astype(bool)
        n_tte_predictions = len(time_vectors)

        motor_tte_time = np.full(
            (
                self.motor_num_time_pieces,
                n_tte_predictions,
                self.tokenizer.motor_tte_vocab_size,
            ),
            fill_value=0.0,
            dtype=np.float32,
        )
        motor_tte_event_indicator = np.zeros(
            (
                self.motor_num_time_pieces,
                n_tte_predictions,
                self.tokenizer.motor_tte_vocab_size,
            ),
            dtype=bool,
        )
        motor_tte_mask = np.zeros(
            (
                self.motor_num_time_pieces,
                n_tte_predictions,
                self.tokenizer.motor_tte_vocab_size,
            ),
            dtype=bool,
        )

        if n_tte_predictions > 0:
            # Putting the event time and censor time into the corresponding time bins
            motor_time_bins = [
                float("inf") if time_bin == float("inf") else time_bin // 3600
                for time_bin in self.motor_time_bins
            ]
            for bin_num, (start, end) in enumerate(
                zip(motor_time_bins, motor_time_bins[1:])
            ):
                time_vectors = time_vectors // 3600
                time_in_bin = np.clip(time_vectors - start, 0, end - start)
                mask = time_in_bin != 0
                time_in_bin[mask] = np.log2(time_in_bin[mask])
                time_in_bin[~mask] = -torch.inf

                motor_tte_time[bin_num] = time_in_bin
                event_indicator = (
                    global_event_indicators
                    & (start <= time_vectors)
                    & (time_vectors < end)
                )
                motor_tte_event_indicator[bin_num] = event_indicator
                motor_tte_mask[bin_num] = mask | event_indicator

        motor_tte_times.append(motor_tte_time.swapaxes(0, 1))
        motor_tte_event_indicators.append(motor_tte_event_indicator.swapaxes(0, 1))
        motor_tte_masks.append(motor_tte_mask.swapaxes(0, 1))
        record["motor_tte_times"] = np.concatenate(motor_tte_times, axis=0)
        record["motor_tte_event_indicators"] = np.concatenate(
            motor_tte_event_indicators, axis=0
        )
        record["motor_tte_masks"] = np.concatenate(motor_tte_masks, axis=0)
        assert (
            sum(record["motor_tte_task_indicators"]) == n_tte_predictions
        ), f'sum(record["motor_tte_task_indicators"]) == n_tte_predictions must be true'
        # Delete the additional inputs that are not required by the model
        del record["motor_tte_tasks"]
        del record["motor_censor_times"]
        del record["motor_tte_label_offsets"]
        return record


class SamplePackingCehrGptDataCollator(CehrGptDataCollator):
    def __init__(self, max_tokens, max_position_embeddings, *args, **kwargs):
        self.max_tokens_per_batch = max_tokens
        self.max_position_embeddings = max_position_embeddings
        self.sample_packing = True
        super(SamplePackingCehrGptDataCollator, self).__init__(*args, **kwargs)

    def __call__(self, examples):
        current_input_ids = []
        current_attention_mask = []
        current_position_ids = []
        current_value_indicators = []
        current_values = []

        # MOTOR inputs
        current_motor_censor_times = []
        current_motor_tte_tasks = []
        current_motor_tte_times = []
        current_motor_tte_label_offsets = []
        current_motor_tte_task_indicators = []

        # Demographics
        current_person_ids = []
        current_index_dates = []

        # Binary classification inputs
        current_prediction_ages = []
        current_labels = []

        for idx, example in enumerate(examples):
            input_ids = example["input_ids"]
            # We add [END] [PAD], we want to attend to [END], adding [END] is important for sequence generation.
            # If the sequence length of the sequence is less than the context window, we add both [END][PAD], otherwise
            # we only add [PAD] token to the end of the sequence because it's not finished
            current_input_ids.extend(list(input_ids) + [self.tokenizer.pad_token_id])
            current_attention_mask.extend(np.ones_like(input_ids).tolist() + [0])
            if "position_ids" in example:
                position_ids = (
                    example["position_ids"].tolist()
                    if isinstance(example["position_ids"], torch.Tensor)
                    else list(example["position_ids"])
                )
                current_position_ids.extend(position_ids + [max(position_ids)])
            else:
                current_position_ids.extend(
                    np.clip(
                        list(range(len(input_ids) + 1)),
                        0,
                        self.max_position_embeddings - 1,
                    )
                )

            if self.include_values:
                current_value_indicators.extend(
                    (
                        example["value_indicators"].tolist()
                        if isinstance(example["value_indicators"], torch.Tensor)
                        else list(example["value_indicators"])
                    )
                    + [False]
                )
                current_values.extend(
                    (
                        example["values"].tolist()
                        if isinstance(example["values"], torch.Tensor)
                        else list(example["values"])
                    )
                    + [self.tokenizer.pad_value_token_id]
                )

            if self.include_motor_time_to_event:
                existing_sample_length = len(current_motor_tte_times)
                current_motor_tte_tasks.extend(
                    example["motor_tte_tasks"].tolist()
                    if isinstance(example["motor_tte_tasks"], torch.Tensor)
                    else list(example["motor_tte_tasks"])
                )
                current_motor_tte_times.extend(
                    example["motor_tte_times"].tolist()
                    if isinstance(example["motor_tte_times"], torch.Tensor)
                    else list(example["motor_tte_times"])
                )
                current_motor_censor_times.extend(
                    example["motor_censor_times"].tolist()
                    if isinstance(example["motor_censor_times"], torch.Tensor)
                    else list(example["motor_censor_times"])
                )
                motor_tte_label_offsets = (
                    example["motor_tte_label_offsets"].tolist()
                    if isinstance(example["motor_tte_label_offsets"], torch.Tensor)
                    else list(example["motor_tte_label_offsets"])
                )
                current_motor_tte_label_offsets.extend(
                    list(
                        map(
                            lambda offset: offset + existing_sample_length,
                            motor_tte_label_offsets,
                        )
                    )
                )
                current_motor_tte_task_indicators.extend(
                    (
                        example["motor_tte_task_indicators"].tolist()
                        if isinstance(
                            example["motor_tte_task_indicators"], torch.Tensor
                        )
                        else list(example["motor_tte_task_indicators"])
                    )
                    + [False]
                )

            if "person_id" in example:
                current_person_ids.append(example["person_id"])

            if "index_date" in example:
                current_index_dates.append(example["index_date"])

            if "age_at_index" in example:
                current_prediction_ages.append(example["age_at_index"])

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
            packed_example.update(
                {"value_indicators": current_value_indicators, "values": current_values}
            )
        if self.include_motor_time_to_event:
            packed_example.update(
                {
                    "motor_censor_times": current_motor_censor_times,
                    "motor_tte_times": current_motor_tte_times,
                    "motor_tte_tasks": current_motor_tte_tasks,
                    "motor_tte_label_offsets": current_motor_tte_label_offsets,
                    "motor_tte_task_indicators": current_motor_tte_task_indicators,
                }
            )

        if current_labels:
            packed_example.update(
                {
                    "person_id": current_person_ids,
                    "index_date": current_index_dates,
                    "age_at_index": current_prediction_ages,
                    "classifier_label": current_labels,
                }
            )

        return super().__call__([packed_example])
