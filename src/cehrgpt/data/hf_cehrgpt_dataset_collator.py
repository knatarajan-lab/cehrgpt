from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.utils import logging

from cehrgpt.data.cehrgpt_data_processor import CehrGptDataProcessor
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

LOG = logging.get_logger("transformers")


class CehrGptDataCollator:
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        max_length: int,
        include_values: bool = False,
        shuffle_records: bool = False,
        include_ttv_prediction: bool = False,
        use_sub_time_tokenization: bool = False,
        include_motor_time_to_event: bool = False,
        motor_tte_vocab_size: int = 0,
        motor_num_time_pieces: int = 8,
        motor_sampling_probability: float = 0.5,
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
        # Convert the time bins to seconds
        self.motor_time_bins = [time_bin * 86400 for time_bin in self.motor_time_bins]
        LOG.info("self.motor_time_bins: %s", self.motor_time_bins)
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

        self.cehrgpt_data_processor = CehrGptDataProcessor(
            tokenizer=tokenizer,
            max_length=self.max_length,
            shuffle_records=shuffle_records,
            include_ttv_prediction=include_ttv_prediction,
            include_values=include_values,
            include_motor_time_to_event=include_motor_time_to_event,
            motor_sampling_probability=motor_sampling_probability,
            pretraining=pretraining,
            add_linear_prob_token=add_linear_prob_token,
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

    def create_time_to_event_tensors_ultra_optimized(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ultra-optimized version using advanced vectorization techniques."""
        motor_row_indices = record["motor_row_indices"]
        motor_col_indices = record["motor_col_indices"]
        motor_values = record["motor_values"]
        motor_censor_times = record["motor_censor_times"]

        if len(motor_row_indices) == 0:
            # Handle empty case - use tuples for better performance
            empty_shape = (
                0,
                self.motor_num_time_pieces,
                self.tokenizer.motor_tte_vocab_size,
            )
            record["motor_tte_times"] = np.zeros(empty_shape, dtype=np.float32)
            record["motor_tte_event_indicators"] = np.zeros(empty_shape, dtype=bool)
            record["motor_tte_masks"] = np.zeros(empty_shape, dtype=bool)
            return record

        # Convert to numpy arrays once and get dimensions
        motor_row_indices = np.asarray(motor_row_indices, dtype=np.int32)
        motor_col_indices = np.asarray(motor_col_indices, dtype=np.int32)
        motor_values = np.asarray(motor_values, dtype=np.float32)
        motor_censor_times = np.asarray(motor_censor_times, dtype=np.float32)

        n_tte_predictions = len(motor_censor_times)  # More direct than unique()
        vocab_size = self.tokenizer.motor_tte_vocab_size
        n_time_pieces = self.motor_num_time_pieces

        # Create time_vectors more efficiently without broadcasting copy
        time_vectors = np.tile(
            motor_censor_times[:, np.newaxis], (1, vocab_size)
        ).astype(np.float32)
        event_indicators = np.zeros((n_tte_predictions, vocab_size), dtype=bool)

        # Vectorized assignment (already optimal)
        time_vectors[motor_row_indices, motor_col_indices] = motor_values
        event_indicators[motor_row_indices, motor_col_indices] = True

        # Early return if no predictions
        if n_tte_predictions == 0:
            empty_shape = (0, n_time_pieces, vocab_size)
            record["motor_tte_times"] = np.zeros(empty_shape, dtype=np.float32)
            record["motor_tte_event_indicators"] = np.zeros(empty_shape, dtype=bool)
            record["motor_tte_masks"] = np.zeros(empty_shape, dtype=bool)
            return record

        # Cache motor_time_bins as numpy array to avoid repeated conversion
        if not hasattr(self, "_motor_time_bins_array"):
            self._motor_time_bins_array = np.asarray(
                self.motor_time_bins, dtype=np.float32
            )

        motor_time_bins = self._motor_time_bins_array
        start_times = motor_time_bins[:-1]
        end_times = motor_time_bins[1:]
        bin_widths = end_times - start_times  # Pre-compute bin widths

        # ELIMINATED TRANSPOSE: Compute directly in target shape (n_pred, n_bins, vocab)
        # Reshape for broadcasting in target order
        time_vectors_3d = time_vectors[:, np.newaxis, :]  # (n_pred, 1, vocab)
        event_indicators_3d = event_indicators[:, np.newaxis, :]  # (n_pred, 1, vocab)

        # Broadcast time bins to match target shape
        start_times_broadcast = start_times[np.newaxis, :, np.newaxis]  # (1, n_bins, 1)
        bin_widths_broadcast = bin_widths[np.newaxis, :, np.newaxis]  # (1, n_bins, 1)

        # Compute directly in target shape (n_pred, n_bins, vocab)
        time_diff = time_vectors_3d - start_times_broadcast
        time_in_bin = np.clip(time_diff, 0, bin_widths_broadcast)

        # Optimized mask computation
        mask = time_in_bin > 0

        # More efficient log computation with better constant
        log_constant = 1e-8  # Better numerical stability than 1e-10
        time_in_bin_log = np.where(
            mask, np.log2(np.maximum(time_in_bin, log_constant)), -np.inf
        )

        # Event indicator computation in target shape
        end_times_broadcast = motor_time_bins[1:][np.newaxis, :, np.newaxis]
        time_in_range = (time_vectors_3d >= start_times_broadcast) & (
            time_vectors_3d < end_times_broadcast
        )
        event_in_bin = event_indicators_3d & time_in_range

        # Combined mask computation
        final_mask = mask | event_in_bin

        # Direct assignment - NO TRANSPOSE NEEDED!
        record["motor_tte_times"] = time_in_bin_log
        record["motor_tte_event_indicators"] = event_in_bin
        record["motor_tte_masks"] = final_mask

        # Validation (keep as is - important for correctness)
        assert (
            sum(record["motor_tte_task_indicators"]) == n_tte_predictions
        ), f'sum(record["motor_tte_task_indicators"]) == n_tte_predictions must be true'

        # Clean up input data
        del record["motor_row_indices"]
        del record["motor_col_indices"]
        del record["motor_values"]

        return record

    def __call__(self, examples):

        if not getattr(self, "sample_packing", False):
            examples = [self.cehrgpt_data_processor.transform(_) for _ in examples]

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

        batch_ages = [
            self._try_reverse_tensor(self._convert_to_tensor(example["ages"]))
            for example in examples
        ]
        # Pad sequences to the max length in the batch
        batch["ages"] = self._try_reverse_tensor(
            pad_sequence(
                batch_ages,
                batch_first=True,
                padding_value=0,
            ).to(torch.int64)
        )

        batch_epoch_times = [
            self._try_reverse_tensor(self._convert_to_tensor(example["epoch_times"]))
            for example in examples
        ]
        # Pad sequences to the max length in the batch
        batch["epoch_times"] = self._try_reverse_tensor(
            pad_sequence(
                batch_epoch_times,
                batch_first=True,
                padding_value=0,
            ).to(torch.float32)
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
                self.create_time_to_event_tensors_ultra_optimized(_) for _ in examples
            ]
            # print(f"Creating MOTOR TTE tensors took {time.time() - start} seconds")
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


class SamplePackingCehrGptDataCollator(CehrGptDataCollator):
    def __init__(self, max_tokens, max_position_embeddings, *args, **kwargs):
        self.max_tokens_per_batch = max_tokens
        self.max_position_embeddings = max_position_embeddings
        self.sample_packing = True
        super(SamplePackingCehrGptDataCollator, self).__init__(*args, **kwargs)
        self.cehrgpt_data_processor.max_length = self.max_position_embeddings

    def __call__(self, examples):
        current_input_ids = []
        current_attention_mask = []
        current_ages = []
        current_epoch_times = []
        current_value_indicators = []
        current_values = []
        current_time_to_visits = []

        # MOTOR inputs
        current_motor_censor_times = []
        current_motor_row_indices = []
        current_motor_col_indices = []
        current_motor_values = []
        current_motor_tte_task_indicators = []

        # Demographics
        current_person_ids = []
        current_index_dates = []

        # Binary classification inputs
        current_prediction_ages = []
        current_labels = []

        for idx, example in enumerate(examples):
            example = self.cehrgpt_data_processor.transform(example)
            input_ids = example["input_ids"]
            # We add [END] [PAD], we want to attend to [END], adding [END] is important for sequence generation.
            # If the sequence length of the sequence is less than the context window, we add both [END][PAD], otherwise
            # we only add [PAD] token to the end of the sequence because it's not finished
            current_input_ids.extend(list(input_ids) + [self.tokenizer.pad_token_id])
            current_attention_mask.extend(np.ones_like(input_ids).tolist() + [0])

            ages = (
                example["ages"].tolist()
                if isinstance(example["ages"], torch.Tensor)
                else list(example["ages"])
            )
            current_ages.extend(ages + [max(ages)])

            epoch_times = (
                example["epoch_times"].tolist()
                if isinstance(example["epoch_times"], torch.Tensor)
                else list(example["epoch_times"])
            )
            current_epoch_times.extend(epoch_times + [max(epoch_times)])

            if self.include_ttv_prediction:
                current_time_to_visits.extend(
                    (
                        example["time_to_visits"].tolist()
                        if isinstance(example["time_to_visits"], torch.Tensor)
                        else list(example["time_to_visits"])
                    )
                    + [-100]
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
                current_max_motor_row_index = len(np.unique(current_motor_row_indices))
                motor_row_indices = (
                    example["motor_row_indices"].tolist()
                    if isinstance(example["motor_row_indices"], torch.Tensor)
                    else list(example["motor_row_indices"])
                )
                current_motor_row_indices.extend(
                    list(
                        map(
                            lambda offset: offset + current_max_motor_row_index,
                            motor_row_indices,
                        )
                    )
                )
                current_motor_col_indices.extend(
                    example["motor_col_indices"].tolist()
                    if isinstance(example["motor_col_indices"], torch.Tensor)
                    else list(example["motor_col_indices"])
                )
                current_motor_values.extend(
                    example["motor_values"].tolist()
                    if isinstance(example["motor_values"], torch.Tensor)
                    else list(example["motor_values"])
                )
                current_motor_censor_times.extend(
                    example["motor_censor_times"].tolist()
                    if isinstance(example["motor_censor_times"], torch.Tensor)
                    else list(example["motor_censor_times"])
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
            "ages": current_ages,
            "epoch_times": current_epoch_times,
        }

        if self.include_ttv_prediction:
            packed_example.update({"time_to_visits": current_time_to_visits})

        if self.include_values:
            packed_example.update(
                {"value_indicators": current_value_indicators, "values": current_values}
            )
        if self.include_motor_time_to_event:
            packed_example.update(
                {
                    "motor_censor_times": current_motor_censor_times,
                    "motor_row_indices": current_motor_row_indices,
                    "motor_col_indices": current_motor_col_indices,
                    "motor_values": current_motor_values,
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
        # print(f"Packing examples took {time.time() - start} seconds")
        return super().__call__([packed_example])
