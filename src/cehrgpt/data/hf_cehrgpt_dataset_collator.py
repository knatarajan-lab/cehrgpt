from typing import Any

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
        motor_sampling_probability: float = 0.5,
        is_data_in_meds: bool = False,
        pretraining: bool = True,
        include_demographics: bool = False,
        add_linear_prob_token: bool = False,
        include_age_at_ve_prediction: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.vs_token_id = tokenizer.vs_token_id
        self.ve_token_id = tokenizer.ve_token_id

        self.include_age_at_ve_prediction = include_age_at_ve_prediction
        if include_age_at_ve_prediction:
            # Precompute valid integer age → vocabulary token id mapping.
            # Only ages whose 'age:<N>' token exists in the vocab are kept.
            self._age_to_token_id = {
                age: tid
                for age in tokenizer.valid_age_values
                if (tid := tokenizer.get_age_token_id(age)) is not None
            }
        else:
            self._age_to_token_id = {}

        self.include_values = include_values
        self.include_ttv_prediction = include_ttv_prediction
        self.use_sub_time_tokenization = use_sub_time_tokenization
        self.pretraining = pretraining
        self.include_demographics = include_demographics
        self.include_motor_time_to_event = include_motor_time_to_event
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
            is_data_in_meds=is_data_in_meds,
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

        if batch["input_ids"].shape[1] > self.max_length:
            LOG.warning(
                "batch[input_ids].shape[1] > self.max_length. batch[input_ids].shape[1]: %s, self.max_length: %s",
                batch["input_ids"].shape[1],
                self.max_length,
            )
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

        if self.include_age_at_ve_prediction and self._age_to_token_id:
            vs_mask = batch["input_ids"] == self.vs_token_id
            # Default all positions to -100 (ignored in loss)
            age_reconstruction_labels = torch.full_like(batch["input_ids"], -100)
            # At each VS position, if the integer age has a valid 'age:<N>' token,
            # set the label to the integer age value.
            ages_int = batch["ages"].to(torch.int64)
            for age_val, _tok_id in self._age_to_token_id.items():
                age_reconstruction_labels = torch.where(
                    vs_mask & (ages_int == age_val),
                    age_val,
                    age_reconstruction_labels,
                )
            batch["age_reconstruction_labels"] = age_reconstruction_labels

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

            motor_row_indices = [example["motor_row_indices"] for example in examples]
            row_index_pads = [0] + [max_row_index + 1 for max_row_index in map(max, motor_row_indices)][:-1]
            row_index_pads = np.cumsum(row_index_pads)
            for i, row_index_pad in enumerate(row_index_pads):
                motor_row_indices[i] = np.asarray(motor_row_indices[i]) + row_index_pad

            motor_row_indices = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(_)
                )
                for _ in motor_row_indices
            ]
            motor_col_indices = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_col_indices"])
                )
                for example in examples
            ]
            motor_values = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_values"])
                )
                for example in examples
            ]
            motor_tte_task_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_tte_task_indicators"])
                )
                for example in examples
            ]
            motor_censor_times = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["motor_censor_times"])
                )
                for example in examples
            ]

            motor_row_indices = torch.concat(motor_row_indices, dim=0)

            # If every example in the batch only contains one visit, there would be no labels generated for MOTOR TTE
            # we only create the labels when any example has more than one visit
            if motor_row_indices.shape[0] <= 1:
                LOG.warning(
                    "There are no MOTOR TTE labels generated for this batch "
                    "because every example in this batch only contains one visit."
                )
            else:
                batch_size = len(examples)
                length = motor_row_indices.shape[0]
                padded_length = batch_size - length % batch_size

                batch["motor_row_indices"] = (
                    torch.concat(
                        [
                            motor_row_indices,
                            torch.full(
                                (padded_length,),
                                0,
                            ),
                        ],
                        dim=0,
                    )
                    .reshape((batch_size, -1))
                    .to(torch.int32)
                )
                batch["motor_col_indices"] = (
                    torch.concat(
                        [
                            torch.concat(motor_col_indices, dim=0),
                            torch.full(
                                (padded_length,),
                                0,
                            ),
                        ],
                        dim=0,
                    )
                    .reshape((batch_size, -1))
                    .to(torch.int32)
                )
                batch["motor_values"] = (
                    torch.concat(
                        [
                            torch.concat(motor_values, dim=0),
                            torch.full(
                                (padded_length,),
                                0.0,
                            ),
                        ],
                        dim=0,
                    )
                    .reshape((batch_size, -1))
                    .to(torch.float32)
                )
                # Input to indicate whether the visit should be included for TTE predictions
                batch["motor_censor_times"] = pad_sequence(
                    motor_censor_times,
                    batch_first=True,
                    padding_value=0.0,
                ).to(torch.float32)
                # Input to indicate whether the visit should be included for TTE predictions
                batch["motor_tte_task_indicators"] = pad_sequence(
                    motor_tte_task_indicators,
                    batch_first=True,
                    padding_value=False,
                ).to(torch.bool)
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

            if batch["value_indicators"].shape[1] > self.max_length:
                LOG.warning(
                    "The total number of values in the packed sequence should be less than %s. "
                    "But the total number of tokens is: %s",
                    self.max_length,
                    batch["value_indicators"].shape[1],
                )

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
            current_ages.extend(ages + [max(ages)] if len(ages) > 0 else [])

            epoch_times = (
                example["epoch_times"].tolist()
                if isinstance(example["epoch_times"], torch.Tensor)
                else list(example["epoch_times"])
            )
            current_epoch_times.extend(
                epoch_times + [max(epoch_times)] if len(epoch_times) > 0 else []
            )

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
                    (
                        example["motor_censor_times"].tolist()
                        if isinstance(example["motor_censor_times"], torch.Tensor)
                        else list(example["motor_censor_times"])
                    )
                    + [0]
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

        if len(current_input_ids) > self.max_tokens_per_batch:
            LOG.warning(
                "The total number of tokens in the packed sequence should be less than %s. "
                "But the total number of tokens is: %s",
                self.max_tokens_per_batch,
                len(current_input_ids),
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
        return super().__call__([packed_example])
