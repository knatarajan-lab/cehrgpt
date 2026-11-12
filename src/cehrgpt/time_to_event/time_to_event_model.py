from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from cehrbert_data.decorators.patient_event_decorator_base import time_month_token
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


@dataclass
class TimeToEvent:
    average_time: float
    median_time: float
    standard_deviation: float
    most_likely_time: str
    num_of_simulations: int
    time_intervals: List[int]
    outcome_events: List[str]
    time_interval_probability_table: List[Dict[str, Any]]
    # The raw generated continuation for every simulated trial (not just the ones that
    # satisfied the stopping criteria), kept for manual investigation of model behavior.
    generated_trajectories: List[List[str]] = field(default_factory=list)
    # The decoded value of every token of generated_trajectories, when the model includes values
    generated_value_trajectories: List[List[str]] = field(default_factory=list)


class TimeToEventStoppingCriteria(StoppingCriteria):
    """Stop each sampled trajectory once its task outcome is determined."""

    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        outcome_events: List[str],
        future_visit_start: int,
        future_visit_end: int,
        prediction_window_start: int,
        prediction_window_end: int,
    ):
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values(), default=-1) + 1
        self._time_days = torch.zeros(vocab_size, dtype=torch.float32)
        self._is_time = torch.zeros(vocab_size, dtype=torch.bool)
        self._is_visit_start = torch.zeros(vocab_size, dtype=torch.bool)
        self._is_outcome = torch.zeros(vocab_size, dtype=torch.bool)
        for token, token_id in vocab.items():
            if is_att_token(token):
                self._is_time[token_id] = True
                self._time_days[token_id] = extract_time_interval_in_days(token)
            if is_visit_start(token):
                self._is_visit_start[token_id] = True
        outcome_ids = [vocab[event] for event in outcome_events if event in vocab]
        if outcome_ids:
            self._is_outcome[outcome_ids] = True
        self.future_visit_start = future_visit_start
        self.future_visit_end = future_visit_end
        self.prediction_window_start = prediction_window_start
        self.prediction_window_end = prediction_window_end
        self._device = None
        self._time_delta = None
        self._visit_count = None
        self._finished = None

    def reset(self) -> None:
        """
        Forget the trajectories of the previous generation, keeping the vocabulary tables.

        The tables take a pass over the whole vocabulary to build, so one criteria is built per
        task window and reset before every generation rather than rebuilt.
        """
        self._time_delta = None
        self._visit_count = None
        self._finished = None

    def _move_to(self, device: torch.device, batch_size: int) -> None:
        if self._device != device:
            self._time_days = self._time_days.to(device)
            self._is_time = self._is_time.to(device)
            self._is_visit_start = self._is_visit_start.to(device)
            self._is_outcome = self._is_outcome.to(device)
            self._device = device
        self._time_delta = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self._visit_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        if self._finished is None or self._finished.shape[0] != batch_size:
            self._move_to(input_ids.device, batch_size)

        next_tokens = input_ids[:, -1]
        self._visit_count.add_(self._is_visit_start[next_tokens].long())
        crossed_visit_limit = (
            self._visit_count > self.future_visit_end
            if self.future_visit_end != -1
            else torch.zeros_like(self._finished)
        )
        crossed_time_limit = (
            self._time_delta > self.prediction_window_end
            if self.prediction_window_end != -1
            else torch.zeros_like(self._finished)
        )
        eligible_outcome = (
            (self._visit_count >= self.future_visit_start)
            & (self._time_delta >= self.prediction_window_start)
            & ~self._is_time[next_tokens]
            & self._is_outcome[next_tokens]
        )
        self._finished |= crossed_visit_limit | crossed_time_limit | eligible_outcome
        self._time_delta.add_(self._time_days[next_tokens])
        return self._finished


def create_time_to_event(
    time_event_tuples: List[Tuple[str, int]],
    num_of_simulations: int,
    generated_trajectories: Optional[List[List[str]]] = None,
    generated_value_trajectories: Optional[List[List[str]]] = None,
) -> TimeToEvent:
    outcome_events, time_intervals = zip(*time_event_tuples)
    time_buckets = [time_month_token(_) for _ in time_intervals]
    time_bucket_counter = Counter(time_buckets)
    most_common_item = time_bucket_counter.most_common(1)[0][0]
    total_count = sum(time_bucket_counter.values())
    # Generate the probability table
    probability_table = {
        item: count / total_count for item, count in time_bucket_counter.items()
    }
    sorted_probability_table = [
        {"time_interval": k, "probability": v}
        for k, v in sorted(probability_table.items(), key=lambda x: x[1], reverse=True)
    ]
    return TimeToEvent(
        time_intervals=time_intervals,
        outcome_events=outcome_events,
        average_time=np.mean(time_intervals),
        median_time=np.median(time_intervals),
        standard_deviation=np.std(time_intervals),
        most_likely_time=most_common_item,
        num_of_simulations=num_of_simulations,
        time_interval_probability_table=sorted_probability_table,
        generated_trajectories=generated_trajectories or [],
        generated_value_trajectories=generated_value_trajectories or [],
    )


class TimeToEventModel:
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        model: CEHRGPT2LMHeadModel,
        outcome_events: List[Union[str, int]],
        generation_config: GenerationConfig,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        store_generated_trajectories: bool = False,
        enable_early_stopping: bool = True,
        max_input_length: Optional[int] = None,
        reuse_prompt_cache: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.generation_config = generation_config
        # Generated tokens are always decoded as strings, so outcome_events (which callers
        # may build from int-typed concept ids, e.g. to query concept_ancestor) needs to be
        # normalized to strings here so token matching in is_outcome_event actually works.
        self.outcome_events = [str(event) for event in outcome_events]
        self.device = device
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.store_generated_trajectories = store_generated_trajectories
        self.enable_early_stopping = enable_early_stopping
        self.max_sequence = model.config.n_positions
        # A patient's samples are generated in batches of batch_size, and again when a trial
        # leaves trajectories undetermined. Every one of those generations starts from the same
        # prompt, so the KV cache of the prompt is kept from the first to the next ones, which
        # then only run the model on the new tokens.
        self.reuse_prompt_cache = reuse_prompt_cache
        # (prompt key, KV cache of one row of the prompt without its last token)
        self._prompt_cache: Optional[Tuple[Any, Tuple[Tuple[torch.Tensor, ...], ...]]] = None
        # One stopping criteria per task window, see _get_stopping_criteria
        self._stopping_criteria: Dict[Tuple[int, int, int, int], TimeToEventStoppingCriteria] = {}

    def is_outcome_event(self, token: str) -> bool:
        return token in self.outcome_events

    def _get_stopping_criteria(
        self,
        future_visit_start: int,
        future_visit_end: int,
        prediction_window_start: int,
        prediction_window_end: int,
    ) -> TimeToEventStoppingCriteria:
        """The stopping criteria of a task window, built once and reset for every generation."""
        key = (
            future_visit_start,
            future_visit_end,
            prediction_window_start,
            prediction_window_end,
        )
        criteria = self._stopping_criteria.get(key)
        if criteria is None:
            criteria = TimeToEventStoppingCriteria(
                tokenizer=self.tokenizer,
                outcome_events=self.outcome_events,
                future_visit_start=future_visit_start,
                future_visit_end=future_visit_end,
                prediction_window_start=prediction_window_start,
                prediction_window_end=prediction_window_end,
            )
            self._stopping_criteria[key] = criteria
        criteria.reset()
        return criteria

    @property
    def include_values(self) -> bool:
        """Whether the model reads and generates a value bin for every lab token."""
        return bool(getattr(self.model.config, "include_values", False))

    def prepare_prompt(self, partial_history: Union[np.ndarray, List[str]]) -> List[str]:
        """Fit a history to the encoded context while preserving demographics."""
        return self.prepare_prompt_with_indices(partial_history)[0]

    def prepare_prompt_with_indices(
        self, partial_history: Union[np.ndarray, List[str]]
    ) -> Tuple[List[str], List[int]]:
        """
        Fit a history to the encoded context while preserving demographics.

        Also returns, for every token of the prompt, its index in partial_history, so per-token
        columns such as the values can follow the tokens that are kept.
        """
        history = list(partial_history)
        indices = list(range(len(history)))
        max_prompt_tokens = (
            self.generation_config.max_length
            - self.generation_config.max_new_tokens
        )
        if self.max_input_length is not None:
            if self.max_input_length <= 0:
                raise ValueError("max_input_length must be greater than zero")
            max_prompt_tokens = min(max_prompt_tokens, self.max_input_length)
        if len(self.tokenizer.encode(history)) <= max_prompt_tokens:
            return history, indices

        demographics = history[:4]
        if len(self.tokenizer.encode(demographics)) > max_prompt_tokens:
            raise ValueError("The demographic prompt does not fit in the model context")

        # Drop old visits, not the year/age/gender/race header. Verify encoded
        # length so tokenizer-added special tokens count toward the context.
        prompt = history
        while len(self.tokenizer.encode(prompt)) > max_prompt_tokens and len(prompt) > 4:
            next_visit = next(
                (
                    index
                    for index in range(5, len(prompt))
                    if is_visit_start(prompt[index])
                ),
                None,
            )
            if next_visit is None:
                remove_index = 5 if len(prompt) > 5 and is_visit_start(prompt[4]) else 4
                prompt.pop(remove_index)
                indices.pop(remove_index)
            else:
                prompt = demographics + prompt[next_visit:]
                indices = indices[:4] + indices[next_visit:]
        return prompt, indices

    def _can_reuse_prompt_cache(self, num_prompt_tokens: int) -> bool:
        config = self.model.config
        return (
            self.reuse_prompt_cache
            and bool(self.generation_config.use_cache)
            # The last token of the prompt is left out of the cache: the model has to read
            # it to predict the first new token
            and num_prompt_tokens > 1
            # Every sequence gets its own random vector, in the middle of the prompt
            and not getattr(config, "causal_sfm", False)
            # The visit-local window of a decoding step is not covered by the tests of the reuse
            and not getattr(config, "use_local_attention", False)
        )

    def _get_prompt_cache(self, prompt_key: Any, num_sequences: int):
        """The KV cache of the prompt for num_sequences sequences, None when there is none."""
        if self._prompt_cache is None or self._prompt_cache[0] != prompt_key:
            return None
        return tuple(
            tuple(state.expand(num_sequences, *state.shape[1:]) for state in layer_past)
            for layer_past in self._prompt_cache[1]
        )

    def _store_prompt_cache(
        self, prompt_key: Any, past_key_values: Any, num_prompt_tokens: int
    ) -> None:
        """
        Keep the prompt part of the KV cache a generation ended with.

        The sequences of a generation share the prompt, so the first row is kept for all of
        them. It is copied, so the cache of the generated tokens can be freed.
        """
        self._prompt_cache = None
        if not isinstance(past_key_values, tuple) or not past_key_values:
            return
        prompt_cache = []
        for layer_past in past_key_values:
            if not isinstance(layer_past, tuple) or any(
                not isinstance(state, torch.Tensor)
                or state.dim() < 3
                or state.shape[-2] < num_prompt_tokens
                for state in layer_past
            ):
                return
            prompt_cache.append(
                tuple(
                    state[:1, ..., : num_prompt_tokens - 1, :].clone()
                    for state in layer_past
                )
            )
        self._prompt_cache = (prompt_key, tuple(prompt_cache))

    def _decode_values(
        self,
        generated_ids: np.ndarray,
        tokens: List[str],
        generated_values: Optional[np.ndarray],
    ) -> List[str]:
        """The decoded value of every token of `tokens`, the decoded generated_ids."""
        if generated_values is None:
            return []
        # decode drops the special tokens such as the padding after a trajectory stopped, so
        # the values of those positions are dropped too to keep one value per token
        all_tokens = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        # Look the bins up one by one: decode_value joins them with spaces and splits them
        # again, which breaks on a categorical value with a space in it
        value_bins = self.__dict__.get("_value_bins")
        if value_bins is None:
            value_bins = {
                value_id: value_bin
                for value_bin, value_id in self.tokenizer.get_value_vocab().items()
            }
            self._value_bins = value_bins
        all_values = [value_bins.get(int(value_id)) for value_id in generated_values]
        skipped = set(all_tokens) - set(tokens)
        return [
            value
            for token, value in zip(all_tokens, all_values)
            if token not in skipped
        ]

    def simulate(
        self,
        partial_history: Union[np.ndarray, List[str]],
        future_visit_start: int = 0,
        future_visit_end: int = -1,
        prediction_window_start: int = 0,
        prediction_window_end: int = 365,
        value_indicators: Optional[Sequence[bool]] = None,
        values: Optional[Sequence[int]] = None,
        return_values: bool = False,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[List[str]]]]:
        """
        Generate continuations of partial_history.

        When the model includes values, value_indicators and values (the value token ids, as
        in training) give the values of the lab tokens of partial_history, which the model
        reads like in training; a prompt without them is read as if no lab had a value. With
        return_values, the decoded values generated with every continuation are also returned.
        """
        if self.include_values and (value_indicators is None) != (values is None):
            raise ValueError("value_indicators and values must be given together")
        partial_history, prompt_indices = self.prepare_prompt_with_indices(
            partial_history
        )
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)

        value_kwargs = {}
        if self.include_values and values is not None:
            if len(value_indicators) != len(values):
                raise ValueError("value_indicators and values must have the same length")
            if len(token_ids) != len(partial_history):
                raise ValueError(
                    "The values follow the tokens, so the tokenizer must not add tokens to the prompt"
                )
            value_kwargs = {
                "value_indicators": torch.tensor(
                    [bool(value_indicators[i]) for i in prompt_indices],
                    dtype=torch.bool,
                ).unsqueeze(0).to(self.device),
                "values": torch.tensor(
                    [int(values[i]) for i in prompt_indices], dtype=torch.long
                ).unsqueeze(0).to(self.device),
            }

        reuse_prompt_cache = self._can_reuse_prompt_cache(len(token_ids))
        prompt_key = (
            tuple(token_ids),
            tuple(value_kwargs["value_indicators"][0].tolist()) if value_kwargs else None,
            tuple(value_kwargs["values"][0].tolist()) if value_kwargs else None,
        )
        if not reuse_prompt_cache:
            self._prompt_cache = None

        simulated_sequences = []
        simulated_values = []
        old_num_return_sequences = self.generation_config.num_return_sequences
        with torch.inference_mode():
            while len(simulated_sequences) < old_num_return_sequences:
                self.generation_config.num_return_sequences = min(
                    self.batch_size,
                    old_num_return_sequences - len(simulated_sequences),
                )
                generation_kwargs = dict(value_kwargs)
                if self.enable_early_stopping:
                    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [
                            self._get_stopping_criteria(
                                future_visit_start,
                                future_visit_end,
                                prediction_window_start,
                                prediction_window_end,
                            )
                        ]
                    )
                prompt_cache = (
                    self._get_prompt_cache(
                        prompt_key, self.generation_config.num_return_sequences
                    )
                    if reuse_prompt_cache
                    else None
                )
                if prompt_cache is not None:
                    generation_kwargs["past_key_values"] = prompt_cache
                results = self.model.generate(
                    inputs=prompt,
                    generation_config=self.generation_config,
                    cehrgpt_tokenizer=self.tokenizer,
                    **generation_kwargs,
                )
                if reuse_prompt_cache and prompt_cache is None:
                    self._store_prompt_cache(
                        prompt_key,
                        getattr(results, "past_key_values", None),
                        len(token_ids),
                    )
                sequence_vals = getattr(results, "sequence_vals", None)
                for i, seq in enumerate(results.sequences):
                    generated_ids = seq[len(token_ids) :].cpu().numpy()
                    tokens = self.tokenizer.decode(generated_ids)
                    simulated_sequences.append(tokens)
                    if return_values:
                        simulated_values.append(
                            self._decode_values(
                                generated_ids,
                                tokens,
                                (
                                    sequence_vals[i, len(token_ids) :].cpu().numpy()
                                    if sequence_vals is not None
                                    else None
                                ),
                            )
                        )

        self.generation_config.num_return_sequences = old_num_return_sequences
        if return_values:
            return simulated_sequences, simulated_values
        return simulated_sequences

    def predict_time_to_events(
        self,
        partial_history: Union[np.ndarray, list],
        future_visit_start: int = 0,
        future_visit_end: int = -1,
        prediction_window_start: int = 0,
        prediction_window_end: int = 365,
        debug: bool = False,
        max_n_trial: int = 2,
        value_indicators: Optional[Sequence[bool]] = None,
        values: Optional[Sequence[int]] = None,
    ) -> Optional[TimeToEvent]:
        time_event_tuples = []
        seqs_failed_to_convert = []
        generated_trajectories = []
        generated_value_trajectories = []
        store_values = self.store_generated_trajectories and self.include_values
        n_trial = 0
        num_return_sequences = self.generation_config.num_return_sequences
        max_new_tokens = self.generation_config.max_new_tokens
        while (
            len(time_event_tuples) < self.generation_config.num_return_sequences
            and n_trial < max_n_trial
        ):
            self.generation_config.num_return_sequences = num_return_sequences - len(
                time_event_tuples
            )
            # self.generation_config.max_new_tokens = max_new_tokens * (n_trial + 1)
            simulated = self.simulate(
                partial_history,
                future_visit_start,
                future_visit_end,
                prediction_window_start,
                prediction_window_end,
                value_indicators=value_indicators,
                values=values,
                return_values=store_values,
            )
            if store_values:
                simulated_seqs, simulated_values = simulated
                generated_value_trajectories.extend(simulated_values)
            else:
                simulated_seqs = simulated
            n_trial += 1
            for seq in simulated_seqs:
                visit_counter = 0
                time_delta = 0
                success = False
                generated_trajectory = seq
                if self.store_generated_trajectories:
                    generated_trajectories.append(generated_trajectory)
                for next_token in generated_trajectory:
                    visit_counter += int(is_visit_start(next_token))
                    if (
                        visit_counter > future_visit_end != -1
                        or time_delta > prediction_window_end != -1
                    ):
                        time_event_tuples.append(("0", time_delta))
                        success = True
                        break
                    if is_att_token(next_token):
                        time_delta += extract_time_interval_in_days(next_token)
                    elif (
                        visit_counter >= future_visit_start
                        and time_delta >= prediction_window_start
                    ):
                        if self.is_outcome_event(next_token):
                            time_event_tuples.append((next_token, time_delta))
                            success = True
                            break
                if not success:
                    # This indicates the generated sequence did not satisfy the criteria
                    if future_visit_end != -1 or prediction_window_end != -1:
                        seqs_failed_to_convert.append(generated_trajectory)
                    else:
                        time_event_tuples.append(("0", time_delta))

        self.generation_config.num_return_sequences = num_return_sequences
        self.generation_config.max_new_tokens = max_new_tokens

        if debug:
            print(f"seqs_failed_to_convert: {seqs_failed_to_convert}")

        # Count the occurrences of each time tokens for each concept
        return (
            create_time_to_event(
                time_event_tuples,
                len(time_event_tuples),
                generated_trajectories,
                generated_value_trajectories,
            )
            if len(time_event_tuples) > 0
            else None
        )

    @staticmethod
    def get_generation_config(
        tokenizer: CehrGptTokenizer,
        max_length: int,
        num_return_sequences: int,
        top_p: float = 1.0,
        top_k: int = 300,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        epsilon_cutoff: float = 0.0,
        max_new_tokens: int = 128,
    ) -> GenerationConfig:
        return GenerationConfig(
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            epsilon_cutoff=epsilon_cutoff,
            top_p=top_p,
            top_k=top_k,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            renormalize_logits=True,
        )
