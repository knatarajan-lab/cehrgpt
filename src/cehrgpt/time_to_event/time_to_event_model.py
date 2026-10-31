from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
        self.store_generated_trajectories = store_generated_trajectories
        self.enable_early_stopping = enable_early_stopping
        self.max_sequence = model.config.n_positions

    def is_outcome_event(self, token: str) -> bool:
        return token in self.outcome_events

    def prepare_prompt(self, partial_history: Union[np.ndarray, List[str]]) -> List[str]:
        """Fit a history to the encoded context while preserving demographics."""
        history = list(partial_history)
        max_prompt_tokens = (
            self.generation_config.max_length
            - self.generation_config.max_new_tokens
        )
        if len(self.tokenizer.encode(history)) <= max_prompt_tokens:
            return history

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
            else:
                prompt = demographics + prompt[next_visit:]
        return prompt

    def simulate(
        self,
        partial_history: Union[np.ndarray, List[str]],
        future_visit_start: int = 0,
        future_visit_end: int = -1,
        prediction_window_start: int = 0,
        prediction_window_end: int = 365,
    ) -> List[List[str]]:
        partial_history = self.prepare_prompt(partial_history)
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)

        simulated_sequences = []
        old_num_return_sequences = self.generation_config.num_return_sequences
        with torch.inference_mode():
            while len(simulated_sequences) < old_num_return_sequences:
                self.generation_config.num_return_sequences = min(
                    self.batch_size,
                    old_num_return_sequences - len(simulated_sequences),
                )
                generation_kwargs = {}
                if self.enable_early_stopping:
                    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [
                            TimeToEventStoppingCriteria(
                                tokenizer=self.tokenizer,
                                outcome_events=self.outcome_events,
                                future_visit_start=future_visit_start,
                                future_visit_end=future_visit_end,
                                prediction_window_start=prediction_window_start,
                                prediction_window_end=prediction_window_end,
                            )
                        ]
                    )
                results = self.model.generate(
                    inputs=prompt,
                    generation_config=self.generation_config,
                    cehrgpt_tokenizer=self.tokenizer,
                    **generation_kwargs,
                )
                # Add the sequences to the result array
                simulated_sequences.extend(
                    [
                        self.tokenizer.decode(seq[len(token_ids) :].cpu().numpy())
                        for seq in results.sequences
                    ]
                )

        self.generation_config.num_return_sequences = old_num_return_sequences
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
    ) -> Optional[TimeToEvent]:
        time_event_tuples = []
        seqs_failed_to_convert = []
        generated_trajectories = []
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
            simulated_seqs = self.simulate(
                partial_history,
                future_visit_start,
                future_visit_end,
                prediction_window_start,
                prediction_window_end,
            )
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
                time_event_tuples, len(time_event_tuples), generated_trajectories
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
