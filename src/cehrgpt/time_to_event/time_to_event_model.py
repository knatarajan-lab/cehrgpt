import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cehrbert_data.decorators.patient_event_decorator_base import time_month_token
from transformers import GenerationConfig

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


def create_time_to_event(
    time_event_tuples: List[Tuple[str, int]], num_of_simulations: int
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
    )


class TimeToEventModel:
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        model: CEHRGPT2LMHeadModel,
        outcome_events: List[str],
        generation_config: GenerationConfig,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
    ):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.generation_config = generation_config
        self.outcome_events = outcome_events
        self.device = device
        self.batch_size = batch_size
        self.max_sequence = model.config.n_positions

    def is_outcome_event(self, token: str):
        return token in self.outcome_events

    def simulate(
        self,
        partial_history: Union[np.ndarray, List[str]],
    ) -> List[List[str]]:
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)

        simulated_sequences = []
        num_iters = max(
            math.ceil(self.generation_config.num_return_sequences / self.batch_size), 1
        )
        old_num_return_sequences = self.generation_config.num_return_sequences
        self.generation_config.num_return_sequences = min(
            self.batch_size, old_num_return_sequences
        )
        with torch.no_grad():
            for _ in range(num_iters):
                results = self.model.generate(
                    inputs=prompt,
                    generation_config=self.generation_config,
                )
                # Clear the cache
                torch.cuda.empty_cache()
                # Add the sequences to the result array
                simulated_sequences.extend(
                    [
                        self.tokenizer.decode(seq.cpu().numpy())
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
        patient_history_length = len(partial_history)
        time_event_tuples = []
        seqs_failed_to_convert = []
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
            simulated_seqs = self.simulate(partial_history)
            n_trial += 1
            for seq in simulated_seqs:
                visit_counter = 0
                time_delta = 0
                success = False
                for next_token in seq[patient_history_length:]:
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
                    ) and self.is_outcome_event(next_token):
                        time_event_tuples.append((next_token, time_delta))
                        success = True
                        break
                if not success:
                    # This indicates the generated sequence did not satisfy the criteria
                    if future_visit_end != -1 or prediction_window_end != -1:
                        seqs_failed_to_convert.append(seq[patient_history_length:])
                    else:
                        time_event_tuples.append(("0", time_delta))

        self.generation_config.num_return_sequences = num_return_sequences
        self.generation_config.max_new_tokens = max_new_tokens

        if debug:
            print(f"seqs_failed_to_convert: {seqs_failed_to_convert}")

        # Count the occurrences of each time tokens for each concept
        return (
            create_time_to_event(time_event_tuples, len(time_event_tuples))
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
