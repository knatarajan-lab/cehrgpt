import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from collections import defaultdict

import numpy as np
import torch
from transformers import GenerationConfig

from ..gpt_utils import is_visit_end, is_att_token, extract_time_interval_in_days
from ..models.hf_cehrgpt import CEHRGPT2LMHeadModel
from ..models.tokenization_hf_cehrgpt import CehrGptTokenizer

from cehrbert_data.decorators.patient_event_decorator import time_month_token


@dataclass
class TimeToEvent:
    average_time: float
    median_time: float
    standard_deviation: float
    most_likely_time: int
    num_of_simulations: int
    time_interval_probability_table: List[Dict[str, Any]]


@dataclass
class ConceptTimeToEvent:
    concept: str
    time_to_events: TimeToEvent


def create_time_to_event(time_intervals: List[int]) -> TimeToEvent:
    time_buckets = [time_month_token(_) for _ in time_intervals]
    time_bucket_counter = Counter(time_buckets)
    most_common_item = time_bucket_counter.most_common(1)[0][0]
    total_count = sum(time_bucket_counter.values())
    # Generate the probability table
    probability_table = {item: count / total_count for item, count in time_bucket_counter.items()}
    sorted_probability_table = [
        {"time_interval": k, "probability": v}
        for k, v in
        sorted(probability_table.items(), key=lambda x: x[1], reverse=True)
    ]
    return TimeToEvent(
        average_time=np.mean(time_intervals),
        median_time=np.median(time_intervals),
        standard_deviation=np.std(time_intervals),
        most_likely_time=most_common_item,
        num_of_simulations=len(time_intervals),
        time_interval_probability_table=sorted_probability_table
    )


class TimeToEventModel:
    def __init__(
            self,
            tokenizer: CehrGptTokenizer,
            model: CEHRGPT2LMHeadModel,
            outcome_events: List[str],
            generation_config: GenerationConfig,
            device: torch.device = torch.device("cpu"),
            batch_size: int = 32
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
            max_new_tokens: int = 0
    ) -> List[List[str]]:

        sequence_is_demographics = len(partial_history) == 4 and partial_history[0].startswith("year")
        sequence_ends_ve = is_visit_end(partial_history[-1])

        if not (sequence_is_demographics | sequence_ends_ve):
            raise ValueError(
                "There are only two types of sequences allowed. 1) the sequence only contains "
                "demographics; 2) the sequence ends on VE;"
            )

        seq_length = len(partial_history)
        old_max_new_tokens = self.generation_config.max_new_tokens

        # Set this to max sequence
        self.generation_config.max_new_tokens = (
            self.model.config.n_positions - seq_length if max_new_tokens == 0 else max_new_tokens
        )
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)

        simulated_sequences = []
        num_iters = max(
            math.ceil(self.generation_config.num_return_sequences / self.batch_size),
            1
        )
        old_num_return_sequences = self.generation_config.num_return_sequences
        self.generation_config.num_return_sequences = min(self.batch_size, old_num_return_sequences)
        with torch.no_grad():
            for _ in range(num_iters):
                results = self.model.generate(
                    inputs=prompt,
                    generation_config=self.generation_config,
                )
                # Clear the cache
                torch.cuda.empty_cache()
            simulated_sequences.extend([self.tokenizer.decode(seq.cpu().numpy()) for seq in results.sequences])

        self.generation_config.num_return_sequences = old_num_return_sequences
        self.generation_config.max_new_tokens = old_max_new_tokens
        return simulated_sequences

    def predict_time_to_events(
            self,
            partial_history: Union[np.ndarray, list],
            n_future_visits: int = 1,
            future_visit_offset: int = 0
    ) -> List[ConceptTimeToEvent]:

        patient_history_length = len(partial_history)
        simulated_seqs = self.simulate(
            partial_history=partial_history,
            max_new_tokens=self.model.config.n_positions - patient_history_length
        )

        events = defaultdict(list)
        for seq in simulated_seqs:
            visit_counter = 0
            time_delta = 0
            for next_token in seq[patient_history_length:]:
                visit_counter += int(is_visit_end(next_token))
                if visit_counter > n_future_visits != -1:
                    break
                if is_att_token(next_token):
                    time_delta += extract_time_interval_in_days(next_token)
                elif visit_counter >= future_visit_offset and self.is_outcome_event(next_token):
                    events[next_token].append(time_delta)

        # Count the occurrences of each time tokens for each concept
        return [
            ConceptTimeToEvent(concept_id, create_time_to_event(time_intervals))
            for concept_id, time_intervals in events.items()
        ]

    @staticmethod
    def get_generation_config(
            tokenizer: CehrGptTokenizer,
            max_length: int,
            num_return_sequences: int,
            top_p: float = 1.0,
            top_k: int = 300,
            temperature: float = 1.0,
            repetition_penalty: float = 1.0,
            epsilon_cutoff: float = 0.0
    ) -> GenerationConfig:
        return GenerationConfig(
            max_length=max_length,
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
            renormalize_logits=True
        )
