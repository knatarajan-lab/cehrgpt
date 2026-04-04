"""
Data collator for CEHR-GPT RL fine-tuning.

Each training example is a tokenized patient sequence.  The collator:
  1. Picks a random visit-boundary split point in each sequence.
  2. Returns the prefix (history up to the split) as padded tensors,
     including value_indicators and values (lab bin IDs) so the forward
     pass during PG loss computation matches the model's training distribution.
  3. Returns the true future conditions (concept_id, days_to_event) after the
     split, capped at max(prediction_windows) days, for use as reward targets.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token

_VISIT_START_TOKEN = "[VS]"
_DEMOGRAPHICS_SIZE = 4  # year, age, gender, race tokens at the start of every sequence


@dataclass
class RLDataCollator:
    """
    Prepares batches for RL training from tokenized CEHR-GPT sequences.

    Args:
        tokenizer: ``CehrGptTokenizer`` instance used to encode prefix tokens.
        target_concept_ids: Set of condition concept ID strings that are reward targets.
        prediction_windows: Prediction horizon windows in days (e.g. [30, 90, 180, 365, 730]).
        max_prefix_length: Prefix is right-truncated to this many tokens before padding.
        min_prefix_visits: Minimum [VS] tokens required in the prefix; examples with
            fewer visits are skipped.
    """

    tokenizer: Any  # CehrGptTokenizer
    target_concept_ids: Set[str]
    prediction_windows: List[int]
    max_prefix_length: int = 1024
    min_prefix_visits: int = 2

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_prefix_ids: List[List[int]] = []
        batch_prefix_ages: List[List[int]] = []
        batch_prefix_times: List[List[float]] = []
        batch_prefix_vals: List[List[int]] = []
        batch_prefix_val_masks: List[List[int]] = []
        batch_future_conditions: List[List[Tuple[str, float]]] = []
        batch_prefix_end_times: List[float] = []
        raw_prefix_lengths: List[int] = []

        for example in examples:
            result = self._prepare_single(example)
            if result is None:
                continue
            (
                prefix_ids, prefix_ages, prefix_times,
                prefix_vals, prefix_val_masks,
                future_conds, prefix_end_time,
            ) = result

            # Truncate to max_prefix_length (keep the most recent tokens)
            if len(prefix_ids) > self.max_prefix_length:
                prefix_ids = prefix_ids[-self.max_prefix_length:]
                prefix_ages = prefix_ages[-self.max_prefix_length:]
                prefix_times = prefix_times[-self.max_prefix_length:]
                prefix_vals = prefix_vals[-self.max_prefix_length:]
                prefix_val_masks = prefix_val_masks[-self.max_prefix_length:]

            batch_prefix_ids.append(prefix_ids)
            batch_prefix_ages.append(prefix_ages)
            batch_prefix_times.append(prefix_times)
            batch_prefix_vals.append(prefix_vals)
            batch_prefix_val_masks.append(prefix_val_masks)
            batch_future_conditions.append(future_conds)
            batch_prefix_end_times.append(prefix_end_time)
            raw_prefix_lengths.append(len(prefix_ids))

        if not batch_prefix_ids:
            return {}

        # Left-pad all sequences to the maximum length in the batch
        max_len = max(raw_prefix_lengths)
        pad_id = self.tokenizer.pad_token_id

        padded_ids, padded_ages, padded_times = [], [], []
        padded_vals, padded_val_masks, attention_masks = [], [], []

        for ids, ages, times, vals, val_masks in zip(
            batch_prefix_ids, batch_prefix_ages, batch_prefix_times,
            batch_prefix_vals, batch_prefix_val_masks,
        ):
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            padded_ages.append([0] * pad_len + ages)
            padded_times.append([0.0] * pad_len + times)
            padded_vals.append([0] * pad_len + vals)
            padded_val_masks.append([0] * pad_len + val_masks)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return {
            "prefix_input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "prefix_ages": torch.tensor(padded_ages, dtype=torch.long),
            "prefix_epoch_times": torch.tensor(padded_times, dtype=torch.float32),
            "prefix_values": torch.tensor(padded_vals, dtype=torch.long),
            "prefix_value_indicators": torch.tensor(padded_val_masks, dtype=torch.bool),
            "prefix_attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            # Actual (unpadded) prefix lengths for slicing during rollout extraction
            "prefix_lengths": torch.tensor(raw_prefix_lengths, dtype=torch.long),
            # Unix timestamp of the last prefix token (float64 to preserve precision)
            "prefix_end_times": torch.tensor(batch_prefix_end_times, dtype=torch.float64),
            # Not a tensor — kept as a Python list of lists for the reward function
            "future_conditions": batch_future_conditions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_single(
        self, example: Dict[str, Any]
    ) -> Optional[Tuple]:
        concept_ids: List[str] = example["concept_ids"]
        epoch_times: List[float] = example["epoch_times"]
        ages: List[int] = example["ages"]

        # Value columns are optional (absent when include_values=False)
        raw_vals: Optional[List[int]] = example.get("values")
        raw_val_masks: Optional[List[int]] = example.get("value_indicators")
        has_values = raw_vals is not None and raw_val_masks is not None

        # Find [VS] positions after the fixed demographic header
        vs_positions = [
            i for i, tok in enumerate(concept_ids)
            if tok == _VISIT_START_TOKEN and i >= _DEMOGRAPHICS_SIZE
        ]

        # Need enough visits to split: at least min_prefix_visits in prefix + 1 in future
        if len(vs_positions) < self.min_prefix_visits + 1:
            return None

        # Choose a random split among the first half of VS positions (so there is always
        # a non-trivial future window)
        max_split_idx = max(1, len(vs_positions) // 2)
        split_vs_pos_idx = random.randint(0, max_split_idx - 1)
        split_token_pos = vs_positions[split_vs_pos_idx]

        # Prefix ends just before the chosen [VS]
        prefix_concept_ids = concept_ids[:split_token_pos]
        prefix_epoch_times = epoch_times[:split_token_pos]
        prefix_ages_list = ages[:split_token_pos]
        prefix_vals = list(raw_vals[:split_token_pos]) if has_values else [0] * split_token_pos
        prefix_val_masks = list(raw_val_masks[:split_token_pos]) if has_values else [0] * split_token_pos

        if not prefix_concept_ids:
            return None

        prefix_end_time = float(prefix_epoch_times[-1])

        # Encode concept ID strings → token IDs (no added special tokens)
        prefix_input_ids = self.tokenizer.convert_tokens_to_ids(prefix_concept_ids)

        # Walk the future tokens to collect conditions with their elapsed days
        max_window = max(self.prediction_windows)
        cumulative_days = 0.0
        future_conditions: List[Tuple[str, float]] = []

        for token in concept_ids[split_token_pos:]:
            if is_att_token(token):
                try:
                    cumulative_days += extract_time_interval_in_days(token)
                except ValueError:
                    pass
                if cumulative_days > max_window:
                    break
            elif token in self.target_concept_ids:
                future_conditions.append((token, cumulative_days))

        return (
            prefix_input_ids,
            list(prefix_ages_list),
            list(prefix_epoch_times),
            prefix_vals,
            prefix_val_masks,
            future_conditions,
            prefix_end_time,
        )
