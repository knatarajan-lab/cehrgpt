from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from cehrgpt.gpt_utils import is_att_token


def convert_month_token_to_upperbound_days(
    month_token: str, time_bucket_size: int = 90
) -> str:
    if is_att_token(month_token):
        if month_token == "LT":
            return ">= 1095 days"
        else:
            base = (int(month_token[1:]) + 1) * 30 // (time_bucket_size + 1)
            return (
                f"{base * time_bucket_size} days - {(base + 1) * time_bucket_size} days"
            )
    raise ValueError(f"month_token: {month_token} is not a valid month token")


def calculate_time_bucket_probability(
    predictions: List[Dict[str, Any]], time_bucket_size: int = 90
) -> List[Tuple[str, Any]]:
    predictions_with_time_buckets = [
        {
            "probability": p["probability"],
            "time_bucket": convert_month_token_to_upperbound_days(
                p["time_interval"], time_bucket_size
            ),
        }
        for p in predictions
    ]
    # Dictionary to store summed probabilities per time bucket
    grouped_probabilities = defaultdict(float)
    # Loop through the data
    for entry in predictions_with_time_buckets:
        time_bucket = entry["time_bucket"]
        probability = entry["probability"]
        grouped_probabilities[time_bucket] += probability
    return sorted(grouped_probabilities.items(), key=lambda item: item[1], reverse=True)


def calculate_accumulative_time_bucket_probability(
    predictions: List[Dict[str, Any]], time_bucket_size: int = 90
) -> List[Tuple[str, Any]]:
    time_bucket_probability = calculate_time_bucket_probability(
        predictions, time_bucket_size
    )
    accumulative_probs = np.cumsum([_[1] for _ in time_bucket_probability])
    return [
        (*_, accumulative_prob)
        for _, accumulative_prob in zip(time_bucket_probability, accumulative_probs)
    ]
