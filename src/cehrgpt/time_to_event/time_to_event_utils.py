from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from cehrgpt.gpt_utils import is_att_token

from src.cehrgpt.gpt_utils import is_visit_start, is_visit_end, is_att_token
from src.cehrgpt.models.tokenization_hf_cehrgpt import END_TOKEN

VISIT_CONCEPT_IDS = [
    '9202', '9203', '581477', '9201', '5083', '262', '38004250', '0', '8883', '38004238', '38004251',
    '38004222', '38004268', '38004228', '32693', '8971', '38004269', '38004193', '32036', '8782'
]
DISCHARGE_CONCEPT_IDS = [

]


def is_clinical_event(token: str) -> bool:
    return token.isnumeric()


def is_artificial_token(token: str) -> bool:
    if token in VISIT_CONCEPT_IDS:
        return True
    if token in DISCHARGE_CONCEPT_IDS:
        return True
    if is_visit_start(token):
        return True
    if is_visit_end(token):
        return True
    if is_att_token(token):
        return True
    if token == END_TOKEN:
        return True
    return False


def convert_month_token_to_upperbound_days(month_token: str, time_bucket_size: int = 90) -> str:
    if is_att_token(month_token):
        if month_token == 'LT':
            return ">= 1095 days"
        else:
            base = (int(month_token[1:]) + 1) * 30 // (time_bucket_size + 1)
            return f"{base * time_bucket_size} days - {(base + 1) * time_bucket_size} days"
    raise ValueError(f"month_token: {month_token} is not a valid month token")


def calculate_time_bucket_probability(predictions: List[Dict[str, Any]], time_bucket_size: int = 90) -> List[
    Tuple[str, Any]]:
    predictions_with_time_buckets = [
        {"probability": p["probability"],
         "time_bucket": convert_month_token_to_upperbound_days(p["time_interval"], time_bucket_size)}
        for p in predictions
    ]
    # Dictionary to store summed probabilities per time bucket
    grouped_probabilities = defaultdict(float)
    # Loop through the data
    for entry in predictions_with_time_buckets:
        time_bucket = entry['time_bucket']
        probability = entry['probability']
        grouped_probabilities[time_bucket] += probability
    return sorted(grouped_probabilities.items(), key=lambda item: item[1], reverse=True)


def calculate_accumulative_time_bucket_probability(
        predictions: List[Dict[str, Any]],
        time_bucket_size: int = 90
) -> List[Tuple[str, Any]]:
    time_bucket_probability = calculate_time_bucket_probability(predictions, time_bucket_size)
    accumulative_probs = np.cumsum([_[1] for _ in time_bucket_probability])
    return [(*_, accumulative_prob) for _, accumulative_prob in zip(time_bucket_probability, accumulative_probs)]
