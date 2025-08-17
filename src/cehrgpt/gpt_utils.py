import random
import re
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from cehrbert_data.const.artificial_tokens import DEATH_TOKEN
from meds import death_code
from transformers.utils import logging

from cehrgpt.cehrgpt_args import SamplingStrategy
from cehrgpt.models.special_tokens import (
    DISCHARGE_CONCEPT_IDS,
    END_TOKEN,
    VISIT_CONCEPT_IDS,
)

# Regular expression pattern to match inpatient attendance tokens
MEDS_CODE_PATTERN = re.compile(r".*/.*")
INPATIENT_ATT_PATTERN = re.compile(r"(?:VS-|i-)D(\d+)(?:-VE)?")
DEMOGRAPHIC_PROMPT_SIZE = 4
logger = logging.get_logger("transformers")


class RandomSampleCache:
    def __init__(
        self,
        data_indices: Sequence[int],
        cache_size: int,
        sample_weights: Sequence[float] = None,
    ):
        """
        Initialize the RandomSampleCache.

        :param data_indices: Sequence of data indices to sample from.
        :param cache_size: Size of the cache.
        :param sample_weights: Optional sequence of weights for sampling.
        """
        self._data_indices = data_indices
        self._sample_weights = sample_weights
        self._cache_size = cache_size
        self._cache = []

        if self._sample_weights is not None:
            assert sum(self._sample_weights) - 1 < 1e-8

    def next(self):
        """
        Get the next sample from the cache.

        If the cache is empty, refill it.

        :return: A sampled data index.
        """
        if not self._cache:
            if self._sample_weights is not None:
                self._cache.extend(
                    random.choices(
                        self._data_indices,
                        k=self._cache_size,
                        weights=self._sample_weights,
                    )
                )
            else:
                self._cache.extend(
                    random.choices(self._data_indices, k=self._cache_size)
                )
        return self._cache.pop()


def construct_time_sequence(
    concept_ids: List[str], epoch_times: Optional[List[Union[int, float]]] = None
) -> List[float]:
    if epoch_times is not None:
        return epoch_times

    if concept_ids[0].lower().startswith("year"):
        year_str = concept_ids[0].split(":")[1]
    else:
        year_str = "1985"

    datetime_cursor = datetime(
        int(year_str), month=1, day=1, hour=0, minute=0, second=0
    ).replace(tzinfo=timezone.utc)
    epoch_times = []
    for concept_id in concept_ids:
        if is_att_token(concept_id):
            att_days = extract_time_interval_in_days(concept_id)
            datetime_cursor += timedelta(days=att_days)
        epoch_times.append(datetime_cursor.timestamp())
    return epoch_times


def construct_age_sequence(
    concept_ids: List[str], ages: Optional[List[int]] = None
) -> List[int]:
    if ages is not None:
        return ages
    elif concept_ids[1].lower().startswith("age"):
        age_str = concept_ids[1].split(":")[1]
        assert age_str.isnumeric(), f"age_str: {age_str}"
        ages = []
        time_delta = 0
        for concept_id in concept_ids:
            if is_att_token(concept_id):
                time_delta += extract_time_interval_in_days(concept_id)
            ages.append(int(age_str) + time_delta // 365)
        return ages
    else:
        logger.warning(
            "The second token is not a valid age token. The first 4 tokens are: %s. "
            "Trying to fall back to ages, but it is not valid either %s. "
            "Fall back to a zero vector [0, 0, 0, ...., 0]",
            concept_ids[:4],
            ages,
        )
        return np.zeros_like(concept_ids, dtype=int).tolist()


def multiple_of_10(n: int) -> int:
    return ((n // 10) + 1) * 10


def encode_demographics(
    age: int, gender: int, race: int, max_age=200, max_gender=10, max_race=10
) -> int:
    assert 0 <= age < max_age, f"age: {age}"
    assert 0 <= gender < max_gender, f"gender: {gender}"
    assert 0 <= race < max_race, f"race: {race}"
    return age + max_age * gender + max_age * max_gender * race


def collect_demographic_prompts_at_visits(patient_history: List[str]):
    demographic_prompts_at_visits = []
    start_year, start_age, start_gender, start_race = patient_history[
        :DEMOGRAPHIC_PROMPT_SIZE
    ]
    try:
        start_year = int(start_year.split(":")[1])
        start_age = int(start_age.split(":")[1])
        valid_prompt = True
    except IndexError | ValueError:
        start_year = 1900
        start_age = 0
        valid_prompt = False
    data_cursor = date(int(start_year), 1, 1)
    birth_date = date(start_year - start_age, 1, 1)
    for i, current_token in enumerate(patient_history):
        if is_visit_start(current_token):
            reconstructed_year = (
                f"year:{data_cursor.year}" if valid_prompt else "year:unknown"
            )
            reconstructed_age = (
                f"age:{data_cursor.year - birth_date.year}"
                if valid_prompt
                else "age:unknown"
            )
            demographic_prompts_at_visits.append(
                (
                    i,
                    (
                        reconstructed_year,
                        reconstructed_age,
                        start_gender,
                        start_race,
                    ),
                )
            )
        elif is_att_token(current_token):
            att_date_delta = extract_time_interval_in_days(current_token)
            data_cursor = data_cursor + timedelta(days=att_date_delta)
    return demographic_prompts_at_visits


def random_slice_gpt_sequence(concept_ids, max_seq_len):
    """
    Randomly slice a GPT sequence.

    :param concept_ids: List of concept IDs.
    :param max_seq_len: Maximum sequence length.
    :return: Tuple containing start index, end index, and demographic tokens.
    """
    seq_length = len(concept_ids)
    starting_points = []
    start_year, start_age, start_gender, start_race = [
        _ for _ in concept_ids[:DEMOGRAPHIC_PROMPT_SIZE]
    ]
    try:
        start_year = int(start_year.split(":")[1])
        start_age = int(start_age.split(":")[1])
        data_cursor = date(int(start_year), 1, 1)
        birth_date = date(start_year - start_age, 1, 1)
        for i in range(
            DEMOGRAPHIC_PROMPT_SIZE,
            min(seq_length, seq_length - max_seq_len + DEMOGRAPHIC_PROMPT_SIZE),
        ):
            current_token = concept_ids[i]
            if is_visit_start(current_token):
                starting_points.append(
                    (i, data_cursor.year, data_cursor.year - birth_date.year)
                )
            elif is_att_token(current_token):
                att_date_delta = extract_time_interval_in_days(current_token)
                data_cursor = data_cursor + timedelta(days=att_date_delta)

        if len(starting_points) == 0:
            return 0, 0, concept_ids[:DEMOGRAPHIC_PROMPT_SIZE]

        random_starting_index, random_starting_year, random_starting_age = (
            random.choice(starting_points)
        )
        demographic_tokens = [
            f"year:{random_starting_year}",
            f"age:{random_starting_age}",
            start_gender,
            start_race,
        ]
        # Remove the number of demographic tokens
        random_end_index = random_starting_index
        for i in reversed(
            range(
                random_starting_index,
                random_starting_index + max_seq_len - DEMOGRAPHIC_PROMPT_SIZE,
            )
        ):
            current_token = concept_ids[i]
            if is_visit_end(current_token):
                random_end_index = i
                break
        return random_starting_index, random_end_index, demographic_tokens

    except Exception:
        return 0, max_seq_len - 1, []


def get_cehrgpt_output_folder(args, cehrgpt_tokenizer) -> str:
    if args.sampling_strategy == SamplingStrategy.TopKStrategy.value:
        folder_name = f"top_k{args.top_k}"
        args.top_p = 1.0
    elif args.sampling_strategy == SamplingStrategy.TopPStrategy.value:
        folder_name = f"top_p{int(args.top_p * 10000)}"
        args.top_k = cehrgpt_tokenizer.vocab_size
    elif args.sampling_strategy == SamplingStrategy.TopMixStrategy.value:
        folder_name = f"top_mix_p{int(args.top_p * 10000)}_k{args.top_k}"
    else:
        raise RuntimeError(
            "sampling_strategy has to be one of the following three options [TopKStrategy, TopPStrategy, TopMixStrategy]"
        )
    if args.temperature != 1.0:
        folder_name = f"{folder_name}_temp_{int(args.temperature * 10000)}"
    if args.repetition_penalty != 1.0:
        folder_name = (
            f"{folder_name}_repetition_penalty_{int(args.repetition_penalty * 10000)}"
        )
    if args.num_beams > 1:
        folder_name = f"{folder_name}_num_beams_{int(args.num_beams)}"
    if args.num_beam_groups > 1:
        folder_name = f"{folder_name}_num_beam_groups_{int(args.num_beam_groups)}"
    if args.epsilon_cutoff > 0.0:
        folder_name = (
            f"{folder_name}_epsilon_cutoff_{int(args.epsilon_cutoff * 100000)}"
        )
    return folder_name


def is_clinical_event(token: str, meds: bool = False) -> bool:
    if token.isnumeric():
        return True
    if token in [DEATH_TOKEN, death_code]:
        return True
    if meds:
        return bool(MEDS_CODE_PATTERN.match(token))
    return False


def is_visit_start(token: str):
    """
    Check if the token indicates the start of a visit.

    :param token: Token to check.
    :return: True if the token is a visit start token, False otherwise.
    """
    return token in ["VS", "[VS]"]


def is_visit_end(token: str) -> bool:
    return token in ["VE", "[VE]"]


def is_inpatient_hour_token(token: str) -> bool:
    return token.startswith("i-H")


def extract_time_interval_in_hours(token: str) -> int:
    try:
        hour = int(token[3:])
        return hour
    except ValueError:
        return 0


def is_att_token(token: str):
    """
    Check if the token is an attention token.

    :param token: Token to check.
    :return: True if the token is an attention token, False otherwise.
    """
    if bool(re.match(r"^D\d+", token)):  # day tokens
        return True
    elif bool(re.match(r"^W\d+", token)):  # week tokens
        return True
    elif bool(re.match(r"^M\d+", token)):  # month tokens
        return True
    elif bool(re.match(r"^Y\d+", token)):  # year tokens
        return True
    elif token == "LT":
        return True
    elif token[:3] == "VS-":  # VS-D7-VE
        return True
    elif token[:2] == "i-" and not token.startswith(
        "i-H"
    ):  # i-D7 and exclude hour tokens
        return True
    return False


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


def is_inpatient_att_token(token: str):
    """
    Check if the token is an inpatient ATT token.

    :param token: Token to check.
    :return: True if the token is an inpatient ATT token, False otherwise.
    """
    return INPATIENT_ATT_PATTERN.match(token)


def extract_time_interval_in_days(token: str):
    """
    Extract the time interval in days from a token.

    :param token: Token to extract from.
    :return: Time interval in days.
    :raises ValueError: If the token is invalid.
    """
    try:
        if token[0] == "D":  # day tokens
            return int(token[1:])
        elif token[0] == "W":  # week tokens
            return int(token[1:]) * 7
        elif token[0] == "M":  # month tokens
            return int(token[1:]) * 30
        elif token[0] == "Y":  # year tokens
            return int(token[1:]) * 365
        elif token == "LT":
            return 365 * 3
        elif token[:3] == "VS-":  # VS-D7-VE
            part = token.split("-")[1]
            if part.startswith("LT"):
                return 365 * 3
            return int(part[1:])
        elif token[:2] == "i-":  # i-D7
            part = token.split("-")[1]
            if part.startswith("LT"):
                return 365 * 3
            return int(token.split("-")[1][1:])
    except Exception:
        raise ValueError(f"Invalid time token: {token}")
    raise ValueError(f"Invalid time token: {token}")


def convert_time_interval_to_time_tuple(
    time_interval: int, is_inpatient: bool
) -> Tuple[str, str, str]:
    """
    Convert a time interval to a tuple of time tokens.

    :param time_interval: Time interval in days.
    :param is_inpatient: Whether the interval is for an inpatient.
    :return: Tuple of year, month, and day tokens.
    """
    assert time_interval >= 0, "the time interval must equal and greater than zero"
    year = time_interval // 365
    month = time_interval % 365 // 30
    day = time_interval % 365 % 30
    year_token = f"year:{year}"
    month_token = f"month:{month}"
    day_token = f"i-day:{day}" if is_inpatient else f"day:{day}"
    return year_token, month_token, day_token


def generate_artificial_time_tokens():
    """
    Generate all the time tokens used in training.

    :return: List of time tokens.
    """
    day_tokens = [f"D{i}" for i in range(2000)]
    week_tokens = [f"W{i}" for i in range(4)]
    month_tokens = [f"M{i}" for i in range(12)]
    long_term_tokens = ["LT"]
    return day_tokens + week_tokens + month_tokens + long_term_tokens
