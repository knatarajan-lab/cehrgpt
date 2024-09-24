import random
import re
from datetime import date, timedelta
from typing import Sequence, Tuple

from src.cehrgpt.cehrgpt_args import SamplingStrategy
from src.cehrgpt.models.special_tokens import END_TOKEN, VISIT_CONCEPT_IDS, DISCHARGE_CONCEPT_IDS

INPATIENT_ATT_PATTERN = re.compile(r'(?:VS-|i-)D(\d+)(?:-VE)?')


class RandomSampleCache:
    def __init__(
            self,
            data_indices: Sequence[int],
            cache_size: int,
            sample_weights: Sequence[float] = None,
    ):
        self._data_indices = data_indices
        self._sample_weights = sample_weights
        self._cache_size = cache_size
        self._cache = []

        if self._sample_weights is not None:
            assert sum(self._sample_weights) - 1 < 1e-8

    def next(self):
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


def random_slice_gpt_sequence(concept_ids, max_seq_len):
    seq_length = len(concept_ids)
    starting_points = []
    [start_year, start_age, start_gender, start_race] = [_ for _ in concept_ids[0:4]]
    try:
        start_year = int(start_year.split(":")[1])
        start_age = int(start_age.split(":")[1])
        data_cursor = date(int(start_year), 1, 1)
        birth_date = date(start_year - start_age, 1, 1)
        for i in range(4, max(5, seq_length - max_seq_len)):
            current_token = concept_ids[i]
            if is_visit_start(current_token):
                starting_points.append(
                    (i, data_cursor.year, data_cursor.year - birth_date.year)
                )
            elif is_att_token(current_token):
                att_date_delta = extract_time_interval_in_days(current_token)
                data_cursor = data_cursor + timedelta(days=att_date_delta)

        if len(starting_points) == 0:
            return 0, 0, concept_ids[0:4]

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
                list(range(random_starting_index, random_starting_index + max_seq_len - 4))
        ):
            current_token = concept_ids[i]
            if current_token == "VE":
                random_end_index = i
                break
        # new_token_ids = demographic_tokens + concept_ids[random_starting_index:random_end_index + 1]
        return random_starting_index, random_end_index, demographic_tokens

    except Exception:
        return 0, max_seq_len - 1, []


def get_cehrgpt_output_folder(args, cehrgpt_tokenizer) -> str:
    if args.sampling_strategy == SamplingStrategy.TopKStrategy.value:
        folder_name = f'top_k{args.top_k}'
        args.top_p = 1.0
    elif args.sampling_strategy == SamplingStrategy.TopPStrategy.value:
        folder_name = f'top_p{int(args.top_p * 1000)}'
        args.top_k = cehrgpt_tokenizer.vocab_size
    elif args.sampling_strategy == SamplingStrategy.TopMixStrategy.value:
        folder_name = f'top_mix_p{int(args.top_p * 1000)}_k{args.top_k}'
    else:
        raise RuntimeError(
            'sampling_strategy has to be one of the following three options [TopKStrategy, TopPStrategy, TopMixStrategy]'
        )
    if args.temperature != 1.0:
        folder_name = f'{folder_name}_temp_{int(args.temperature * 1000)}'
    if args.repetition_penalty != 1.0:
        folder_name = f'{folder_name}_repetition_penalty_{int(args.repetition_penalty * 1000)}'
    if args.num_beams > 1:
        folder_name = f'{folder_name}_num_beams_{int(args.num_beams)}'
    if args.num_beam_groups > 1:
        folder_name = f'{folder_name}_num_beam_groups_{int(args.num_beam_groups)}'
    if args.epsilon_cutoff > 0.0:
        folder_name = f'{folder_name}_epsilon_cutoff_{int(args.epsilon_cutoff * 100000)}'
    return folder_name


def is_clinical_event(token: str) -> bool:
    return token.isnumeric()


def is_visit_start(token: str) -> bool:
    return token in ['VS', '[VS]']


def is_visit_end(token: str) -> bool:
    return token in ['VE', '[VE]']


def is_att_token(token: str):
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
    if token[:3] == "VS-":  # VS-D7-VE
        return True
    elif token[:2] == "i-":  # i-D7
        return True
    return False


def extract_time_interval_in_days(token: str):
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

    :return:
    """
    day_tokens = [f"D{i}" for i in range(2000)]
    week_tokens = [f"W{i}" for i in range(4)]
    month_tokens = [f"M{i}" for i in range(12)]
    long_term_tokens = ["LT"]
    return day_tokens + week_tokens + month_tokens + long_term_tokens
