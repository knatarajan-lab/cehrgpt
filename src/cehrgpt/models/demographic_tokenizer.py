import pickle
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Union

from cehrbert.models.hf_models.tokenization_utils import agg_helper
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm

from cehrgpt.generation.omop_converter_batch import START_TOKEN_SIZE
from cehrgpt.models.tokenization_hf_cehrgpt import OUT_OF_VOCABULARY_TOKEN


class DemographicTokenizer:
    """
    A tokenizer for converting demographic data into tokenized forms suitable for model training.

    Attributes:
        initial_year_mapping (Dict[str, int]): Mapping from start year to unique identifier.
        initial_age_mapping (Dict[str, int]): Mapping from start age to unique identifier.
        gender_mapping (Dict[str, int]): Mapping from gender descriptor to unique identifier.
        race_mapping (Dict[str, int]): Mapping from race descriptor to unique identifier.
    """

    def __init__(
        self,
        initial_year_tokenizer: Tokenizer,
        initial_age_tokenizer: Tokenizer,
        gender_tokenizer: Tokenizer,
        race_tokenizer: Tokenizer,
    ):
        self.initial_year_tokenizer = initial_year_tokenizer
        self.initial_age_tokenizer = initial_age_tokenizer
        self.gender_tokenizer = gender_tokenizer
        self.race_tokenizer = race_tokenizer

    @property
    def num_initial_years(self) -> int:
        return self.initial_year_tokenizer.get_vocab_size()

    @property
    def num_initial_ages(self) -> int:
        return self.initial_age_tokenizer.get_vocab_size()

    @property
    def num_genders(self) -> int:
        return self.gender_tokenizer.get_vocab_size()

    @property
    def num_races(self) -> int:
        return self.race_tokenizer.get_vocab_size()

    def encode(
        self,
        initial_year: Union[List[str], str],
        initial_age: Union[List[str], str],
        gender: Union[List[str], str],
        race: Union[List[str], str],
    ) -> Dict[str, Any]:
        """
        Encodes demographic information into tokens.

        Parameters:
            initial_year (Union[List[str], str]): The initial year or years to encode.
            initial_age (Union[List[str], str]): The initial age or ages to encode.
            gender (Union[List[str], str]): The gender or genders to encode.
            race (Union[List[str], str]): The race or races to encode.

        Returns:
            Dict[str, List[int]]: A dictionary with encoded tokens for each demographic category.
        """
        if isinstance(initial_year, str):
            initial_year = [initial_year]
        if isinstance(initial_age, str):
            initial_age = [initial_age]
        if isinstance(gender, str):
            gender = [gender]
        if isinstance(race, str):
            race = [race]
        return {
            "initial_year": self.initial_year_tokenizer.encode(
                initial_year, is_pretokenized=True
            ),
            "initial_age": self.initial_age_tokenizer.encode(
                initial_age, is_pretokenized=True
            ),
            "gender": self.gender_tokenizer.encode(gender, is_pretokenized=True),
            "race": self.race_tokenizer.encode(race, is_pretokenized=True),
        }

    def decode(self, encoded_data: Dict[str, List[int]]) -> Dict[str, List[str]]:
        """
        Decodes token indices back to their respective string representations.

        Parameters:
            encoded_data (Dict[str, List[int]]): Dictionary containing lists of token indices for each demographic category.

        Returns:
            Dict[str, List[str]]: A dictionary with decoded strings for each demographic category.
        """
        return {
            "initial_year": self.initial_year_tokenizer.decode(
                encoded_data["initial_year"]
            ),
            "initial_age": self.initial_age_tokenizer.decode(
                encoded_data["initial_age"]
            ),
            "gender": self.gender_tokenizer.decode(encoded_data["gender"]),
            "race": self.race_tokenizer.decode(encoded_data["race"]),
        }

    @classmethod
    def train_tokenizer(cls, dataset: Dataset, data_args: DataTrainingArguments):
        """
        Class method to train the tokenizer on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to process.
            data_args (DataTrainingArguments): Configuration arguments for data preprocessing.

        Returns:
            DemographicTokenizer: An instance of DemographicTokenizer initialized with trained data mappings.
        """
        map_fun_args = {
            "batched": True,
            "batch_size": data_args.preprocessing_batch_size,
            "keep_in_memory": True,
            "new_fingerprint": "invalid",
            "remove_columns": dataset.column_names,
            "num_proc": (
                None if data_args.streaming else data_args.preprocessing_num_workers
            ),
        }
        parts = dataset.map(
            partial(agg_helper, map_func=map_demographics), **map_fun_args
        )
        current = None
        for stat in tqdm(parts, desc="Aggregating the demographics info"):
            fixed_stat = pickle.loads(stat["data"])
            if current is None:
                current = fixed_stat
            else:
                for demographic_name, tokens in fixed_stat.items():
                    current[demographic_name].extend(tokens)

        start_year_token_id_mapping = build_token_index_map(current["start_year"])
        start_age_token_id_mapping = build_token_index_map(current["start_age"])
        gender_token_id_mapping = build_token_index_map(current["gender"])
        race_token_id_mapping = build_token_index_map(current["race"])

        return DemographicTokenizer(
            initial_year_tokenizer=start_year_token_id_mapping,
            initial_age_tokenizer=start_age_token_id_mapping,
            gender_tokenizer=gender_token_id_mapping,
            race_tokenizer=race_token_id_mapping,
        )


def build_token_index_map(tokens: List[str]) -> Tokenizer:
    """
    Creates a mapping from each unique token to a unique index.

    Parameters:
        tokens (List[str]): A list of tokens from which to create the mapping.

    Returns:
        Dict[str, int]: A dictionary mapping each token to a unique index.
    """
    tokenizer = Tokenizer(WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict()))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train_from_iterator(
        tokens,
        trainer=WordLevelTrainer(
            special_tokens=[OUT_OF_VOCABULARY_TOKEN],
            vocab_size=len(set(tokens)) + 1,
        ),
    )
    return tokenizer


def map_demographics(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps batch data to demographic information needed for tokenizer training.

    Parameters:
        batch (Dict[str, Any]): Batch from the dataset containing demographic information.

    Returns:
        Dict[str, Any]: A dictionary with demographic categories as keys and list of tokens as values.
    """
    demographic_tokens = defaultdict(list)
    for concept_ids in batch["concept_ids"]:
        start_year, start_age, gender, race = concept_ids[:START_TOKEN_SIZE]
        if start_year not in demographic_tokens["start_year"]:
            demographic_tokens["start_year"].append(start_year)
        if start_age not in demographic_tokens["start_age"]:
            demographic_tokens["start_age"].append(start_age)
        if gender not in demographic_tokens["gender"]:
            demographic_tokens["gender"].append(gender)
        if race not in demographic_tokens["race"]:
            demographic_tokens["race"].append(race)
    return demographic_tokens
