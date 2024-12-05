import collections
import copy
import json
import os
import pickle
from functools import partial
from itertools import islice
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import transformers
from cehrbert.models.hf_models.tokenization_utils import (
    agg_helper,
    agg_statistics,
    load_json_file,
    map_statistics,
)
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict
from femr.stat_utils import OnlineStatistics
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from cehrgpt.gpt_utils import (
    convert_time_interval_to_time_tuple,
    extract_time_interval_in_days,
    is_att_token,
    is_inpatient_att_token,
)
from cehrgpt.models.special_tokens import (
    END_TOKEN,
    OUT_OF_VOCABULARY_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
)

NA = "N/A"
TOKENIZER_FILE_NAME = "cehrgpt_tokenizer.json"
TIME_TOKENIZER_FILE_NAME = "cehrgpt_time_tokenizer.json"
TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME = "token_to_sub_time_token_mapping.json"
LAB_STATS_FILE_NAME = "cehrgpt_lab_stats.json"
CONCEPT_MAPPING_FILE_NAME = "concept_name_mapping.json"


def create_numeric_concept_unit_mapping(
    lab_stats: List[Dict[str, Any]]
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    numeric_concept_unit_mapping = collections.defaultdict(list)
    for each_lab_stat in lab_stats:
        numeric_concept_unit_mapping[each_lab_stat["concept_id"]].append(
            (each_lab_stat["count"], each_lab_stat["unit"])
        )

    concept_prob_mapping = dict()
    concept_unit_mapping = dict()
    for concept_id in numeric_concept_unit_mapping.keys():
        counts, units = zip(*numeric_concept_unit_mapping[concept_id])
        total_count = sum(counts)
        probs = [float(c) / total_count for c in counts]
        concept_prob_mapping[concept_id] = probs
        concept_unit_mapping[concept_id] = units
    return concept_prob_mapping, concept_unit_mapping


class NumericEventStatistics:
    def __init__(self, lab_stats: List[Dict[str, Any]]):
        self._lab_stats = lab_stats
        self._lab_stats_mapping = {
            (lab_stat["concept_id"], lab_stat["unit"]): {
                "unit": lab_stat["unit"],
                "mean": lab_stat["mean"],
                "std": lab_stat["std"],
                "value_outlier_std": lab_stat["value_outlier_std"],
                "lower_bound": lab_stat["lower_bound"],
                "upper_bound": lab_stat["upper_bound"],
            }
            for lab_stat in lab_stats
        }
        self._concept_prob_mapping, self._concept_unit_mapping = (
            create_numeric_concept_unit_mapping(lab_stats)
        )

    def get_numeric_concept_ids(self) -> List[str]:
        return [_["concept_id"] for _ in self._lab_stats]

    def get_random_unit(self, concept_id: str) -> str:
        if concept_id in self._concept_prob_mapping:
            unit_probs = self._concept_prob_mapping[concept_id]
            return np.random.choice(
                self._concept_unit_mapping[concept_id], p=unit_probs
            )
        return NA

    def normalize(self, concept_id: str, unit: str, concept_value: float) -> float:
        if (concept_id, unit) in self._lab_stats_mapping:
            concept_unit_stats = self._lab_stats_mapping[(concept_id, unit)]
            mean_ = concept_value - concept_unit_stats["mean"]
            std = concept_unit_stats["std"]
            if std > 0:
                value_outlier_std = concept_unit_stats["value_outlier_std"]
                normalized_value = mean_ / std
                # Clip the value between the lower and upper bounds of the corresponding lab
                normalized_value = max(
                    -value_outlier_std, min(value_outlier_std, normalized_value)
                )
            else:
                # If there is not a valid standard deviation,
                # we just the normalized value to the mean of the standard normal
                normalized_value = 0.0
            return normalized_value
        return concept_value

    def denormalize(self, concept_id: str, value: float) -> Tuple[float, str]:
        unit = self.get_random_unit(concept_id)
        if (concept_id, unit) in self._lab_stats_mapping:
            stats = self._lab_stats_mapping[(concept_id, unit)]
            value = value * stats["std"] + stats["mean"]
        return value, unit


class CehrGptTokenizer(PreTrainedTokenizer):

    def __init__(
        self,
        tokenizer: Tokenizer,
        att_tokenizer: Tokenizer,
        token_to_sub_time_token_mapping: Dict[str, List[str]],
        lab_stats: List[Dict[str, Any]],
        concept_name_mapping: Dict[str, str],
    ):
        self._tokenizer = tokenizer
        self._att_tokenizer = att_tokenizer
        self._token_to_sub_time_token_mapping = token_to_sub_time_token_mapping
        self._lab_stats = lab_stats
        self._numeric_event_statistics = NumericEventStatistics(lab_stats)
        self._concept_name_mapping = concept_name_mapping
        self._oov_token_id = self._tokenizer.token_to_id(OUT_OF_VOCABULARY_TOKEN)
        self._padding_token_id = self._tokenizer.token_to_id(PAD_TOKEN)
        self._start_token_id = self._tokenizer.token_to_id(START_TOKEN)
        self._end_token_id = self._tokenizer.token_to_id(END_TOKEN)

        super().__init__()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def time_token_vocab_size(self) -> int:
        return self._att_tokenizer.get_vocab_size()

    @property
    def start_token_id(self):
        return self._start_token_id

    @property
    def end_token_id(self):
        return self._end_token_id

    @property
    def end_token(self):
        return END_TOKEN

    @property
    def pad_token_id(self):
        return self._padding_token_id

    @property
    def pad_token(self):
        return PAD_TOKEN

    @property
    def lab_token_ids(self):
        reserved_tokens = [START_TOKEN, PAD_TOKEN, END_TOKEN, OUT_OF_VOCABULARY_TOKEN]
        return self.encode(
            [
                concept_id
                for concept_id in self._numeric_event_statistics.get_numeric_concept_ids()
                if concept_id not in reserved_tokens
            ]
        )

    @property
    def token_to_time_token_mapping(self) -> Dict[int, List[int]]:
        default_mapping = {-1: [0, 0, 0]}
        default_mapping.update(
            {
                self._tokenizer.token_to_id(time_token): list(
                    map(self._att_tokenizer.token_to_id, sub_time_tokens)
                )
                for time_token, sub_time_tokens in self._token_to_sub_time_token_mapping.items()
            }
        )
        return default_mapping

    def encode(self, concept_ids, **kwargs) -> Sequence[int]:
        encoded = self._tokenizer.encode(concept_ids, is_pretokenized=True)
        return encoded.ids

    def decode(
        self, concept_token_ids: List[int], skip_special_tokens: bool = True, **kwargs
    ) -> List[str]:
        return self._tokenizer.decode(
            concept_token_ids, skip_special_tokens=skip_special_tokens
        ).split(" ")

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        token_id = self._tokenizer.token_to_id(token)
        return token_id if token_id else self._oov_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._tokenizer.id_to_token(index)
        return token if token else OUT_OF_VOCABULARY_TOKEN

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join([self._concept_name_mapping[t] for t in tokens])
        return out_string

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the Cehrbert tokenizer.

        This method make sure the batch processor can then be re-loaded using the
        .from_pretrained class method.

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`PushToHubMixin.push_to_hub`] method.
        """
        assert not os.path.isfile(
            save_directory
        ), f"Provided path ({save_directory}) should be a directory, not a file"

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", str(save_directory).split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        self._tokenizer.save(os.path.join(save_directory, TOKENIZER_FILE_NAME))

        self._att_tokenizer.save(os.path.join(save_directory, TIME_TOKENIZER_FILE_NAME))

        with open(
            os.path.join(save_directory, TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME), "w"
        ) as f:
            json.dump(self._token_to_sub_time_token_mapping, f)

        with open(os.path.join(save_directory, LAB_STATS_FILE_NAME), "w") as f:
            json.dump(self._lab_stats, f)

        with open(os.path.join(save_directory, CONCEPT_MAPPING_FILE_NAME), "w") as f:
            json.dump(self._concept_name_mapping, f)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Load the CehrBert tokenizer.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing tokenization data saved using
                      [`save_pretrained`], e.g., `./my_data_directory/`.
            kwargs: Arguments for loading to pass to transformers.utils.hub.cached_file

        Returns:
            A CehrBert Tokenizer
        """

        tokenizer_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TOKENIZER_FILE_NAME, **kwargs
        )

        if not tokenizer_file:
            return None

        tokenizer = Tokenizer.from_file(tokenizer_file)

        att_tokenizer_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TIME_TOKENIZER_FILE_NAME, **kwargs
        )
        if not att_tokenizer_file:
            return None

        att_tokenizer = Tokenizer.from_file(att_tokenizer_file)

        token_to_sub_time_token_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path,
            TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME,
            **kwargs,
        )
        if not token_to_sub_time_token_mapping_file:
            return None

        lab_stats_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, LAB_STATS_FILE_NAME, **kwargs
        )
        if not lab_stats_file:
            return None

        concept_name_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_MAPPING_FILE_NAME, **kwargs
        )
        if not concept_name_mapping_file:
            return None

        token_to_sub_time_token_mapping = load_json_file(
            token_to_sub_time_token_mapping_file
        )

        concept_name_mapping = load_json_file(concept_name_mapping_file)

        lab_stats = load_json_file(lab_stats_file)

        return CehrGptTokenizer(
            tokenizer,
            att_tokenizer,
            token_to_sub_time_token_mapping,
            lab_stats,
            concept_name_mapping,
        )

    @classmethod
    def expand_trained_tokenizer(
        cls,
        cehrgpt_tokenizer,
        dataset: Union[Dataset, DatasetDict],
        feature_names: List[str],
        concept_name_mapping: Dict[str, str],
        data_args: DataTrainingArguments,
    ):
        if not isinstance(cehrgpt_tokenizer, CehrGptTokenizer):
            raise ValueError(
                "The existing cehrgpt must be an instance of CehrGptTokenizer"
            )

        cehrgpt_tokenizer_copy = copy.deepcopy(cehrgpt_tokenizer)

        new_tokenizer = CehrGptTokenizer.train_tokenizer(
            dataset=dataset,
            feature_names=feature_names,
            concept_name_mapping=concept_name_mapping,
            data_args=data_args,
        )

        new_tokens = list(new_tokenizer._tokenizer.get_vocab().keys())
        new_att_tokens = list(new_tokenizer._att_tokenizer.get_vocab().keys())
        new_token_to_sub_time_token_mapping = (
            new_tokenizer._token_to_sub_time_token_mapping
        )
        new_lab_stats = new_tokenizer._lab_stats
        new_concept_name_mapping = new_tokenizer._concept_name_mapping

        # Add new tokens to the existing tokenizer
        cehrgpt_tokenizer_copy._tokenizer.add_tokens(
            [
                AddedToken(token, single_word=True, normalized=False)
                for token in new_tokens
            ]
        )
        # Add new time tokens to the existing att tokenizer
        cehrgpt_tokenizer_copy._att_tokenizer.add_tokens(
            [
                AddedToken(token, single_word=True, normalized=False)
                for token in new_att_tokens
            ]
        )
        # Merge the time_token -> List[sub_time_tokens] mapping
        for time_token, sub_time_tokens in new_token_to_sub_time_token_mapping.items():
            if (
                time_token
                not in cehrgpt_tokenizer_copy._token_to_sub_time_token_mapping
            ):
                cehrgpt_tokenizer_copy._token_to_sub_time_token_mapping[time_token] = (
                    sub_time_tokens
                )

        # Merge lab_stats
        cehrgpt_tokenizer_copy._lab_stats = cls.merge_lab_stats(
            cehrgpt_tokenizer_copy._lab_stats,
            new_lab_stats,
        )

        # Merge concept_name_mapping
        for token, concept_name in new_concept_name_mapping.items():
            if token not in cehrgpt_tokenizer_copy._concept_name_mapping:
                cehrgpt_tokenizer_copy._concept_name_mapping[token] = concept_name

        return CehrGptTokenizer(
            tokenizer=cehrgpt_tokenizer_copy._tokenizer,
            att_tokenizer=cehrgpt_tokenizer_copy._att_tokenizer,
            token_to_sub_time_token_mapping=cehrgpt_tokenizer_copy._token_to_sub_time_token_mapping,
            lab_stats=cehrgpt_tokenizer_copy._lab_stats,
            concept_name_mapping=cehrgpt_tokenizer_copy._concept_name_mapping,
        )

    @classmethod
    def merge_lab_stats(
        cls,
        lab_stats_existing: List[Dict[str, Any]],
        lab_stats_new: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        lab_stats_existing_mapping = {
            (lab_stat["concept_id"], lab_stat["unit"]): lab_stat
            for lab_stat in lab_stats_existing
        }
        for lab_stat in lab_stats_new:
            concept_unit_pair = (lab_stat["concept_id"], lab_stat["unit"])
            if concept_unit_pair in lab_stats_existing_mapping:
                existing = OnlineStatistics()
                existing.count = lab_stats_existing_mapping[concept_unit_pair]["count"]
                existing.current_mean = lab_stats_existing_mapping[concept_unit_pair][
                    "mean"
                ]
                existing.variance = (
                    lab_stats_existing_mapping[concept_unit_pair]["std"] ** 2
                    * existing.count
                )
                new = OnlineStatistics()
                new.count = lab_stat["count"]
                new.current_mean = lab_stat["mean"]
                new.variance = lab_stat["std"] ** 2 * new.count
                existing.combine(new)
                lab_stats_existing_mapping[concept_unit_pair]["mean"] = existing.mean()
                lab_stats_existing_mapping[concept_unit_pair][
                    "std"
                ] = existing.standard_deviation()
                lab_stats_existing_mapping[concept_unit_pair]["count"] = existing.count
                lab_stats_existing_mapping[concept_unit_pair]["lower_bound"] = min(
                    lab_stats_existing_mapping[concept_unit_pair]["lower_bound"],
                    lab_stat["lower_bound"],
                )
                lab_stats_existing_mapping[concept_unit_pair]["upper_bound"] = max(
                    lab_stats_existing_mapping[concept_unit_pair]["upper_bound"],
                    lab_stat["upper_bound"],
                )
            else:
                if lab_stat["count"] > 0:
                    lab_stats_existing_mapping[concept_unit_pair] = lab_stat

        return list(lab_stats_existing_mapping.values())

    @classmethod
    def train_tokenizer(
        cls,
        dataset: Union[Dataset, DatasetDict],
        feature_names: List[str],
        concept_name_mapping: Dict[str, str],
        data_args: DataTrainingArguments,
    ):
        """
        Train a huggingface word level tokenizer.

        To use their tokenizer, we need to concatenate all the concepts
        together and treat it as a sequence.
        """

        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

        lab_stats = []
        # Use the Fast Tokenizer from the Huggingface tokenizers Rust implementation.
        # https://github.com/huggingface/tokenizers
        concept_tokenizer = Tokenizer(
            WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict())
        )
        concept_tokenizer.pre_tokenizer = WhitespaceSplit()
        concept_trainer = WordLevelTrainer(
            special_tokens=[PAD_TOKEN, OUT_OF_VOCABULARY_TOKEN, START_TOKEN, END_TOKEN],
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.min_frequency,
            show_progress=True,
        )
        for feature_name in feature_names:
            batch_concat_concepts_partial_func = partial(
                cls.batch_concat_concepts, feature_name=feature_name
            )
            if data_args.streaming:
                concatenated_features = dataset.map(
                    batch_concat_concepts_partial_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                )

                def batched_generator():
                    iterator = iter(concatenated_features)
                    while True:
                        batch = list(
                            islice(iterator, data_args.preprocessing_batch_size)
                        )
                        if not batch:
                            break
                        yield [example[feature_name] for example in batch]

                # We pass a generator of list of texts (concatenated concept_ids) to train_from_iterator
                # for efficient training
                generator = batched_generator()
            else:
                concatenated_features = dataset.map(
                    batch_concat_concepts_partial_func,
                    num_proc=data_args.preprocessing_num_workers,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=dataset.column_names,
                )
                generator = concatenated_features[feature_name]

            concept_tokenizer.train_from_iterator(generator, trainer=concept_trainer)

            map_statistics_partial = partial(
                map_statistics,
                capacity=data_args.offline_stats_capacity,
                value_outlier_std=data_args.value_outlier_std,
            )

            if data_args.streaming:
                parts = dataset.map(
                    partial(agg_helper, map_func=map_statistics_partial),
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    keep_in_memory=True,
                    new_fingerprint="invalid",
                    remove_columns=dataset.column_names,
                )
            else:
                parts = dataset.map(
                    partial(agg_helper, map_func=map_statistics_partial),
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=dataset.column_names,
                    num_proc=data_args.preprocessing_num_workers,
                    keep_in_memory=True,
                    new_fingerprint="invalid",
                )
            current = None
            for stat in tqdm(parts, desc="Aggregating the lab statistics"):
                fixed_stat = pickle.loads(stat["data"])
                if current is None:
                    current = fixed_stat
                else:
                    current = agg_statistics(current, fixed_stat)

            lab_stats = [
                {
                    "concept_id": concept_id,
                    "unit": unit,
                    "mean": online_stats.mean(),
                    "std": online_stats.standard_deviation(),
                    "count": online_stats.count,
                    "value_outlier_std": data_args.value_outlier_std,
                    "lower_bound": online_stats.mean()
                    - data_args.value_outlier_std * online_stats.standard_deviation(),
                    "upper_bound": online_stats.mean()
                    + data_args.value_outlier_std * online_stats.standard_deviation(),
                }
                for (concept_id, unit), online_stats in current[
                    "numeric_stats_by_lab"
                ].items()
                if online_stats.count > 0
            ]

        # We will train a tokenizer specifically for time intervals
        sub_time_token_data = []
        token_to_sub_time_token_mapping = collections.defaultdict(list)
        for token, token_id in concept_tokenizer.get_vocab().items():
            if is_att_token(token):
                time_interval = extract_time_interval_in_days(token)
                time_tuple = convert_time_interval_to_time_tuple(
                    time_interval, is_inpatient_att_token(token)
                )
                token_to_sub_time_token_mapping[token] = list(time_tuple)
                sub_time_token_data.append(" ".join(time_tuple))

        att_tokenizer = Tokenizer(
            WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict())
        )
        att_tokenizer.pre_tokenizer = WhitespaceSplit()
        att_trainer = WordLevelTrainer(
            special_tokens=[OUT_OF_VOCABULARY_TOKEN],
            vocab_size=data_args.vocab_size,
            min_frequency=0,
            show_progress=True,
        )
        att_tokenizer.train_from_iterator(sub_time_token_data, trainer=att_trainer)

        return CehrGptTokenizer(
            concept_tokenizer,
            att_tokenizer,
            token_to_sub_time_token_mapping,
            lab_stats,
            concept_name_mapping,
        )

    def normalize(self, concept_id: str, unit: str, concept_value: float) -> float:
        return self._numeric_event_statistics.normalize(concept_id, unit, concept_value)

    def denormalize(self, concept_id: str, value: float) -> Tuple[float, str]:
        return self._numeric_event_statistics.denormalize(concept_id, value)

    @classmethod
    def batch_concat_concepts(
        cls, records: Dict[str, List], feature_name
    ) -> Dict[str, List]:
        return {feature_name: [" ".join(map(str, _)) for _ in records[feature_name]]}
