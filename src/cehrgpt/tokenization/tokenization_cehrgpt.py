import collections
import copy
import json
import os
import re
import pickle
from functools import partial
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import transformers
from cehrbert.models.hf_models.tokenization_utils import load_json_file
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset, DatasetDict
from femr.stat_utils import OnlineStatistics
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizer
from transformers.utils import logging

from cehrgpt.gpt_utils import (
    convert_time_interval_to_time_tuple,
    extract_time_interval_in_days,
    is_att_token,
    is_clinical_event,
    is_inpatient_att_token,
)
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.tokenization.special_tokens import (
    END_TOKEN,
    LINEAR_PROB_TOKEN,
    OUT_OF_VOCABULARY_TOKEN,
    PAD_TOKEN,
    RANDOM_TOKEN,
    START_TOKEN,
)
from cehrgpt.tokenization.tokenization_bin_utils import (
    create_bins_with_spline,
    create_sample_from_bins,
    create_value_bin,
)
from cehrgpt.tokenization.tokenization_constants import (
    CONCEPT_MAPPING_FILE_NAME,
    CONCEPT_STATS_FILE_NAME,
    DEGREE_OF_FREEDOM,
    DEMOGRAPHICS_STATS_FILE_NAME,
    LAB_STATS_FILE_NAME,
    LEGACY_LAB_STATS_FILE_NAME,
    MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME,
    NONE_BIN,
    NUM_OF_BINS,
    ONTOLOGY_FILE_NAME,
    TIME_TOKENIZER_FILE_NAME,
    TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME,
    TOKENIZER_FILE_NAME,
    UNKNOWN_BIN,
    VALUE_TOKENIZER_FILE_NAME,
)
from cehrgpt.tokenization.tokenization_numeric_stats import NumericEventStatistics
from cehrgpt.tokenization.tokenization_statistics import (
    compute_motor_tte_statistics,
    compute_statistics,
    get_allowed_motor_codes,
)
from cehrgpt.omop.ontology import Ontology

LOG = logging.get_logger("transformers")
MAX_CLINICAL_AGE = 120


class _LazyAttribute:
    """
    Loads a group of attributes from disk the first time one of them is read.

    It is a non-data descriptor, so once the value is in the instance ``__dict__`` (loaded, or
    assigned explicitly) the instance value wins and the descriptor is never consulted again.
    """

    def __init__(self, group: str):
        self.group = group
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        obj._load_lazy_group(self.group)
        try:
            return obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None


class CehrGptTokenizer(PreTrainedTokenizer):
    # The statistics, MOTOR info and ontology can be hundreds of MB and are only needed for
    # training, generation and to bin numeric values, so a tokenizer loaded with
    # `from_pretrained` reads them from disk on first use (see `lazy_files`).
    _numeric_lab_stats = _LazyAttribute("lab_stats")
    _categorical_lab_stats = _LazyAttribute("lab_stats")
    _numeric_event_statistics = _LazyAttribute("lab_stats")
    _numeric_concept_ids = _LazyAttribute("lab_stats")
    _categorical_concept_ids = _LazyAttribute("lab_stats")
    _motor_task_info = _LazyAttribute("motor_task_info")
    _motor_time_to_event_codes = _LazyAttribute("motor_task_info")
    _motor_code_to_id_mapping = _LazyAttribute("motor_task_info")
    _ontology = _LazyAttribute("ontology")

    def __init__(
            self,
            tokenizer: Tokenizer,
            value_tokenizer: Tokenizer,
            att_tokenizer: Tokenizer,
            token_to_sub_time_token_mapping: Dict[str, List[str]],
            concept_code_stats: Dict[str, Any],
            numeric_lab_stats: Optional[List[Dict[str, Any]]],
            categorical_lab_stats: Optional[Dict[Tuple[str, str], int]],
            concept_name_mapping: Dict[str, str],
            pretrained_concept_embedding_model: PretrainedEmbeddings = None,
            motor_task_info: Optional[Dict[str, Any]] = None,
            gender_map: Optional[Dict[str, int]] = None,
            race_map: Optional[Dict[str, int]] = None,
            ontology: Optional[Ontology] = None,
            lazy_files: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        """
        ``lazy_files`` maps a group ("lab_stats", "motor_task_info", "ontology") to a
        (kind, path) tuple, kind being "pickle" or "legacy" (the json lab stats of old
        tokenizers). The group is then read from ``path`` on first use, and the matching
        constructor arguments (numeric_lab_stats / categorical_lab_stats, motor_task_info,
        ontology) are ignored.
        """
        self._lazy_files = dict(lazy_files) if lazy_files else {}
        # The objects that were read from disk, so that they can be dropped again
        self._lazy_loaded: Dict[str, Any] = {}
        self._tokenizer = tokenizer
        self._value_tokenizer = value_tokenizer
        self._att_tokenizer = att_tokenizer
        self._token_to_sub_time_token_mapping = token_to_sub_time_token_mapping
        self._concept_code_stats = concept_code_stats
        if "lab_stats" not in self._lazy_files:
            self._numeric_lab_stats = numeric_lab_stats
            self._numeric_event_statistics = NumericEventStatistics(numeric_lab_stats)
            self._categorical_lab_stats = categorical_lab_stats
            self._numeric_concept_ids = (
                self._numeric_event_statistics.get_numeric_concept_ids()
            )
            self._categorical_concept_ids = list(
                {t[0] for t in self._categorical_lab_stats.keys()}
            )
        self._concept_name_mapping = concept_name_mapping
        self._oov_token_id = self._tokenizer.token_to_id(OUT_OF_VOCABULARY_TOKEN)
        self._padding_token_id = self._tokenizer.token_to_id(PAD_TOKEN)
        self._start_token_id = self._tokenizer.token_to_id(START_TOKEN)
        self._end_token_id = self._tokenizer.token_to_id(END_TOKEN)

        # Backward compatible with the old tokenizer
        self._linear_token_id = None
        if LINEAR_PROB_TOKEN in self._tokenizer.get_vocab():
            self._linear_token_id = self._tokenizer.token_to_id(LINEAR_PROB_TOKEN)
        else:
            self._linear_token_id = self._oov_token_id

        self._random_token_id = None
        if RANDOM_TOKEN in self._tokenizer.get_vocab():
            self._random_token_id = self._tokenizer.token_to_id(RANDOM_TOKEN)
        else:
            self._random_token_id = self._oov_token_id

        self._padding_value_token_id = self._value_tokenizer.token_to_id(PAD_TOKEN)
        self._pretrained_concept_embedding_model = (
            pretrained_concept_embedding_model
            if pretrained_concept_embedding_model
            else PretrainedEmbeddings(None)
        )
        self._pretrained_concept_ids = [
            _
            for _ in self.get_vocab().keys()
            if self._pretrained_concept_embedding_model.is_concept_available(_)
        ]
        if "motor_task_info" not in self._lazy_files:
            self._motor_task_info: Dict[str, Any] = (
                motor_task_info if motor_task_info is not None else {}
            )
            self._motor_time_to_event_codes = self._motor_task_info.get(
                "motor_time_to_event_codes", []
            )
            self._motor_code_to_id_mapping = {
                code: i
                for i, code in enumerate(sorted(self._motor_time_to_event_codes))
            }
        self._gender_map = gender_map if gender_map else {}
        self._race_map = race_map if race_map else {}
        if "ontology" not in self._lazy_files:
            self._ontology = ontology
        super().__init__()

    def _load_lazy_group(self, group: str) -> None:
        """Read a group of lazily loaded attributes from disk, keeping any assigned explicitly."""
        source = self.__dict__.get("_lazy_files", {}).get(group)
        if source is None:
            raise AttributeError(f"{group} is not available in this tokenizer")
        kind, path = source
        values: Dict[str, Any] = {}
        if group == "lab_stats":
            if kind == "legacy":
                # Support the old version of the numeric lab stats file
                numeric_lab_stats, categorical_lab_stats = load_json_file(path), dict()
            else:
                with open(path, "rb") as file:
                    lab_stats = pickle.load(file)
                numeric_lab_stats = lab_stats["numeric_lab_stats"]
                categorical_lab_stats = lab_stats["categorical_lab_stats"]
            numeric_event_statistics = NumericEventStatistics(numeric_lab_stats)
            values = {
                "_numeric_lab_stats": numeric_lab_stats,
                "_categorical_lab_stats": categorical_lab_stats,
                "_numeric_event_statistics": numeric_event_statistics,
                "_numeric_concept_ids": numeric_event_statistics.get_numeric_concept_ids(),
                "_categorical_concept_ids": list(
                    {t[0] for t in categorical_lab_stats.keys()}
                ),
            }
        elif group == "motor_task_info":
            with open(path, "rb") as file:
                motor_task_info = pickle.load(file)
            motor_task_info = motor_task_info if motor_task_info is not None else {}
            codes = motor_task_info.get("motor_time_to_event_codes", [])
            values = {
                "_motor_task_info": motor_task_info,
                "_motor_time_to_event_codes": codes,
                "_motor_code_to_id_mapping": {
                    code: i for i, code in enumerate(sorted(codes))
                },
            }
        elif group == "ontology":
            with open(path, "rb") as file:
                values = {"_ontology": pickle.load(file)}
        for name, value in values.items():
            # An attribute assigned in the meantime wins over what is on disk
            if name not in self.__dict__:
                self.__dict__[name] = value
                self._lazy_loaded[name] = value

    def release_lazy_data(self) -> None:
        """
        Drop the lazily loaded data that has not been replaced, so it is read again from disk
        on next use. Keeps the tokenizer small to pickle or to copy to worker processes.
        """
        for name, value in list(self.__dict__.get("_lazy_loaded", {}).items()):
            if self.__dict__.get(name) is value:
                del self.__dict__[name]
        self._lazy_loaded = {}

    def __getstate__(self):
        # Lazily loaded data is re-read from disk by the unpickled copy
        state = self.__dict__.copy()
        for name, value in state.get("_lazy_loaded", {}).items():
            if state.get(name) is value:
                del state[name]
        state["_lazy_loaded"] = {}
        return state

    @property
    def pretrained_concept_ids(self):
        return self._pretrained_concept_ids

    @property
    def pretrained_token_ids(self):
        return self.encode(self._pretrained_concept_ids)

    @property
    def pretrained_embeddings(self):
        return np.asarray(
            [
                self._pretrained_concept_embedding_model.get_concept_embeddings(_)
                for _ in self._pretrained_concept_ids
            ]
        )

    @property
    def motor_tte_vocab_size(self) -> int:
        return len(self._motor_code_to_id_mapping)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def value_vocab_size(self) -> int:
        return self._value_tokenizer.get_vocab_size()

    @property
    def time_token_vocab_size(self) -> int:
        return self._att_tokenizer.get_vocab_size()

    @property
    def gender_size(self) -> int:
        # Plus one for the unknown
        return len(self._gender_map) + 1

    @property
    def race_size(self) -> int:
        # Plus one for the unknown
        return len(self._race_map) + 1

    @property
    def pad_value_token_id(self):
        return self._padding_value_token_id

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
    def eos_token(self):
        return END_TOKEN

    @property
    def eos_token_id(self):
        return self._end_token_id

    @property
    def pad_token_id(self):
        return self._padding_token_id

    @property
    def oov_token_id(self):
        return self._oov_token_id

    @property
    def pad_token(self):
        return PAD_TOKEN

    @property
    def linear_token_id(self) -> Optional[int]:
        return self._linear_token_id

    @property
    def linear_token(self) -> Optional[str]:
        if LINEAR_PROB_TOKEN in self._tokenizer.get_vocab():
            return LINEAR_PROB_TOKEN
        return None

    @property
    def random_token_id(self) -> Optional[int]:
        return self._random_token_id

    @property
    def random_token(self) -> Optional[str]:
        if RANDOM_TOKEN in self._tokenizer.get_vocab():
            return RANDOM_TOKEN
        return None

    @property
    def vs_token_id(self):
        # We used VS for the historical data, currently, we use the new [VS] for the newer data
        # so we need to check both cases.
        if "VS" in self._tokenizer.get_vocab():
            return self._convert_token_to_id("VS")
        elif "[VS]" in self._tokenizer.get_vocab():
            return self._convert_token_to_id("[VS]")
        else:
            # Some tokenization schemes (e.g. ETHOS/CoMET) do not use VS/[VS] tokens at all,
            # so fall back to pad_token_id instead of failing every caller.
            LOG.warning("The tokenizer does not contain either VS or [VS]")
            return self.pad_token_id

    @property
    def att_token_ids(self) -> List[int]:
        """Return the list of integer IDs for all ATT (time-interval) tokens in the vocabulary."""
        return [token_id for token, token_id in self._tokenizer.get_vocab().items() if is_att_token(token)]

    @property
    def ve_token_id(self):
        # We used VE for the historical data, currently, we use the new [VE] for the newer data
        # so we need to check both cases.
        if "VE" in self._tokenizer.get_vocab():
            return self._convert_token_to_id("VE")
        elif "[VE]" in self._tokenizer.get_vocab():
            return self._convert_token_to_id("[VE]")
        else:
            # Some tokenization schemes (e.g. ETHOS/CoMET) do not use VE/[VE] tokens at all,
            # so fall back to pad_token_id instead of failing every caller.
            LOG.warning("The tokenizer does not contain either VE or [VE]")
            return self.pad_token_id

    @property
    def valid_age_values(self) -> frozenset:
        """
        Returns the frozenset of integer ages (capped at 120) that have a corresponding
        'age:<int>' token in the vocabulary (case-insensitive).
        Ages above 120 are excluded as data-quality outliers.
        Used for constructing age reconstruction labels.
        """
        pattern = re.compile(r"^age:(\d+)$", re.IGNORECASE)
        return frozenset(
            age
            for token in self._tokenizer.get_vocab()
            if (m := pattern.match(token))
            and (age := int(m.group(1))) <= MAX_CLINICAL_AGE
        )

    @property
    def age_at_vs_vocab_size(self) -> int:
        """
        Returns max(valid_ages) + 1 so that actual age values used directly as
        class indices are always in [0, age_at_vs_vocab_size).

        Age tokens in the vocabulary are often sparse (only ages present in the
        training data), so len(ages) < max(ages) + 1.  Using len(ages) as the
        number of classes causes an IndexError when a patient's real age equals
        or exceeds that count even though the age token is not out of range.
        """
        ages = self.valid_age_values
        if not ages:
            raise RuntimeError(
                "No age tokens matching 'age:<int>' found in the vocabulary. "
                "Cannot derive age_at_vs_vocab_size."
            )
        return max(ages) + 1

    def get_age_token_id(self, age: int) -> Optional[int]:
        """
        Returns the vocabulary token ID for 'age:<age>' (tries common casings),
        or None if the token is not found in the vocabulary.
        """
        if age > MAX_CLINICAL_AGE:
            return None

        vocab = self._tokenizer.get_vocab()
        for candidate in (f"age:{age}", f"Age:{age}", f"AGE:{age}"):
            if candidate in vocab:
                return vocab[candidate]
        return None

    @property
    def valid_year_values(self) -> frozenset:
        """Returns the frozenset of integer calendar years that have a corresponding
        'year:<int>' token in the vocabulary (case-insensitive)."""
        pattern = re.compile(r"^year:(\d+)$", re.IGNORECASE)
        return frozenset(
            int(m.group(1))
            for token in self._tokenizer.get_vocab()
            if (m := pattern.match(token))
        )

    @property
    def year_at_vs_vocab_size(self) -> int:
        """Returns max(valid_years) + 1 so that year values used directly as class
        indices are always in [0, year_at_vs_vocab_size)."""
        years = self.valid_year_values
        if not years:
            raise RuntimeError(
                "No year tokens matching 'year:<int>' found in the vocabulary. "
                "Cannot derive year_at_vs_vocab_size."
            )
        return max(years) + 1

    def get_year_token_id(self, year: int) -> Optional[int]:
        """Returns the vocabulary token ID for 'year:<year>' (tries common casings),
        or None if the token is not found in the vocabulary."""
        vocab = self._tokenizer.get_vocab()
        for candidate in (f"year:{year}", f"Year:{year}", f"YEAR:{year}"):
            if candidate in vocab:
                return vocab[candidate]
        return None

    @property
    def numeric_concept_ids(self):
        return self._numeric_concept_ids

    @property
    def categorical_concept_ids(self):
        return self._categorical_concept_ids

    @property
    def lab_token_ids(self):
        reserved_tokens = [
            START_TOKEN,
            PAD_TOKEN,
            END_TOKEN,
            OUT_OF_VOCABULARY_TOKEN,
            LINEAR_PROB_TOKEN,
        ]
        lab_token_ids = self.encode(
            [
                concept_id
                for concept_id in self._numeric_concept_ids
                                  + self._categorical_concept_ids
                if concept_id not in reserved_tokens
            ]
        )
        return list(
            filter(lambda token_id: token_id != self._oov_token_id, lab_token_ids)
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

    @property
    def pretrained_concept_embedding_model(self):
        return self._pretrained_concept_embedding_model

    def get_motor_task_prevalence_rates(self) -> List[float]:
        """
        Returns the per-task hazard rate for each MOTOR task in sorted code order.
        Rate = frac_events / mean_event_tte (days), matching the original MOTOR
        implementation. Used to initialize task_layer bias in MotorTaskHead.
        """
        task_tte = self._motor_task_info.get("task_tte_stats", {})
        task_censor = self._motor_task_info.get("task_censor_stats", {})
        task_tte_time = self._motor_task_info.get("task_tte_time_stats", {})
        rates = []
        for code in sorted(self._motor_time_to_event_codes):
            events = task_tte.get(code, 0)
            censored = task_censor.get(code, 0)
            total = events + censored
            frac_events = events / total if total > 0 else 1e-6
            tte_stats = task_tte_time.get(code)
            if tte_stats is not None and tte_stats.count > 0:
                mean_tte = tte_stats.mean()
                rate = frac_events / mean_tte if mean_tte > 0 else 1e-6
            else:
                rate = 1e-6
            rates.append(max(rate, 1e-6))
        return rates

    def get_motor_time_bins(self, motor_num_time_pieces: int) -> List[int]:
        if "motor_event_times" not in self._motor_task_info:
            return []
        time_bins = np.percentile(
            self._motor_task_info["motor_event_times"].samples,
            np.linspace(0, 100, motor_num_time_pieces + 1),
        )
        time_bins[0] = 0
        time_bins[-1] = float("inf")
        return list(time_bins)

    def get_motor_token_id(self, concept_id: str) -> int:
        if not self.is_motor_time_to_event_code(concept_id):
            raise RuntimeError(f"Invalid motor concept id: {concept_id}")
        return self._motor_code_to_id_mapping[concept_id]

    def is_motor_time_to_event_code(self, future_concept_id: str) -> bool:
        if (
                self._motor_time_to_event_codes
                and future_concept_id in self._motor_time_to_event_codes
        ):
            return True
        return False

    def get_motor_parents(self, concept_id: str) -> List[str]:
        motor_codes = []
        if self._ontology is None:
            if self.is_motor_time_to_event_code(concept_id):
                motor_codes.append(concept_id)
        else:
            motor_codes.extend(
                [
                    p
                    for p in self._ontology.get_all_parents(concept_id)
                    if self.is_motor_time_to_event_code(p)
                ]
            )
        return motor_codes

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()

    def get_value_vocab(self) -> Dict[str, int]:
        return self._value_tokenizer.get_vocab()

    def encode(self, concept_ids, **kwargs) -> Sequence[int]:
        encoded = self._tokenizer.encode(concept_ids, is_pretokenized=True)
        return encoded.ids

    def decode(
            self, concept_token_ids: List[int], skip_special_tokens: bool = True, **kwargs
    ) -> List[str]:
        return self._tokenizer.decode(
            concept_token_ids, skip_special_tokens=skip_special_tokens
        ).split(" ")

    def encode_value(self, concept_values: Sequence[str]) -> Sequence[int]:
        encoded = self._value_tokenizer.encode(concept_values, is_pretokenized=True)
        return encoded.ids

    def decode_value(
            self, concept_value_token_ids: List[int], skip_special_tokens: bool = True
    ) -> List[str]:
        return self._value_tokenizer.decode(
            concept_value_token_ids, skip_special_tokens=skip_special_tokens
        ).split(" ")

    def encode_gender(self, gender: str) -> int:
        return self._gender_map.get(gender, 0)

    def encode_race(self, race: str) -> int:
        return self._race_map.get(race, 0)

    def ensure_visit_tokens(self) -> bool:
        """Add [VS] and [VE] to the vocabulary if they are not already present.

        Returns True if any tokens were added (i.e. the tokenizer was modified and
        should be re-saved), False if both were already in the vocabulary.
        """
        missing = [t for t in ("[VS]", "[VE]") if t not in self.get_vocab()]
        if not missing:
            return False
        LOG.warning(
            "The following visit tokens are missing from the tokenizer vocabulary "
            "and will be added: %s. Re-save the tokenizer afterwards.",
            missing,
        )
        self.add_token(missing)
        return True

    def add_token(self, tokens: Union[str, List[str]]) -> None:
        if isinstance(tokens, str):
            tokens = [tokens]
        vocab = self.get_vocab()
        self._tokenizer.add_tokens(
            [
                AddedToken(token, single_word=True, normalized=False)
                for token in tokens
                if token not in vocab
            ]
        )

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

        self._value_tokenizer.save(
            os.path.join(save_directory, VALUE_TOKENIZER_FILE_NAME)
        )

        self._att_tokenizer.save(os.path.join(save_directory, TIME_TOKENIZER_FILE_NAME))

        with open(
                os.path.join(save_directory, TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME), "w"
        ) as f:
            json.dump(self._token_to_sub_time_token_mapping, f)

        with open(os.path.join(save_directory, CONCEPT_STATS_FILE_NAME), "w") as f:
            json.dump(self._concept_code_stats, f)

        with open(os.path.join(save_directory, LAB_STATS_FILE_NAME), "wb") as f:
            lab_stats = {
                "numeric_lab_stats": self._numeric_lab_stats,
                "categorical_lab_stats": self._categorical_lab_stats,
            }
            pickle.dump(lab_stats, f)

        with open(
                os.path.join(save_directory, DEMOGRAPHICS_STATS_FILE_NAME), "wb"
        ) as f:
            pickle.dump(
                {
                    "gender_map": self._gender_map,
                    "race_map": self._race_map,
                },
                f,
            )

        with open(os.path.join(save_directory, CONCEPT_MAPPING_FILE_NAME), "w") as f:
            json.dump(self._concept_name_mapping, f)

        with open(
                os.path.join(save_directory, MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME), "wb"
        ) as f:
            pickle.dump(self._motor_task_info, f)

        if self._ontology is not None:
            with open(os.path.join(save_directory, ONTOLOGY_FILE_NAME), "wb") as f:
                pickle.dump(self._ontology, f)

        self._pretrained_concept_embedding_model.save(save_directory)

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

        is_legacy_tokenizer = CehrGptTokenizer.is_legacy_tokenizer(
            pretrained_model_name_or_path, **kwargs
        )

        # Load the concept tokenizer
        tokenizer_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TOKENIZER_FILE_NAME, **kwargs
        )
        if not tokenizer_file:
            return None
        tokenizer = Tokenizer.from_file(tokenizer_file)

        # Load the concept_value_tokenizer
        if is_legacy_tokenizer:
            value_tokenizer = Tokenizer(
                WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict())
            )
        else:
            value_tokenizer_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path, VALUE_TOKENIZER_FILE_NAME, **kwargs
            )
            if not value_tokenizer_file:
                return None
            value_tokenizer = Tokenizer.from_file(value_tokenizer_file)

        # Load the ttt tokenizer
        att_tokenizer_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TIME_TOKENIZER_FILE_NAME, **kwargs
        )
        if not att_tokenizer_file:
            return None
        att_tokenizer = Tokenizer.from_file(att_tokenizer_file)

        # Load the sub time token json file
        token_to_sub_time_token_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path,
            TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME,
            **kwargs,
        )
        if not token_to_sub_time_token_mapping_file:
            return None
        token_to_sub_time_token_mapping = load_json_file(
            token_to_sub_time_token_mapping_file
        )

        # The lab stats can be hundreds of MB, they are read when first needed
        lazy_files: Dict[str, Tuple[str, str]] = {}
        if is_legacy_tokenizer:
            legacy_lab_stats_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path, LEGACY_LAB_STATS_FILE_NAME, **kwargs
            )
            if not legacy_lab_stats_file:
                return None
            lazy_files["lab_stats"] = ("legacy", legacy_lab_stats_file)
        else:
            lab_stats_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path, LAB_STATS_FILE_NAME, **kwargs
            )
            if not lab_stats_file:
                return None
            lazy_files["lab_stats"] = ("pickle", lab_stats_file)
        try:
            # Load the demographics stats json file
            demographics_stats_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path, DEMOGRAPHICS_STATS_FILE_NAME, **kwargs
            )
            if demographics_stats_file:
                with open(demographics_stats_file, "rb") as file:
                    demographics_stats = pickle.load(file)
            else:
                demographics_stats = None
        except EnvironmentError:
            LOG.warning(
                f"The %s files does not exist in %s, setting demographics_stats to None",
                DEMOGRAPHICS_STATS_FILE_NAME,
                pretrained_model_name_or_path,
            )
            demographics_stats = None

        # Load the concept_name json file
        concept_name_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_MAPPING_FILE_NAME, **kwargs
        )
        if not concept_name_mapping_file:
            return None
        concept_name_mapping = load_json_file(concept_name_mapping_file)

        # Load the concept_code_stats json file
        concept_code_stats_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_STATS_FILE_NAME, **kwargs
        )
        if not concept_code_stats_mapping_file:
            return None
        concept_code_stats = load_json_file(concept_code_stats_mapping_file)

        try:
            # Load the MOTOR time to event codes file
            motor_tte_task_info_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path,
                MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME,
                **kwargs,
            )
            if motor_tte_task_info_file:
                lazy_files["motor_task_info"] = ("pickle", motor_tte_task_info_file)
            motor_task_info = None
        except EnvironmentError:
            LOG.warning(
                f"The %s files does not exist in %s, setting motor_task_info to None",
                MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME,
                pretrained_model_name_or_path,
            )
            motor_task_info = None

        ontology = None
        try:
            ontology_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path,
                ONTOLOGY_FILE_NAME,
                **kwargs,
            )
            if ontology_file:
                lazy_files["ontology"] = ("pickle", ontology_file)
        except EnvironmentError | OSError:
            LOG.warning(
                "The ontology file %s does not existing in %s",
                ONTOLOGY_FILE_NAME,
                pretrained_model_name_or_path,
            )

        pretrained_embedding_model = PretrainedEmbeddings(pretrained_model_name_or_path)

        return CehrGptTokenizer(
            tokenizer,
            value_tokenizer,
            att_tokenizer,
            token_to_sub_time_token_mapping,
            concept_code_stats,
            None,
            None,
            concept_name_mapping,
            pretrained_embedding_model,
            motor_task_info,
            demographics_stats["gender_map"] if demographics_stats else None,
            demographics_stats["race_map"] if demographics_stats else None,
            ontology=ontology,
            lazy_files=lazy_files,
        )

    @classmethod
    def is_legacy_tokenizer(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        try:
            legacy_lab_stats_file = transformers.utils.hub.cached_file(
                pretrained_model_name_or_path, LEGACY_LAB_STATS_FILE_NAME, **kwargs
            )
            return legacy_lab_stats_file is not None
        except Exception:
            return False

    @classmethod
    def expand_trained_tokenizer(
            cls,
            cehrgpt_tokenizer,
            dataset: Union[Dataset, DatasetDict],
            concept_name_mapping: Dict[str, str],
            data_args: DataTrainingArguments,
            pretrained_concept_embedding_model: PretrainedEmbeddings = None,
            num_motor_tasks: Optional[int] = None,
            apply_entropy_filter: bool = False,
            min_patient_count: int = 0,
            ontology: Optional[Ontology] = None,
    ):
        if not isinstance(cehrgpt_tokenizer, CehrGptTokenizer):
            raise ValueError(
                "The existing cehrgpt must be an instance of CehrGptTokenizer"
            )

        cehrgpt_tokenizer_copy = copy.deepcopy(cehrgpt_tokenizer)

        new_tokenizer = CehrGptTokenizer.train_tokenizer(
            dataset=dataset,
            concept_name_mapping=concept_name_mapping,
            data_args=data_args,
            num_motor_tasks=num_motor_tasks,
            apply_entropy_filter=apply_entropy_filter,
            min_patient_count=min_patient_count,
            ontology=ontology,
        )

        new_tokens = set(new_tokenizer.get_vocab().keys()) - set(
            cehrgpt_tokenizer_copy.get_vocab().keys()
        )
        new_value_tokens = set(new_tokenizer.get_value_vocab().keys()) - set(
            cehrgpt_tokenizer_copy.get_value_vocab().keys()
        )
        new_att_tokens = set(new_tokenizer._att_tokenizer.get_vocab().keys()) - set(
            cehrgpt_tokenizer_copy._att_tokenizer.get_vocab().keys()
        )
        new_token_to_sub_time_token_mapping = (
            new_tokenizer._token_to_sub_time_token_mapping
        )
        new_numeric_lab_stats = new_tokenizer._numeric_lab_stats
        new_categorical_lab_stats = new_tokenizer._categorical_lab_stats
        new_concept_name_mapping = new_tokenizer._concept_name_mapping
        new_motor_time_to_event_codes = new_tokenizer._motor_time_to_event_codes

        # Add new tokens to the existing tokenizer
        cehrgpt_tokenizer_copy._tokenizer.add_tokens(
            [
                AddedToken(token, single_word=True, normalized=False)
                for token in new_tokens
            ]
        )
        # Add new tokens to the existing value tokenizer
        cehrgpt_tokenizer_copy._value_tokenizer.add_tokens(
            [
                AddedToken(token, single_word=True, normalized=False)
                for token in new_value_tokens
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

        # Merge numeric lab_stats
        cehrgpt_tokenizer_copy._numeric_lab_stats = cls.merge_numeric_lab_stats(
            cehrgpt_tokenizer_copy._numeric_lab_stats,
            new_numeric_lab_stats,
        )
        # Merge categorical lab_stats
        cehrgpt_tokenizer_copy._categorical_lab_stats = cls.merge_categorical_lab_stats(
            cehrgpt_tokenizer_copy._categorical_lab_stats,
            new_categorical_lab_stats,
        )

        # Merge concept_name_mapping
        for token, concept_name in new_concept_name_mapping.items():
            if token not in cehrgpt_tokenizer_copy._concept_name_mapping:
                cehrgpt_tokenizer_copy._concept_name_mapping[token] = concept_name

        # Merge motor_time_to_event_codes
        if (
                new_motor_time_to_event_codes
                and cehrgpt_tokenizer_copy._motor_time_to_event_codes
        ):
            for motor_time_to_event_code in new_motor_time_to_event_codes:
                if (
                        motor_time_to_event_code
                        not in cehrgpt_tokenizer_copy._motor_time_to_event_codes
                ):
                    cehrgpt_tokenizer_copy._motor_time_to_event_codes.append(
                        motor_time_to_event_code
                    )

        for gender in new_tokenizer._gender_map.keys():
            if gender not in cehrgpt_tokenizer_copy._gender_map:
                cehrgpt_tokenizer_copy._gender_map[gender] = len(
                    cehrgpt_tokenizer_copy._gender_map
                )

        for race in new_tokenizer._race_map.keys():
            if race not in cehrgpt_tokenizer_copy._race_map:
                cehrgpt_tokenizer_copy._race_map[race] = len(
                    cehrgpt_tokenizer_copy._race_map
                )

        return CehrGptTokenizer(
            tokenizer=cehrgpt_tokenizer_copy._tokenizer,
            value_tokenizer=cehrgpt_tokenizer_copy._value_tokenizer,
            att_tokenizer=cehrgpt_tokenizer_copy._att_tokenizer,
            token_to_sub_time_token_mapping=cehrgpt_tokenizer_copy._token_to_sub_time_token_mapping,
            concept_code_stats=cehrgpt_tokenizer_copy._concept_code_stats,
            numeric_lab_stats=cehrgpt_tokenizer_copy._numeric_lab_stats,
            categorical_lab_stats=cehrgpt_tokenizer_copy._categorical_lab_stats,
            concept_name_mapping=cehrgpt_tokenizer_copy._concept_name_mapping,
            pretrained_concept_embedding_model=pretrained_concept_embedding_model,
            motor_task_info=cehrgpt_tokenizer_copy._motor_task_info,
            gender_map=cehrgpt_tokenizer_copy._gender_map,
            race_map=cehrgpt_tokenizer_copy._race_map,
            ontology=cehrgpt_tokenizer_copy._ontology,
        )

    @classmethod
    def merge_numeric_lab_stats(
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
                # recreate the bins
                sample = create_sample_from_bins(
                    lab_stats_existing_mapping[concept_unit_pair]["bins"]
                )
                sample.extend(create_sample_from_bins(lab_stat["bins"]))
                lab_stats_existing_mapping[concept_unit_pair]["bins"] = (
                    create_bins_with_spline(sample, NUM_OF_BINS, DEGREE_OF_FREEDOM)
                )

            else:
                if lab_stat["count"] > 0:
                    lab_stats_existing_mapping[concept_unit_pair] = lab_stat

        return list(lab_stats_existing_mapping.values())

    @classmethod
    def merge_categorical_lab_stats(
            cls,
            categorical_lab_stats_existing: Dict[Tuple[str, str], int],
            categorical_lab_stats_new: Dict[Tuple[str, str], int],
    ) -> Dict[Tuple[str, str], int]:
        for (concept_id, concept_as_value), count in categorical_lab_stats_new.items():
            if (concept_id, concept_as_value) not in categorical_lab_stats_new:
                categorical_lab_stats_existing[(concept_id, concept_as_value)] = 0
            categorical_lab_stats_existing[(concept_id, concept_as_value)] += count
        return categorical_lab_stats_existing

    @classmethod
    def train_tokenizer(
            cls,
            dataset: Union[Dataset, DatasetDict],
            concept_name_mapping: Dict[str, str],
            data_args: DataTrainingArguments,
            pretrained_concept_embedding_model: PretrainedEmbeddings = None,
            num_motor_tasks: Optional[int] = None,
            apply_entropy_filter: bool = False,
            ontology: Optional[Ontology] = None,
            min_patient_count: int = 0,
    ):
        """
        Train a huggingface word level tokenizer.

        To use their tokenizer, we need to concatenate all the concepts
        together and treat it as a sequence.

        Args:
            min_patient_count: a clinical token must be had by at least this many patients of the
                dataset. With an ontology a patient counts for a token when they have the token or
                any of its descendants, once however many; without one, only the token itself
                counts. 0 keeps every token. Tokens that are not clinical events (time and visit
                markers, year and age) and the gender and race tokens of the demographic prompt are
                never removed.
            apply_entropy_filter: also keep only the clinical tokens whose entropy is at least that
                of a token that occurs in min_patient_count of the patients (as a share of the
                dataset). It applies on top of min_patient_count.
        """
        if min_patient_count is None:
            min_patient_count = 0
        if min_patient_count < 0:
            raise ValueError(f"min_patient_count cannot be negative, got {min_patient_count}")

        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

        LOG.info("Calculating data statistics")
        cehrgpt_data_statistics = compute_statistics(dataset, data_args, ontology)
        numeric_lab_stats = cehrgpt_data_statistics["numeric_lab_stats"]
        categorical_lab_stats = cehrgpt_data_statistics["categorical_lab_stats"]
        original_concept_codes = cehrgpt_data_statistics["original_concept_codes"]
        all_concept_code_stats = cehrgpt_data_statistics["all_concept_code_stats"]
        all_concept_code_entropies = cehrgpt_data_statistics[
            "all_concept_code_entropies"
        ]

        # Only the real clinical events are filtered. The gender and race tokens are numeric concept
        # ids as well, but they are the fixed demographic prompt of every patient: a rare race removed
        # from the vocabulary would be dropped from the sequence and leave the prompt a token short.
        demographic_tokens = set(cehrgpt_data_statistics["gender_list"]) | set(
            cehrgpt_data_statistics["race_list"]
        )

        def is_filtered(token: str) -> bool:
            return (
                is_clinical_event(token, data_args.is_data_in_meds)
                and token not in demographic_tokens
            )

        if apply_entropy_filter:
            # The cut-off is the entropy of a token that occurs in min_patient_count of the patients
            threshold_share = max(1e-8, min_patient_count / cehrgpt_data_statistics["total"])
            min_entropy = (
                    np.log(1 - threshold_share) * (1 - threshold_share)
                    + np.log(threshold_share) * threshold_share
            )
            qualified_codes = [
                k
                for k in original_concept_codes
                if all_concept_code_entropies[k] <= min_entropy or not is_filtered(k)
            ]
        else:
            qualified_codes = list(original_concept_codes)

        if min_patient_count > 0:
            patient_counts = cehrgpt_data_statistics["concept_patient_counts"]
            before = len(qualified_codes)
            qualified_codes = [
                k
                for k in qualified_codes
                if not is_filtered(k) or patient_counts.get(k, 0) >= min_patient_count
            ]
            LOG.info(
                "min_patient_count=%s removed %s of the %s tokens that were left",
                min_patient_count,
                before - len(qualified_codes),
                before,
            )

        # Create the tokenizer now
        LOG.info("Training the tokenizer for concepts")
        concept_tokenizer = cls.train_concept_tokenizer_from_counts(
            cehrgpt_data_statistics["concept_token_counts"],
            special_tokens=[
                PAD_TOKEN,
                OUT_OF_VOCABULARY_TOKEN,
                START_TOKEN,
                END_TOKEN,
                LINEAR_PROB_TOKEN,
            ],
            unk_token=OUT_OF_VOCABULARY_TOKEN,
            data_args=data_args,
            qualified_codes=qualified_codes,
        )
        LOG.info("Training the tokenizer for values")
        value_tokenizer = cls.train_concept_tokenizer_from_counts(
            cehrgpt_data_statistics["value_token_counts"],
            special_tokens=[OUT_OF_VOCABULARY_TOKEN, PAD_TOKEN],
            unk_token=OUT_OF_VOCABULARY_TOKEN,
            data_args=data_args,
        )
        value_tokenizer.add_tokens(
            [
                AddedToken(_, single_word=True, normalized=False)
                for _ in [create_value_bin(_) for _ in range(NUM_OF_BINS)]
                         + [UNKNOWN_BIN, NONE_BIN]
            ]
        )

        # We will train a tokenizer specifically for time intervals
        sub_time_token_data = []
        token_to_sub_time_token_mapping = collections.defaultdict(list)
        vocab = concept_tokenizer.get_vocab()
        for token, token_id in vocab.items():
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

        # Prune concept_name_mapping
        concept_name_mapping = {
            concept_id: concept_name_mapping[concept_id]
            for concept_id in vocab.keys()
            if concept_id in concept_name_mapping
        }

        motor_task_info = None
        if num_motor_tasks:
            LOG.info("Computing the MOTOR TTE statistics")
            measurement_concept_ids = set(
                [stat["concept_id"] for stat in numeric_lab_stats]
                + [t[0] for t in categorical_lab_stats.keys()]
            )
            allowed_motor_codes = get_allowed_motor_codes(
                original_concept_codes, data_args, ontology, measurement_concept_ids
            )
            if min_patient_count > 0:
                # A token that was removed from the vocabulary must not be a task either, and a token
                # that stayed only because it is a gender or race (the exemption above) is not a
                # clinical event to predict
                kept = set(qualified_codes)
                allowed_motor_codes = [
                    c
                    for c in allowed_motor_codes
                    if c in kept and c not in demographic_tokens
                ]
            motor_tte_statistics = compute_motor_tte_statistics(
                dataset, data_args, allowed_motor_codes, ontology
            )
            motor_time_to_event_codes = []
            for concept_id, _ in sorted(
                    all_concept_code_entropies.items(), key=lambda t: t[1]
            ):
                if concept_id not in allowed_motor_codes:
                    continue
                tte_stats = motor_tte_statistics["task_tte_stats"][concept_id]
                censor_stats = motor_tte_statistics["task_censor_stats"][concept_id]
                frac_events = tte_stats / (tte_stats + censor_stats)

                if frac_events < 1 / 1000:
                    LOG.info(
                        "Ran into very rare task %s with %s", concept_id, frac_events
                    )
                    continue

                if len(motor_time_to_event_codes) < num_motor_tasks:
                    motor_time_to_event_codes.append(concept_id)
                else:
                    break

            LOG.info(
                f"{len(motor_time_to_event_codes)} number of tasks have been added as MOTOR tasks"
            )
            motor_task_info = {
                "motor_event_times": motor_tte_statistics["motor_event_times"],
                "task_tte_stats": motor_tte_statistics["task_tte_stats"],
                "task_censor_stats": motor_tte_statistics["task_censor_stats"],
                "motor_time_to_event_codes": motor_time_to_event_codes,
            }

        gender_map = {
            gender: i + 1
            for i, gender in enumerate(sorted(cehrgpt_data_statistics["gender_list"]))
        }
        race_map = {
            race: i + 1
            for i, race in enumerate(sorted(cehrgpt_data_statistics["race_list"]))
        }

        return CehrGptTokenizer(
            concept_tokenizer,
            value_tokenizer,
            att_tokenizer,
            token_to_sub_time_token_mapping,
            all_concept_code_stats,
            numeric_lab_stats,
            categorical_lab_stats,
            concept_name_mapping,
            pretrained_concept_embedding_model,
            motor_task_info,
            gender_map,
            race_map,
            ontology,
        )

    @classmethod
    def train_concept_tokenizer_from_counts(
            cls,
            token_counts: Dict[str, int],
            special_tokens: List[str],
            unk_token: str,
            data_args: DataTrainingArguments,
            qualified_codes: Optional[Sequence[str]] = None,
    ) -> Tokenizer:
        """Construct a WordLevel tokenizer from counts gathered during statistics.

        ``WordLevelTrainer`` assigns special tokens first, then orders observed tokens by
        decreasing frequency and lexicographically for ties. Reproducing that ordering avoids
        two full ``dataset.map`` passes while preserving the token ids produced by the trainer.
        """
        special_tokens = list(dict.fromkeys(special_tokens))
        special_token_set = set(special_tokens)
        qualified_code_set = set(qualified_codes) if qualified_codes else None
        word_counts = collections.defaultdict(int)
        for value, count in token_counts.items():
            # The existing map filters complete values before joining them. Split only after this
            # check to retain that behavior if a value itself contains whitespace.
            if qualified_code_set is not None and value not in qualified_code_set:
                continue
            for token in value.split():
                word_counts[token] += count
        candidates = [
            (token, count)
            for token, count in word_counts.items()
            if token not in special_token_set
            and count >= data_args.min_frequency
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        available_slots = max(0, data_args.vocab_size - len(special_tokens))
        candidates = candidates[:available_slots]

        vocab = {token: token_id for token_id, token in enumerate(special_tokens)}
        vocab.update(
            (token, len(special_tokens) + token_id)
            for token_id, (token, _) in enumerate(candidates)
        )
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token, vocab=vocab))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        # Mark the already indexed tokens as special so decoding keeps the trainer's behavior.
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    @classmethod
    def train_concept_tokenizer(
            cls,
            dataset,
            feature_name,
            special_tokens: List[str],
            unk_token,
            data_args,
            qualified_codes: Optional[List[str]] = None,
    ):
        # Use the Fast Tokenizer from the Huggingface tokenizers Rust implementation.
        # https://github.com/huggingface/tokenizers
        concept_tokenizer = Tokenizer(WordLevel(unk_token=unk_token, vocab=dict()))
        concept_tokenizer.pre_tokenizer = WhitespaceSplit()
        concept_trainer = WordLevelTrainer(
            special_tokens=special_tokens,
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.min_frequency,
            show_progress=True,
        )
        # This collection is queried for every token in every patient sequence. Keeping it as a
        # list makes tokenization O(number of tokens * vocabulary size).
        qualified_code_set = (
            set(qualified_codes) if qualified_codes is not None else None
        )
        batch_concat_concepts_partial_func = partial(
            cls.batch_concat_concepts,
            feature_name=feature_name,
            qualified_codes=qualified_code_set,
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
                    batch = list(islice(iterator, data_args.preprocessing_batch_size))
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
        return concept_tokenizer

    def normalize(self, concept_id: str, unit: str, concept_value: float) -> str:
        return self._numeric_event_statistics.normalize(concept_id, unit, concept_value)

    def denormalize(self, concept_id: str, value_bin: str) -> Tuple[float, str]:
        return self._numeric_event_statistics.denormalize(concept_id, value_bin)

    @classmethod
    def batch_concat_concepts(
            cls,
            records: Dict[str, List],
            feature_name: str,
            qualified_codes: Optional[Set[str]] = None,
    ) -> Dict[str, List]:
        def filter_token(t: str) -> bool:
            """
            If the token is None or not string, return False.

            When qualified_codes is provided, t must be in
            qualified_codes to be valid, otherwise the tokens are always valid

            :param t:
            :return:
            """
            if t is None or not isinstance(t, str):
                return False
            if qualified_codes:
                return t in qualified_codes
            return True

        return {
            feature_name: [
                " ".join(filter(filter_token, tokens))
                for tokens in records[feature_name]
            ]
        }
