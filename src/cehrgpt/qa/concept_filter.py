"""
Which concepts the language model is asked about.

The questions cover the clinical events of the tokenizer vocabulary (conditions, procedures,
medications), not the measurement and observation records, and not the artificial tokens
the model sequence is built from ([VS], [VE], time tokens, year/age/gender/race, ...).
"""

import json
import logging
import os
import pickle
from typing import Collection, Dict, Optional, Set

import pandas as pd

from cehrgpt.tokenization.tokenization_constants import (
    CONCEPT_MAPPING_FILE_NAME,
    LAB_STATS_FILE_NAME,
    MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME,
)

LOG = logging.getLogger(__name__)

DEFAULT_ALLOWED_DOMAINS = ("Condition", "Procedure", "Drug")


def load_concept_names(tokenizer_path: str) -> Dict[str, str]:
    """token -> readable name for the whole vocabulary of a saved tokenizer."""
    with open(os.path.join(tokenizer_path, CONCEPT_MAPPING_FILE_NAME)) as f:
        return json.load(f)


def load_motor_codes(tokenizer_path: str) -> Set[str]:
    """The concepts that are MOTOR tasks; empty if the tokenizer has no task information."""
    path = os.path.join(tokenizer_path, MOTOR_TIME_TO_EVENT_TASK_INFO_FILE_NAME)
    if not os.path.exists(path):
        LOG.warning(
            "%s not found, so no query can be tagged as inside the MOTOR task set", path
        )
        return set()
    with open(path, "rb") as f:
        info = pickle.load(f) or {}
    return {str(code) for code in info.get("motor_time_to_event_codes", [])}


def is_artificial_token(token: str) -> bool:
    """
    Whether a token is not a concept id.

    Concept tokens are the OMOP concept id as digits. Everything else is built by the
    pipeline: [VS]/[VE]/[PAD]..., time tokens (D5, W2, M3, LT), year:2020, age:50,
    Gender/F, Race/5 and so on.
    """
    return not token.isdigit()


def load_value_concepts(
    tokenizer_path: str, include_categorical: bool = False
) -> Set[str]:
    """
    Concepts that carry a value, i.e. measurements (numeric lab statistics).

    With include_categorical the concepts with categorical results are added as well; they
    are observation-like records.
    """
    path = os.path.join(tokenizer_path, LAB_STATS_FILE_NAME)
    if not os.path.exists(path):
        LOG.warning(
            "%s not found, so measurement concepts cannot be recognised from the tokenizer",
            path,
        )
        return set()
    with open(path, "rb") as f:
        lab_stats = pickle.load(f) or {}
    concepts = {str(stat["concept_id"]) for stat in lab_stats.get("numeric_lab_stats", [])}
    if include_categorical:
        concepts |= {
            str(key[0]) for key in lab_stats.get("categorical_lab_stats", {}).keys()
        }
    return concepts


def load_concept_domains(path: str) -> Dict[str, str]:
    """concept_id -> domain_id from an OMOP concept table (parquet/csv file or folder)."""
    if os.path.isdir(path):
        table = pd.concat(
            [
                pd.read_parquet(os.path.join(path, name), columns=["concept_id", "domain_id"])
                for name in sorted(os.listdir(path))
                if name.endswith(".parquet")
            ]
        )
    elif path.endswith(".csv"):
        table = pd.read_csv(path, usecols=["concept_id", "domain_id"])
    else:
        table = pd.read_parquet(path, columns=["concept_id", "domain_id"])
    return dict(zip(table["concept_id"].astype(str), table["domain_id"]))


def eligible_concepts(
    concept_names: Dict[str, str],
    value_concepts: Collection[str] = (),
    concept_domains: Optional[Dict[str, str]] = None,
    allowed_domains: Collection[str] = DEFAULT_ALLOWED_DOMAINS,
) -> Dict[str, str]:
    """
    The concepts, with their names, that questions may be about.

    Args:
        concept_names: token -> readable name for the whole vocabulary
        value_concepts: measurement (value carrying) concepts to leave out
        concept_domains: concept id -> OMOP domain. Without it the domains cannot be told
            apart, so visit, observation and device concepts that are not value carrying
            remain; with it only allowed_domains are kept
        allowed_domains: the domains to keep when concept_domains is given
    """
    excluded = set(value_concepts)
    allowed = set(allowed_domains)
    eligible = {}
    for token, name in concept_names.items():
        if is_artificial_token(token) or token in excluded or not name:
            continue
        if concept_domains is not None and concept_domains.get(token) not in allowed:
            continue
        eligible[token] = name
    if concept_domains is None:
        LOG.warning(
            "No concept table given: visit, observation and device concepts that carry no "
            "value are still eligible. Pass one to keep only the domains %s",
            sorted(allowed),
        )
    return eligible
