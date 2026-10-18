"""
Choose how a MEDS dataset is converted, without asking the caller for its shape.

The conversion of cehrbert (MedsToBertMimic4, MedsToCehrbertOMOP, ...) is a set of
rules that find the visits in the event codes of a dataset: the codes of an ED
registration, an admission, a discharge, a visit. The dataset decides which rules
apply, so they are found in the data. A sample of the event codes is matched against
the rules of every conversion that is registered, and the conversion whose
rules match the most events is used.

    * A conversion type that is set explicitly (anything but the default) is used.
    * The default, MedsToCehrbertOMOP, stands for "not set": it is replaced by the
      conversion that fits the data.
    * Data that no conversion matches keeps the default, as before, with a warning.

A new conversion needs no change here: it is a subclass of MedsToCehrBertConversion
and is scored with its own rules.
"""

import itertools
import os
import re
from typing import Dict, Iterable, List, Optional

import meds_reader
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import (
    MedsToCehrBertConversion,
    MedsToCehrbertOMOP,
)
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    get_meds_to_cehrbert_conversion_cls,
)
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from transformers.utils import logging

LOG = logging.get_logger("transformers")

DEFAULT_CONVERSION_TYPE = MedsToCehrbertOMOP.__name__
NUM_SAMPLE_SUBJECTS = 200


def registered_conversion_types() -> List[str]:
    return sorted(cls.__name__ for cls in MedsToCehrBertConversion.__subclasses__())


def visit_rules(conversion_type: str) -> List[re.Pattern]:
    """The patterns with which a conversion finds the visits (re.match, as it does)."""
    conversion = get_meds_to_cehrbert_conversion_cls(
        conversion_type, default_visit_id=1
    )
    rules = (
        conversion.get_other_visit_matching_rules()
        + conversion.get_ed_admission_matching_rules()
        + conversion.get_admission_matching_rules()
        + conversion.get_discharge_matching_rules()
    )
    return [re.compile(rule) for rule in rules]


def score_conversion_types(codes: Iterable[str]) -> Dict[str, int]:
    """For each conversion, the number of the event codes that its visit rules match."""
    codes = list(codes)
    scores = {}
    for conversion_type in registered_conversion_types():
        patterns = visit_rules(conversion_type)
        scores[conversion_type] = sum(
            1 for code in codes if any(pattern.match(code) for pattern in patterns)
        )
    return scores


def best_conversion_type(scores: Dict[str, int]) -> Optional[str]:
    """The conversion with the highest score; None if none scores or two tie."""
    best = max(scores.values(), default=0)
    winners = [name for name, score in scores.items() if score == best]
    return winners[0] if best > 0 and len(winners) == 1 else None


def detect_conversion_type(codes: Iterable[str]) -> Optional[str]:
    """The conversion whose rules match the most of the event codes, if one does."""
    return best_conversion_type(score_conversion_types(codes))


def sample_event_codes(
    meds_reader_path: str, num_subjects: int = NUM_SAMPLE_SUBJECTS
) -> List[str]:
    """The event codes of the first subjects of a meds_reader database."""
    codes: List[str] = []
    with meds_reader.SubjectDatabase(meds_reader_path) as database:
        for subject_id in itertools.islice(iter(database), num_subjects):
            codes.extend(event.code for event in database[subject_id].events)
    return codes


def resolve_meds_conversion_type(
    data_args: DataTrainingArguments, num_subjects: int = NUM_SAMPLE_SUBJECTS
) -> str:
    """
    Set the conversion type of data_args to the one that fits its MEDS dataset.

    The dataset is the meds_reader database in data_args.data_folder. A conversion
    type other than the default was asked for and stays, with a warning if the data
    looks like another. Returns the conversion type that is used.
    """
    requested = data_args.meds_to_cehrbert_conversion_type
    if hasattr(requested, "name"):  # an enum member
        requested = requested.name
    folder = os.path.expanduser(data_args.data_folder)
    scores = score_conversion_types(sample_event_codes(folder, num_subjects))
    detected = best_conversion_type(scores)
    LOG.info(
        "Events that match the visit rules of each MEDS conversion (%s subjects): %s",
        num_subjects,
        scores,
    )

    if requested != DEFAULT_CONVERSION_TYPE:
        if detected is not None and detected != requested:
            LOG.warning(
                "meds_to_cehrbert_conversion_type is %s but the data looks like %s; "
                "using %s as asked",
                requested,
                detected,
                requested,
            )
        return requested
    if detected is None:
        LOG.warning(
            "No MEDS conversion matches the events of %s (%s); using the default %s. "
            "Set meds_to_cehrbert_conversion_type if the dataset has another shape.",
            data_args.data_folder,
            scores,
            DEFAULT_CONVERSION_TYPE,
        )
        return requested
    if detected != requested:
        LOG.info("Using the MEDS conversion %s, which matches the data", detected)
        data_args.meds_to_cehrbert_conversion_type = detected
    return detected
