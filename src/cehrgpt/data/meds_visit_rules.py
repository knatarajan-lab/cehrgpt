"""
The rules that turn the events of a MEDS dataset into visits: which events are a visit, an admission, an ED admission and
a discharge. They depend on how the dataset was made and they change, so they are data and not code:

  * CEHR-GPT's ``defaults.yaml`` provides the standard OMOP and MIMIC patterns;
  * a yaml file of the user, given with ``--meds_visit_rules_file``, is applied on top of them for a dataset that
    codes its visits differently.

The MEDS dataset itself carries nothing for this: its ``metadata`` folder is the standard one, and cehrgpt neither
writes nor reads a rules file there.

    visit_events: ["Visit/"]      # the event that is the visit: its code is the visit type, its time the start
    ed_admission_events: ["Visit/ER"]
    admission_events: ["Visit/IP", "Visit/ERIP"]
    discharge_events: ["CMS Place of Service/12", "PCORNet/Generic-NI"]
    text_events:                  # optional: an event whose text value holds numbers, e.g. a blood pressure
      - code: "Blood Pressure"
        pattern: '(\\d+)/(\\d+)'
        labels: ["Systolic Blood Pressure", "Diastolic Blood Pressure"]

The codes are regular expressions that are matched from the start of the code of an event (re.match), so a code is
written as it is, and ``"Visit/"`` matches every code that starts with it.

For ``visit_id`` data, the time of an event is always the ground truth. An event that has the id of a visit but happened
more than a day before its start or after its end (a problem list entry from years ago, a drug that is recorded long
after the visit) is taken out of that visit and put where its own time is.

``max_stay_gap_days`` (default 3): a stay with an admission but no discharge event ends at its last event once this many
days pass without any, instead of lasting until the next admission or the end of the record. Stays that have a discharge
event are not affected, however long they are.

Visits are built from the visit ids that the events carry and, for the events that have none, from the admission and
discharge events and the time between them (see ``meds_dataset._build_patient_blocks``).
"""

import dataclasses
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)
from transformers.utils import logging

LOG = logging.get_logger("transformers")

DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visit_rules", "defaults.yaml")
PATTERN_LISTS = ("visit_events", "ed_admission_events", "admission_events", "discharge_events")
KEYS = PATTERN_LISTS + ("text_events", "max_stay_gap_days")
DEFAULT_MAX_STAY_GAP_DAYS = 3.0


class VisitRulesError(ValueError):
    """The rules are not valid."""


@dataclasses.dataclass(frozen=True)
class TextEventRule:
    code: str
    pattern: str
    labels: List[str]


@dataclasses.dataclass(frozen=True)
class VisitRules:
    visit_events: List[str] = dataclasses.field(default_factory=list)
    ed_admission_events: List[str] = dataclasses.field(default_factory=list)
    admission_events: List[str] = dataclasses.field(default_factory=list)
    discharge_events: List[str] = dataclasses.field(default_factory=list)
    text_events: List[TextEventRule] = dataclasses.field(default_factory=list)
    # a stay that has no discharge event ends after this many days without an event
    max_stay_gap_days: float = DEFAULT_MAX_STAY_GAP_DAYS
    # where the rules came from, for the log and for error messages
    source: str = ""

    @classmethod
    def from_dict(cls, values: Dict[str, Any], source: str = "") -> "VisitRules":
        unknown = sorted(set(values) - set(KEYS))
        if unknown:
            raise VisitRulesError(f"Unknown keys in the visit rules ({source}): {unknown}; the keys are {list(KEYS)}")
        lists = {}
        for key in PATTERN_LISTS:
            patterns = values.get(key) or []
            if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
                raise VisitRulesError(f"{key} ({source}) must be a list of codes, got {patterns!r}")
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise VisitRulesError(f"{key} ({source}): {pattern!r} is not a valid regular expression: {error}")
            lists[key] = list(patterns)
        text_events = []
        for entry in values.get("text_events") or []:
            if not isinstance(entry, dict) or set(entry) != {"code", "pattern", "labels"}:
                raise VisitRulesError(f"A text event ({source}) needs exactly code, pattern and labels, got {entry!r}")
            try:
                groups = re.compile(entry["pattern"]).groups
            except re.error as error:
                raise VisitRulesError(f"text_events ({source}): {entry['pattern']!r} is not a valid regular expression: {error}")
            if groups != len(entry["labels"]):
                raise VisitRulesError(
                    f"text_events ({source}): the pattern of {entry['code']!r} has {groups} groups but there are "
                    f"{len(entry['labels'])} labels"
                )
            text_events.append(TextEventRule(entry["code"], entry["pattern"], list(entry["labels"])))
        gap = values.get("max_stay_gap_days", DEFAULT_MAX_STAY_GAP_DAYS)
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or gap <= 0:
            raise VisitRulesError(f"max_stay_gap_days ({source}) must be a number of days above 0, got {gap!r}")
        return cls(text_events=text_events, max_stay_gap_days=float(gap), source=source, **lists)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in PATTERN_LISTS:
            result[key] = list(getattr(self, key))
        if self.text_events:
            result["text_events"] = [dataclasses.asdict(rule) for rule in self.text_events]
        result["max_stay_gap_days"] = self.max_stay_gap_days
        return result


def load_rules_file(path: str) -> Dict[str, Any]:
    """The keys of a visit rules file (YAML, which includes JSON)."""
    path = os.path.expanduser(path)
    with open(path) as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict):
        raise VisitRulesError(f"{path} must be a mapping of keys to values")
    return values


def merge_rule_values(*layers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge rule files so later files add patterns; a repeated text-event code replaces its earlier definition."""
    merged: Dict[str, Any] = {}
    for layer in layers:
        if layer is None:
            continue
        for key, value in layer.items():
            if key in PATTERN_LISTS:
                if not isinstance(value, list):
                    merged[key] = value
                    continue
                existing = merged.get(key, [])
                if not isinstance(existing, list):
                    existing = []
                combined = list(existing)
                for pattern in value:
                    if pattern not in combined:
                        combined.append(pattern)
                merged[key] = combined
            elif key == "text_events" and isinstance(value, list):
                existing = merged.get(key, [])
                if not isinstance(existing, list):
                    existing = []
                replacements = {entry.get("code") for entry in value if isinstance(entry, dict)}
                merged[key] = [
                    entry
                    for entry in existing
                    if not isinstance(entry, dict) or entry.get("code") not in replacements
                ] + list(value)
            else:
                merged[key] = value
    return merged


def resolve_visit_rules(rules_file: Optional[str] = None) -> VisitRules:
    """
    The visit rules: the standard defaults, with the yaml file of the user (rules_file) applied on top.

    Without a file the annotated OMOP and MIMIC defaults are used as they are.
    """
    override = load_rules_file(rules_file) if rules_file else None
    sources = [DEFAULT_RULES_PATH]
    if override is not None:
        sources.append(os.path.expanduser(rules_file))
    rules = VisitRules.from_dict(
        merge_rule_values(load_rules_file(DEFAULT_RULES_PATH), override),
        source=" + ".join(sources),
    )
    LOG.info("Visit rules from %s: %s", rules.source, rules.to_dict())
    return rules


class YamlDrivenConversion(MedsToCehrBertConversion):
    """One generic CEHR-BERT conversion whose complete behavior is supplied by YAML rules."""

    def __init__(self, rules: VisitRules, default_visit_id: int = 1, **kwargs):
        self.rules = rules
        self.default_visit_id = default_visit_id
        # Event time is always the ground truth for visit_id data.
        self.disconnect_problem_list_events = True
        super().__init__(**kwargs)

    def _create_visit_matching_rules(self) -> List[str]:
        return list(self.rules.visit_events)

    def _create_ed_admission_matching_rules(self) -> List[str]:
        return list(self.rules.ed_admission_events)

    def _create_admission_matching_rules(self) -> List[str]:
        return list(self.rules.admission_events)

    def _create_discharge_matching_rules(self) -> List[str]:
        return list(self.rules.discharge_events)

    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        return [
            EventConversionRule(
                code=rule.code,
                parsing_pattern=re.compile(rule.pattern),
                mapped_event_labels=list(rule.labels),
            )
            for rule in self.rules.text_events
        ]


def build_conversion(rules: VisitRules, default_visit_id: int = 1, **kwargs) -> MedsToCehrBertConversion:
    """Build the generic conversion from YAML; no dataset-specific conversion class is used."""
    kwargs.pop("disconnect_problem_list_events", None)
    return YamlDrivenConversion(rules, default_visit_id=default_visit_id, **kwargs)
