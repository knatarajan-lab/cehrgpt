"""
The rules that turn the events of a MEDS dataset into visits: which events are a visit, an admission, an ED admission and
a discharge. They depend on how the dataset was made and they change, so they are data and not code:

  * the dataset declares them in ``<meds_reader database>/metadata/visit_rules.yaml`` (the file comes from the
    ``metadata`` folder of the MEDS dataset, which meds_reader_convert copies);
  * CEHR-GPT's ``defaults.yaml`` provides the standard OMOP and MIMIC patterns;
  * a yaml file given with ``--meds_visit_rules_file`` adds dataset-specific patterns. ``omop`` and ``mimic4`` stand
    for the two compatibility files that come with cehrgpt (visit_rules/).

The two files are the rules of the two kinds of datasets that cehrgpt is used on: MEDS made from OMOP (``omop``) and
MEDS made from MIMIC-IV with MEDS_transforms (``mimic4``). A dataset that declares nothing is read with the one that
matches ``--meds_to_cehrbert_conversion_type``.

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

Visit assignment is detected automatically. Events with visit ids (or explicit visit events) use those visit boundaries;
otherwise visits are built from the admission and discharge events and the time between them.
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

METADATA_FILE = os.path.join("metadata", "visit_rules.yaml")
PACKAGED_RULES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visit_rules")
DEFAULT_RULES_PATH = os.path.join(PACKAGED_RULES_FOLDER, "defaults.yaml")
# The old conversion names remain only as a backwards-compatible way to choose a packaged YAML file when a dataset
# does not declare rules. They are not instantiated by this module.
PACKAGED_RULES = {"omop": "MedsToCehrbertOMOP", "mimic4": "MedsToBertMimic4"}
PATTERN_LISTS = ("visit_events", "ed_admission_events", "admission_events", "discharge_events")
KEYS = PATTERN_LISTS + ("text_events",)


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
        return cls(text_events=text_events, source=source, **lists)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in PATTERN_LISTS:
            result[key] = list(getattr(self, key))
        if self.text_events:
            result["text_events"] = [dataclasses.asdict(rule) for rule in self.text_events]
        return result


def packaged_rules_path(name: str) -> str:
    """The path of the rules file that comes with cehrgpt under this name ("omop" or "mimic4")."""
    if name not in PACKAGED_RULES:
        raise VisitRulesError(f"There are no packaged visit rules called {name!r}; they are {sorted(PACKAGED_RULES)}")
    return os.path.join(PACKAGED_RULES_FOLDER, f"{name}.yaml")


def packaged_rules_for_conversion(conversion_type: str) -> VisitRules:
    """
    The packaged rules that take the place of a conversion of cehrbert that is named by its class
    (meds_to_cehrbert_conversion_type), for a dataset that does not declare rules.
    """
    for name, conversion in PACKAGED_RULES.items():
        if conversion == conversion_type:
            path = packaged_rules_path(name)
            return VisitRules.from_dict(
                merge_rule_values(load_rules_file(DEFAULT_RULES_PATH), load_rules_file(path)),
                source=f"{DEFAULT_RULES_PATH} + {path}",
            )
    raise VisitRulesError(
        f"The dataset declares no visit rules and {conversion_type!r} is not a conversion with packaged rules "
        f"({sorted(PACKAGED_RULES.values())}). Declare the rules of the dataset in {METADATA_FILE} or pass "
        "--meds_visit_rules_file."
    )


def load_rules_file(path: str) -> Dict[str, Any]:
    """The keys of a visit rules file (yaml, which includes json), or of a packaged one by its name."""
    if path in PACKAGED_RULES:
        path = packaged_rules_path(path)
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


def resolve_visit_rules(data_folder: str, rules_file: Optional[str] = None) -> Optional[VisitRules]:
    """
    The visit rules of a dataset: standard defaults plus patterns declared in its metadata and rules_file.

    Returns None when there are neither, which is the case of a dataset that does not say how its visits are made.
    """
    metadata_path = os.path.join(os.path.expanduser(data_folder), METADATA_FILE)
    declared = load_rules_file(metadata_path) if os.path.exists(metadata_path) else None
    override = load_rules_file(rules_file) if rules_file else None
    if declared is None and override is None:
        return None
    sources = []
    if declared is not None:
        sources.append(metadata_path)
    if override is not None:
        sources.append(packaged_rules_path(rules_file) if rules_file in PACKAGED_RULES else os.path.expanduser(rules_file))
    sources.insert(0, DEFAULT_RULES_PATH)
    rules = VisitRules.from_dict(
        merge_rule_values(load_rules_file(DEFAULT_RULES_PATH), declared, override),
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
