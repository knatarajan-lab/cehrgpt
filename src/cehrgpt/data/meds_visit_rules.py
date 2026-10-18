"""
The rules that turn the events of a MEDS dataset into visits: which events are a visit, an admission, an ED admission and
a discharge. They depend on how the dataset was made and they change, so they are data and not code:

  * the dataset declares them in ``<meds_reader database>/metadata/visit_rules.yaml`` (the file comes from the
    ``metadata`` folder of the MEDS dataset, which meds_reader_convert copies);
  * a yaml file given with ``--meds_visit_rules_file`` overrides them: the keys that it has replace the ones of the
    dataset, the others stay.

    strategy: visit_id            # how the events are put into visits, see STRATEGIES
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

The strategies are the two ways of cehrbert to find the visits; the rules above are the same for both:
  * ``visit_id``: the events carry the id of their visit (a ``visit_id`` property), as in a dataset made from OMOP.
  * ``admission_window``: there are no visit ids; the visits are built from the admission and discharge events and the
    time between them, as in MIMIC-IV.
"""

import dataclasses
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import (
    MedsToBertMimic4,
    MedsToCehrbertOMOP,
)
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)
from transformers.utils import logging

LOG = logging.get_logger("transformers")

METADATA_FILE = os.path.join("metadata", "visit_rules.yaml")
STRATEGY_VISIT_ID = "visit_id"
STRATEGY_ADMISSION_WINDOW = "admission_window"
STRATEGIES = (STRATEGY_VISIT_ID, STRATEGY_ADMISSION_WINDOW)
PATTERN_LISTS = ("visit_events", "ed_admission_events", "admission_events", "discharge_events")
KEYS = ("strategy",) + PATTERN_LISTS + ("text_events",)


class VisitRulesError(ValueError):
    """The rules are not valid."""


@dataclasses.dataclass(frozen=True)
class TextEventRule:
    code: str
    pattern: str
    labels: List[str]


@dataclasses.dataclass(frozen=True)
class VisitRules:
    strategy: str
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
        strategy = values.get("strategy")
        if strategy not in STRATEGIES:
            raise VisitRulesError(f"The visit rules ({source}) need a strategy, one of {list(STRATEGIES)}; got {strategy!r}")
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
        return cls(strategy=strategy, text_events=text_events, source=source, **lists)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"strategy": self.strategy}
        for key in PATTERN_LISTS:
            result[key] = list(getattr(self, key))
        if self.text_events:
            result["text_events"] = [dataclasses.asdict(rule) for rule in self.text_events]
        return result


def load_rules_file(path: str) -> Dict[str, Any]:
    """The keys of a visit rules file (yaml, which includes json)."""
    path = os.path.expanduser(path)
    with open(path) as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict):
        raise VisitRulesError(f"{path} must be a mapping of keys to values")
    return values


def resolve_visit_rules(data_folder: str, rules_file: Optional[str] = None) -> Optional[VisitRules]:
    """
    The visit rules of a dataset: the ones declared in its metadata, with the keys of rules_file on top.

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
        sources.append(os.path.expanduser(rules_file))
    rules = VisitRules.from_dict({**(declared or {}), **(override or {})}, source=" + ".join(sources))
    LOG.info("Visit rules from %s: %s", rules.source, rules.to_dict())
    return rules


class _ConfiguredRules:
    """The rule hooks of a cehrbert conversion, from VisitRules instead of from code."""

    def _configure(self, rules: VisitRules):
        self._visit_rules = rules

    def _create_visit_matching_rules(self) -> List[str]:
        return list(self._visit_rules.visit_events)

    def _create_ed_admission_matching_rules(self) -> List[str]:
        return list(self._visit_rules.ed_admission_events)

    def _create_admission_matching_rules(self) -> List[str]:
        return list(self._visit_rules.admission_events)

    def _create_discharge_matching_rules(self) -> List[str]:
        return list(self._visit_rules.discharge_events)

    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        return [
            EventConversionRule(code=rule.code, parsing_pattern=re.compile(rule.pattern), mapped_event_labels=list(rule.labels))
            for rule in self._visit_rules.text_events
        ]


class VisitIdConversion(_ConfiguredRules, MedsToCehrbertOMOP):
    """The visits of the events that carry a visit id (cehrbert's OMOP algorithm), with configured rules."""

    def __init__(self, rules: VisitRules, **kwargs):
        self._configure(rules)
        super().__init__(**kwargs)


class AdmissionWindowConversion(_ConfiguredRules, MedsToBertMimic4):
    """The visits built from admissions and discharges (cehrbert's MIMIC-IV algorithm), with configured rules."""

    def __init__(self, rules: VisitRules, default_visit_id: int = 1, **kwargs):
        self._configure(rules)
        super().__init__(default_visit_id, **kwargs)


def build_conversion(rules: VisitRules, default_visit_id: int = 1, **kwargs) -> MedsToCehrBertConversion:
    """The cehrbert conversion that follows the rules."""
    if rules.strategy == STRATEGY_VISIT_ID:
        return VisitIdConversion(rules, **kwargs)
    return AdmissionWindowConversion(rules, default_visit_id=default_visit_id, **kwargs)


def rules_of_conversion(conversion: MedsToCehrBertConversion) -> VisitRules:
    """The rules that a conversion of cehrbert has in its code, e.g. to write them out as a rules file."""
    strategy = STRATEGY_ADMISSION_WINDOW if isinstance(conversion, MedsToBertMimic4) else STRATEGY_VISIT_ID
    return VisitRules(
        strategy=strategy,
        visit_events=list(conversion.get_other_visit_matching_rules()),
        ed_admission_events=list(conversion.get_ed_admission_matching_rules()),
        admission_events=list(conversion.get_admission_matching_rules()),
        discharge_events=list(conversion.get_discharge_matching_rules()),
        text_events=[
            TextEventRule(rule.code, rule.parsing_pattern.pattern, list(rule.mapped_event_labels))
            for rule in conversion._text_event_numeric_event_map.values()
        ],
        source=type(conversion).__name__,
    )
