#!/usr/bin/env python3
"""
The outcomes of a study: concept sets, and whether an event has to happen during an inpatient or ER
visit.  This module reads them and turns them into the events in the token sequences of CEHR-GPT.

A study (``study_definition.py``) holds its outcomes in ``outcomes.csv`` (format in
``studies/README.md``).  One row is a concept set::

    name,kind,concept_ids,exact_concept_ids,exclude_concept_ids,exclude_exact_concept_ids,inpatient_or_er,note
    Heart failure,primary,316139,,,,true,
    Rash,negative_control,140648,,,,,

* ``concept_ids`` count with all their descendants, ``exact_concept_ids`` only as themselves; the two
  ``exclude`` columns take concepts out again.  Several ids are separated by ``;``.
* ``inpatient_or_er`` (``true``/``false``, default ``false``): the event only counts during an
  inpatient or ER visit.
* ``kind`` is ``primary`` (default) or ``negative_control`` (an outcome the exposures are not
  expected to affect; its hazard ratio should be 1).
* The rows of one outcome are its concept sets: an event of any of them counts, and each has its own
  exclusions.  ``concept_ids`` ``[DEATH]`` is the death of the patient.

``resolve_outcomes`` expands the descendants with the vocabulary and writes ``outcome_definitions.json``
(the concepts that mark an event, per outcome), which the extraction and hazard ratio steps read with
``--outcome_definitions_path``.

Usage
-----
python outcome_definitions.py --study legend_hypertension --vocab_path /path/to/omop_vocab \\
    --output_dir /path/to/outcome_definitions
"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from cehrgpt.analysis.counterfactual.study_definition import DEFAULT_STUDY, label_of, study_dir

DEATH_TOKEN = "[DEATH]"

PRIMARY = "primary"
NEGATIVE_CONTROL = "negative_control"
KINDS = (PRIMARY, NEGATIVE_CONTROL)

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0", ""}


@dataclass
class ConceptSet:
    """
    ``include_descendants`` count with all their descendants, ``include`` only as themselves; the
    ``exclude`` pair is the same for the concepts that are taken out again.
    """

    include_descendants: List[int] = field(default_factory=list)
    include: List[int] = field(default_factory=list)
    exclude_descendants: List[int] = field(default_factory=list)
    exclude: List[int] = field(default_factory=list)

    def concept_ids(self) -> Set[int]:
        return set(self.include_descendants + self.include + self.exclude_descendants + self.exclude)


@dataclass
class Outcome:
    name: str
    kind: str
    concept_sets: List[ConceptSet]
    inpatient_or_er: bool = False
    death: bool = False
    note: str = ""


def _ids(cell: Optional[str], row_number: int, column: str) -> List[int]:
    try:
        return [int(token) for token in (cell or "").replace(",", ";").split(";") if token.strip()]
    except ValueError:
        raise ValueError(f"outcomes.csv row {row_number}: {column} has to hold concept ids, got {cell!r}")


def _flag(cell: Optional[str], row_number: int) -> bool:
    value = (cell or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"outcomes.csv row {row_number}: inpatient_or_er has to be true or false, got {cell!r}")


def parse_outcomes(rows: List[Dict[str, str]]) -> List[Outcome]:
    """The outcomes of the rows of ``outcomes.csv``: rows with the same name are one outcome."""
    outcomes: Dict[str, Outcome] = {}
    for row_number, row in enumerate(rows, start=2):  # the header is row 1
        name = (row.get("name") or "").strip()
        if not name:
            raise ValueError(f"outcomes.csv row {row_number}: the name is empty")
        kind = (row.get("kind") or PRIMARY).strip().lower()
        if kind not in KINDS:
            raise ValueError(f"outcomes.csv row {row_number}: kind has to be one of {', '.join(KINDS)}, got {kind!r}")
        inpatient = _flag(row.get("inpatient_or_er"), row_number)
        death = (row.get("concept_ids") or "").strip() == DEATH_TOKEN
        concept_set = ConceptSet(
            include_descendants=[] if death else _ids(row.get("concept_ids"), row_number, "concept_ids"),
            include=_ids(row.get("exact_concept_ids"), row_number, "exact_concept_ids"),
            exclude_descendants=_ids(row.get("exclude_concept_ids"), row_number, "exclude_concept_ids"),
            exclude=_ids(row.get("exclude_exact_concept_ids"), row_number, "exclude_exact_concept_ids"),
        )
        if death:
            if inpatient or concept_set.include or concept_set.exclude or concept_set.exclude_descendants:
                raise ValueError(f"outcomes.csv row {row_number}: {DEATH_TOKEN} takes no other column")
        elif not (concept_set.include_descendants or concept_set.include):
            raise ValueError(f"outcomes.csv row {row_number}: {name} has no concept ids")
        outcome = outcomes.get(name)
        if outcome is None:
            outcomes[name] = Outcome(
                name, kind, [] if death else [concept_set], inpatient, death, (row.get("note") or "").strip()
            )
            continue
        if (outcome.kind, outcome.inpatient_or_er, outcome.death) != (kind, inpatient, death):
            raise ValueError(
                f"outcomes.csv row {row_number}: the rows of {name} differ in kind, inpatient_or_er or {DEATH_TOKEN}"
            )
        if not death:
            outcome.concept_sets.append(concept_set)
    labels: Dict[str, str] = {}
    for name in outcomes:
        label = label_of(name)
        if label in labels:
            raise ValueError(f"outcomes {labels[label]!r} and {name!r} have the same label {label!r}")
        labels[label] = name
    return list(outcomes.values())


def read_outcomes(study: str = DEFAULT_STUDY) -> List[Outcome]:
    """The outcomes of a study (the name of a bundled study, or a study folder)."""
    with open(study_dir(study) / "outcomes.csv", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"name", "concept_ids"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"outcomes.csv lacks the column(s) {', '.join(sorted(missing))}")
        return parse_outcomes(list(reader))


# ---------------------------------------------------------------------------
# Concepts of an outcome in the token sequences
# ---------------------------------------------------------------------------


def descendants_of(vocab_path: str, concept_ids) -> Dict[int, Set[int]]:
    """``{ancestor: {descendants, including the ancestor}}`` from ``concept_ancestor``."""
    import polars as pl

    wanted = sorted(set(concept_ids))
    frame = (
        pl.scan_parquet(os.path.join(vocab_path, "concept_ancestor", "*.parquet"))
        .filter(pl.col("ancestor_concept_id").is_in(wanted))
        .select("ancestor_concept_id", "descendant_concept_id")
        .collect()
    )
    result: Dict[int, Set[int]] = {c: {c} for c in wanted}
    for ancestor, descendant in frame.iter_rows():
        result[int(ancestor)].add(int(descendant))
    return result


def resolve_concept_set(concept_set: ConceptSet, descendants: Dict[int, Set[int]]) -> Set[str]:
    """The concepts of a concept set: includes with their descendants, minus the excluded."""
    included = set(concept_set.include)
    excluded = set(concept_set.exclude)
    for ancestor in concept_set.include_descendants:
        included |= descendants[ancestor]
    for ancestor in concept_set.exclude_descendants:
        excluded |= descendants[ancestor]
    return {str(c) for c in included - excluded}


@dataclass
class ResolvedOutcome:
    label: str
    name: str
    role: str
    concepts: List[str]
    inpatient_only: bool
    note: str = ""


def resolve_outcome(outcome: Outcome, descendants: Dict[int, Set[int]]) -> ResolvedOutcome:
    """The concepts that mark an event of *outcome*."""
    if outcome.death:
        concepts = [DEATH_TOKEN]
    else:
        union: Set[str] = set()
        for concept_set in outcome.concept_sets:
            union |= resolve_concept_set(concept_set, descendants)
        concepts = sorted(union)
        if not concepts:
            raise ValueError(f"outcome {outcome.name!r}: the exclusions leave no concept")
    return ResolvedOutcome(
        label=label_of(outcome.name),
        name=outcome.name,
        role=outcome.kind,
        concepts=concepts,
        inpatient_only=outcome.inpatient_or_er,
        note=outcome.note,
    )


def resolve_outcomes(outcomes: List[Outcome], vocab_path: str) -> List[ResolvedOutcome]:
    ids = sorted({c for o in outcomes for s in o.concept_sets for c in s.concept_ids()})
    descendants = descendants_of(vocab_path, ids)
    return [resolve_outcome(o, descendants) for o in outcomes]


def write_resolved(resolved: List[ResolvedOutcome], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in resolved], indent=1))


def load_outcome_definitions(path: str) -> List[Dict]:
    """
    The outcomes of an ``outcome_definitions.json`` as dicts with ``label``, ``role``, ``concepts``
    and ``inpatient_only``.
    """
    entries = json.loads(Path(path).read_text())
    labels = [e["label"] for e in entries]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path} has duplicate outcome labels")
    return entries


def definitions_fingerprint(path: str) -> str:
    """Content hash recorded by the extraction, so a changed definition file is noticed."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="The outcomes of a study, resolved for a vocabulary")
    parser.add_argument("--study", default=DEFAULT_STUDY, help="Name of a bundled study, or a study folder")
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="OMOP vocabulary root with concept_ancestor/ (to expand the descendants)",
    )
    parser.add_argument("--output_dir", required=True, help="Where outcome_definitions.json is written")
    args = parser.parse_args()
    outcomes = read_outcomes(args.study)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_outcomes(outcomes, args.vocab_path)
    write_resolved(resolved, output_dir / "outcome_definitions.json")
    controls = [r for r in resolved if r.role == NEGATIVE_CONTROL]
    print(f"{len(resolved) - len(controls)} outcomes, {len(controls)} negative controls → {output_dir}")
    inpatient = [r.name for r in resolved if r.inpatient_only and r.role == PRIMARY]
    print(f"{len(inpatient)} outcomes must occur in an inpatient or ER visit:")
    for name in inpatient:
        print(f"  {name}")


if __name__ == "__main__":
    main()
