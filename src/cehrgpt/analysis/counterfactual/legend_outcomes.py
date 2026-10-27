#!/usr/bin/env python3
"""
The outcome definitions of the LEGEND study, bundled with this package.

Every outcome of the OHDSI LEGEND study (https://github.com/OHDSI/Legend) is a cohort definition
that the SQL files in ``inst/sql/sql_server`` spell out.  The hypertension outcomes and negative
controls are bundled as data in ``legend_hypertension/`` (``outcomes.json`` and
``negative_controls.csv``; ``source.json`` names the LEGEND version).  How they were read from the
LEGEND repository, and how to read them again, is written down in
``legend_hypertension/README.md``; there is no code that parses LEGEND.

What this module does with the bundled data:

* ``read_outcomes`` and ``negative_controls`` load them: the concept sets of each outcome (the
  concepts that count as they are, those that count with all their descendants, and the excluded
  ones), the domain and concept sets of the criteria that select the events (condition, death,
  visit, ...) and the correlated criteria (for example "during an inpatient or ER visit").
* ``resolve_outcomes`` turns the concept sets into the concepts that mark an event in the token
  sequences of CEHR-GPT (descendants expanded with the vocabulary, exclusions removed) and says which
  outcomes can be evaluated that way.  The result, ``outcome_definitions.json``, is what the
  extraction and hazard ratio steps read with ``--outcome_definitions_path``.  An outcome is
  supported when its events are records of a concept set (condition, observation, procedure),
  optionally during an inpatient or ER visit, or the death record; outcomes that need a confirming
  second record, a drug or a measured value are listed with the reason and left out.  Two outcomes
  (see ``PARTIAL_DEFINITIONS``) have one branch that depends on a measured value; the other branch
  is kept and the dropped one is recorded in the ``approximation`` of the definition.

Usage
-----
python legend_outcomes.py --vocab_path /path/to/omop_vocab --output_dir /path/to/outcome_definitions
"""

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

BUNDLED_DIR = Path(__file__).resolve().parent / "legend_hypertension"
BUNDLED_INDICATION = "Hypertension"
LEGEND_REPOSITORY = "https://github.com/OHDSI/Legend"
_VISIT_DOMAIN = "Visit Occurrence"


@dataclass
class Codeset:
    """
    A concept set.  ``include`` holds the concepts that count only as themselves,
    ``include_descendants`` those that count with all their descendants; the ``exclude``
    pair is the same for the concepts that are removed again.
    """

    codeset_id: int
    include: List[int] = field(default_factory=list)
    include_descendants: List[int] = field(default_factory=list)
    exclude: List[int] = field(default_factory=list)
    exclude_descendants: List[int] = field(default_factory=list)


@dataclass
class Criterion:
    domain: str
    codeset_ids: List[int]


@dataclass
class OutcomeDefinition:
    name: str
    codesets: Dict[int, Codeset]
    primary: List[Criterion]
    correlated: List[Criterion]
    cohort_id: Optional[int] = None
    atlas_id: Optional[int] = None

    @property
    def visit_codeset_ids(self) -> List[int]:
        """Concept sets of visit criteria that correlate with a primary event of another domain."""
        if all(c.domain == _VISIT_DOMAIN for c in self.primary):
            return []
        return [i for c in self.correlated if c.domain == _VISIT_DOMAIN for i in c.codeset_ids]

    @property
    def is_condition_outcome(self) -> bool:
        """Condition records only, optionally restricted to the visits of a visit concept set."""
        return (
            all(c.domain == "Condition Occurrence" for c in self.primary)
            and all(c.domain == _VISIT_DOMAIN for c in self.correlated)
        )

    def primary_codesets(self) -> List[Codeset]:
        return [self.codesets[i] for c in self.primary for i in c.codeset_ids if i in self.codesets]


def outcome_from_dict(entry: Dict) -> OutcomeDefinition:
    return OutcomeDefinition(
        name=entry["name"],
        codesets={int(i): Codeset(**cs) for i, cs in entry["codesets"].items()},
        primary=[Criterion(**c) for c in entry["primary"]],
        correlated=[Criterion(**c) for c in entry["correlated"]],
        cohort_id=entry["cohort_id"],
        atlas_id=entry["atlas_id"],
    )


def read_outcomes() -> List[OutcomeDefinition]:
    """The bundled hypertension outcomes, with the ids of LEGEND's ``OutcomesOfInterest.csv``."""
    entries = json.loads((BUNDLED_DIR / "outcomes.json").read_text())
    return [outcome_from_dict(entry) for entry in entries]


def negative_controls() -> List[Tuple[int, int, str]]:
    """``(concept_id, cohort_id, name)`` of the bundled negative control outcomes."""
    with open(BUNDLED_DIR / "negative_controls.csv") as f:
        return [(int(r["concept_id"]), int(r["cohort_id"]), r["name"]) for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Concepts of an outcome in the token sequences
# ---------------------------------------------------------------------------

# Domains whose records are tokens of a concept set in the sequences of CEHR-GPT
TOKEN_DOMAINS = {"Condition Occurrence", "Observation", "Procedure Occurrence"}
DEATH_TOKEN = "[DEATH]"

# Outcomes with a branch that needs a measured value, which the sequences do not carry: the
# concept sets of the branches that are kept, and what is left out.
PARTIAL_DEFINITIONS: Dict[str, Tuple[List[int], str]] = {
    "Rhabdomyolysis": (
        [10],
        "left out: muscle disorder records with a creatine kinase above 5 x the upper limit of normal",
    ),
    "Hyperkalemia": (
        [0],
        "left out: potassium measurements above 5.6 mmol/L",
    ),
}
PRIMARY = "primary"
NEGATIVE_CONTROL = "negative_control"


def label_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


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


def codeset_concept_ids(codesets: List[Codeset]) -> List[int]:
    return sorted(
        {
            c
            for cs in codesets
            for c in cs.include + cs.include_descendants + cs.exclude + cs.exclude_descendants
        }
    )


def resolve_codeset(codeset: Codeset, descendants: Dict[int, Set[int]]) -> Set[str]:
    """The concepts of a concept set: includes with their descendants, minus the excluded."""
    included = set(codeset.include)
    excluded = set(codeset.exclude)
    for ancestor in codeset.include_descendants:
        included |= descendants[ancestor]
    for ancestor in codeset.exclude_descendants:
        excluded |= descendants[ancestor]
    return {str(c) for c in included - excluded}


@dataclass
class ResolvedOutcome:
    label: str
    name: str
    role: str
    concepts: List[str]
    inpatient_only: bool
    supported: bool
    cohort_id: Optional[int] = None
    unsupported_reason: Optional[str] = None
    approximation: Optional[str] = None


def resolve_outcome(outcome: OutcomeDefinition, descendants: Dict[int, Set[int]]) -> ResolvedOutcome:
    """
    The concepts that mark an event of *outcome*, and whether the outcome can be evaluated on
    token sequences (see the module docstring).
    """

    def union(ids: List[int]) -> List[str]:
        concepts: Set[str] = set()
        for i in ids:
            concepts |= resolve_codeset(outcome.codesets[i], descendants)
        return sorted(concepts)

    def result(concepts, inpatient_only=False, reason=None, approximation=None) -> ResolvedOutcome:
        return ResolvedOutcome(
            label=label_of(outcome.name),
            name=outcome.name,
            role=PRIMARY,
            concepts=concepts,
            inpatient_only=inpatient_only,
            supported=reason is None,
            cohort_id=outcome.cohort_id,
            unsupported_reason=reason,
            approximation=approximation,
        )

    if outcome.name in PARTIAL_DEFINITIONS:
        kept_ids, left_out = PARTIAL_DEFINITIONS[outcome.name]
        return result(
            union(kept_ids), inpatient_only=bool(outcome.visit_codeset_ids), approximation=left_out
        )

    primary_domains = {c.domain for c in outcome.primary}
    correlated_domains = {c.domain for c in outcome.correlated}
    primary_ids = [i for c in outcome.primary for i in c.codeset_ids]
    correlated_ids = [i for c in outcome.correlated for i in c.codeset_ids]

    if primary_domains == {"Death"} and not primary_ids and not outcome.correlated:
        return result([DEATH_TOKEN])
    if primary_domains == {"Death"}:
        return result([], reason="death with a cause: the cause is not part of the death record")
    if primary_domains == {_VISIT_DOMAIN}:
        # An inpatient or ER visit that contains a record of a concept set
        record_ids = [
            i for c in outcome.correlated if c.domain in TOKEN_DOMAINS for i in c.codeset_ids
        ]
        if record_ids and correlated_domains <= TOKEN_DOMAINS | {_VISIT_DOMAIN}:
            return result(union(record_ids), inpatient_only=True)
        return result([], reason="visit without a condition record to find in the sequence")
    unsupported_domains = primary_domains - TOKEN_DOMAINS
    if unsupported_domains:
        return result([], reason=f"{', '.join(sorted(unsupported_domains))} criteria cannot be matched on tokens")
    if correlated_domains - {_VISIT_DOMAIN}:
        return result(
            [],
            reason="needs a confirming record ("
            + ", ".join(sorted(correlated_domains - {_VISIT_DOMAIN}))
            + ")",
        )
    return result(union(primary_ids), inpatient_only=bool(outcome.visit_codeset_ids))


def resolve_negative_control(
    concept_id: int, cohort_id: int, name: str, descendants: Dict[int, Set[int]]
) -> ResolvedOutcome:
    """A negative control is the first condition record of the concept or a descendant."""
    return ResolvedOutcome(
        label=str(concept_id),
        name=name,
        role=NEGATIVE_CONTROL,
        concepts=sorted(str(c) for c in descendants[concept_id]),
        inpatient_only=False,
        supported=True,
        cohort_id=cohort_id,
    )


def resolve_outcomes(
    outcomes: List[OutcomeDefinition], controls: List[Tuple[int, int, str]], vocab_path: str
) -> List[ResolvedOutcome]:
    ids = codeset_concept_ids([cs for o in outcomes for cs in o.codesets.values()])
    descendants = descendants_of(vocab_path, ids + [c[0] for c in controls])
    return [resolve_outcome(o, descendants) for o in outcomes] + [
        resolve_negative_control(concept_id, cohort_id, name, descendants)
        for concept_id, cohort_id, name in controls
    ]


def write_resolved(resolved: List[ResolvedOutcome], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in resolved], indent=1))


def load_outcome_definitions(path: str) -> List[Dict]:
    """
    The supported outcomes of an ``outcome_definitions.json`` as dicts with ``label``, ``role``,
    ``concepts`` and ``inpatient_only``.  The outcomes that cannot be evaluated are reported.
    """
    entries = json.loads(Path(path).read_text())
    skipped = [e for e in entries if not e["supported"]]
    if skipped:
        print(f"{len(skipped)} outcome(s) in {path} cannot be evaluated and are left out:")
        for entry in skipped:
            print(f"  {entry['name']}: {entry['unsupported_reason']}")
    partial = [e for e in entries if e["supported"] and e.get("approximation")]
    if partial:
        print(f"{len(partial)} outcome(s) in {path} use only part of their definition:")
        for entry in partial:
            print(f"  {entry['name']}: {entry['approximation']}")
    labels = [e["label"] for e in entries if e["supported"]]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path} has duplicate outcome labels")
    return [e for e in entries if e["supported"]]


def definitions_fingerprint(path: str) -> str:
    """Content hash recorded by the extraction, so a changed definition file is noticed."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="The LEGEND outcome concepts, resolved for a vocabulary")
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="OMOP vocabulary root with concept_ancestor/ (to expand the descendants)",
    )
    parser.add_argument("--output_dir", required=True, help="Where outcome_definitions.json is written")
    args = parser.parse_args()
    outcomes = read_outcomes()
    controls = negative_controls()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_outcomes(outcomes, controls, args.vocab_path)
    write_resolved(resolved, output_dir / "outcome_definitions.json")
    supported = [r for r in resolved if r.supported]
    print(f"{len(outcomes)} outcomes, {len(controls)} negative controls → {output_dir}")
    print(f"{len(supported)} of {len(resolved)} can be evaluated on token sequences:")
    for r in resolved:
        if not r.supported:
            print(f"  not supported: {r.name}: {r.unsupported_reason}")
        elif r.approximation:
            print(f"  partial: {r.name}: {r.approximation}")
    inpatient = [r.name for r in supported if r.inpatient_only and r.role == PRIMARY]
    print(f"{len(inpatient)} supported outcomes must occur in an inpatient or ER visit:")
    for name in inpatient:
        print(f"  {name}")


if __name__ == "__main__":
    main()
