#!/usr/bin/env python3
"""
Read the outcome definitions of the LEGEND study from its generated SQL.

Every outcome of the OHDSI LEGEND study (https://github.com/OHDSI/Legend) is a cohort
definition that the SQL files in ``inst/sql/sql_server`` spell out.  This module parses
those files, so the concepts of an outcome come from the study itself rather than from a
hand-typed list:

* the concept sets (``#Codesets``): the concepts that are included as they are, the ones
  included with all their descendants, and the excluded ones;
* the criteria that select the events: the domain (condition, death, measurement, ...) and
  the concept set of the primary criteria, plus the concept sets of the correlated criteria
  (for example "during an inpatient or ER visit").

``read_outcomes`` lists the outcomes of one indication (``Hypertension``) in the order and
with the ids of ``OutcomesOfInterest.csv``.  ``negative_controls`` reads ``NegativeControls.csv``.

``resolve_outcomes`` turns the concept sets into the concepts that mark an event in the token
sequences of CEHR-GPT (descendants expanded, exclusions removed) and says which outcomes can
be evaluated that way.  The result, ``outcome_definitions.json``, is what the extraction and
hazard ratio steps read with ``--outcome_definitions_path``.  An outcome is supported when its
events are records of a concept set (condition, observation, procedure), optionally during an
inpatient or ER visit, or the death record; outcomes that need a confirming second record, a
drug or a measured value are listed with the reason and left out.  Two outcomes (see
``PARTIAL_DEFINITIONS``) have one branch that depends on a measured value; the other branch is
kept and the dropped one is recorded in the ``approximation`` of the definition.

Usage
-----
python legend_outcomes.py \\
    --legend_dir  /path/to/Legend \\
    --indication  Hypertension \\
    --output_dir  /path/to/outcome_definitions
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

SQL_FOLDER = Path("inst") / "sql" / "sql_server"
SETTINGS_FOLDER = Path("inst") / "settings"

_CODESET = re.compile(
    r"INSERT INTO #Codesets \(codeset_id, concept_id\)\s*SELECT\s+(\d+)\s+as\s+codeset_id,"
    r".*?select distinct I\.concept_id FROM(.*?)\)\s*C;",
    re.S,
)
_LISTED = re.compile(r"CONCEPT\s+where\s+concept_id\s+in\s*\(([^)]*)\)", re.S)
_WITH_DESCENDANTS = re.compile(r"ancestor_concept_id\s+in\s*\(([^)]*)\)", re.S)
_CRITERIA = re.compile(r"-- Begin ([A-Za-z ]+?) Criteria(.*?)-- End \1 Criteria", re.S)
_CODESET_USE = re.compile(r"codesets\.codeset_id\s*=\s*(\d+)")
_VISIT_DOMAIN = "Visit Occurrence"


def _ids(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


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


def parse_codesets(sql: str) -> Dict[int, Codeset]:
    codesets: Dict[int, Codeset] = {}
    for match in _CODESET.finditer(sql):
        codeset_id, body = int(match.group(1)), match.group(2)
        included, _, excluded = body.partition("LEFT JOIN")
        codeset = Codeset(codeset_id)
        for part, plain, with_descendants in (
            (included, codeset.include, codeset.include_descendants),
            (excluded, codeset.exclude, codeset.exclude_descendants),
        ):
            if not part:
                continue
            listed = [i for m in _LISTED.findall(part) for i in _ids(m)]
            descendants = [i for m in _WITH_DESCENDANTS.findall(part) for i in _ids(m)]
            with_descendants.extend(sorted(set(descendants)))
            plain.extend(sorted(set(listed) - set(descendants)))
        codesets[codeset_id] = codeset
    return codesets


def _criteria(text: str) -> List[Criterion]:
    criteria = []
    for domain, block in _CRITERIA.findall(text):
        if domain == "Correlated":
            continue
        used = []
        for codeset_id in map(int, _CODESET_USE.findall(block)):
            if codeset_id not in used:
                used.append(codeset_id)
        criteria.append(Criterion(domain.strip(), used))
    return criteria


def parse_outcome(name: str, sql: str) -> OutcomeDefinition:
    start = sql.index("-- Begin Primary Events")
    end = sql.index("-- End Primary Events")
    # Correlated criteria wrap domain criteria of their own, inside or after the primary events
    correlated: List[Criterion] = []
    for match in re.finditer(r"-- Begin Correlated Criteria(.*?)-- End Correlated Criteria", sql, re.S):
        for criterion in _criteria(match.group(1)):
            if criterion not in correlated:
                correlated.append(criterion)
    return OutcomeDefinition(
        name=name,
        codesets=parse_codesets(sql),
        primary=_criteria(sql[start:end]),
        correlated=correlated,
    )


def read_outcomes(legend_dir: Path, indication: str = "Hypertension") -> List[OutcomeDefinition]:
    """The outcomes of *indication*, with the ids of ``OutcomesOfInterest.csv``."""
    legend_dir = Path(legend_dir)
    with open(legend_dir / SETTINGS_FOLDER / "OutcomesOfInterest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["indicationId"] == indication]
    outcomes = []
    for row in rows:
        sql_path = legend_dir / SQL_FOLDER / f"{row['name']}.sql"
        outcome = parse_outcome(row["name"], sql_path.read_text())
        outcome.cohort_id = int(row["cohortId"])
        outcome.atlas_id = int(row["atlasId"])
        outcomes.append(outcome)
    return outcomes


def negative_controls(legend_dir: Path, indication: str = "Hypertension") -> List[Tuple[int, int, str]]:
    """``(concept_id, cohort_id, name)`` of the negative control outcomes of *indication*."""
    with open(Path(legend_dir) / SETTINGS_FOLDER / "NegativeControls.csv") as f:
        return [
            (int(r["conceptId"]), int(r["cohortId"]), r["name"])
            for r in csv.DictReader(f)
            if r["indicationId"] == indication
        ]


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


def write_definitions(outcomes: List[OutcomeDefinition], controls, output_dir: Path) -> None:
    """One row per concept of every concept set, one row per criterion, and the controls."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "outcome_concepts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort_id", "outcome", "codeset_id", "role", "include_descendants", "concept_id"])
        for o in outcomes:
            for cs in o.codesets.values():
                for role, flag, ids in (
                    ("include", False, cs.include),
                    ("include", True, cs.include_descendants),
                    ("exclude", False, cs.exclude),
                    ("exclude", True, cs.exclude_descendants),
                ):
                    for concept_id in ids:
                        writer.writerow([o.cohort_id, o.name, cs.codeset_id, role, int(flag), concept_id])
    with open(output_dir / "outcome_criteria.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort_id", "outcome", "kind", "domain", "codeset_ids"])
        for o in outcomes:
            for kind, criteria in (("primary", o.primary), ("correlated", o.correlated)):
                for c in criteria:
                    writer.writerow([o.cohort_id, o.name, kind, c.domain, ";".join(map(str, c.codeset_ids))])
    with open(output_dir / "negative_controls.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort_id", "concept_id", "name"])
        for concept_id, cohort_id, name in controls:
            writer.writerow([cohort_id, concept_id, name])
    with open(output_dir / "outcomes.json", "w") as f:
        json.dump(
            [
                {
                    "name": o.name,
                    "cohort_id": o.cohort_id,
                    "atlas_id": o.atlas_id,
                    "codesets": {str(i): cs.__dict__ for i, cs in o.codesets.items()},
                    "primary": [c.__dict__ for c in o.primary],
                    "correlated": [c.__dict__ for c in o.correlated],
                }
                for o in outcomes
            ],
            f,
            indent=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the LEGEND outcome concepts from its SQL")
    parser.add_argument("--legend_dir", required=True, help="Checkout of OHDSI/Legend")
    parser.add_argument("--indication", default="Hypertension")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--vocab_path",
        default=None,
        help="OMOP vocabulary root with concept_ancestor/: also writes outcome_definitions.json",
    )
    args = parser.parse_args()
    outcomes = read_outcomes(Path(args.legend_dir), args.indication)
    controls = negative_controls(Path(args.legend_dir), args.indication)
    write_definitions(outcomes, controls, Path(args.output_dir))
    print(f"{len(outcomes)} outcomes, {len(controls)} negative controls → {args.output_dir}")
    if args.vocab_path:
        resolved = resolve_outcomes(outcomes, controls, args.vocab_path)
        write_resolved(resolved, Path(args.output_dir) / "outcome_definitions.json")
        supported = [r for r in resolved if r.supported]
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
