#!/usr/bin/env python3
"""
The study definition of LEGEND-HTN: drug classes, their ingredients and the cohort rules.

The OHDSI LEGEND study (https://github.com/OHDSI/Legend) defines its exposures in
``inst/settings/ExposuresOfInterest.csv`` and the rules of its cohorts in
``CreateExposureCohortsHypertension.sql``, ``R/CreateExposureCohorts.R`` and
``cmAnalysisListHypertension.json``.  They are bundled as data in ``legend_hypertension/``
(``exposures.csv`` and ``study.json``, next to the outcomes of ``legend_outcomes.py``;
``source.json`` names the LEGEND version).  ``legend_hypertension/README.md`` describes how they were
read from the LEGEND repository; this module has no code that parses LEGEND.

The analysis scripts take the two drug classes to compare by name instead of carrying lists of
concept ids::

    _STUDY_ENV="$(python legend_study.py --target 'ACE inhibitors' \\
                                         --comparator 'Thiazide or thiazide-like diuretics')"
    eval "${_STUDY_ENV}"

prints the shell variables (concept ids of both classes, the exclusion list of all other
antihypertensive ingredients, the indication and the windows) that the scripts use.  ``--list``
shows the available classes.
"""

import argparse
import csv
import json
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional

from cehrgpt.analysis.counterfactual.legend_outcomes import BUNDLED_DIR, label_of

DEFAULT_TYPE = "Drug class"
INGREDIENT_TYPE = "Drug"
EXPOSURE_TYPES = ("Drug class", "Drug major class", "Drug")


@dataclass
class Exposure:
    type: str
    name: str
    cohort_id: int
    concept_id: Optional[int]  # only drugs have one; a class is its list of ingredients
    included_concept_ids: List[int]

    @property
    def ingredients(self) -> List[int]:
        """The ingredients of a class, or the drug itself."""
        return self.included_concept_ids or [self.concept_id]


# ---------------------------------------------------------------------------
# Using the bundled definition
# ---------------------------------------------------------------------------

def load_study() -> Dict:
    return json.loads((BUNDLED_DIR / "study.json").read_text())


def load_exposures() -> List[Exposure]:
    with open(BUNDLED_DIR / "exposures.csv") as f:
        return [
            Exposure(
                type=r["type"],
                name=r["name"],
                cohort_id=int(r["cohort_id"]),
                concept_id=int(r["concept_id"]) if r["concept_id"] else None,
                included_concept_ids=[int(x) for x in r["included_concept_ids"].split(";") if x],
            )
            for r in csv.DictReader(f)
        ]


def find_exposure(name: str, exposure_type: str = DEFAULT_TYPE) -> Exposure:
    """
    The exposure called *name* among those of *exposure_type*: an exact name (not case
    sensitive), or else a part of exactly one name.
    """
    candidates = [e for e in load_exposures() if e.type == exposure_type]
    wanted = name.strip().lower()
    exact = [e for e in candidates if e.name.lower() == wanted]
    matches = exact or [e for e in candidates if wanted in e.name.lower()]
    if len(matches) != 1:
        options = ", ".join(sorted(e.name for e in candidates))
        problem = "no" if not matches else "more than one"
        raise ValueError(f"{problem} {exposure_type!r} matches {name!r}; choose one of: {options}")
    return matches[0]


def antihypertensive_ingredients() -> List[int]:
    """All the ingredients of the study, in the order of the exposure file."""
    return [e.concept_id for e in load_exposures() if e.type == INGREDIENT_TYPE]


def resolve_pair(target: str, comparator: str, exposure_type: str = DEFAULT_TYPE) -> Dict:
    """
    The two exposures to compare, and everything the scripts derive from them: their ingredients
    and the exclusion list, which holds every other ingredient of the study (a patient that took
    one of them before the index date is not a new user of the target or the comparator).
    """
    first = find_exposure(target, exposure_type)
    second = find_exposure(comparator, exposure_type)
    shared = set(first.ingredients) & set(second.ingredients)
    if shared:
        raise ValueError(
            f"{first.name!r} and {second.name!r} share ingredients {sorted(shared)}; "
            "compare two exposures that do not overlap"
        )
    used = set(first.ingredients) | set(second.ingredients)
    return {
        "target": first,
        "comparator": second,
        "exclusion": [i for i in antihypertensive_ingredients() if i not in used],
    }


def environment(target: str, comparator: str, exposure_type: str = DEFAULT_TYPE) -> List[str]:
    """
    Shell assignments for the analysis scripts.  Values the user may want to change
    (windows, the labels of the arms, the indication) keep a value that is already set.
    """
    study = load_study()
    pair = resolve_pair(target, comparator, exposure_type)

    def ids(values: List[int]) -> str:
        return ",".join(str(v) for v in values)

    def keep(name: str, value, colon: bool = True) -> str:
        operator = ":-" if colon else "-"
        return f"{name}=\"${{{name}{operator}{value}}}\""

    lines = [
        f"STUDY_NAME={shlex.quote(study['name'])}",
        f"TARGET_NAME={shlex.quote(pair['target'].name)}",
        keep("TARGET_LABEL", label_of(pair["target"].name)),
        f"TARGET_CONCEPT_IDS={shlex.quote(ids(pair['target'].ingredients))}",
        f"COMPARATOR_NAME={shlex.quote(pair['comparator'].name)}",
        keep("COMPARATOR_LABEL", label_of(pair["comparator"].name)),
        f"COMPARATOR_CONCEPT_IDS={shlex.quote(ids(pair['comparator'].ingredients))}",
        # every other antihypertensive ingredient: exposure to it before the index date, or in the
        # monotherapy window, takes the patient out of the cohort
        keep("EXCLUSION_CONCEPT_IDS", ids(pair["exclusion"]), colon=False),
        keep("POPULATION_CONCEPT_IDS", ids(study["indication"]["concept_ids"]), colon=False),
        keep("POPULATION_WINDOW_DAYS", study["indication"]["window_days"]),
        keep("MIN_PRIOR_OBSERVATION_DAYS", study["washout_days"]),
        keep("ERA_GAP_DAYS", study["era_gap_days"]),
        keep("MONOTHERAPY_WINDOW_DAYS", study["monotherapy_window_days"], colon=False),
        keep("MIN_DAYS_AT_RISK", study["time_at_risk"]["min_days_at_risk"]),
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="The LEGEND-HTN study definition for the analysis scripts")
    parser.add_argument("--target", help="Name of the target drug class, e.g. 'ACE inhibitors'")
    parser.add_argument("--comparator", help="Name of the comparator drug class")
    parser.add_argument(
        "--exposure_type", default=DEFAULT_TYPE, choices=EXPOSURE_TYPES,
        help="Level of the two exposures (default: %(default)s)",
    )
    parser.add_argument("--list", action="store_true", help="List the available exposures and exit")
    args = parser.parse_args()

    if args.list:
        for exposure_type in EXPOSURE_TYPES:
            print(f"{exposure_type}:")
            for e in load_exposures():
                if e.type == exposure_type:
                    print(f"  {e.name}  ({len(e.ingredients)} ingredient(s))")
        return
    if not (args.target and args.comparator):
        parser.error("give --target and --comparator, or --list")
    print("\n".join(environment(args.target, args.comparator, args.exposure_type)))


if __name__ == "__main__":
    main()
