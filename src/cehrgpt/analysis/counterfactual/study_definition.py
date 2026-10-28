#!/usr/bin/env python3
"""
A study definition: the exposures to compare, the cohort rules and the outcomes of a study.

The analysis scripts do not know any study.  A study is a folder (see ``studies/README.md`` for the
files and their format) that is either bundled with the package, in ``studies/<name>``, or anywhere
else on disk:

* ``study.json``: the name, the levels of the exposures, the pair to compare by default, the
  indication, the washout, the exposure gap, the monotherapy window and the time at risk;
* ``exposures.csv``: the drugs and the classes of drugs of the study;
* ``outcomes.csv``: the outcomes and negative controls (see ``outcome_definitions.py``);
* ``reference_results.csv`` (optional): published results to compare with.  A result of the site
  that is replicated (``source`` other than ``Meta-analysis``, e.g. CUMC) is used where there is one.

This module resolves two exposures of the study by name into everything a run needs: their
ingredients, the exclusion list (every other drug of the study; a patient that took one of them
before the index date, or in the monotherapy window, is not a new user of the target or the
comparator), the indication and the windows.  ``environment`` prints them as shell assignments::

    _STUDY_ENV="$(python study_definition.py --study legend_hypertension \\
                      --target 'ACE inhibitors' --comparator 'Thiazide or thiazide-like diuretics')"
    eval "${_STUDY_ENV}"

Without ``--target`` and ``--comparator`` the pair of ``default_pair`` is used.  ``--list`` shows the
exposures of a study, ``--studies`` the bundled studies and ``--reference`` the reference results
for a pair.
"""

import argparse
import csv
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

STUDIES_DIR = Path(__file__).resolve().parent / "studies"
DEFAULT_STUDY = "legend_hypertension"
REQUIRED_FILES = ("study.json", "exposures.csv", "outcomes.csv")


def label_of(name: str) -> str:
    """A name as a label for file names and arms: lower case, letters and digits, underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def bundled_studies() -> List[str]:
    return sorted(p.name for p in STUDIES_DIR.iterdir() if (p / "study.json").is_file())


def study_dir(study: str = DEFAULT_STUDY) -> Path:
    """The folder of a study: the name of a bundled study, or the path of a study folder."""
    bundled = STUDIES_DIR / study
    folder = bundled if bundled.is_dir() else Path(study)
    if not folder.is_dir():
        raise ValueError(
            f"no study {study!r}: give the name of a bundled study ({', '.join(bundled_studies())}) or a folder"
        )
    missing = [f for f in REQUIRED_FILES if not (folder / f).is_file()]
    if missing:
        raise ValueError(f"the study folder {folder} lacks {', '.join(missing)} (see studies/README.md)")
    return folder


def load_study(study: str = DEFAULT_STUDY) -> Dict:
    return json.loads((study_dir(study) / "study.json").read_text())


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


def load_exposures(study: str = DEFAULT_STUDY) -> List[Exposure]:
    with open(study_dir(study) / "exposures.csv") as f:
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


def find_exposure(name: str, level: Optional[str] = None, study: str = DEFAULT_STUDY) -> Exposure:
    """
    The exposure called *name* among those of *level* (default: the default level of the
    study): an exact name (not case sensitive), or else a part of exactly one name.
    """
    level = level or load_study(study)["exposures"]["default_level"]
    candidates = [e for e in load_exposures(study) if e.type == level]
    wanted = name.strip().lower()
    exact = [e for e in candidates if e.name.lower() == wanted]
    matches = exact or [e for e in candidates if wanted in e.name.lower()]
    if len(matches) != 1:
        options = ", ".join(sorted(e.name for e in candidates))
        problem = "no" if not matches else "more than one"
        raise ValueError(f"{problem} {level!r} matches {name!r}; choose one of: {options}")
    return matches[0]


def drug_universe(study: str = DEFAULT_STUDY) -> List[int]:
    """All the drugs (ingredients) of the study, in the order of the exposure file."""
    drug_level = load_study(study)["exposures"]["drug_level"]
    return [e.concept_id for e in load_exposures(study) if e.type == drug_level]


def resolve_pair(
    target: Optional[str] = None,
    comparator: Optional[str] = None,
    level: Optional[str] = None,
    study: str = DEFAULT_STUDY,
) -> Dict:
    """
    The two exposures to compare (the default pair of the study when none is given), and the
    exclusion list: every other drug of the study.
    """
    default = load_study(study).get("default_pair") or {}
    target = target or default.get("target")
    comparator = comparator or default.get("comparator")
    if not (target and comparator):
        raise ValueError("give a target and a comparator; the study has no default pair")
    first = find_exposure(target, level, study)
    second = find_exposure(comparator, level, study)
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
        "exclusion": [i for i in drug_universe(study) if i not in used],
    }


def _invert(row: Dict) -> Dict:
    """The result of the opposite comparison: 1/hr, the interval inverted and swapped."""
    lower, upper = row["ci_lower"], row["ci_upper"]
    return {
        **row,
        "target": row["comparator"],
        "comparator": row["target"],
        "hr": 1 / row["hr"],
        "ci_lower": None if upper is None else 1 / upper,
        "ci_upper": None if lower is None else 1 / lower,
    }


def reference_results(pair: Dict, study: str = DEFAULT_STUDY, time_at_risk: Optional[str] = None) -> List[Dict]:
    """
    The reference results of the study for this pair, if it has any (``time_at_risk``: only the
    results of that design, ``on_treatment`` or ``intent_to_treat``).  The results of the opposite
    comparison are inverted.
    """
    path = study_dir(study) / "reference_results.csv"
    if not path.is_file():
        return []
    number = lambda cell: float(cell) if (cell or "").strip() else None
    with open(path, newline="") as f:
        rows = [
            {**r, "hr": float(r["hr"]), "ci_lower": number(r["ci_lower"]), "ci_upper": number(r["ci_upper"])}
            for r in csv.DictReader(f)
        ]
    target, comparator = pair["target"].name, pair["comparator"].name
    results = [r for r in rows if (r["target"], r["comparator"]) == (target, comparator)]
    results += [_invert(r) for r in rows if (r["target"], r["comparator"]) == (comparator, target)]
    if time_at_risk:
        results = [r for r in results if r["time_at_risk"] == time_at_risk]
    return results


def _interval(row: Dict) -> str:
    lower = "?" if row["ci_lower"] is None else f"{row['ci_lower']:.2f}"
    upper = "?" if row["ci_upper"] is None else f"{row['ci_upper']:.2f}"
    return f"{lower}–{upper}"


def reference_lines(pair: Dict, study: str = DEFAULT_STUDY, time_at_risk: Optional[str] = None) -> List[str]:
    """The results of the replicated site, and how many results of meta-analyses the file holds."""
    results = reference_results(pair, study, time_at_risk)
    if not results:
        return []
    site = [r for r in results if r["source"] != "Meta-analysis"]
    header = f"Reference results of {load_study(study)['name']} ({pair['target'].name} vs {pair['comparator'].name}):"
    lines = [header] + [
        f"  {r['outcome']:<36}: {r['hr']:.2f}  (95% CI {_interval(r)})  {r['source']}, "
        f"{r['time_at_risk'].replace('_', '-')}, PS-{r['adjustment']}"
        for r in site
    ]
    others = len(results) - len(site)
    if others:
        lines.append(f"  and {others} results of meta-analyses in reference_results.csv of the study")
    return lines


def environment(
    target: Optional[str] = None,
    comparator: Optional[str] = None,
    level: Optional[str] = None,
    study: str = DEFAULT_STUDY,
) -> List[str]:
    """
    Shell assignments for the analysis scripts.  Values the user may want to change
    (windows, the labels of the arms, the indication) keep a value that is already set.
    """
    folder = study_dir(study)
    definition = load_study(study)
    pair = resolve_pair(target, comparator, level, study)
    indication = definition.get("indication") or {}

    def ids(values: List[int]) -> str:
        return ",".join(str(v) for v in values)

    def keep(name: str, value, colon: bool = True) -> str:
        # the value is the default of the variable, in double quotes: it must need no quoting
        if not re.fullmatch(r"[A-Za-z0-9_,.\-]*", str(value)):
            raise ValueError(f"{name} would be {value!r}; only letters, digits and _ , . - are allowed")
        operator = ":-" if colon else "-"
        return f"{name}=\"${{{name}{operator}{value}}}\""

    time_at_risk = definition.get("time_at_risk", {})
    return [
        f"STUDY_NAME={shlex.quote(definition['name'])}",
        f"STUDY_DIR={shlex.quote(str(folder))}",
        keep("STUDY_LABEL", label_of(definition["name"])),
        f"TARGET_NAME={shlex.quote(pair['target'].name)}",
        keep("TARGET_LABEL", label_of(pair["target"].name)),
        f"TARGET_CONCEPT_IDS={shlex.quote(ids(pair['target'].ingredients))}",
        f"COMPARATOR_NAME={shlex.quote(pair['comparator'].name)}",
        keep("COMPARATOR_LABEL", label_of(pair["comparator"].name)),
        f"COMPARATOR_CONCEPT_IDS={shlex.quote(ids(pair['comparator'].ingredients))}",
        # every other drug of the study: exposure to it before the index date, or in the
        # monotherapy window, takes the patient out of the cohort
        keep("EXCLUSION_CONCEPT_IDS", ids(pair["exclusion"]), colon=False),
        keep("POPULATION_CONCEPT_IDS", ids(indication.get("concept_ids", [])), colon=False),
        keep("POPULATION_WINDOW_DAYS", indication.get("window_days", "")),
        keep("MIN_PRIOR_OBSERVATION_DAYS", definition.get("washout_days", 0)),
        keep("ERA_GAP_DAYS", definition.get("era_gap_days", 30)),
        keep("MONOTHERAPY_WINDOW_DAYS", definition.get("monotherapy_window_days") or "", colon=False),
        keep("MIN_DAYS_AT_RISK", time_at_risk.get("min_days_at_risk", 1)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="A study definition for the analysis scripts")
    parser.add_argument("--study", default=DEFAULT_STUDY, help="Name of a bundled study, or a study folder")
    parser.add_argument("--target", help="Name of the target exposure (default: the pair of the study)")
    parser.add_argument("--comparator", help="Name of the comparator exposure")
    parser.add_argument("--level", help="Level of the two exposures, e.g. 'Drug class' (default: that of the study)")
    parser.add_argument("--list", action="store_true", help="List the exposures of the study and exit")
    parser.add_argument("--studies", action="store_true", help="List the bundled studies and exit")
    parser.add_argument("--reference", action="store_true", help="Print the reference results for the pair")
    parser.add_argument(
        "--time_at_risk", choices=["on_treatment", "intent_to_treat"], help="With --reference: only this design"
    )
    args = parser.parse_args()

    if args.studies:
        print("\n".join(bundled_studies()))
        return
    if args.list:
        definition = load_study(args.study)
        for level in definition["exposures"]["levels"]:
            print(f"{level}:")
            for e in load_exposures(args.study):
                if e.type == level:
                    print(f"  {e.name}  ({len(e.ingredients)} ingredient(s))")
        return
    if args.reference:
        pair = resolve_pair(args.target, args.comparator, args.level, args.study)
        print("\n".join(reference_lines(pair, args.study, args.time_at_risk)))
        return
    print("\n".join(environment(args.target, args.comparator, args.level, args.study)))


if __name__ == "__main__":
    main()
