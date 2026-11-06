#!/usr/bin/env python3
"""
Generated against observed hazard ratios, side by side.

Step 4 writes two summaries into the results folder: ``hazard_ratio_summary.csv`` (the generated
trajectories of the two arms) and ``observed_hazard_ratio_summary.csv`` (the real patients of the two
groups, an unadjusted Cox model).  This joins them per outcome into
``generated_vs_observed_hazard_ratios.csv`` and prints the table: the primary outcomes of the study
first, then its safety outcomes, both in the order of the publication (``primary_outcomes`` of
study.json, then the order of reference_results.csv), then the other outcomes of the study and the
negative controls.

Usage
-----
python compare_hazard_ratios.py --results_dir /path/to/results --study legend_hypertension \\
    --arm_a thiazide_or_thiazide_like_diuretics --arm_b ace_inhibitors
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

from cehrgpt.analysis.counterfactual.outcome_definitions import NEGATIVE_CONTROL, read_outcomes
from cehrgpt.analysis.counterfactual.study_definition import DEFAULT_STUDY, label_of, outcome_order

GENERATED_FILE = "hazard_ratio_summary.csv"
OBSERVED_FILE = "observed_hazard_ratio_summary.csv"
COMPARISON_FILE = "generated_vs_observed_hazard_ratios.csv"

GROUP_TITLES = {
    "primary": "Primary outcomes",
    "safety": "Safety outcomes",
    "other": "Other outcomes",
    "negative_control": "Negative controls",
}


def ordered_outcomes(study: str = DEFAULT_STUDY) -> List[Tuple[str, str, str]]:
    """(label, name, group) of every outcome of the study, in the order to show them."""
    order = outcome_order(study)
    outcomes = {label_of(o.name): o for o in read_outcomes(study)}
    primary = [label_of(n) for n in order["primary"]]
    safety = [label_of(n) for n in order["safety"]]
    labels = [(label, "primary") for label in primary] + [(label, "safety") for label in safety]
    labels += [(label, "other") for label in outcomes if label not in primary + safety]
    return [
        (label, outcomes[label].name.strip(), "negative_control" if outcomes[label].kind == NEGATIVE_CONTROL else group)
        for label, group in labels
        if label in outcomes  # an outcome of the publication that the study does not have is not in the results
    ]


def _side(summary: pl.DataFrame, prefix: str, arm_a: str, arm_b: str) -> pl.DataFrame:
    """The columns of one summary for the join, named with ``prefix``."""
    return summary.select(
        pl.col("outcome_concept_id").cast(pl.Utf8).alias("label"),
        pl.col(f"n_events_{arm_a}").alias(f"{prefix}_events_{arm_a}"),
        pl.col(f"n_events_{arm_b}").alias(f"{prefix}_events_{arm_b}"),
        pl.col("hr").alias(f"{prefix}_hr"),
        pl.col("hr_lower_95").alias(f"{prefix}_ci_lower"),
        pl.col("hr_upper_95").alias(f"{prefix}_ci_upper"),
    )


def build_comparison(
    generated: pl.DataFrame,
    observed: pl.DataFrame,
    arm_a: str,
    arm_b: str,
    outcomes: List[Tuple[str, str, str]],
) -> pl.DataFrame:
    """
    One row per outcome of ``outcomes`` (label, name, group) that either summary has, in that order:
    the events and the hazard ratio (arm_a vs arm_b) of both, their ratio, and whether the 95% interval
    of the generated hazard ratio contains the observed one.
    """
    order = pl.DataFrame(
        {
            "label": [o[0] for o in outcomes],
            "outcome": [o[1] for o in outcomes],
            "group": [o[2] for o in outcomes],
            "position": list(range(len(outcomes))),
        }
    )
    known = set(order["label"].to_list())
    # outcomes given by concept id are not outcomes of the study; they go after it, in the order of the file
    extra = [
        label
        for label in dict.fromkeys(
            generated["outcome_concept_id"].cast(pl.Utf8).to_list() + observed["outcome_concept_id"].cast(pl.Utf8).to_list()
        )
        if label not in known
    ]
    if extra:
        order = pl.concat(
            [
                order,
                pl.DataFrame(
                    {
                        "label": extra,
                        "outcome": extra,
                        "group": ["other"] * len(extra),
                        "position": list(range(len(outcomes), len(outcomes) + len(extra))),
                    }
                ),
            ]
        )
    table = (
        order.join(_side(generated, "generated", arm_a, arm_b), on="label", how="left")
        .join(_side(observed, "observed", arm_a, arm_b), on="label", how="left")
        .filter(pl.col("generated_hr").is_not_null() | pl.col("observed_hr").is_not_null())
        .sort("position")
        .with_columns(
            (pl.col("generated_hr") / pl.col("observed_hr")).alias("generated_over_observed"),
            (
                (pl.col("generated_ci_lower") <= pl.col("observed_hr"))
                & (pl.col("observed_hr") <= pl.col("generated_ci_upper"))
            ).alias("generated_ci_contains_observed"),
        )
        .drop("position", "label")
    )
    return table


def _hr(hr: Optional[float], lower: Optional[float], upper: Optional[float]) -> str:
    if hr is None or not math.isfinite(hr):
        return "-"
    interval = "?" if lower is None or not math.isfinite(lower) else f"{lower:.2f}"
    interval += "-" + ("?" if upper is None or not math.isfinite(upper) else f"{upper:.2f}")
    return f"{hr:.2f} ({interval})"


def _events(a: Optional[int], b: Optional[int]) -> str:
    return "-" if a is None or b is None else f"{a}/{b}"


def format_comparison(table: pl.DataFrame, arm_a: str, arm_b: str) -> str:
    """The table for the console: a block per group, primary outcomes first."""
    width = max([len("outcome")] + [len(o) for o in table["outcome"].to_list()])
    header = (
        f"{'outcome':<{width}}  {'obs events':>10}  {'obs HR (95% CI)':>20}  "
        f"{'gen events':>10}  {'gen HR (95% CI)':>20}  {'gen/obs':>7}  obs in gen CI"
    )
    lines = [f"Hazard ratio of {arm_a} vs {arm_b} (events: {arm_a}/{arm_b}); obs = observed patients, unadjusted; gen = generated trajectories"]
    for group, title in GROUP_TITLES.items():
        rows = table.filter(pl.col("group") == group).to_dicts()
        if not rows:
            continue
        if group == "negative_control":
            covered = sum(1 for r in rows if r["generated_ci_contains_observed"])
            comparable = sum(1 for r in rows if r["generated_ci_contains_observed"] is not None)
            lines += ["", f"{title}: {len(rows)} outcomes, {covered}/{comparable} generated intervals contain the observed hazard ratio ({COMPARISON_FILE} has each)"]
            continue
        lines += ["", f"{title}", header, "-" * len(header)]
        for r in rows:
            ratio = r["generated_over_observed"]
            inside = r["generated_ci_contains_observed"]
            lines.append(
                f"{r['outcome']:<{width}}  "
                f"{_events(r[f'observed_events_{arm_a}'], r[f'observed_events_{arm_b}']):>10}  "
                f"{_hr(r['observed_hr'], r['observed_ci_lower'], r['observed_ci_upper']):>20}  "
                f"{_events(r[f'generated_events_{arm_a}'], r[f'generated_events_{arm_b}']):>10}  "
                f"{_hr(r['generated_hr'], r['generated_ci_lower'], r['generated_ci_upper']):>20}  "
                f"{'-' if ratio is None or not math.isfinite(ratio) else f'{ratio:.2f}':>7}  "
                f"{'-' if inside is None else ('yes' if inside else 'no')}"
            )
    return "\n".join(lines)


def compare(results_dir: str, arm_a: str, arm_b: str, study: str = DEFAULT_STUDY) -> Optional[pl.DataFrame]:
    """Write the comparison into ``results_dir`` and print it; None if one of the summaries is missing."""
    folder = Path(results_dir)
    missing = [f for f in (GENERATED_FILE, OBSERVED_FILE) if not (folder / f).is_file()]
    if missing:
        print(f"No comparison of generated and observed hazard ratios: {', '.join(missing)} not in {folder}")
        return None
    table = build_comparison(
        pl.read_csv(folder / GENERATED_FILE, infer_schema_length=None),
        pl.read_csv(folder / OBSERVED_FILE, infer_schema_length=None),
        arm_a,
        arm_b,
        ordered_outcomes(study),
    )
    table.write_csv(folder / COMPARISON_FILE)
    print(format_comparison(table, arm_a, arm_b))
    print(f"\nComparison written to {folder / COMPARISON_FILE}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Generated against observed hazard ratios, side by side")
    parser.add_argument("--results_dir", required=True, help="Folder with hazard_ratio_summary.csv and observed_hazard_ratio_summary.csv")
    parser.add_argument("--arm_a", required=True, help="The arm of the numerator, as in the summaries (n_events_<arm_a>)")
    parser.add_argument("--arm_b", required=True)
    parser.add_argument("--study", default=DEFAULT_STUDY, help="Name of a bundled study, or a study folder (for the order of the outcomes)")
    args = parser.parse_args()
    compare(args.results_dir, args.arm_a, args.arm_b, args.study)


if __name__ == "__main__":
    main()
