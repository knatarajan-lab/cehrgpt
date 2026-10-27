#!/usr/bin/env python3
"""
The preference score plot: do the two groups of patients overlap?

The swap step (``create_drug_swap_context.py --preference_score_trim``) saves, for every patient, the
model's probabilities of the two drug classes at the index position and the preference score

    P(comparator class) / (P(source class) + P(comparator class)),

where *source* is the class the patient actually started.  This tool draws, for both groups, the
distribution of the **preference for the target**, P(target) / (P(target) + P(comparator)):

* for the patients who started the target the score file holds the preference for the comparator,
  so the preference for the target is ``1 - score``;
* for the patients who started the comparator it is the score itself.

The two histograms (density) overlap where patients could plausibly have received either drug; the
area where they overlap is reported as the overlap coefficient (1 means identical distributions,
0 means no patient in common).  The dashed lines are the retention band ``[trim, 1 - trim]`` of the
overlap check: patients outside it are dropped from every arm.  With only the scores of the target
patients (the ``target_initiators`` design) a single distribution is drawn.

Usage
-----
python overlap_plot.py \\
    --target_scores     swap_contexts/target_pts_overlap_scores.parquet \\
    --comparator_scores swap_contexts/comparator_pts_overlap_scores.parquet \\
    --trim 0.05 --target_name "ACE inhibitors" --comparator_name "Thiazides" \\
    --output results/preference_score_overlap.png

Next to the plot the tool writes ``<output stem>_summary.csv`` with, for each group, the number of
patients, how many were retained, and the quantiles of the preference for the target.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

BINS = 40


def preference_for_target(scores: pl.DataFrame, started_target: bool) -> np.ndarray:
    """The preference for the target of each scored patient (the unscorable ones are left out)."""
    values = scores["preference_score"].drop_nulls().to_numpy().astype(float)
    return 1.0 - values if started_target else values


def overlap_coefficient(first: np.ndarray, second: np.ndarray, bins: int = BINS) -> float:
    """The area under the minimum of the two density histograms on [0, 1]."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    density_a, _ = np.histogram(first, bins=edges, density=True)
    density_b, _ = np.histogram(second, bins=edges, density=True)
    return float(np.minimum(density_a, density_b).sum() * (1.0 / bins))


def summarize(scores: pl.DataFrame, started_target: bool, group: str) -> Dict:
    values = preference_for_target(scores, started_target)
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(values, q)) if len(values) else float("nan")
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }
    return {
        "group": group,
        "n_patients": len(scores),
        "n_scored": len(values),
        "n_retained": int(scores["retained"].sum()),
        "share_retained": float(scores["retained"].mean()) if len(scores) else float("nan"),
        **quantiles,
    }


def plot_preference_scores(
    target_scores: pl.DataFrame,
    comparator_scores: Optional[pl.DataFrame],
    trim: Optional[float],
    output_path: Path,
    target_name: str = "target",
    comparator_name: str = "comparator",
) -> pl.DataFrame:
    """
    Draw the plot to *output_path* (a png) and write the summary next to it.  Returns the summary.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(f"Started {target_name}", target_scores, True, "#1f77b4")]
    if comparator_scores is not None:
        groups.append((f"Started {comparator_name}", comparator_scores, False, "#d62728"))

    figure, axis = plt.subplots(figsize=(8, 4.8))
    edges = np.linspace(0.0, 1.0, BINS + 1)
    rows: List[Dict] = []
    values: List[np.ndarray] = []
    for label, scores, started_target, color in groups:
        group_values = preference_for_target(scores, started_target)
        values.append(group_values)
        rows.append(summarize(scores, started_target, label))
        retained = f"{rows[-1]['share_retained']:.0%} retained" if trim is not None else ""
        axis.hist(
            group_values, bins=edges, density=True, alpha=0.45, color=color,
            label=f"{label} (n={len(scores):,}{', ' + retained if retained else ''})",
        )
        axis.hist(group_values, bins=edges, density=True, histtype="step", color=color, linewidth=1.5)

    if trim is not None:
        for edge in (trim, 1.0 - trim):
            axis.axvline(edge, color="black", linestyle="--", linewidth=1)
        axis.axvspan(0.0, trim, color="grey", alpha=0.15)
        axis.axvspan(1.0 - trim, 1.0, color="grey", alpha=0.15)
        axis.text(0.5, 0.97, f"retained: [{trim:g}, {1 - trim:g}]", transform=axis.transAxes,
                  ha="center", va="top", fontsize=9)

    title = "Preference score overlap"
    if len(values) == 2:
        title += f" (overlap coefficient {overlap_coefficient(values[0], values[1]):.2f})"
    axis.set_title(title)
    axis.set_xlabel(f"Preference for {target_name}:  P({target_name}) / (P({target_name}) + P({comparator_name}))")
    axis.set_ylabel("Density")
    axis.set_xlim(0.0, 1.0)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9), fontsize=9, frameon=False)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    summary = pl.DataFrame(rows)
    if len(values) == 2:
        summary = summary.with_columns(
            pl.lit(overlap_coefficient(values[0], values[1])).alias("overlap_coefficient")
        )
    summary.write_csv(str(output_path.with_name(output_path.stem + "_summary.csv")))
    return summary


def _read(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the overlap of the preference scores of two groups")
    parser.add_argument("--target_scores", required=True,
                        help="Scores parquet of the patients who started the target (swap step)")
    parser.add_argument("--comparator_scores", default=None,
                        help="Scores parquet of the patients who started the comparator (symmetric design)")
    parser.add_argument("--trim", type=float, default=None, help="The preference score trim of the overlap check")
    parser.add_argument("--target_name", default="target")
    parser.add_argument("--comparator_name", default="comparator")
    parser.add_argument("--output", required=True, help="Path of the png")
    args = parser.parse_args()
    summary = plot_preference_scores(
        _read(args.target_scores),
        _read(args.comparator_scores) if args.comparator_scores else None,
        args.trim,
        Path(args.output),
        args.target_name,
        args.comparator_name,
    )
    print(summary)
    print(f"Plot → {args.output}")


if __name__ == "__main__":
    main()
