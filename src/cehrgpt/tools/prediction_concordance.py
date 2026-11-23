"""
Agreement of two sets of predictions (e.g. zero-shot against the linear probe) on the samples both have.

    python -m cehrgpt.tools.prediction_concordance \
        --reference <probe output_dir>/logistic/test_predictions \
        --candidate <zero-shot output_folder>/top_p10000/<task_name> --output_file concordance.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from cehrgpt.tools.bootstrap_ci import compute_metrics

KEY = ["subject_id", "prediction_time"]


def read_predictions(path: str, match_on: str = "exact") -> pl.DataFrame:
    """One row per (subject_id, prediction_time); a sample predicted twice gets the mean."""
    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    time = pl.col("prediction_time").cast(pl.Datetime("us"))
    return (
        pl.concat(
            pl.scan_parquet(file).select(
                "subject_id",
                time.dt.date() if match_on == "date" else time,
                pl.col("predicted_boolean_probability").alias("probability"),
                pl.col("boolean_value").cast(pl.Int8).alias("label"),
            )
            for file in files
        )
        .drop_nulls()
        .group_by(KEY)
        .agg(pl.col("probability").mean(), pl.col("label").max())
        .collect()
    )


def concordance(reference: pl.DataFrame, candidate: pl.DataFrame, nearest: bool = False) -> dict:
    if nearest:
        # each candidate sample with the reference sample of the subject closest in time, a day at most
        matched = (
            candidate.rename({"probability": "probability_candidate", "label": "label_candidate"})
            .sort("prediction_time")
            .join_asof(
                reference.sort("prediction_time"),
                on="prediction_time",
                by="subject_id",
                strategy="nearest",
                tolerance="1d",
            )
            .drop_nulls("probability")
        )
    else:
        matched = reference.join(candidate, on=KEY, suffix="_candidate").sort(KEY)
    ref = matched["probability"].to_numpy()
    cand = matched["probability_candidate"].to_numpy()
    label = matched["label"].to_numpy()
    candidate_label = matched["label_candidate"].to_numpy()
    result = {
        "num_reference": reference.height,
        "num_candidate": candidate.height,
        "num_intersection": matched.height,
        "num_labels_that_differ": int((label != candidate_label).sum()),
        "num_positive_in_reference_only": int(((label == 1) & (candidate_label == 0)).sum()),
        "num_positive_in_candidate_only": int(((label == 0) & (candidate_label == 1)).sum()),
    }
    if matched.height < 2 or len(np.unique(label)) < 2:
        return result
    top = max(1, matched.height // 10)
    result.update(
        {
            "spearman": float(spearmanr(ref, cand).statistic),
            # within a class: agreement that does not come from both separating the cases
            "spearman_cases": float(spearmanr(ref[label == 1], cand[label == 1]).statistic),
            "spearman_controls": float(spearmanr(ref[label == 0], cand[label == 0]).statistic),
            "reference_mean": float(ref.mean()),
            "candidate_mean": float(cand.mean()),
            "observed_rate": float(label.mean()),
            "mean_absolute_difference": float(np.abs(ref - cand).mean()),
            "share_of_reference_top_10pct_in_candidate_top_10pct": len(
                set(np.argsort(-ref)[:top]) & set(np.argsort(-cand)[:top])
            )
            / top,
        }
    )
    # the label of the reference is used for both
    for name, score in (("reference", ref), ("candidate", cand)):
        roc_auc, pr_auc = compute_metrics(label, score)
        result[f"{name}_roc_auc"], result[f"{name}_pr_auc"] = float(roc_auc), float(pr_auc)
    return result


def main(args: argparse.Namespace) -> dict:
    result = concordance(
        read_predictions(args.reference, args.match_on),
        read_predictions(args.candidate, args.match_on),
        nearest=args.match_on == "nearest",
    )
    print(json.dumps(result, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(result, indent=2))
    return result


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--reference", required=True, help="Parquet file or folder, e.g. the probe")
    parser.add_argument("--candidate", required=True, help="Parquet file or folder, e.g. the zero-shot")
    parser.add_argument("--output_file")
    parser.add_argument(
        "--match_on",
        choices=["exact", "date", "nearest"],
        default="exact",
        help="date: the same day; nearest: the closest prediction time of the subject within a day",
    )
    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
