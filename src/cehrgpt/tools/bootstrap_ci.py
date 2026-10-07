"""
Patient-level bootstrap confidence intervals of ROC-AUC and PR-AUC.

A cohort can have several labels per patient (one per visit / index date), which are correlated,
so the bootstrap resamples patients with all their labels, not single labels. Each replicate draws
as many patients as there are in the predictions, with replacement; a patient drawn twice counts
twice (as a sample weight). The interval is the percentile interval of the replicates. This is the
same procedure as ethos-ares' scripts/linear_prob/bootstrap_ci.py.

PR-AUC is the trapezoid area under the precision-recall curve by default, as in
train_with_cehrgpt_features.py, so the point estimate matches the metrics.json of a linear probing
run. `--pr_auc average_precision` uses the average precision instead.

It reads the parquet files of a linear probing run (<output_dir>/logistic/test_predictions/) or of
a zero-shot run (<output_folder>/<sampling>/<task_name>/, shard_<i> folders included). They need
the columns subject_id, boolean_value and predicted_boolean_probability.

Usage:
    python -m cehrgpt.tools.bootstrap_ci \
        --predictions <output_dir>/logistic/test_predictions \
        --output_file <output_dir>/logistic/bootstrap_ci.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

LABEL_COLUMN = "boolean_value"
SCORE_COLUMN = "predicted_boolean_probability"
SUBJECT_COLUMN = "subject_id"


def read_predictions(
    paths: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the predictions of one or more parquet files or folders (searched recursively).

    Returns the subject_id, the label and the predicted probability of every prediction. Only the
    three needed columns are read, because zero-shot files also hold the generated trajectories.
    """
    files: List[str] = []
    for path in map(Path, paths):
        if path.is_dir():
            files.extend(str(f) for f in sorted(path.rglob("*.parquet")))
        else:
            files.append(str(path))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {list(paths)}")
    df = (
        pl.scan_parquet(files)
        .select(SUBJECT_COLUMN, LABEL_COLUMN, SCORE_COLUMN)
        .drop_nulls()
        .collect()
    )
    return (
        df[SUBJECT_COLUMN].to_numpy(),
        df[LABEL_COLUMN].cast(pl.Int8).to_numpy(),
        df[SCORE_COLUMN].to_numpy(),
    )


def compute_metrics(
    y: np.ndarray,
    score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    pr_auc: str = "trapezoid",
) -> Tuple[float, float]:
    """Return (ROC-AUC, PR-AUC), optionally with a weight per label."""
    roc = roc_auc_score(y, score, sample_weight=sample_weight)
    if pr_auc == "average_precision":
        return roc, average_precision_score(y, score, sample_weight=sample_weight)
    precision, recall, _ = precision_recall_curve(
        y, score, sample_weight=sample_weight
    )
    return roc, auc(recall, precision)


def _replicates(
    subject_idx: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    num_subjects: int,
    seeds: Sequence[int],
    pr_auc: str,
) -> List[Tuple[float, float]]:
    replicates = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, num_subjects, num_subjects)
        weight = np.bincount(draws, minlength=num_subjects)[subject_idx]
        keep = weight > 0
        # A replicate without both classes has no AUC
        if len(np.unique(y[keep])) < 2:
            replicates.append((np.nan, np.nan))
            continue
        replicates.append(compute_metrics(y[keep], score[keep], weight[keep], pr_auc))
    return replicates


def bootstrap_confidence_intervals(
    subject_ids: Sequence[Any],
    y: Sequence[int],
    score: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    pr_auc: str = "trapezoid",
    plus_minus: str = "ci",
    n_jobs: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    ROC-AUC and PR-AUC with patient-level bootstrap confidence intervals.

    Args:
        subject_ids: The patient of every label; labels of one patient are resampled together.
        y: The binary labels.
        score: The predicted probabilities.
        n_boot: The number of bootstrap replicates.
        alpha: The significance level, 0.05 gives a 95% interval.
        pr_auc: "trapezoid" (area under the precision-recall curve) or "average_precision".
        plus_minus: The ± of the text fields: "ci" is half the width of the interval, "std" the
            standard deviation of the replicates.
        n_jobs: The number of processes the replicates are computed in. The result does not
            depend on it.
        seed: The seed of the replicates.

    Returns:
        The number of labels / patients, the prevalence, and for roc_auc and pr_auc the point
        estimate, `_low` / `_high` (percentile interval), `_boot_mean`, `_boot_std` and `_text`,
        e.g. 65.8±0.6%.
    """
    if pr_auc not in ("trapezoid", "average_precision"):
        raise ValueError(f"Unknown pr_auc {pr_auc!r}")
    if plus_minus not in ("ci", "std"):
        raise ValueError(f"Unknown plus_minus {plus_minus!r}")
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=float)
    # Index of the patient of every label
    _, subject_idx = np.unique(np.asarray(subject_ids), return_inverse=True)
    subject_idx = subject_idx.reshape(-1)
    num_subjects = int(subject_idx.max()) + 1

    point_estimates = compute_metrics(y, score, None, pr_auc)

    seeds = np.random.SeedSequence(seed).generate_state(n_boot)
    chunks = [c for c in np.array_split(seeds, max(n_jobs, 1) * 4) if len(c)]
    replicates = Parallel(n_jobs=n_jobs)(
        delayed(_replicates)(subject_idx, y, score, num_subjects, chunk, pr_auc)
        for chunk in chunks
    )
    replicates = np.array([r for chunk in replicates for r in chunk], dtype=float)
    replicates = replicates[~np.isnan(replicates).any(axis=1)]
    if len(replicates) == 0:
        raise ValueError("None of the bootstrap replicates contains both classes")
    lower, upper = 100 * alpha / 2, 100 * (1 - alpha / 2)

    result: Dict[str, Any] = {
        "n_labels": len(y),
        "n_patients": num_subjects,
        "prevalence": float(y.mean()),
        "n_boot": len(replicates),
        "alpha": alpha,
        "pr_auc_kind": pr_auc,
    }
    for name, i in (("roc_auc", 0), ("pr_auc", 1)):
        point = float(point_estimates[i])
        low, high = (float(v) for v in np.percentile(replicates[:, i], [lower, upper]))
        std = float(replicates[:, i].std(ddof=1)) if len(replicates) > 1 else 0.0
        half = (high - low) / 2 if plus_minus == "ci" else std
        result.update(
            {
                name: point,
                f"{name}_low": low,
                f"{name}_high": high,
                f"{name}_boot_mean": float(replicates[:, i].mean()),
                f"{name}_boot_std": std,
                f"{name}_text": f"{100 * point:.1f}±{100 * half:.1f}%",
            }
        )
    return result


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patient-level bootstrap confidence intervals of ROC-AUC and PR-AUC"
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Parquet files or folders (searched recursively) with the columns subject_id, "
        "boolean_value and predicted_boolean_probability",
    )
    parser.add_argument(
        "--output_file", help="Write the result to this json file (optional)"
    )
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap replicates")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="0.05 gives a 95%% interval"
    )
    parser.add_argument(
        "--plus_minus",
        choices=["ci", "std"],
        default="ci",
        help="The ± of the text fields (65.8±0.6%%): half the width of the interval, or the "
        "standard deviation of the replicates",
    )
    parser.add_argument(
        "--pr_auc", choices=["trapezoid", "average_precision"], default="trapezoid"
    )
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(args: argparse.Namespace) -> Dict[str, Any]:
    subject_ids, y, score = read_predictions(args.predictions)
    result = bootstrap_confidence_intervals(
        subject_ids,
        y,
        score,
        n_boot=args.n_boot,
        alpha=args.alpha,
        pr_auc=args.pr_auc,
        plus_minus=args.plus_minus,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    print(
        f"n={result['n_labels']:,} patients={result['n_patients']:,} "
        f"ROC-AUC {result['roc_auc_text']}  PR-AUC {result['pr_auc_text']}"
    )
    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, indent=2))
        print(f"Wrote {output_file}")
    return result


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
