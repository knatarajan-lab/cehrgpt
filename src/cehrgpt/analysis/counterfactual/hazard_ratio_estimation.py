#!/usr/bin/env python3
"""
Step 4: Estimate Hazard Ratios from generated treated vs non-treated trajectories.

For each specified outcome concept_id the script:
  1. Loads generated trajectories for both arms.
  2. Identifies the first occurrence of each outcome event in every
     trajectory and computes time-to-event (days from drug_epoch_time).
  3. Censors patients who never experience the outcome at *follow_up_days*.
  4. Fits a Cox Proportional Hazards model (via partial-likelihood
     maximisation with scipy.optimize) with a single binary treatment
     covariate to estimate the Hazard Ratio (HR).
  5. Performs a log-rank test between the two arms.
  6. Writes a summary CSV and per-outcome Kaplan-Meier data CSV.

Requirements: numpy, polars, scipy  (no external survival-analysis library)

Usage
-----
python hazard_ratio_estimation.py \\
    --trajectories_dir   /path/to/trajectories \\
    --drug_info_path     /path/to/context_dir/drug_info.parquet \\
    --outcome_concept_ids 4329847,312327 \\
    --follow_up_days     365 \\
    --output_dir         /path/to/hr_results
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default arm names for the ACEi vs Thiazide LEGEND-HTN study.
# These match the --arm_context names passed to generate_counterfactual_sequences.py.
# Override via --arm_a / --arm_b if you use different names.
DEFAULT_ARM_A = "acei"       # the "treated" arm (actual drug received)
DEFAULT_ARM_B = "thiazide"   # the "comparator" arm (counterfactual)
EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(trajectories_dir: str, arm: str) -> pl.DataFrame:
    """
    Load and concatenate all parquet batches for a given *arm*.

    Expects files at ``<trajectories_dir>/<arm>/<traj_id>/batch_*.parquet``.
    """
    arm_dir = os.path.join(trajectories_dir, arm)
    if not os.path.isdir(arm_dir):
        raise FileNotFoundError(f"Arm directory not found: {arm_dir}")

    parquet_files = list(Path(arm_dir).rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {arm_dir}")

    dfs = [pl.read_parquet(str(f)) for f in sorted(parquet_files)]
    df = pl.concat(dfs)
    return df


# ---------------------------------------------------------------------------
# Time-to-event computation
# ---------------------------------------------------------------------------

def compute_tte(
    trajectories: pl.DataFrame,
    drug_info: pl.DataFrame,
    outcome_concept_ids: List[str],
    follow_up_days: float,
) -> pl.DataFrame:
    """
    For each (subject_id, trajectory_id) compute:
      - event      : 1 if outcome occurred, 0 if censored
      - time_days  : days from drug_epoch_time to first outcome (or follow-up)

    Parameters
    ----------
    trajectories
        Generated trajectory events.  Expected columns:
        subject_id, trajectory_id, prediction_time, time, code
    drug_info
        Parquet with (person_id, drug_epoch_time).
    outcome_concept_ids
        List of OMOP concept_id strings to treat as the outcome.
    follow_up_days
        Maximum follow-up window in days.

    Returns
    -------
    DataFrame with columns: subject_id, trajectory_id, event, time_days
    """
    outcome_set = set(outcome_concept_ids)
    follow_up_seconds = follow_up_days * 86_400

    # Attach drug_epoch_time to each trajectory event
    drug_lf = drug_info.lazy().rename({"person_id": "subject_id"})
    traj_lf = (
        trajectories
        .lazy()
        .join(drug_lf.select(["subject_id", "drug_epoch_time"]), on="subject_id", how="left")
    )

    # Keep only events within the follow-up window and before drug_epoch_time + window
    traj_lf = traj_lf.with_columns(
        pl.col("time").cast(pl.Datetime).dt.epoch(time_unit="s").alias("time_epoch_s")
    ).filter(
        pl.col("time_epoch_s") > pl.col("drug_epoch_time")
    ).with_columns(
        ((pl.col("time_epoch_s") - pl.col("drug_epoch_time")) / 86_400).alias("days_since_drug")
    ).filter(
        pl.col("days_since_drug") <= follow_up_days
    )

    # Flag outcome events
    outcome_events = (
        traj_lf
        .filter(pl.col("code").is_in(list(outcome_set)))
        .group_by(["subject_id", "trajectory_id"])
        .agg(pl.col("days_since_drug").min().alias("time_days"))
        .with_columns(pl.lit(1).alias("event"))
    )

    # All unique (subject_id, trajectory_id) pairs
    all_pairs = (
        trajectories
        .lazy()
        .select(["subject_id", "trajectory_id"])
        .unique()
    )

    # Left join: censored patients get event=0 and time=follow_up_days
    tte = (
        all_pairs
        .join(outcome_events, on=["subject_id", "trajectory_id"], how="left")
        .with_columns(
            pl.col("event").fill_null(0),
            pl.col("time_days").fill_null(follow_up_days),
        )
        .collect()
    )
    return tte


# ---------------------------------------------------------------------------
# Cox PH via partial likelihood  (single binary covariate)
# ---------------------------------------------------------------------------

def _cox_partial_log_likelihood(beta: float, time: np.ndarray, event: np.ndarray, x: np.ndarray) -> float:
    """
    Negative partial log-likelihood for a Cox PH model with a single
    covariate *x* (binary: 1=treated, 0=non-treated).

    L(β) = Σ_i { event_i * [β x_i - log(Σ_{j: t_j ≥ t_i} exp(β x_j))] }
    """
    order = np.argsort(time)
    t = time[order]
    e = event[order]
    xi = x[order]

    n = len(t)
    log_risk = beta * xi  # shape (n,)

    # Compute log(Σ risk sets) efficiently with a reverse cumsum
    # At each event time t_i, the risk set is {j : t_j >= t_i}
    exp_xb = np.exp(log_risk)
    # Reverse cumsum gives Σ_{j>=i} exp(xb_j) for each position i
    risk_set_sum = np.cumsum(exp_xb[::-1])[::-1]

    pll = np.sum(e * (log_risk - np.log(risk_set_sum + EPSILON)))
    return -pll   # minimise negative log-likelihood


def fit_cox(
    time: np.ndarray,
    event: np.ndarray,
    treatment: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Estimate HR = exp(β) for a binary treatment covariate.

    Returns
    -------
    (hr, hr_lower_95, hr_upper_95, p_value)
    """
    # Minimise NLL over β
    result = minimize_scalar(
        _cox_partial_log_likelihood,
        bounds=(-10, 10),
        method="bounded",
        args=(time, event, treatment),
    )
    beta_hat = result.x

    # Numerical Hessian (second derivative) at β̂ for variance estimate
    h = 1e-5
    f0 = _cox_partial_log_likelihood(beta_hat, time, event, treatment)
    fp = _cox_partial_log_likelihood(beta_hat + h, time, event, treatment)
    fm = _cox_partial_log_likelihood(beta_hat - h, time, event, treatment)
    hess = (fp - 2 * f0 + fm) / (h ** 2)
    var_beta = 1.0 / (hess + EPSILON)
    se_beta = math.sqrt(abs(var_beta))

    MAX_EXP = 700  # math.exp overflows above ~709
    hr = math.exp(max(-MAX_EXP, min(MAX_EXP, beta_hat)))
    hr_lower = math.exp(max(-MAX_EXP, min(MAX_EXP, beta_hat - 1.96 * se_beta)))
    hr_upper = math.exp(max(-MAX_EXP, min(MAX_EXP, beta_hat + 1.96 * se_beta)))

    # Wald test p-value
    z = beta_hat / (se_beta + EPSILON)
    p_value = 2 * (1 - chi2.cdf(z ** 2, df=1))

    return hr, hr_lower, hr_upper, p_value


# ---------------------------------------------------------------------------
# Log-rank test
# ---------------------------------------------------------------------------

def log_rank_test(
    time_a: np.ndarray,
    event_a: np.ndarray,
    time_b: np.ndarray,
    event_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample log-rank test comparing arm A vs arm B.

    Returns (test_statistic, p_value).
    """
    # Pool all unique event times
    all_times = np.unique(np.concatenate([time_a[event_a == 1], time_b[event_b == 1]]))

    O_a = O_b = E_a = E_b = 0.0

    for t in all_times:
        # Number at risk
        n_a = np.sum(time_a >= t)
        n_b = np.sum(time_b >= t)
        n = n_a + n_b
        if n == 0:
            continue

        # Observed events
        o_a = np.sum((time_a == t) & (event_a == 1))
        o_b = np.sum((time_b == t) & (event_b == 1))
        o = o_a + o_b

        # Expected under H0
        e_a = o * n_a / n
        e_b = o * n_b / n

        O_a += o_a
        O_b += o_b
        E_a += e_a
        E_b += e_b

    if E_a == 0 and E_b == 0:
        return 0.0, 1.0

    numerator = (O_a - E_a) ** 2 / (E_a + EPSILON) + (O_b - E_b) ** 2 / (E_b + EPSILON)
    p_value = float(1 - chi2.cdf(numerator, df=1))
    return float(numerator), p_value


# ---------------------------------------------------------------------------
# Kaplan-Meier estimator
# ---------------------------------------------------------------------------

def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Kaplan-Meier survival estimate.

    Returns (time_points, survival_probabilities).
    """
    order = np.argsort(time)
    t = time[order]
    e = event[order]

    unique_times = np.unique(t[e == 1])
    n_total = len(t)

    km_times = [0.0]
    km_surv = [1.0]

    n_at_risk = n_total
    prev_t = 0.0
    for ut in unique_times:
        # Update at-risk count for any censored/events between prev_t and ut
        n_at_risk -= np.sum((t >= prev_t) & (t < ut))
        d = np.sum((t == ut) & (e == 1))
        if n_at_risk > 0:
            km_surv.append(km_surv[-1] * (1 - d / n_at_risk))
        else:
            km_surv.append(km_surv[-1])
        km_times.append(float(ut))
        n_at_risk -= d
        prev_t = ut

    return np.array(km_times), np.array(km_surv)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse_outcome(
    outcome_concept_id: str,
    traj_arm_a: pl.DataFrame,
    traj_arm_b: pl.DataFrame,
    drug_info: pl.DataFrame,
    follow_up_days: float,
    arm_a_label: str,
    arm_b_label: str,
    output_dir: Path,
) -> Dict:
    """
    Run full HR analysis for a single outcome concept.

    HR is reported as arm_a relative to arm_b (e.g. ACEi vs Thiazide).
    Since both arms share the same patients (within-patient counterfactual),
    each patient contributes one aggregated time-to-event value per arm.
    """
    print(f"\n  Outcome: {outcome_concept_id}")

    tte_a = compute_tte(traj_arm_a, drug_info, [outcome_concept_id], follow_up_days)
    tte_b = compute_tte(traj_arm_b, drug_info, [outcome_concept_id], follow_up_days)

    # Aggregate N trajectories per patient:
    #   event = 1 if outcome occurred in ANY trajectory
    #   time  = mean time-to-event across trajectories
    def aggregate(tte: pl.DataFrame) -> pl.DataFrame:
        return (
            tte
            .group_by("subject_id")
            .agg(
                pl.col("event").max().alias("event"),
                pl.col("time_days").mean().alias("time_days"),
            )
        )

    agg_a = aggregate(tte_a)
    agg_b = aggregate(tte_b)

    time_a  = agg_a["time_days"].to_numpy()
    event_a = agg_a["event"].to_numpy().astype(int)
    time_b  = agg_b["time_days"].to_numpy()
    event_b = agg_b["event"].to_numpy().astype(int)

    n_events_a = int(event_a.sum())
    n_events_b = int(event_b.sum())
    print(f"    {arm_a_label}: {len(time_a):,} patients, {n_events_a:,} events")
    print(f"    {arm_b_label}: {len(time_b):,} patients, {n_events_b:,} events")

    if n_events_a + n_events_b == 0:
        print("    Skipping: no events observed in either arm.")
        return None

    # Cox PH: treatment=1 for arm_a (ACEi), treatment=0 for arm_b (Thiazide)
    # HR > 1 means arm_a has higher hazard than arm_b
    time_all      = np.concatenate([time_a, time_b])
    event_all     = np.concatenate([event_a, event_b])
    treatment_all = np.concatenate([np.ones(len(time_a)), np.zeros(len(time_b))])

    hr, hr_lower, hr_upper, cox_p = fit_cox(time_all, event_all, treatment_all)
    lr_stat, lr_p = log_rank_test(time_a, event_a, time_b, event_b)

    print(f"    HR ({arm_a_label} vs {arm_b_label}) = {hr:.3f}  "
          f"(95% CI: {hr_lower:.3f}–{hr_upper:.3f})  Cox p={cox_p:.4f}")
    print(f"    Log-rank stat={lr_stat:.3f}  p={lr_p:.4f}")

    # Kaplan-Meier survival curves
    km_t_a, km_s_a = kaplan_meier(time_a, event_a)
    km_t_b, km_s_b = kaplan_meier(time_b, event_b)

    km_df = pl.concat([
        pl.DataFrame({"time_days": km_t_a, "survival": km_s_a, "arm": arm_a_label}),
        pl.DataFrame({"time_days": km_t_b, "survival": km_s_b, "arm": arm_b_label}),
    ])
    km_df.write_csv(str(output_dir / f"km_{outcome_concept_id}.csv"))

    return {
        "outcome_concept_id": outcome_concept_id,
        f"n_{arm_a_label}": len(time_a),
        f"n_{arm_b_label}": len(time_b),
        f"n_events_{arm_a_label}": int(event_a.sum()),
        f"n_events_{arm_b_label}": int(event_b.sum()),
        "hr": hr,
        "hr_lower_95": hr_lower,
        "hr_upper_95": hr_upper,
        "cox_p_value": cox_p,
        "log_rank_stat": lr_stat,
        "log_rank_p_value": lr_p,
        "hr_interpretation": f"HR of {arm_a_label} vs {arm_b_label} (>1 = higher hazard in {arm_a_label})",
    }


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcome_ids = [x.strip() for x in args.outcome_concept_ids.split(",")]

    print(f"Loading {args.arm_a} trajectories …")
    traj_a = load_trajectories(args.trajectories_dir, args.arm_a)
    print(f"  {len(traj_a):,} events")

    print(f"Loading {args.arm_b} trajectories …")
    traj_b = load_trajectories(args.trajectories_dir, args.arm_b)
    print(f"  {len(traj_b):,} events")

    print("Loading drug info …")
    drug_info = pl.read_parquet(os.path.join(args.drug_info_path, "*.parquet") if os.path.isdir(args.drug_info_path) else args.drug_info_path)

    results = []
    for outcome_id in outcome_ids:
        result = analyse_outcome(
            outcome_concept_id=outcome_id,
            traj_arm_a=traj_a,
            traj_arm_b=traj_b,
            drug_info=drug_info,
            follow_up_days=args.follow_up_days,
            arm_a_label=args.arm_a,
            arm_b_label=args.arm_b,
            output_dir=output_dir,
        )
        if result is not None:
            results.append(result)

    summary = pl.DataFrame(results)
    summary_path = output_dir / "hazard_ratio_summary.csv"
    summary.write_csv(str(summary_path))

    print(f"\nSummary written to {summary_path}")
    print(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate hazard ratios from generated counterfactual trajectories"
    )
    parser.add_argument(
        "--trajectories_dir",
        required=True,
        help="Root directory of generated trajectories",
    )
    parser.add_argument(
        "--drug_info_path",
        required=True,
        help="drug_info.parquet from extract_drug_initiation_sequences.py",
    )
    parser.add_argument(
        "--outcome_concept_ids",
        required=True,
        help="Comma-separated OMOP concept_ids for outcomes, e.g. 4329847,316139,4110192,376713",
    )
    parser.add_argument(
        "--follow_up_days",
        type=float,
        default=365.0,
        help="Maximum follow-up window in days (default: 365)",
    )
    parser.add_argument(
        "--arm_a",
        default=DEFAULT_ARM_A,
        help=f"Name of arm A (the 'treated' arm); must match sub-directory under "
             f"trajectories_dir (default: {DEFAULT_ARM_A})",
    )
    parser.add_argument(
        "--arm_b",
        default=DEFAULT_ARM_B,
        help=f"Name of arm B (the 'comparator' arm); must match sub-directory under "
             f"trajectories_dir (default: {DEFAULT_ARM_B})",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output CSV files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
