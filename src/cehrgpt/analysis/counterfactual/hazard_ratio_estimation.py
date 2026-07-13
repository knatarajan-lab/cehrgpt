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
from typing import Dict, List, Optional, Set, Tuple

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
# Drug concept expansion and contamination filtering
# ---------------------------------------------------------------------------

def expand_to_descendants(vocab_path: str, concept_ids: List[int]) -> Set[str]:
    """Return all descendant concept_ids (as strings) via concept_ancestor."""
    ancestor_glob = os.path.join(vocab_path, "concept_ancestor", "*.parquet")
    descendants: List[str] = (
        pl.scan_parquet(ancestor_glob)
        .filter(pl.col("ancestor_concept_id").is_in(concept_ids))
        .select(pl.col("descendant_concept_id").cast(pl.String))
        .collect()
        ["descendant_concept_id"]
        .to_list()
    )
    result = set(descendants)
    result.update(str(c) for c in concept_ids)
    return result


def filter_contaminated_trajectories(
    trajectories: pl.DataFrame,
    forbidden_concepts: Set[str],
    arm: str,
) -> pl.DataFrame:
    """
    Remove entire trajectories (subject_id, trajectory_id) where any
    generated event code belongs to *forbidden_concepts*.

    For example, ACEi-arm trajectories that contain thiazide events are
    invalid counterfactuals and must be excluded before HR estimation.
    """
    if not forbidden_concepts:
        return trajectories

    contaminated = (
        trajectories
        .filter(pl.col("code").is_in(list(forbidden_concepts)))
        .select(["subject_id", "trajectory_id"])
        .unique()
    )
    n_contaminated = len(contaminated)
    if n_contaminated > 0:
        n_total = trajectories.select(["subject_id", "trajectory_id"]).n_unique()
        print(f"  [{arm}] Filtering {n_contaminated:,} / {n_total:,} contaminated trajectories "
              f"({n_contaminated / n_total:.1%})")
        trajectories = trajectories.join(
            contaminated, on=["subject_id", "trajectory_id"], how="anti"
        )
    return trajectories


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
# Observed-outcome baseline analysis
# ---------------------------------------------------------------------------

def analyse_observed_outcomes(
    observed_outcomes_path: str,
    drug_info_path: str,
    outcome_concept_ids: List[str],
    follow_up_days: float,
    output_dir: Path,
    source_arm_label: str = "acei",
    comparator_arm_label: str = "thiazide",
) -> None:
    """
    Compute hazard ratios from observed (real) patient outcomes extracted
    during Step 1.  These serve as a baseline to compare against the
    model-generated counterfactual HRs.

    Parameters
    ----------
    observed_outcomes_path
        Directory or parquet with columns:
        person_id, drug_epoch_time, <one column per outcome_concept_id>
        (each outcome column contains the epoch_time of the first event or null).
    drug_info_path
        Directory or parquet with columns:
        person_id, drug_concept_id, drug_epoch_time, arm  ('source'/'comparator').
    outcome_concept_ids
        Strings of OMOP concept_ids to analyse.
    follow_up_days
        Maximum follow-up in days; patients without an event are censored here.
    output_dir
        Directory for output CSVs.
    source_arm_label / comparator_arm_label
        Labels used in output files (default: acei / thiazide).
    """
    print("\n" + "=" * 60)
    print("Observed baseline HR (real patient outcomes)")
    print("=" * 60)

    # Load data
    def _read(path: str) -> pl.DataFrame:
        return pl.read_parquet(
            os.path.join(path, "*.parquet") if os.path.isdir(path) else path
        )

    obs  = _read(observed_outcomes_path)
    info = _read(drug_info_path).select(["person_id", "arm"])

    # Join arm label onto observed outcomes
    df = obs.join(info, on="person_id", how="left")

    source_df     = df.filter(pl.col("arm") == "source")
    comparator_df = df.filter(pl.col("arm") == "comparator")
    print(f"  Source ({source_arm_label}):     {len(source_df):,} patients")
    print(f"  Comparator ({comparator_arm_label}): {len(comparator_df):,} patients")

    results = []
    for oc_id in outcome_concept_ids:
        if oc_id not in df.columns:
            print(f"  Outcome {oc_id}: column not found in observed_outcomes — skipping")
            continue

        print(f"\n  Outcome: {oc_id}")

        def _tte(arm_df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            """Compute (time_days, event) arrays for one arm."""
            times, events = [], []
            for row in arm_df.select(["drug_epoch_time", oc_id]).iter_rows():
                drug_t, outcome_t = row
                if outcome_t is not None and drug_t is not None:
                    t_days = (float(outcome_t) - float(drug_t)) / 86_400
                    if 0 < t_days <= follow_up_days:
                        times.append(t_days)
                        events.append(1)
                        continue
                times.append(follow_up_days)
                events.append(0)
            return np.array(times, dtype=float), np.array(events, dtype=int)

        time_a, event_a = _tte(source_df)
        time_b, event_b = _tte(comparator_df)

        n_ev_a = int(event_a.sum())
        n_ev_b = int(event_b.sum())
        print(f"    {source_arm_label}: {len(time_a):,} patients, {n_ev_a:,} events")
        print(f"    {comparator_arm_label}: {len(time_b):,} patients, {n_ev_b:,} events")

        if n_ev_a + n_ev_b == 0:
            print("    Skipping: no events in either arm.")
            continue

        time_all      = np.concatenate([time_a, time_b])
        event_all     = np.concatenate([event_a, event_b])
        treatment_all = np.concatenate([np.ones(len(time_a)), np.zeros(len(time_b))])

        hr, hr_lower, hr_upper, cox_p = fit_cox(time_all, event_all, treatment_all)
        lr_stat, lr_p = log_rank_test(time_a, event_a, time_b, event_b)

        print(f"    HR ({source_arm_label} vs {comparator_arm_label}) = {hr:.3f}  "
              f"(95% CI: {hr_lower:.3f}–{hr_upper:.3f})  Cox p={cox_p:.4f}")
        print(f"    Log-rank stat={lr_stat:.3f}  p={lr_p:.4f}")

        # KM curves
        km_t_a, km_s_a = kaplan_meier(time_a, event_a)
        km_t_b, km_s_b = kaplan_meier(time_b, event_b)
        km_df = pl.concat([
            pl.DataFrame({"time_days": km_t_a, "survival": km_s_a, "arm": source_arm_label}),
            pl.DataFrame({"time_days": km_t_b, "survival": km_s_b, "arm": comparator_arm_label}),
        ])
        km_df.write_csv(str(output_dir / f"observed_km_{oc_id}.csv"))

        results.append({
            "outcome_concept_id": oc_id,
            f"n_{source_arm_label}": len(time_a),
            f"n_{comparator_arm_label}": len(time_b),
            f"n_events_{source_arm_label}": n_ev_a,
            f"n_events_{comparator_arm_label}": n_ev_b,
            "hr": hr,
            "hr_lower_95": hr_lower,
            "hr_upper_95": hr_upper,
            "cox_p_value": cox_p,
            "log_rank_stat": lr_stat,
            "log_rank_p_value": lr_p,
            "hr_interpretation": (
                f"Observed HR of {source_arm_label} vs {comparator_arm_label} "
                f"(>1 = higher hazard in {source_arm_label})"
            ),
        })

    if results:
        out_path = output_dir / "observed_hazard_ratio_summary.csv"
        pl.DataFrame(results).write_csv(str(out_path))
        print(f"\nObserved HR summary written to {out_path}")


# ---------------------------------------------------------------------------
# Faithfulness check
# ---------------------------------------------------------------------------

def faithfulness_check_outcome(
    outcome_concept_id: str,
    traj_comparator: pl.DataFrame,
    observed_outcomes: pl.DataFrame,
    drug_info_generated: pl.DataFrame,
    follow_up_days: float,
    comparator_label: str,
    output_dir: Path,
) -> Optional[Dict]:
    """
    Compare generated comparator trajectories against real observed outcomes
    for the same drug class.  HR should be ≈ 1.0 if the model faithfully
    reproduces comparator (e.g. Thiazide) outcomes.

    Parameters
    ----------
    traj_comparator
        Generated trajectories for the comparator arm (e.g. thiazide/).
    observed_outcomes
        Observed outcomes parquet from Step 1 (comparator patients only),
        with one row per patient and one column per outcome concept_id.
    drug_info_generated
        drug_info for the generated arm (anchors drug_epoch_time).
    drug_info_observed
        drug_info for the observed comparator patients.
    """
    print(f"\n  [Faithfulness] Outcome: {outcome_concept_id}")

    # --- Generated comparator TTE ---
    tte_gen = compute_tte(
        traj_comparator, drug_info_generated, [outcome_concept_id], follow_up_days
    )
    agg_gen = (
        tte_gen
        .group_by("subject_id")
        .agg(
            pl.col("event").max().alias("event"),
            pl.col("time_days").mean().alias("time_days"),
        )
    )

    # --- Observed comparator TTE (from observed_outcomes parquet) ---
    if outcome_concept_id not in observed_outcomes.columns:
        print(f"    Outcome column {outcome_concept_id} not in observed_outcomes — skipping.")
        return None

    # observed_outcomes already carries drug_epoch_time from Step 1 — no join needed
    obs_times, obs_events = [], []
    for row in observed_outcomes.select(["drug_epoch_time", outcome_concept_id]).iter_rows():
        drug_t, outcome_t = row
        if outcome_t is not None and drug_t is not None:
            t_days = (float(outcome_t) - float(drug_t)) / 86_400
            if 0 < t_days <= follow_up_days:
                obs_times.append(t_days)
                obs_events.append(1)
                continue
        obs_times.append(follow_up_days)
        obs_events.append(0)

    time_gen  = agg_gen["time_days"].to_numpy()
    event_gen = agg_gen["event"].to_numpy().astype(int)
    time_obs  = np.array(obs_times, dtype=float)
    event_obs = np.array(obs_events, dtype=int)

    n_ev_gen = int(event_gen.sum())
    n_ev_obs = int(event_obs.sum())
    print(f"    Generated {comparator_label}: {len(time_gen):,} patients, {n_ev_gen:,} events")
    print(f"    Observed  {comparator_label}: {len(time_obs):,} patients, {n_ev_obs:,} events")

    if n_ev_gen + n_ev_obs == 0:
        print("    Skipping: no events in either group.")
        return None

    # HR: generated=1, observed=0  (expect ≈ 1.0)
    time_all      = np.concatenate([time_gen, time_obs])
    event_all     = np.concatenate([event_gen, event_obs])
    treatment_all = np.concatenate([np.ones(len(time_gen)), np.zeros(len(time_obs))])

    hr, hr_lower, hr_upper, cox_p = fit_cox(time_all, event_all, treatment_all)
    lr_stat, lr_p = log_rank_test(time_gen, event_gen, time_obs, event_obs)

    print(f"    HR (generated vs observed {comparator_label}) = {hr:.3f}  "
          f"(95% CI: {hr_lower:.3f}–{hr_upper:.3f})  Cox p={cox_p:.4f}  "
          f"[expect ≈ 1.0]")
    print(f"    Log-rank stat={lr_stat:.3f}  p={lr_p:.4f}")

    # KM curves
    km_t_g, km_s_g = kaplan_meier(time_gen, event_gen)
    km_t_o, km_s_o = kaplan_meier(time_obs, event_obs)
    km_df = pl.concat([
        pl.DataFrame({"time_days": km_t_g, "survival": km_s_g,
                      "arm": f"generated_{comparator_label}"}),
        pl.DataFrame({"time_days": km_t_o, "survival": km_s_o,
                      "arm": f"observed_{comparator_label}"}),
    ])
    km_df.write_csv(str(output_dir / f"faithfulness_km_{outcome_concept_id}.csv"))

    return {
        "outcome_concept_id": outcome_concept_id,
        f"n_generated_{comparator_label}": len(time_gen),
        f"n_observed_{comparator_label}": len(time_obs),
        f"n_events_generated_{comparator_label}": n_ev_gen,
        f"n_events_observed_{comparator_label}": n_ev_obs,
        "hr_generated_vs_observed": hr,
        "hr_lower_95": hr_lower,
        "hr_upper_95": hr_upper,
        "cox_p_value": cox_p,
        "log_rank_stat": lr_stat,
        "log_rank_p_value": lr_p,
        "note": f"HR should be ≈ 1.0 if model faithfully reproduces {comparator_label} outcomes",
    }


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

    # ------------------------------------------------------------------
    # Observed baseline HR (runs when --observed_outcomes_path is given)
    # ------------------------------------------------------------------
    if args.observed_outcomes_path:
        analyse_observed_outcomes(
            observed_outcomes_path=args.observed_outcomes_path,
            drug_info_path=args.drug_info_path,
            outcome_concept_ids=outcome_ids,
            follow_up_days=args.follow_up_days,
            output_dir=output_dir,
            source_arm_label=args.arm_a,
            comparator_arm_label=args.arm_b,
        )

    # ------------------------------------------------------------------
    # Generated-trajectory HR
    # ------------------------------------------------------------------
    if args.trajectories_dir is None:
        print("\nNo --trajectories_dir provided; skipping generated-trajectory HR.")
        return

    # Expand drug concept IDs to descendants for contamination filtering
    arm_a_forbidden: Set[str] = set()
    arm_b_forbidden: Set[str] = set()
    if args.vocab_path and args.arm_a_concept_ids and args.arm_b_concept_ids:
        arm_a_ids = [int(x.strip()) for x in args.arm_a_concept_ids.split(",")]
        arm_b_ids = [int(x.strip()) for x in args.arm_b_concept_ids.split(",")]
        print("Expanding drug concepts for contamination filtering …")
        arm_a_concepts = expand_to_descendants(args.vocab_path, arm_a_ids)
        arm_b_concepts = expand_to_descendants(args.vocab_path, arm_b_ids)
        arm_a_forbidden = arm_b_concepts
        arm_b_forbidden = arm_a_concepts
        print(f"  Arm A forbidden concepts (comparator drugs): {len(arm_a_forbidden):,}")
        print(f"  Arm B forbidden concepts (source drugs):     {len(arm_b_forbidden):,}")

    print(f"Loading {args.arm_a} trajectories …")
    traj_a = load_trajectories(args.trajectories_dir, args.arm_a)
    traj_a = filter_contaminated_trajectories(traj_a, arm_a_forbidden, args.arm_a)
    print(f"  {len(traj_a):,} events")

    print(f"Loading {args.arm_b} trajectories …")
    traj_b = load_trajectories(args.trajectories_dir, args.arm_b)
    traj_b = filter_contaminated_trajectories(traj_b, arm_b_forbidden, args.arm_b)
    print(f"  {len(traj_b):,} events")

    def _read_parquet(path: str) -> pl.DataFrame:
        return pl.read_parquet(
            os.path.join(path, "*.parquet") if os.path.isdir(path) else path
        )

    print("Loading drug info …")
    drug_info = _read_parquet(args.drug_info_path)

    # ------------------------------------------------------------------
    # Faithfulness check: generated comparator vs observed comparator
    # HR ≈ 1.0 means the model faithfully reproduces arm_b outcomes.
    # ------------------------------------------------------------------
    if args.observed_outcomes_path:
        print("\n" + "=" * 60)
        print(f"Faithfulness check: generated {args.arm_b} vs observed {args.arm_b}")
        print("(HR should be ≈ 1.0 if the model is faithful)")
        print("=" * 60)

        observed_outcomes = _read_parquet(args.observed_outcomes_path)
        # Keep only observed comparator patients (e.g. Thiazide initiators)
        comparator_person_ids = (
            drug_info.filter(pl.col("arm") == "comparator").select("person_id")
        )
        obs_comparator = observed_outcomes.join(comparator_person_ids, on="person_id", how="inner")

        faithfulness_results = []
        for outcome_id in outcome_ids:
            res = faithfulness_check_outcome(
                outcome_concept_id=outcome_id,
                traj_comparator=traj_b,
                observed_outcomes=obs_comparator,
                drug_info_generated=drug_info,
                follow_up_days=args.follow_up_days,
                comparator_label=args.arm_b,
                output_dir=output_dir,
            )
            if res is not None:
                faithfulness_results.append(res)

        if faithfulness_results:
            faith_path = output_dir / "faithfulness_summary.csv"
            pl.DataFrame(faithfulness_results).write_csv(str(faith_path))
            print(f"\nFaithfulness summary written to {faith_path}")

    # ------------------------------------------------------------------
    # Main HR: generated arm_a vs generated arm_b (counterfactual HR)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Main HR: generated {args.arm_a} vs generated {args.arm_b}")
    print("=" * 60)

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
        default=None,
        help="Root directory of generated trajectories. "
             "Optional when --observed_outcomes_path is provided.",
    )
    parser.add_argument(
        "--observed_outcomes_path",
        default=None,
        help="Directory or parquet of observed outcomes from Step 1 "
             "(observed_outcomes/). When provided, computes baseline "
             "HR from real patient data before the generated-trajectory HR.",
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
    parser.add_argument(
        "--vocab_path",
        default=None,
        help="OMOP vocabulary root (concept_ancestor/ parquets). Required for "
             "contamination filtering via --arm_a_concept_ids / --arm_b_concept_ids.",
    )
    parser.add_argument(
        "--arm_a_concept_ids",
        default=None,
        help="Comma-separated ingredient concept_ids for arm A drug class. "
             "Arm B trajectories containing these codes will be removed.",
    )
    parser.add_argument(
        "--arm_b_concept_ids",
        default=None,
        help="Comma-separated ingredient concept_ids for arm B drug class. "
             "Arm A trajectories containing these codes will be removed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
