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

# Tokens that explicitly mark the end of a patient's record in generated sequences.
_RECORD_END_TOKENS = {"[END]", "[DEATH]"}


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


# ---------------------------------------------------------------------------
# Time-to-event computation
# ---------------------------------------------------------------------------

def _find_era_gap_end(target_days: List[float], era_gap_days: float) -> Optional[float]:
    """
    Given a **sorted** list of target drug event times (days since drug start),
    return the time at which the drug era ends due to a gap > *era_gap_days*.

    Mirrors the gap-detection logic of ``compute_drug_era_end_time`` in
    ``extract_drug_initiation_sequences.py``: the era ends at
    ``last_target_time + era_gap_days`` when consecutive target events are
    separated by more than *era_gap_days* days.

    Returns ``None`` when no such gap is found (era is still active).
    """
    for i in range(1, len(target_days)):
        if target_days[i] - target_days[i - 1] > era_gap_days:
            return target_days[i - 1] + era_gap_days
    return None


def compute_tte(
    trajectories: pl.DataFrame,
    drug_info: pl.DataFrame,
    outcome_concept_ids: List[str],
    follow_up_days: float,
    target_concept_ids: Optional[Set[str]] = None,
    competing_concept_ids: Optional[Set[str]] = None,
    era_gap_days: float = 30.0,
) -> pl.DataFrame:
    """
    For each (subject_id, trajectory_id) compute:
      - event      : 1 if outcome occurred, 0 if censored
      - time_days  : days from drug_epoch_time to first outcome (or follow-up)

    Follow-up is censored at the earliest of three conditions, mirroring
    ``compute_drug_era_end_time`` in ``extract_drug_initiation_sequences.py``:

    1. **Era gap**: consecutive target drug events are > *era_gap_days* apart.
       Follow-up ends at ``last_target_event + era_gap_days``.
    2. **Treatment switch**: a concept from *competing_concept_ids* appears in
       the generated trajectory.  Follow-up ends at that event.
    3. **End of sequence / no discontinuation**: right-censored at *follow_up_days*.

    Parameters
    ----------
    trajectories
        Generated trajectory events.  Expected columns:
        subject_id, trajectory_id, prediction_time, time, code
    drug_info
        Parquet with columns (person_id, drug_epoch_time, …).
    outcome_concept_ids
        List of OMOP concept_id strings to treat as the outcome.
    follow_up_days
        Maximum follow-up window in days.
    target_concept_ids
        OMOP concept_ids for the arm's own drug class (used for era-gap
        detection).  If None, era-gap censoring is skipped.
    competing_concept_ids
        OMOP concept_ids for competing / exclusion drugs (opposite arm plus
        other excluded antihypertensives).  A trajectory event matching these
        censors follow-up at that point (treatment switch).  If None, switch
        censoring is skipped.
    era_gap_days
        Persistence window in days (default: 30).  Consecutive target drug
        events more than this many days apart terminate the era.

    Returns
    -------
    DataFrame with columns: subject_id, trajectory_id, event, time_days
    """
    outcome_set = set(outcome_concept_ids)

    drug_lf = drug_info.lazy().rename({"person_id": "subject_id"})

    # Attach drug_epoch_time to each event and compute days_since_drug.
    # Only events strictly after the drug start and within follow_up_days matter.
    traj_lf = (
        trajectories.lazy()
        .join(drug_lf.select(["subject_id", "drug_epoch_time"]), on="subject_id", how="left")
        .with_columns(
            pl.col("time").cast(pl.Datetime).dt.epoch(time_unit="s").alias("time_epoch_s"),
        )
        .filter(pl.col("time_epoch_s") > pl.col("drug_epoch_time"))
        .with_columns(
            ((pl.col("time_epoch_s") - pl.col("drug_epoch_time")) / 86_400).alias("days_since_drug")
        )
        .filter(pl.col("days_since_drug") <= follow_up_days)
    )

    # All unique (subject_id, trajectory_id) pairs; effective follow-up starts
    # at follow_up_days and may be reduced by era-gap or competing events below.
    all_pairs = (
        trajectories.lazy()
        .select(["subject_id", "trajectory_id"])
        .unique()
        .with_columns(pl.lit(follow_up_days).alias("effective_follow_up_days"))
    )

    # --- Condition 1: era gap ---
    # Find the first gap > era_gap_days between consecutive target drug events
    # in the generated trajectory; censor at last_target_time + era_gap_days.
    if target_concept_ids:
        target_era_end = (
            traj_lf
            .filter(pl.col("code").is_in(list(target_concept_ids)))
            .group_by(["subject_id", "trajectory_id"])
            .agg(pl.col("days_since_drug").sort().alias("target_days"))
            .with_columns(
                pl.col("target_days").map_elements(
                    lambda days: _find_era_gap_end(days, era_gap_days),
                    return_dtype=pl.Float64,
                ).alias("era_gap_end_days")
            )
            .select(["subject_id", "trajectory_id", "era_gap_end_days"])
        )
        all_pairs = (
            all_pairs
            .join(target_era_end, on=["subject_id", "trajectory_id"], how="left")
            .with_columns(
                pl.min_horizontal(
                    "effective_follow_up_days",
                    pl.col("era_gap_end_days").fill_null(follow_up_days),
                ).alias("effective_follow_up_days")
            )
            .drop("era_gap_end_days")
        )

    # --- Condition 2: treatment switch ---
    # Censor at the first generated event whose code belongs to a competing /
    # exclusion drug class (opposite arm or other excluded antihypertensives).
    if competing_concept_ids:
        first_competing = (
            traj_lf
            .filter(pl.col("code").is_in(list(competing_concept_ids)))
            .group_by(["subject_id", "trajectory_id"])
            .agg(pl.col("days_since_drug").min().alias("competing_days"))
        )
        all_pairs = (
            all_pairs
            .join(first_competing, on=["subject_id", "trajectory_id"], how="left")
            .with_columns(
                pl.min_horizontal(
                    "effective_follow_up_days",
                    pl.col("competing_days").fill_null(follow_up_days),
                ).alias("effective_follow_up_days")
            )
            .drop("competing_days")
        )

    # --- Condition 3: end of record / death ---
    # Censor at [END] or [DEATH] if the model generates one of these tokens,
    # mirroring compute_drug_era_end_time in extract_drug_initiation_sequences.py.
    end_of_record = (
        traj_lf
        .filter(pl.col("code").is_in(list(_RECORD_END_TOKENS)))
        .group_by(["subject_id", "trajectory_id"])
        .agg(pl.col("days_since_drug").min().alias("end_of_record_days"))
    )
    all_pairs = (
        all_pairs
        .join(end_of_record, on=["subject_id", "trajectory_id"], how="left")
        .with_columns(
            pl.min_horizontal(
                "effective_follow_up_days",
                pl.col("end_of_record_days").fill_null(follow_up_days),
            ).alias("effective_follow_up_days")
        )
        .drop("end_of_record_days")
    )

    # --- Outcome events within effective follow-up ---
    outcome_events = (
        traj_lf
        .join(
            all_pairs.select(["subject_id", "trajectory_id", "effective_follow_up_days"]),
            on=["subject_id", "trajectory_id"],
            how="left",
        )
        .filter(pl.col("days_since_drug") <= pl.col("effective_follow_up_days"))
        .filter(pl.col("code").is_in(list(outcome_set)))
        .group_by(["subject_id", "trajectory_id"])
        .agg(pl.col("days_since_drug").min().alias("time_days"))
        .with_columns(pl.lit(1).alias("event"))
    )

    # Left join: censored patients get event=0, time=effective_follow_up_days
    tte = (
        all_pairs
        .join(outcome_events, on=["subject_id", "trajectory_id"], how="left")
        .with_columns(
            pl.col("event").fill_null(0),
            pl.col("time_days").fill_null(pl.col("effective_follow_up_days")),
        )
        .select(["subject_id", "trajectory_id", "event", "time_days"])
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
    # drug_epoch_time is already in obs; only pull arm and era_end from drug_info
    # to avoid duplicate-column conflicts on join.
    _info_raw = _read(drug_info_path)
    if "era_end_epoch_time" not in _info_raw.columns:
        _info_raw = _info_raw.with_columns(pl.lit(None, dtype=pl.Float64).alias("era_end_epoch_time"))
    info = _info_raw.select(["person_id", "arm", "era_end_epoch_time"])

    # Join arm label and era end onto observed outcomes
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
            """Compute (time_days, event) arrays for one arm.

            Censoring at the earlier of follow_up_days and era_end_epoch_time
            (treatment discontinuation or switch to another antihypertensive).
            """
            times, events = [], []
            for row in arm_df.select(["drug_epoch_time", "era_end_epoch_time", oc_id]).iter_rows():
                drug_t, era_end_t, outcome_t = row
                # Effective censoring = min(follow_up_days, era_days)
                if drug_t is not None and era_end_t is not None:
                    era_days = (float(era_end_t) - float(drug_t)) / 86_400
                    censor_days = min(follow_up_days, max(era_days, 0.0))
                else:
                    censor_days = follow_up_days

                if outcome_t is not None and drug_t is not None:
                    t_days = (float(outcome_t) - float(drug_t)) / 86_400
                    if 0 < t_days <= censor_days:
                        times.append(t_days)
                        events.append(1)
                        continue
                times.append(censor_days)
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
    traj_source: pl.DataFrame,
    observed_outcomes: pl.DataFrame,
    drug_info: pl.DataFrame,
    follow_up_days: float,
    source_label: str,
    output_dir: Path,
    target_concept_ids: Optional[Set[str]] = None,
    competing_concept_ids: Optional[Set[str]] = None,
    era_gap_days: float = 30.0,
) -> Optional[Dict]:
    """
    Compare generated source-arm trajectories against the same patients'
    observed outcomes.  HR should be ≈ 1.0 if the model faithfully
    reproduces what actually happened to the source (e.g. ACEi) patients.

    Parameters
    ----------
    traj_source
        Generated trajectories for the source arm (e.g. acei/).
    observed_outcomes
        Observed outcomes from Step 1, filtered to source patients only.
        Columns: person_id, drug_epoch_time, <outcome_concept_id>, …
    drug_info
        drug_info parquet (anchors drug_epoch_time for generated TTE).
    target_concept_ids
        OMOP concept_ids for the source arm's drug class (for era-gap
        censoring in generated trajectories).
    competing_concept_ids
        Concepts for which a generated event censors the source-arm follow-up
        (opposite-arm drugs plus exclusion antihypertensives).
    era_gap_days
        Persistence window in days (default 30).
    """
    print(f"\n  [Faithfulness] Outcome: {outcome_concept_id}")

    # --- Generated source-arm TTE ---
    tte_gen = compute_tte(
        traj_source, drug_info, [outcome_concept_id], follow_up_days,
        target_concept_ids=target_concept_ids,
        competing_concept_ids=competing_concept_ids,
        era_gap_days=era_gap_days,
    )
    agg_gen = (
        tte_gen
        .group_by("subject_id")
        .agg(
            pl.col("event").max().alias("event"),
            pl.col("time_days").mean().alias("time_days"),
        )
    )

    # --- Observed source-arm TTE (drug_epoch_time already in observed_outcomes) ---
    if outcome_concept_id not in observed_outcomes.columns:
        print(f"    Outcome column {outcome_concept_id} not in observed_outcomes — skipping.")
        return None

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
    print(f"    Generated {source_label}: {len(time_gen):,} patients, {n_ev_gen:,} events")
    print(f"    Observed  {source_label}: {len(time_obs):,} patients, {n_ev_obs:,} events")

    if n_ev_gen + n_ev_obs == 0:
        print("    Skipping: no events in either group.")
        return None

    # HR: generated=1, observed=0  (expect ≈ 1.0)
    time_all      = np.concatenate([time_gen, time_obs])
    event_all     = np.concatenate([event_gen, event_obs])
    treatment_all = np.concatenate([np.ones(len(time_gen)), np.zeros(len(time_obs))])

    hr, hr_lower, hr_upper, cox_p = fit_cox(time_all, event_all, treatment_all)
    lr_stat, lr_p = log_rank_test(time_gen, event_gen, time_obs, event_obs)

    print(f"    HR (generated vs observed {source_label}) = {hr:.3f}  "
          f"(95% CI: {hr_lower:.3f}–{hr_upper:.3f})  Cox p={cox_p:.4f}  "
          f"[expect ≈ 1.0]")
    print(f"    Log-rank stat={lr_stat:.3f}  p={lr_p:.4f}")

    # KM curves
    km_t_g, km_s_g = kaplan_meier(time_gen, event_gen)
    km_t_o, km_s_o = kaplan_meier(time_obs, event_obs)
    km_df = pl.concat([
        pl.DataFrame({"time_days": km_t_g, "survival": km_s_g,
                      "arm": f"generated_{source_label}"}),
        pl.DataFrame({"time_days": km_t_o, "survival": km_s_o,
                      "arm": f"observed_{source_label}"}),
    ])
    km_df.write_csv(str(output_dir / f"faithfulness_km_{outcome_concept_id}.csv"))

    return {
        "outcome_concept_id": outcome_concept_id,
        f"n_generated_{source_label}": len(time_gen),
        f"n_observed_{source_label}": len(time_obs),
        f"n_events_generated_{source_label}": n_ev_gen,
        f"n_events_observed_{source_label}": n_ev_obs,
        "hr_generated_vs_observed": hr,
        "hr_lower_95": hr_lower,
        "hr_upper_95": hr_upper,
        "cox_p_value": cox_p,
        "log_rank_stat": lr_stat,
        "log_rank_p_value": lr_p,
        "note": f"HR should be ≈ 1.0 if model faithfully reproduces {source_label} outcomes",
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
    target_concept_ids_a: Optional[Set[str]] = None,
    target_concept_ids_b: Optional[Set[str]] = None,
    competing_concept_ids_a: Optional[Set[str]] = None,
    competing_concept_ids_b: Optional[Set[str]] = None,
    era_gap_days: float = 30.0,
) -> Dict:
    """
    Run full HR analysis for a single outcome concept.

    HR is reported as arm_a relative to arm_b (e.g. ACEi vs Thiazide).
    Since both arms share the same patients (within-patient counterfactual),
    each patient contributes one aggregated time-to-event value per arm.

    Parameters
    ----------
    target_concept_ids_a / target_concept_ids_b
        OMOP concept_ids for each arm's own drug class (era-gap censoring).
    competing_concept_ids_a / competing_concept_ids_b
        Concepts whose appearance in a generated trajectory censors follow-up
        (opposite-arm drugs + exclusion antihypertensives).
    era_gap_days
        Persistence window in days (default 30).
    """
    print(f"\n  Outcome: {outcome_concept_id}")

    tte_a = compute_tte(
        traj_arm_a, drug_info, [outcome_concept_id], follow_up_days,
        target_concept_ids=target_concept_ids_a,
        competing_concept_ids=competing_concept_ids_a,
        era_gap_days=era_gap_days,
    )
    tte_b = compute_tte(
        traj_arm_b, drug_info, [outcome_concept_id], follow_up_days,
        target_concept_ids=target_concept_ids_b,
        competing_concept_ids=competing_concept_ids_b,
        era_gap_days=era_gap_days,
    )

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


def _print_era_duration_stats(
    drug_info: pl.DataFrame,
    follow_up_days: float,
    arm_a_label: str,
    arm_b_label: str,
) -> None:
    """Print descriptive statistics of drug era duration by arm."""
    if "era_end_epoch_time" not in drug_info.columns:
        print("  (era_end_epoch_time not available — skipping era duration stats)")
        return

    print("\n" + "=" * 60)
    print("Drug era duration statistics")
    print("=" * 60)

    arm_map = {"source": arm_a_label, "comparator": arm_b_label}
    for arm_key, arm_label in arm_map.items():
        arm_df = drug_info.filter(pl.col("arm") == arm_key)
        if arm_df.is_empty():
            continue

        era_days = (
            arm_df
            .with_columns(
                # Clamp to 0 to handle any residual timestamp noise in drug_info
                (
                    (pl.col("era_end_epoch_time") - pl.col("drug_epoch_time")) / 86_400
                ).clip(lower_bound=0.0).alias("era_days")
            )
            .select("era_days")
        )

        n_total      = len(arm_df)
        n_censored   = era_days.filter(pl.col("era_days").is_null()).height
        n_terminated = n_total - n_censored

        print(f"\n  {arm_label}  (n={n_total:,})")
        print(f"    Right-censored (no era end) : {n_censored:,}  ({n_censored/n_total:.1%})")
        print(f"    Era terminated (gap/switch) : {n_terminated:,}  ({n_terminated/n_total:.1%})")

        if n_terminated > 0:
            stats = (
                era_days
                .filter(pl.col("era_days").is_not_null())
                .select([
                    pl.col("era_days").mean().alias("mean"),
                    pl.col("era_days").median().alias("median"),
                    pl.col("era_days").quantile(0.25).alias("p25"),
                    pl.col("era_days").quantile(0.75).alias("p75"),
                    pl.col("era_days").min().alias("min"),
                    pl.col("era_days").max().alias("max"),
                ])
                .row(0, named=True)
            )
            print(f"    Era duration (days) among terminated patients:")
            print(f"      mean={stats['mean']:.1f}  median={stats['median']:.1f}"
                  f"  IQR=[{stats['p25']:.1f}, {stats['p75']:.1f}]"
                  f"  min={stats['min']:.1f}  max={stats['max']:.1f}")

    print("=" * 60)


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

    # Expand drug concept IDs to descendants for competing-event censoring.
    arm_a_concepts: Set[str] = set()
    arm_b_concepts: Set[str] = set()
    exclusion_concepts: Set[str] = set()
    competing_for_arm_a: Optional[Set[str]] = None
    competing_for_arm_b: Optional[Set[str]] = None

    if args.vocab_path and args.arm_a_concept_ids and args.arm_b_concept_ids:
        arm_a_ids = [int(x.strip()) for x in args.arm_a_concept_ids.split(",")]
        arm_b_ids = [int(x.strip()) for x in args.arm_b_concept_ids.split(",")]
        print("Expanding drug concepts …")
        arm_a_concepts = expand_to_descendants(args.vocab_path, arm_a_ids)
        arm_b_concepts = expand_to_descendants(args.vocab_path, arm_b_ids)
        print(f"  Arm A concepts: {len(arm_a_concepts):,}")
        print(f"  Arm B concepts: {len(arm_b_concepts):,}")

    if args.vocab_path and args.exclusion_concept_ids:
        excl_ids = [int(x.strip()) for x in args.exclusion_concept_ids.split(",")]
        print("Expanding exclusion concepts for competing-event censoring …")
        exclusion_concepts = expand_to_descendants(args.vocab_path, excl_ids)
        print(f"  Exclusion concepts (other antihypertensives): {len(exclusion_concepts):,}")

    # Build per-arm competing sets (union of opposite-arm + exclusion drugs).
    # These censor follow-up when a generated trajectory contains a switch event;
    # trajectories are NOT removed — only their follow-up is shortened.
    if arm_a_concepts or arm_b_concepts or exclusion_concepts:
        competing_for_arm_a = (arm_b_concepts | exclusion_concepts) or None
        competing_for_arm_b = (arm_a_concepts | exclusion_concepts) or None

    print(f"Loading {args.arm_a} trajectories …")
    traj_a = load_trajectories(args.trajectories_dir, args.arm_a)
    print(f"  {len(traj_a):,} events")

    print(f"Loading {args.arm_b} trajectories …")
    traj_b = load_trajectories(args.trajectories_dir, args.arm_b)
    print(f"  {len(traj_b):,} events")

    def _read_parquet(path: str) -> pl.DataFrame:
        return pl.read_parquet(
            os.path.join(path, "*.parquet") if os.path.isdir(path) else path
        )

    print("Loading drug info …")
    drug_info = _read_parquet(args.drug_info_path)
    _print_era_duration_stats(drug_info, args.follow_up_days, args.arm_a, args.arm_b)

    # ------------------------------------------------------------------
    # Faithfulness check: generated comparator vs observed comparator
    # HR ≈ 1.0 means the model faithfully reproduces arm_b outcomes.
    # ------------------------------------------------------------------
    if args.observed_outcomes_path:
        print("\n" + "=" * 60)
        print(f"Faithfulness check: generated {args.arm_a} vs observed {args.arm_a}")
        print("(Same patients, same drug — HR should be ≈ 1.0 if the model is faithful)")
        print("=" * 60)

        observed_outcomes = _read_parquet(args.observed_outcomes_path)
        # Keep only observed source patients (e.g. ACEi initiators)
        source_person_ids = drug_info.filter(pl.col("arm") == "source").select("person_id")
        obs_source = observed_outcomes.join(source_person_ids, on="person_id", how="inner")

        faithfulness_results = []
        for outcome_id in outcome_ids:
            res = faithfulness_check_outcome(
                outcome_concept_id=outcome_id,
                traj_source=traj_a,           # generated ACEi trajectories
                observed_outcomes=obs_source,  # same ACEi patients' real outcomes
                drug_info=drug_info,
                follow_up_days=args.follow_up_days,
                source_label=args.arm_a,
                output_dir=output_dir,
                target_concept_ids=arm_a_concepts or None,
                competing_concept_ids=competing_for_arm_a,
                era_gap_days=args.era_gap_days,
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
            target_concept_ids_a=arm_a_concepts or None,
            target_concept_ids_b=arm_b_concepts or None,
            competing_concept_ids_a=competing_for_arm_a,
            competing_concept_ids_b=competing_for_arm_b,
            era_gap_days=args.era_gap_days,
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
    parser.add_argument(
        "--exclusion_concept_ids",
        default=None,
        help="Comma-separated ingredient concept_ids for excluded antihypertensives "
             "(ARBs, dCCBs, ndCCBs, 2nd-line agents). When a generated trajectory "
             "contains one of these codes, follow-up is censored at that event.",
    )
    parser.add_argument(
        "--era_gap_days",
        type=float,
        default=30.0,
        help="Persistence window in days for drug era computation (default: 30). "
             "Consecutive target drug events separated by more than this many days "
             "terminate the era; follow-up is censored at last_event + era_gap_days.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
