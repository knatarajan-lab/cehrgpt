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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from cehrgpt.analysis.counterfactual.legend_outcomes import load_outcome_definitions

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
INPATIENT_ER_VISIT_CONCEPTS: Set[str] = {"9201", "9203", "262"}


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
    df = pl.concat(dfs).with_row_index("position")
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

_KEYS = ["subject_id", "trajectory_id"]


def _with_days_since_drug(trajectories: pl.DataFrame, drug_info: pl.DataFrame) -> pl.LazyFrame:
    """Attach ``days_since_drug`` (relative to ``drug_epoch_time``) to every trajectory event."""
    drug_lf = drug_info.lazy().rename({"person_id": "subject_id"})
    return (
        trajectories.lazy()
        .join(drug_lf.select(["subject_id", "drug_epoch_time"]), on="subject_id", how="left")
        .with_columns(
            pl.col("time").cast(pl.Datetime).dt.epoch(time_unit="s").alias("time_epoch_s"),
        )
        .with_columns(
            ((pl.col("time_epoch_s") - pl.col("drug_epoch_time")) / 86_400).alias("days_since_drug")
        )
    )


def compute_follow_up(
    trajectories: pl.DataFrame,
    drug_info: pl.DataFrame,
    follow_up_days: float,
    target_concept_ids: Optional[Set[str]] = None,
    competing_concept_ids: Optional[Set[str]] = None,
    era_gap_days: float = 30.0,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Work out how long each generated trajectory is observed.

    The generated trajectory is the only data we have for a counterfactual
    patient, so the time of its **last event** (any code) is its last
    observation: nothing is known after it, and follow-up cannot extend past it.
    This is the same rule as for a real record, which is censored at the last
    recorded event.  Follow-up ends at the earliest of

    1. the last generated event, or *follow_up_days* when that comes first;
    2. the end of the drug era: the first gap of more than *era_gap_days*
       between target-drug events.  The index drug at day 0 starts the chain and
       the last observation closes it, so a patient who never refills is
       censored at ``era_gap_days`` and one who stops refilling is censored
       ``era_gap_days`` after the last refill;
    3. the first event of a competing concept (treatment switch);
    4. a record-end token (``[END]`` / ``[DEATH]``).

    Returns ``(events, follow_up)``.  *events* are the trajectory events after
    the index drug and within *follow_up_days*, with ``days_since_drug``.
    *follow_up* has one row per (subject_id, trajectory_id) with
    ``last_observed_days``, ``effective_follow_up_days`` and ``terminated``
    (follow-up ended before the end of the observed data).
    """
    n_unmatched = len(set(trajectories["subject_id"].to_list()) - set(drug_info["person_id"].to_list()))
    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} trajectory subjects are not in drug_info; they have no index "
            f"time and are given zero follow-up."
        )
    days = _with_days_since_drug(trajectories, drug_info)
    after_index = days.filter(pl.col("days_since_drug") > 0)
    events = after_index.filter(pl.col("days_since_drug") <= follow_up_days)

    last_observed = after_index.group_by(_KEYS).agg(
        pl.col("days_since_drug").max().alias("last_observed_days")
    )
    follow_up = (
        trajectories.lazy()
        .select(_KEYS)
        .unique()
        .join(last_observed, on=_KEYS, how="left")
        .with_columns(
            pl.col("last_observed_days").fill_null(0.0).clip(upper_bound=follow_up_days)
        )
        .with_columns(pl.col("last_observed_days").alias("effective_follow_up_days"))
    )

    if target_concept_ids:
        refills = events.filter(pl.col("code").is_in(list(target_concept_ids))).select(
            _KEYS + [pl.col("days_since_drug").alias("day")]
        )
        index_day = follow_up.select(_KEYS + [pl.lit(0.0).alias("day")])
        last_day = follow_up.select(_KEYS + [pl.col("last_observed_days").alias("day")])
        era_gap_end = (
            pl.concat([index_day, refills, last_day])
            .sort(_KEYS + ["day"])
            .with_columns(pl.col("day").shift(1).over(_KEYS).alias("previous_day"))
            .filter(pl.col("day") - pl.col("previous_day") > era_gap_days)
            .group_by(_KEYS)
            .agg((pl.col("previous_day").min() + era_gap_days).alias("era_gap_end_days"))
        )
        follow_up = (
            follow_up.join(era_gap_end, on=_KEYS, how="left")
            .with_columns(
                pl.min_horizontal("effective_follow_up_days", "era_gap_end_days").alias(
                    "effective_follow_up_days"
                )
            )
            .drop("era_gap_end_days")
        )

    if competing_concept_ids:
        first_competing = (
            events.filter(pl.col("code").is_in(list(competing_concept_ids)))
            .group_by(_KEYS)
            .agg(pl.col("days_since_drug").min().alias("competing_days"))
        )
        follow_up = (
            follow_up.join(first_competing, on=_KEYS, how="left")
            .with_columns(
                pl.min_horizontal("effective_follow_up_days", "competing_days").alias(
                    "effective_follow_up_days"
                )
            )
            .drop("competing_days")
        )

    end_of_record = (
        events.filter(pl.col("code").is_in(list(_RECORD_END_TOKENS)))
        .group_by(_KEYS)
        .agg(pl.col("days_since_drug").min().alias("end_of_record_days"))
    )
    follow_up = (
        follow_up.join(end_of_record, on=_KEYS, how="left")
        .with_columns(
            pl.min_horizontal("effective_follow_up_days", "end_of_record_days").alias(
                "effective_follow_up_days"
            ),
            (
                (pl.col("effective_follow_up_days") < pl.col("last_observed_days"))
                | pl.col("end_of_record_days").is_not_null()
            ).alias("terminated"),
        )
        .drop("end_of_record_days")
    )
    return events, follow_up


def _find_first_inpatient_outcome(
    codes: List[str],
    times: List[float],
    outcome_set: Set[str],
) -> Optional[float]:
    """
    Return the time of the first outcome event that is **linked to** an
    inpatient or ER visit, i.e. the outcome must be contained within the same
    ``[VS]…[VE]`` block as one of the ``INPATIENT_ER_VISIT_CONCEPTS``
    (9201, 9203, 262).

    The visit-type concept always precedes outcome events within a block, so a
    single forward pass suffices: ``in_qualifying`` is set when the visit-type
    token is seen and reset on ``[VS]`` or ``[VE]``.
    """
    in_qualifying = False
    for code, t in zip(codes, times):
        if code == "[VS]":
            in_qualifying = False
        elif code in INPATIENT_ER_VISIT_CONCEPTS:
            in_qualifying = True
        elif code == "[VE]":
            in_qualifying = False
        elif in_qualifying and code in outcome_set:
            return t
    return None


def compute_tte(
    trajectories: pl.DataFrame,
    drug_info: pl.DataFrame,
    outcome_concept_ids: List[str],
    follow_up_days: float,
    target_concept_ids: Optional[Set[str]] = None,
    competing_concept_ids: Optional[Set[str]] = None,
    era_gap_days: float = 30.0,
    inpatient_only: bool = False,
    expanded_outcome_set: Optional[Set[str]] = None,
) -> pl.DataFrame:
    """
    For each (subject_id, trajectory_id) compute:
      - event      : 1 if outcome occurred, 0 if censored
      - time_days  : days from drug_epoch_time to first outcome (or follow-up)

    Follow-up ends at the last generated event (the last observation of the
    counterfactual patient), the end of the drug era, a treatment switch, a
    record-end token or *follow_up_days*, whichever comes first.  See
    :func:`compute_follow_up` for the exact rules.

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
    inpatient_only
        When True, outcomes are only counted when they occur inside an
        inpatient or ER visit in the generated trajectory (visit concept
        9201/9203/262 immediately follows the ``[VS]`` token).
    expanded_outcome_set
        Pre-expanded set of descendant concept_ids (strings) for the outcome.
        When provided this is used instead of ``outcome_concept_ids`` for code
        matching, so that all descendant codes count toward the ancestor
        outcome.  Build with :func:`expand_to_descendants`.

    Returns
    -------
    DataFrame with columns: subject_id, trajectory_id, event, time_days
    """
    outcome_set = expanded_outcome_set if expanded_outcome_set is not None else set(outcome_concept_ids)

    traj_lf, all_pairs = compute_follow_up(
        trajectories,
        drug_info,
        follow_up_days,
        target_concept_ids,
        competing_concept_ids,
        era_gap_days,
    )

    # --- Outcome events within effective follow-up ---
    if inpatient_only:
        # Sequential visit-type detection requires Python-level iteration.
        # Pre-filter to relevant codes before collecting to keep memory low.
        _relevant = outcome_set | INPATIENT_ER_VISIT_CONCEPTS | {"[VS]", "[VE]"}
        _inpatient_df = (
            traj_lf
            .filter(pl.col("code").is_in(list(_relevant)))
            .collect()
            .sort(["subject_id", "trajectory_id", "days_since_drug", "position"])
        )

        _outcome_rows: List[dict] = []
        for _grp in _inpatient_df.partition_by(
            ["subject_id", "trajectory_id"], maintain_order=True
        ):
            _t = _find_first_inpatient_outcome(
                _grp["code"].to_list(), _grp["days_since_drug"].to_list(), outcome_set
            )
            if _t is not None:
                _outcome_rows.append({
                    "subject_id":    _grp["subject_id"][0],
                    "trajectory_id": _grp["trajectory_id"][0],
                    "time_days":     _t,
                    "event":         1,
                })

        if _outcome_rows:
            _raw_outcomes = pl.DataFrame(_outcome_rows).lazy()
        else:
            _raw_outcomes = pl.DataFrame({
                "subject_id":    pl.Series([], dtype=_inpatient_df["subject_id"].dtype),
                "trajectory_id": pl.Series([], dtype=_inpatient_df["trajectory_id"].dtype),
                "time_days":     pl.Series([], dtype=pl.Float64),
                "event":         pl.Series([], dtype=pl.Int32),
            }).lazy()
        # Apply effective follow-up ceiling
        outcome_events = (
            _raw_outcomes
            .join(
                all_pairs.select(["subject_id", "trajectory_id", "effective_follow_up_days"]),
                on=["subject_id", "trajectory_id"],
                how="left",
            )
            .filter(pl.col("time_days") <= pl.col("effective_follow_up_days"))
            .drop("effective_follow_up_days")
        )
    else:
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
# Cox PH via partial likelihood  (single covariate, Breslow ties)
# ---------------------------------------------------------------------------

def _breslow_terms(
    time: np.ndarray, event: np.ndarray, x: np.ndarray, beta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The distinct event times and, for each, the number of events and the sums over its risk set
    (S0 = sum exp(b x), S1 = sum x exp(b x), S2 = sum x^2 exp(b x)); the risk set of a time is every
    subject with a time at or after it, so tied times share one risk set (Breslow).
    """
    order = np.argsort(time, kind="stable")
    t, e, xs = time[order], event[order], x[order]
    w = np.exp(beta * xs)
    s0_all = np.cumsum(w[::-1])[::-1]
    s1_all = np.cumsum((w * xs)[::-1])[::-1]
    s2_all = np.cumsum((w * xs * xs)[::-1])[::-1]
    event_times, d = np.unique(t[e == 1], return_counts=True)
    first = np.searchsorted(t, event_times, side="left")
    return event_times, d, s0_all[first], s1_all[first], s2_all[first]


def _cox_partial_log_likelihood(beta: float, time: np.ndarray, event: np.ndarray, x: np.ndarray) -> float:
    """
    Negative Breslow partial log-likelihood of a Cox PH model with a single covariate *x*:

        l(b) = sum_i event_i * b * x_i  -  sum_k d_k * log(sum_{j: t_j >= t_k} exp(b x_j))

    over the distinct event times t_k with d_k events each.
    """
    _, d, s0, _, _ = _breslow_terms(time, event, x, beta)
    return float(-(beta * np.sum(event * x) - np.sum(d * np.log(s0 + EPSILON))))


def _cox_score_residuals(
    time: np.ndarray, event: np.ndarray, x: np.ndarray, beta: float
) -> Tuple[np.ndarray, float]:
    """
    The score residual of every subject at *beta* (they sum to zero at the maximum) and the information.

        r_i = event_i (x_i - xbar(T_i)) - exp(b x_i) sum_{k: t_k <= T_i} d_k (x_i - xbar_k) / S0_k
    """
    event_times, d, s0, s1, s2 = _breslow_terms(time, event, x, beta)
    xbar = s1 / s0
    information = float(np.sum(d * (s2 / s0 - xbar ** 2)))
    cum0 = np.cumsum(d / s0)
    cum1 = np.cumsum(d * xbar / s0)
    last = np.searchsorted(event_times, time, side="right") - 1          # last event time <= T_i
    a0 = np.where(last >= 0, cum0[np.maximum(last, 0)], 0.0)
    a1 = np.where(last >= 0, cum1[np.maximum(last, 0)], 0.0)
    own = np.minimum(np.searchsorted(event_times, time, side="left"), len(event_times) - 1)
    residual = event * (x - xbar[own]) - np.exp(beta * x) * (x * a0 - a1)
    return residual, information


def fit_cox_details(
    time: np.ndarray,
    event: np.ndarray,
    treatment: np.ndarray,
    clusters: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Fit the Cox model with one covariate and return the estimate with its standard errors.

    The standard error of the model (the inverse of the information) assumes independent observations. When
    *clusters* is given (one label per observation, e.g. the subject id), the robust sandwich standard error with
    the observations of a cluster summed is returned as well, and it is what hr_lower/hr_upper/p_value use.
    Generated trajectories of one patient, and the two arms of the same patient, are correlated, so the model-based
    interval is too narrow for them.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    x = np.asarray(treatment, dtype=float)
    result = minimize_scalar(
        _cox_partial_log_likelihood,
        bounds=(-10, 10),
        method="bounded",
        args=(time, event, x),
    )
    beta = float(result.x)
    residual, information = _cox_score_residuals(time, event, x, beta)
    se_model = 1.0 / math.sqrt(information + EPSILON)
    se_robust = None
    n_clusters = None
    if clusters is not None:
        _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
        n_clusters = int(inverse.max()) + 1
        cluster_scores = np.bincount(inverse, weights=residual, minlength=n_clusters)
        correction = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1.0
        se_robust = math.sqrt(correction * float(np.sum(cluster_scores ** 2))) / (information + EPSILON)
    se = se_robust if se_robust is not None else se_model

    MAX_EXP = 700  # math.exp overflows above ~709
    bound = lambda v: math.exp(max(-MAX_EXP, min(MAX_EXP, v)))
    z = beta / (se + EPSILON)
    return {
        "beta": beta,
        "hr": bound(beta),
        "hr_lower_95": bound(beta - 1.96 * se),
        "hr_upper_95": bound(beta + 1.96 * se),
        "p_value": float(chi2.sf(z ** 2, df=1)),
        "se_model": se_model,
        "se_robust": se_robust,
        "n_clusters": n_clusters,
        "hr_lower_95_model": bound(beta - 1.96 * se_model),
        "hr_upper_95_model": bound(beta + 1.96 * se_model),
    }


def fit_cox(
    time: np.ndarray,
    event: np.ndarray,
    treatment: np.ndarray,
    clusters: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """
    Estimate HR = exp(β) for a binary treatment covariate, see :func:`fit_cox_details`.

    Returns
    -------
    (hr, hr_lower_95, hr_upper_95, p_value)
    """
    fit = fit_cox_details(time, event, treatment, clusters)
    return fit["hr"], fit["hr_lower_95"], fit["hr_upper_95"], fit["p_value"]


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

    At every distinct event time t_k the survival is multiplied by (1 - d_k / n_k), with n_k the number of subjects
    whose time is at or after t_k (events and censored ones alike).

    Returns (time_points, survival_probabilities), starting at (0, 1).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    t = np.sort(time)
    event_times, d = np.unique(time[event == 1], return_counts=True)
    at_risk = len(t) - np.searchsorted(t, event_times, side="left")
    survival = np.cumprod(1.0 - d / at_risk)
    return np.concatenate([[0.0], event_times]), np.concatenate([[1.0], survival])


# ---------------------------------------------------------------------------
# Observed-outcome baseline analysis
# ---------------------------------------------------------------------------

def observed_tte(
    observed: pl.DataFrame, outcome_concept_id: str, follow_up_days: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (time_days, event) of one outcome for patients from ``observed_outcomes``.

    *observed* has ``drug_epoch_time``, the outcome column (epoch time of the
    first outcome, null if none) and, joined from ``drug_info``,
    ``era_end_epoch_time`` and ``last_observed_epoch_time`` (both nullable).
    Follow-up is censored at the earliest of *follow_up_days*, the end of the
    drug era (discontinuation or switch) and the last observation of the record,
    the same rules as for generated trajectories (:func:`compute_follow_up`).
    """
    for column in ("era_end_epoch_time", "last_observed_epoch_time"):
        if column not in observed.columns:
            observed = observed.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))

    def days(column: str) -> pl.Expr:
        return (pl.col(column).cast(pl.Float64) - pl.col("drug_epoch_time")) / 86_400

    frame = observed.select(
        pl.min_horizontal(
            pl.lit(float(follow_up_days)),
            days("era_end_epoch_time").clip(lower_bound=0.0),
            days("last_observed_epoch_time").clip(lower_bound=0.0),
        ).alias("censor_days"),
        days(outcome_concept_id).alias("outcome_days"),
    ).with_columns(
        (
            (pl.col("outcome_days") > 0) & (pl.col("outcome_days") <= pl.col("censor_days"))
        ).fill_null(False).alias("event")
    )
    time_days = frame.select(
        pl.when(pl.col("event")).then(pl.col("outcome_days")).otherwise(pl.col("censor_days"))
    ).to_series()
    return time_days.to_numpy().astype(float), frame["event"].to_numpy().astype(int)


# ---------------------------------------------------------------------------
# Outcome types: primary outcomes and negative / positive controls
# ---------------------------------------------------------------------------

PRIMARY = "primary"
NEGATIVE_CONTROL = "negative_control"
POSITIVE_CONTROL = "positive_control"


def parse_outcome_specs(
    primary: str,
    negative_controls: Optional[str] = None,
    positive_controls: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
    """
    Parse the outcome options.

    *primary* and *negative_controls* are comma-separated concept_ids.  A negative
    control has no effect of the drug by construction, so its expected HR is 1.
    *positive_controls* entries are ``concept_id`` or ``concept_id:expected_hr``;
    the expected HR is the known effect and may be left out when only the
    direction is known.

    Returns ``(all_outcome_ids, type_by_outcome, expected_hr_by_outcome)``.  An
    outcome can have only one role.
    """
    types: Dict[str, str] = {}
    expected: Dict[str, float] = {}

    def add(outcome_id: str, outcome_type: str, expected_hr: Optional[float] = None) -> None:
        if outcome_id in types:
            raise ValueError(
                f"outcome {outcome_id} is listed as both {types[outcome_id]} and {outcome_type}"
            )
        types[outcome_id] = outcome_type
        if expected_hr is not None:
            expected[outcome_id] = expected_hr

    for token in primary.split(","):
        if token.strip():
            add(token.strip(), PRIMARY)
    for token in (negative_controls or "").split(","):
        if token.strip():
            add(token.strip(), NEGATIVE_CONTROL, 1.0)
    for token in (positive_controls or "").split(","):
        if not token.strip():
            continue
        outcome_id, _, hr = token.partition(":")
        if hr.strip() and float(hr) <= 0:
            raise ValueError(f"expected HR of positive control {outcome_id} must be > 0, got {hr}")
        add(outcome_id.strip(), POSITIVE_CONTROL, float(hr) if hr.strip() else None)
    return list(types), types, expected


def label_outcomes(
    summary: pl.DataFrame, types: Dict[str, str], expected_hr: Dict[str, float]
) -> pl.DataFrame:
    """
    Add ``outcome_type``, ``expected_hr`` and ``ci_covers_expected`` (does the
    95% CI contain the expected HR) after ``outcome_concept_id``.
    """
    if summary.is_empty():
        return summary
    labels = pl.DataFrame(
        {
            "outcome_concept_id": list(types),
            "outcome_type": [types[o] for o in types],
            "expected_hr": [expected_hr.get(o) for o in types],
        },
        schema={
            "outcome_concept_id": pl.String,
            "outcome_type": pl.String,
            "expected_hr": pl.Float64,
        },
    )
    labelled = summary.with_columns(pl.col("outcome_concept_id").cast(pl.String)).join(
        labels, on="outcome_concept_id", how="left"
    )
    labelled = labelled.with_columns(
        (
            (pl.col("hr_lower_95") <= pl.col("expected_hr"))
            & (pl.col("expected_hr") <= pl.col("hr_upper_95"))
        ).alias("ci_covers_expected")
    )
    front = ["outcome_concept_id", "outcome_type", "expected_hr", "ci_covers_expected"]
    return labelled.select(front + [c for c in labelled.columns if c not in front])


def summarize_controls(labelled: pl.DataFrame) -> pl.DataFrame:
    """
    One row per control type: how many controls, how many have a 95% CI that
    contains the expected HR, and the mean and spread of log(HR / expected HR).
    For negative controls the mean is the systematic bias of the analysis and a
    coverage well below 95% means the intervals are too narrow.
    """
    controls = labelled.filter(pl.col("outcome_type") != PRIMARY)
    if controls.is_empty():
        return controls
    return (
        controls.with_columns(
            (pl.col("hr") / pl.col("expected_hr")).log().alias("log_hr_ratio")
        )
        .group_by("outcome_type", maintain_order=True)
        .agg(
            pl.len().alias("n_controls"),
            pl.col("ci_covers_expected").drop_nulls().len().alias("n_with_expected_hr"),
            pl.col("ci_covers_expected").sum().alias("n_ci_covers_expected"),
            pl.col("log_hr_ratio").mean().alias("mean_log_hr_ratio"),
            pl.col("log_hr_ratio").std().alias("sd_log_hr_ratio"),
        )
    )


def write_labelled_summary(
    results: List[Dict],
    summary_path: Path,
    control_summary_path: Path,
    types: Optional[Dict[str, str]],
    expected_hr: Optional[Dict[str, float]],
) -> pl.DataFrame:
    """Write the per-outcome summary (labelled by outcome type) and the control summary."""
    summary = pl.DataFrame(results)
    if types:
        summary = label_outcomes(summary, types, expected_hr or {})
    summary.write_csv(str(summary_path))
    if types and not summary.is_empty():
        controls = summarize_controls(summary)
        if not controls.is_empty():
            controls.write_csv(str(control_summary_path))
            print(f"\nControl outcomes ({control_summary_path.name}):")
            print(controls)
    return summary


def read_dropped_person_ids(overlap_scores_paths: List[str]) -> Set[int]:
    """
    person_ids that ``create_drug_swap_context.py`` dropped because their
    preference score fell outside the trimming band.

    Each path is a scores parquet (or a directory of them) written by the swap
    step; a patient that is absent from every file was never scored and is kept.
    """
    frames = [
        pl.read_parquet(os.path.join(path, "*.parquet") if os.path.isdir(path) else path)
        for path in overlap_scores_paths
    ]
    scores = pl.concat(frames, how="vertical_relaxed")
    return set(scores.filter(~pl.col("retained"))["person_id"].to_list())


def analyse_observed_outcomes(
    observed_outcomes_path: str,
    drug_info_path: str,
    outcome_concept_ids: List[str],
    follow_up_days: float,
    output_dir: Path,
    source_arm_label: str = "acei",
    comparator_arm_label: str = "thiazide",
    excluded_person_ids: Optional[Set[int]] = None,
    outcome_types: Optional[Dict[str, str]] = None,
    expected_hr: Optional[Dict[str, float]] = None,
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
    excluded_person_ids
        Patients dropped by the overlap (preference score) check; they are
        removed from the observed baseline as well so both analyses use the
        same patients.
    outcome_types / expected_hr
        Role of each outcome (primary, negative or positive control) and the
        expected HR of the controls; see ``parse_outcome_specs``.
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
    if excluded_person_ids:
        excluded = list(excluded_person_ids)
        n_before = len(obs)
        obs = obs.filter(~pl.col("person_id").is_in(excluded))
        print(f"  Overlap check: dropped {n_before - len(obs):,} observed patients")
    # drug_epoch_time is already in obs; only pull arm and era_end from drug_info
    # to avoid duplicate-column conflicts on join.
    _info_raw = _read(drug_info_path)
    for _column in ("era_end_epoch_time", "last_observed_epoch_time"):
        if _column not in _info_raw.columns:
            print(f"  Warning: drug_info has no {_column}; re-run the extraction to censor on it.")
            _info_raw = _info_raw.with_columns(pl.lit(None, dtype=pl.Float64).alias(_column))
    info = _info_raw.select(
        ["person_id", "arm", "era_end_epoch_time", "last_observed_epoch_time"]
    )

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

        time_a, event_a = observed_tte(source_df, oc_id, follow_up_days)
        time_b, event_b = observed_tte(comparator_df, oc_id, follow_up_days)

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
        write_labelled_summary(
            results,
            out_path,
            output_dir / "observed_control_outcome_summary.csv",
            outcome_types,
            expected_hr,
        )
        print(f"\nObserved HR summary written to {out_path}")


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
    inpatient_only: bool = False,
    expanded_outcome_set: Optional[Set[str]] = None,
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
    inpatient_only
        When True, outcomes only counted inside inpatient/ER visits.
    expanded_outcome_set
        Descendant-expanded concept_ids for the outcome; see
        :func:`expand_to_descendants`.
    """
    print(f"\n  Outcome: {outcome_concept_id}")

    tte_a = compute_tte(
        traj_arm_a, drug_info, [outcome_concept_id], follow_up_days,
        target_concept_ids=target_concept_ids_a,
        competing_concept_ids=competing_concept_ids_a,
        era_gap_days=era_gap_days,
        inpatient_only=inpatient_only,
        expanded_outcome_set=expanded_outcome_set,
    )
    tte_b = compute_tte(
        traj_arm_b, drug_info, [outcome_concept_id], follow_up_days,
        target_concept_ids=target_concept_ids_b,
        competing_concept_ids=competing_concept_ids_b,
        era_gap_days=era_gap_days,
        inpatient_only=inpatient_only,
        expanded_outcome_set=expanded_outcome_set,
    )

    # Use each (subject_id, trajectory_id) as a separate observation in the
    # Cox model.  This preserves trajectory-level variance and avoids ad-hoc
    # aggregation choices (max event / mean time).  Trajectories of a patient
    # share their pre-drug context, and the two arms contain the same patients,
    # so observations are clustered by subject_id and the CI and p-value use the
    # cluster-robust variance.
    time_a  = tte_a["time_days"].to_numpy()
    event_a = tte_a["event"].to_numpy().astype(int)
    time_b  = tte_b["time_days"].to_numpy()
    event_b = tte_b["event"].to_numpy().astype(int)

    n_patients_a = tte_a["subject_id"].n_unique()
    n_patients_b = tte_b["subject_id"].n_unique()
    n_events_a = int(event_a.sum())
    n_events_b = int(event_b.sum())
    print(f"    {arm_a_label}: {n_patients_a:,} patients / {len(time_a):,} trajectories, {n_events_a:,} events")
    print(f"    {arm_b_label}: {n_patients_b:,} patients / {len(time_b):,} trajectories, {n_events_b:,} events")

    if n_events_a + n_events_b == 0:
        print("    Skipping: no events observed in either arm.")
        return None

    # Cox PH: treatment=1 for arm_a (ACEi), treatment=0 for arm_b (Thiazide)
    # HR > 1 means arm_a has higher hazard than arm_b
    time_all      = np.concatenate([time_a, time_b])
    event_all     = np.concatenate([event_a, event_b])
    treatment_all = np.concatenate([np.ones(len(time_a)), np.zeros(len(time_b))])

    clusters = np.concatenate([tte_a["subject_id"].to_numpy(), tte_b["subject_id"].to_numpy()])

    cox = fit_cox_details(time_all, event_all, treatment_all, clusters=clusters)
    hr, hr_lower, hr_upper, cox_p = cox["hr"], cox["hr_lower_95"], cox["hr_upper_95"], cox["p_value"]
    lr_stat, lr_p = log_rank_test(time_a, event_a, time_b, event_b)

    print(f"    HR ({arm_a_label} vs {arm_b_label}) = {hr:.3f}  "
          f"(95% CI: {hr_lower:.3f}–{hr_upper:.3f})  Cox p={cox_p:.4f}  "
          f"[robust, {cox['n_clusters']:,} clusters]")
    print(f"    Model-based 95% CI: {cox['hr_lower_95_model']:.3f}–{cox['hr_upper_95_model']:.3f} "
          f"(ignores correlation between trajectories)")
    print(f"    Log-rank stat={lr_stat:.3f}  p={lr_p:.4f} (ignores correlation)")

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
        f"n_patients_{arm_a_label}": n_patients_a,
        f"n_patients_{arm_b_label}": n_patients_b,
        f"n_trajectories_{arm_a_label}": len(time_a),
        f"n_trajectories_{arm_b_label}": len(time_b),
        f"n_events_{arm_a_label}": int(event_a.sum()),
        f"n_events_{arm_b_label}": int(event_b.sum()),
        "hr": hr,
        "hr_lower_95": hr_lower,
        "hr_upper_95": hr_upper,
        "cox_p_value": cox_p,
        "n_clusters": cox["n_clusters"],
        "hr_lower_95_model": cox["hr_lower_95_model"],
        "hr_upper_95_model": cox["hr_upper_95_model"],
        "log_rank_stat": lr_stat,
        "log_rank_p_value": lr_p,
        "hr_interpretation": f"HR of {arm_a_label} vs {arm_b_label} (>1 = higher hazard in {arm_a_label})",
    }


def _compute_generated_era_end(
    trajectories: pl.DataFrame,
    drug_info: pl.DataFrame,
    follow_up_days: float,
    target_concept_ids: Optional[Set[str]],
    competing_concept_ids: Optional[Set[str]],
    era_gap_days: float,
) -> pl.DataFrame:
    """
    For each (subject_id, trajectory_id) in *trajectories*, compute the
    effective follow-up in days from drug start with the same rules as
    ``compute_tte`` (see :func:`compute_follow_up`).

    Returns a DataFrame with columns:
        subject_id, trajectory_id, effective_follow_up_days, terminated (bool)
    """
    _, follow_up = compute_follow_up(
        trajectories,
        drug_info,
        follow_up_days,
        target_concept_ids,
        competing_concept_ids,
        era_gap_days,
    )
    return follow_up.collect()


def _print_generated_era_duration_stats(
    trajectories: pl.DataFrame,
    drug_info: pl.DataFrame,
    follow_up_days: float,
    arm_label: str,
    target_concept_ids: Optional[Set[str]],
    competing_concept_ids: Optional[Set[str]],
    era_gap_days: float,
) -> None:
    """Print descriptive statistics of drug era duration from generated trajectories."""
    era_df = _compute_generated_era_end(
        trajectories, drug_info, follow_up_days,
        target_concept_ids, competing_concept_ids, era_gap_days,
    )

    # Aggregate per patient: mean effective_follow_up_days across trajectories;
    # a patient is "terminated" if ANY trajectory was terminated.
    per_patient = (
        era_df
        .group_by("subject_id")
        .agg(
            pl.col("effective_follow_up_days").mean().alias("era_days"),
            pl.col("terminated").any().alias("terminated"),
        )
    )

    n_total      = len(per_patient)
    n_terminated = per_patient["terminated"].sum()
    n_censored   = n_total - n_terminated

    print(f"\n  {arm_label} [generated]  (n={n_total:,} patients, {len(era_df):,} trajectories)")
    print(f"    Right-censored (reached follow-up) : {n_censored:,}  ({n_censored/n_total:.1%})")
    print(f"    Era terminated (gap/switch/end)    : {n_terminated:,}  ({n_terminated/n_total:.1%})")

    terminated_df = per_patient.filter(pl.col("terminated"))
    if len(terminated_df) > 0:
        stats = (
            terminated_df
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

    outcome_ids, outcome_types, expected_hr = parse_outcome_specs(
        args.outcome_concept_ids or "",
        args.negative_control_outcome_ids,
        args.positive_control_outcome_ids,
    )
    defined_concepts: Dict[str, Set[str]] = {}
    defined_inpatient: Set[str] = set()
    if args.outcome_definitions_path:
        for entry in load_outcome_definitions(args.outcome_definitions_path):
            label = entry["label"]
            if label in outcome_types:
                raise ValueError(f"outcome {label} is given twice")
            outcome_ids.append(label)
            outcome_types[label] = entry["role"]
            if entry["role"] == NEGATIVE_CONTROL:
                expected_hr[label] = 1.0
            defined_concepts[label] = set(entry["concepts"])
            if entry["inpatient_only"]:
                defined_inpatient.add(label)
    if not outcome_ids:
        raise ValueError("give --outcome_concept_ids and/or --outcome_definitions_path")
    for outcome_type in (PRIMARY, NEGATIVE_CONTROL, POSITIVE_CONTROL):
        members = [o for o in outcome_ids if outcome_types[o] == outcome_type]
        if members:
            shown = members if len(members) <= 10 else members[:10] + ["..."]
            print(f"{len(members)} {outcome_type} outcomes: {shown}")

    inpatient_outcome_ids: Set[str] = set()
    if args.inpatient_outcome_concept_ids:
        inpatient_outcome_ids = {
            x.strip() for x in args.inpatient_outcome_concept_ids.split(",")
        }
        print(f"Inpatient-only outcomes: {sorted(inpatient_outcome_ids)}")
    inpatient_outcome_ids |= defined_inpatient

    # Expand each outcome concept to all descendants so that any descendant
    # code in a generated trajectory counts toward the ancestor outcome.
    outcome_expanded: Dict[str, Set[str]] = dict(defined_concepts)
    if args.vocab_path:
        print("Expanding outcome concepts …")
        for oid in outcome_ids:
            if oid in defined_concepts:
                continue
            outcome_expanded[oid] = expand_to_descendants(args.vocab_path, [int(oid)])
            print(f"  {oid}: {len(outcome_expanded[oid]):,} concepts (including descendants)")
    else:
        for oid in outcome_ids:
            if oid not in defined_concepts:
                outcome_expanded[oid] = {oid}

    dropped_person_ids: Set[int] = set()
    if args.overlap_scores_path:
        dropped_person_ids = read_dropped_person_ids(args.overlap_scores_path)
        print(f"Overlap check: {len(dropped_person_ids):,} patients dropped from every analysis")

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
            excluded_person_ids=dropped_person_ids,
            outcome_types=outcome_types,
            expected_hr=expected_hr,
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

    # Build per-arm competing sets (union of opposite-arm + exclusion drugs).  These censor follow-up when a generated trajectory
    # contains a switch event; trajectories are NOT removed — only their
    # follow-up is shortened.
    if arm_a_concepts or arm_b_concepts or exclusion_concepts:
        competing_for_arm_a = (arm_b_concepts | exclusion_concepts) or None
        competing_for_arm_b = (arm_a_concepts | exclusion_concepts) or None

    print(f"Loading {args.arm_a} trajectories …")
    traj_a = load_trajectories(args.trajectories_dir, args.arm_a)
    print(f"  {len(traj_a):,} events")

    print(f"Loading {args.arm_b} trajectories …")
    traj_b = load_trajectories(args.trajectories_dir, args.arm_b)
    print(f"  {len(traj_b):,} events")

    if dropped_person_ids:
        # Generation may predate the overlap check; keep only retained patients.
        dropped = list(dropped_person_ids)
        traj_a = traj_a.filter(~pl.col("subject_id").is_in(dropped))
        traj_b = traj_b.filter(~pl.col("subject_id").is_in(dropped))

    def _read_parquet(path: str) -> pl.DataFrame:
        return pl.read_parquet(
            os.path.join(path, "*.parquet") if os.path.isdir(path) else path
        )

    print("Loading drug info …")
    drug_info = _read_parquet(args.drug_info_path)
    if dropped_person_ids:
        drug_info = drug_info.filter(~pl.col("person_id").is_in(list(dropped_person_ids)))
    _print_era_duration_stats(drug_info, args.follow_up_days, args.arm_a, args.arm_b)

    print("\n" + "=" * 60)
    print("Drug era duration statistics (generated trajectories)")
    print("=" * 60)
    _print_generated_era_duration_stats(
        traj_a, drug_info, args.follow_up_days, args.arm_a,
        target_concept_ids=arm_a_concepts or None,
        competing_concept_ids=competing_for_arm_a,
        era_gap_days=args.era_gap_days,
    )
    _print_generated_era_duration_stats(
        traj_b, drug_info, args.follow_up_days, args.arm_b,
        target_concept_ids=arm_b_concepts or None,
        competing_concept_ids=competing_for_arm_b,
        era_gap_days=args.era_gap_days,
    )
    print("=" * 60)

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
            inpatient_only=(outcome_id in inpatient_outcome_ids),
            expanded_outcome_set=outcome_expanded[outcome_id],
        )
        if result is not None:
            results.append(result)

    summary_path = output_dir / "hazard_ratio_summary.csv"
    summary = write_labelled_summary(
        results,
        summary_path,
        output_dir / "control_outcome_summary.csv",
        outcome_types,
        expected_hr,
    )

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
        "--overlap_scores_path",
        nargs="+",
        default=None,
        help="Scores parquet(s) written by create_drug_swap_context.py "
             "(--overlap_scores_path there), one per swap direction. Patients the "
             "overlap check dropped are removed from the observed baseline, the "
             "trajectories and drug_info, so every analysis uses the same patients.",
    )
    parser.add_argument(
        "--drug_info_path",
        required=True,
        help="drug_info.parquet from extract_drug_initiation_sequences.py",
    )
    parser.add_argument(
        "--outcome_concept_ids",
        default=None,
        help="Comma-separated OMOP concept_ids of the primary outcomes, "
             "e.g. 4329847,316139,4110192,376713. Needed unless "
             "--outcome_definitions_path gives the outcomes.",
    )
    parser.add_argument(
        "--outcome_definitions_path",
        default=None,
        help="outcome_definitions.json from legend_outcomes.py: outcomes (and negative "
             "controls) defined by concept sets, with their inpatient/ER restriction. "
             "Analysed next to the outcomes given here; use the file of the extraction step.",
    )
    parser.add_argument(
        "--negative_control_outcome_ids",
        default=None,
        help="Optional comma-separated concept_ids of negative control outcomes: "
             "outcomes the drugs are not expected to affect (expected HR = 1). Analysed "
             "like the primary outcomes and reported with outcome_type, ci_covers_expected "
             "and a summary of the systematic bias in control_outcome_summary.csv. "
             "They must also be extracted (pass them to the extraction step too).",
    )
    parser.add_argument(
        "--positive_control_outcome_ids",
        default=None,
        help="Optional comma-separated positive control outcomes with a known effect, "
             "each as concept_id or concept_id:expected_hr (e.g. 4329847:1.5). Reported "
             "like the negative controls; the expected HR is used when given. "
             "They must also be extracted.",
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
        help="OMOP vocabulary root (concept_ancestor/ parquets). Required to expand "
             "--arm_a_concept_ids, --arm_b_concept_ids, --exclusion_concept_ids "
             "and the outcome concepts to their descendants.",
    )
    parser.add_argument(
        "--arm_a_concept_ids",
        default=None,
        help="Comma-separated ingredient concept_ids for arm A drug class. Used for the "
             "era gap of arm A and as treatment-switch censoring for arm B.",
    )
    parser.add_argument(
        "--arm_b_concept_ids",
        default=None,
        help="Comma-separated ingredient concept_ids for arm B drug class. Used for the "
             "era gap of arm B and as treatment-switch censoring for arm A.",
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
    parser.add_argument(
        "--inpatient_outcome_concept_ids",
        default=None,
        help="Comma-separated OMOP concept_ids for outcomes that must occur "
             "inside an inpatient or ER visit (9201/9203/262). Used for "
             "hospitalisation outcomes such as HF (316139) and AMI (312327).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
