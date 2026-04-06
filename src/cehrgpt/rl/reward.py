"""
Reward computation for CEHR-GPT RL fine-tuning.

Implements the patient-level reward:

    R_i = Σ_{(c,τ)∈F_i} α(c,w(τ)) · p̂(c,w(τ))
          ─────────────────────────────────────────
               Σ_{(c,τ)∈F_i} α(c,w(τ))

where

    α(c,w) = min(α_max, (1/(π_{c,w}+ε))^γ) · (1 + η·log(1 + w/w_ref))

    p̂(c,w) = (1/K) Σ_k 1{c appears in rollout k within window w}
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from transformers.utils import logging

LOG = logging.get_logger("transformers")


# ---------------------------------------------------------------------------
# Event weight
# ---------------------------------------------------------------------------

def compute_event_weight(
    prevalence: float,
    window_days: int,
    gamma: float,
    alpha_max: float,
    eta: float,
    w_ref: float,
    epsilon: float = 1e-4,
) -> float:
    """α(c, w) = min(α_max, (1/(π+ε))^γ) · (1 + η·log(1 + w/w_ref))."""
    rarity = min(alpha_max, (1.0 / (prevalence + epsilon)) ** gamma)
    window_factor = 1.0 + eta * math.log(1.0 + window_days / w_ref)
    return rarity * window_factor


# ---------------------------------------------------------------------------
# Condition extraction from a single rollout
# ---------------------------------------------------------------------------

def extract_conditions_from_rollout(
    rollout_tokens: List[str],
    prefix_len: int,
    target_concept_ids: Set[str],
    windows: List[int],
) -> Dict[int, Set[str]]:
    """
    Parse the generated (non-prefix) portion of a rollout trajectory, tracking
    cumulative time via ATT tokens, and record which target conditions appear
    within each prediction window.

    Args:
        rollout_tokens: Full decoded token list (prefix + generated), as returned
            by ``generate_single_batch``.
        prefix_len: Number of tokens belonging to the prefix (excluded from search).
        target_concept_ids: Set of condition concept ID strings to watch for.
        windows: Prediction horizon windows in days, e.g. [30, 90, 180, 365, 730].

    Returns:
        Dict mapping window_days → set of condition concept IDs found within that window.
    """
    from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token

    found: Dict[int, Set[str]] = {w: set() for w in windows}
    cumulative_days = 0.0
    max_window = max(windows)

    for token in rollout_tokens[prefix_len:]:
        if is_att_token(token):
            try:
                cumulative_days += extract_time_interval_in_days(token)
            except ValueError:
                pass
            if cumulative_days > max_window:
                break
        elif token in target_concept_ids:
            for w in windows:
                if cumulative_days <= w:
                    found[w].add(token)

    return found


# ---------------------------------------------------------------------------
# Patient-level reward
# ---------------------------------------------------------------------------

def compute_patient_reward(
    future_conditions: List[Tuple[str, float]],
    rollout_conditions_list: List[Dict[int, Set[str]]],
    prevalence_stats: Dict[Tuple[str, int], float],
    windows: List[int],
    gamma: float,
    alpha_max: float,
    eta: float,
    w_ref: float,
    epsilon: float,
    false_positive_lambda: float = 0.0,
) -> float:
    """
    Compute R_i for a single patient.

    Args:
        future_conditions: List of (concept_id, days_to_event) for true future events.
        rollout_conditions_list: K dicts, each mapping window_days → set of concept IDs
            found in that rollout within the window.
        prevalence_stats: Dict mapping (concept_id, window_days) → π_{c,w}.
        windows: Prediction horizon windows in days.
        gamma, alpha_max, eta, w_ref, epsilon: Weight hyperparameters.
        false_positive_lambda: Penalty weight for false-positive generated conditions.

    Returns:
        Scalar reward R_i ∈ [0, 1] (higher is better).
    """
    if not future_conditions:
        return 0.0

    K = len(rollout_conditions_list)
    if K == 0:
        return 0.0

    def _assign_window(days: float) -> Optional[int]:
        """Return the smallest window that covers `days`, or None if beyond all."""
        for w in sorted(windows):
            if days <= w:
                return w
        return None

    numerator = 0.0
    denominator = 0.0

    for concept_id, days_to_event in future_conditions:
        w = _assign_window(days_to_event)
        if w is None:
            continue
        prev = prevalence_stats.get((concept_id, w), epsilon)
        weight = compute_event_weight(prev, w, gamma, alpha_max, eta, w_ref, epsilon)

        hit_count = sum(
            1 for rollout in rollout_conditions_list
            if concept_id in rollout.get(w, set())
        )
        p_hat = hit_count / K

        numerator += weight * p_hat
        denominator += weight

    r_true = numerator / denominator if denominator > 0 else 0.0

    if false_positive_lambda <= 0.0:
        return r_true

    # Optional false-positive penalty
    true_set = {cid for cid, _ in future_conditions}
    fp_num = 0.0
    fp_den = 0.0
    for rollout in rollout_conditions_list:
        for w in windows:
            for found_cid in rollout.get(w, set()):
                if found_cid not in true_set:
                    prev = prevalence_stats.get((found_cid, w), epsilon)
                    weight = compute_event_weight(prev, w, gamma, alpha_max, eta, w_ref, epsilon)
                    fp_num += weight
    fp_den = max(fp_den, 1.0)
    r_extra = fp_num / fp_den

    return r_true - false_positive_lambda * r_extra


# ---------------------------------------------------------------------------
# Prevalence statistics
# ---------------------------------------------------------------------------

def compute_prevalence_stats(
    dataset,
    target_concept_ids: Set[str],
    windows: List[int],
    prediction_split_fraction: float = 0.5,
    num_proc: int = 1,
    batch_size: int = 1000,
) -> Dict[Tuple[str, int], float]:
    """
    Estimate π_{c,w} = P(condition c within window w days of a random split point)
    by scanning a HuggingFace Dataset of tokenized CEHR-GPT sequences.

    Uses ``dataset.map`` for parallel processing.  Each example emits a list of
    ``"concept_id:window"`` hit strings; these are aggregated with a Counter after
    the map pass.

    Args:
        dataset: HuggingFace Dataset with ``concept_ids`` (List[str]) column.
        target_concept_ids: Condition concept ID strings to track.
        windows: Prediction horizon windows in days.
        prediction_split_fraction: Where to place the split point in each sequence
            (0.5 = midpoint).
        num_proc: Number of parallel workers for ``dataset.map``.
        batch_size: Batch size passed to ``dataset.map``.

    Returns:
        Dict mapping (concept_id, window_days) → estimated prevalence.
    """
    from tqdm import tqdm

    from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token

    _DEMOGRAPHICS_SIZE = 4
    max_window = max(windows)
    sorted_windows = sorted(windows)

    def _map_hits(examples):
        """Return a ``_hits`` column: list-of-strings per example."""
        from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token

        result = []
        for concept_ids in examples["concept_ids"]:
            hits = []
            if len(concept_ids) <= _DEMOGRAPHICS_SIZE:
                result.append(hits)
                continue

            split_idx = max(
                _DEMOGRAPHICS_SIZE,
                int(len(concept_ids) * prediction_split_fraction),
            )
            cumulative_days = 0.0
            found: Dict[int, Set[str]] = {w: set() for w in sorted_windows}

            for token in concept_ids[split_idx:]:
                if is_att_token(token):
                    try:
                        cumulative_days += extract_time_interval_in_days(token)
                    except ValueError:
                        pass
                    if cumulative_days > max_window:
                        break
                elif token in target_concept_ids:
                    for w in sorted_windows:
                        if cumulative_days <= w:
                            found[w].add(token)
                            break  # assign to smallest covering window only

            for w in sorted_windows:
                for c in found[w]:
                    hits.append(f"{c}:{w}")
            result.append(hits)
        return {"_hits": result}

    LOG.info(
        "Computing prevalence stats over %d examples (num_proc=%d)…",
        len(dataset),
        num_proc,
    )
    mapped = dataset.map(
        _map_hits,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=[c for c in dataset.column_names if c != "_hits"],
        desc="prevalence stats",
    )

    counter: Counter = Counter()
    n_patients = 0
    for hits in tqdm(mapped["_hits"], desc="aggregating prevalence", unit="patient"):
        counter.update(hits)
        n_patients += 1

    if n_patients == 0:
        LOG.warning("compute_prevalence_stats: dataset is empty, returning zero prevalences.")
        return {(c, w): 0.0 for c in target_concept_ids for w in windows}

    LOG.info("Computed prevalence stats over %d patients.", n_patients)
    return {
        (c, w): counter[f"{c}:{w}"] / n_patients
        for c in target_concept_ids
        for w in windows
    }
