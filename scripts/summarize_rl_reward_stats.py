"""
Compute per-(concept, window) reward summary statistics from the training set
and plot average reward weight for a target concept (default: pancreatic cancer)
across prediction horizons.

Usage
-----
python scripts/summarize_rl_reward_stats.py \
    --tokenized_dataset_path /path/to/tokenized_dataset \
    --tokenizer_path /path/to/tokenizer \
    --output_dir /path/to/output \
    [--vocab_dir /path/to/athena_parquet] \
    [--concept_name "pancreatic cancer"] \
    [--prediction_windows 30 90 180 365 730 1095 1460 1825] \
    [--rarity_gamma 0.5] \
    [--alpha_max 10.0] \
    [--window_eta 0.5] \
    [--window_ref_days 365.0] \
    [--prevalence_epsilon 0.0001] \
    [--num_proc 16] \
    [--batch_size 5000]
"""

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized_dataset_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_dir", default=None,
                        help="Path to Athena parquet folder (concept/, concept_ancestor/, etc.)")
    parser.add_argument("--concept_name", default="pancreatic cancer",
                        help="Concept name to highlight in the scatter plot (case-insensitive substring match)")
    parser.add_argument("--prediction_windows", nargs="+", type=int,
                        default=[30, 90, 180, 365, 730, 1095, 1460, 1825])
    parser.add_argument("--rarity_gamma", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=10.0)
    parser.add_argument("--window_eta", type=float, default=0.5)
    parser.add_argument("--window_ref_days", type=float, default=365.0)
    parser.add_argument("--prevalence_epsilon", type=float, default=1e-4)
    parser.add_argument("--prediction_split_fraction", type=float, default=0.5,
                        help="Fraction along the sequence used as the split point")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=5000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reward weight (mirrors reward.py)
# ---------------------------------------------------------------------------

def compute_event_weight(
    prevalence: float,
    window_days: int,
    gamma: float,
    alpha_max: float,
    eta: float,
    w_ref: float,
    epsilon: float,
) -> float:
    rarity = min(alpha_max, (1.0 / (prevalence + epsilon)) ** gamma)
    window_factor = 1.0 + eta * math.log(1.0 + window_days / w_ref)
    return rarity * window_factor


# ---------------------------------------------------------------------------
# Target concept ID loading (mirrors hf_cehrgpt_rl_runner.py)
# ---------------------------------------------------------------------------

def load_target_concept_ids(tokenizer, vocab_dir) -> Set[str]:
    if vocab_dir:
        vocab_dir = os.path.expanduser(vocab_dir)
        concept_parquet_dir = os.path.join(vocab_dir, "concept")
        if os.path.isdir(concept_parquet_dir):
            try:
                from cehrgpt.omop.ontology import Ontology
                ontology = Ontology(vocab_dir)
                all_vocab_tokens = set(tokenizer.get_vocab().keys())
                condition_ids = {
                    tok for tok in all_vocab_tokens
                    if tok.isnumeric() and ontology.get_domain(tok) == "Condition"
                }
                if condition_ids:
                    print(f"Loaded {len(condition_ids)} Condition concept IDs from ontology.")
                    return condition_ids
            except Exception as exc:
                print(f"Warning: could not load ontology ({exc}); falling back.")

    fallback = set(tokenizer._motor_time_to_event_codes)
    fallback.discard("0")
    print(f"Using {len(fallback)} _motor_time_to_event_codes as target concept IDs.")
    return fallback


# ---------------------------------------------------------------------------
# Prevalence stats via dataset.map
# ---------------------------------------------------------------------------

def compute_prevalence_stats(
    dataset,
    target_concept_ids: Set[str],
    windows: List[int],
    prediction_split_fraction: float,
    num_proc: int,
    batch_size: int,
) -> Tuple[Dict[Tuple[str, int], float], int]:
    """
    Returns (prevalence_dict, n_patients) where
    prevalence_dict maps (concept_id, window_days) -> fraction of patients
    who have that concept within that window after the split point.
    """
    from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token

    _DEMOGRAPHICS_SIZE = 4
    max_window = max(windows)
    sorted_windows = sorted(windows)
    target_set = target_concept_ids  # closure for map fn

    def _map_hits(examples):
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
                elif token in target_set:
                    for w in sorted_windows:
                        if cumulative_days <= w:
                            found[w].add(token)
            for w in sorted_windows:
                for c in found[w]:
                    hits.append(f"{c}:{w}")
            result.append(hits)
        return {"_hits": result}

    print(f"Running dataset.map over {len(dataset)} examples (num_proc={num_proc})...")
    mapped = dataset.map(
        _map_hits,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=[c for c in dataset.column_names if c != "_hits"],
        desc="computing reward stats",
    )

    counter: Counter = Counter()
    n_patients = 0
    for hits in tqdm(mapped["_hits"], desc="aggregating", unit="patient"):
        counter.update(hits)
        n_patients += 1

    if n_patients == 0:
        return {}, 0

    prevalence = {
        (c, w): counter[f"{c}:{w}"] / n_patients
        for c in target_concept_ids
        for w in windows
    }
    return prevalence, n_patients


# ---------------------------------------------------------------------------
# Concept name lookup
# ---------------------------------------------------------------------------

def find_concept_ids_by_name(vocab_dir: str, name_query: str) -> Dict[str, str]:
    """Return {concept_id: concept_name} where name contains name_query (case-insensitive)."""
    concept_parquet = os.path.join(vocab_dir, "concept")
    if not os.path.isdir(concept_parquet):
        return {}
    df = pd.read_parquet(concept_parquet, columns=["concept_id", "concept_name"])
    mask = df["concept_name"].str.contains(name_query, case=False, na=False)
    return {str(row.concept_id): row.concept_name for row in df[mask].itertuples()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
    tokenizer = CehrGptTokenizer.from_pretrained(os.path.expanduser(args.tokenizer_path))
    print(f"Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

    # Load training dataset
    dataset_path = os.path.expanduser(args.tokenized_dataset_path)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"] if hasattr(dataset, "__getitem__") and "train" in dataset else dataset
    print(f"Training set size: {len(train_dataset)}")

    # Target concept IDs
    target_concept_ids = load_target_concept_ids(tokenizer, args.vocab_dir)

    # Compute prevalence stats
    prevalence_stats, n_patients = compute_prevalence_stats(
        train_dataset,
        target_concept_ids,
        args.prediction_windows,
        args.prediction_split_fraction,
        args.num_proc,
        args.batch_size,
    )
    print(f"Computed prevalence stats over {n_patients} patients.")

    # Build summary DataFrame: (concept_id, window, count, prevalence, reward_weight)
    rows = []
    for (concept_id, window), prev in prevalence_stats.items():
        weight = compute_event_weight(
            prev,
            window,
            args.rarity_gamma,
            args.alpha_max,
            args.window_eta,
            args.window_ref_days,
            args.prevalence_epsilon,
        )
        rows.append({
            "concept_id": concept_id,
            "window_days": window,
            "n_patients": n_patients,
            "count": round(prev * n_patients),
            "prevalence": prev,
            "reward_weight": weight,
        })

    df = pd.DataFrame(rows).sort_values(["concept_id", "window_days"])

    # Save CSV
    csv_path = os.path.join(args.output_dir, "concept_window_reward_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")

    # Save prevalence JSON (compatible with RL runner format)
    json_path = os.path.join(args.output_dir, "prevalence_stats.json")
    with open(json_path, "w") as f:
        json.dump(
            [{"concept_id": cid, "window": w, "prevalence": p}
             for (cid, w), p in prevalence_stats.items()],
            f, indent=2,
        )
    print(f"Saved prevalence JSON to {json_path}")

    # -----------------------------------------------------------------------
    # Scatter plot: reward weight for target concept across prediction windows
    # -----------------------------------------------------------------------
    target_concept_map: Dict[str, str] = {}

    if args.vocab_dir:
        vocab_dir = os.path.expanduser(args.vocab_dir)
        target_concept_map = find_concept_ids_by_name(vocab_dir, args.concept_name)
        # Restrict to concepts that are actually in target_concept_ids
        target_concept_map = {k: v for k, v in target_concept_map.items() if k in target_concept_ids}
        if not target_concept_map:
            print(f"No concepts matching '{args.concept_name}' found in vocab/target set.")
    else:
        print("No --vocab_dir provided; cannot look up concept names for the plot.")

    if target_concept_map:
        fig, ax = plt.subplots(figsize=(8, 5))
        for concept_id, concept_name in sorted(target_concept_map.items()):
            sub = df[df["concept_id"] == concept_id].sort_values("window_days")
            if sub.empty:
                continue
            ax.scatter(sub["window_days"], sub["reward_weight"], label=f"{concept_id}: {concept_name[:50]}", s=80)
            ax.plot(sub["window_days"], sub["reward_weight"], alpha=0.5)

        ax.set_xlabel("Prediction Window (days)")
        ax.set_ylabel("Reward Weight α(c, w)")
        ax.set_title(f"Reward Weight vs Prediction Horizon\n({args.concept_name})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(args.output_dir, "reward_weight_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}")
    else:
        print("Skipping plot (no matching concepts found).")


if __name__ == "__main__":
    main()
