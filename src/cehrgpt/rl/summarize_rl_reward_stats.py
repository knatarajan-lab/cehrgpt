"""
Compute per-(concept, window) reward summary statistics from the training set
and plot reward weight for a user-specified concept_id across prediction horizons.

Usage
-----
python -m cehrgpt.rl.summarize_rl_reward_stats \
    --tokenized_dataset_path /path/to/tokenized_dataset \
    --tokenizer_path /path/to/tokenizer \
    --output_dir /path/to/output \
    [--vocab_dir /path/to/athena_parquet] \
    [--plot_concept_ids 4021791 443454] \
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
import os
from typing import Dict, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk

from cehrgpt.rl.reward import compute_event_weight, compute_prevalence_stats


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
    parser.add_argument("--plot_concept_ids", nargs="+", default=None,
                        help="One or more concept IDs to plot reward weight vs prediction horizon for")
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
# Concept name lookup
# ---------------------------------------------------------------------------

def load_concept_name_map(vocab_dir: str) -> Dict[str, str]:
    """Return {concept_id: concept_name} for all concepts in the Athena parquet."""
    concept_parquet = os.path.join(vocab_dir, "concept")
    if not os.path.isdir(concept_parquet):
        return {}
    df = pd.read_parquet(concept_parquet, columns=["concept_id", "concept_name"])
    return {str(row.concept_id): row.concept_name for row in df.itertuples(index=False)}


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
    n_patients = len(train_dataset)
    prevalence_stats = compute_prevalence_stats(
        train_dataset,
        target_concept_ids,
        args.prediction_windows,
        args.prediction_split_fraction,
        args.num_proc,
        args.batch_size,
    )
    print(f"Computed prevalence stats over {n_patients} patients.")

    # Load concept names from vocab if available
    concept_name_map: Dict[str, str] = {}
    if args.vocab_dir:
        print("Loading concept names from vocab...")
        concept_name_map = load_concept_name_map(os.path.expanduser(args.vocab_dir))
        print(f"Loaded {len(concept_name_map)} concept names.")

    # Build summary DataFrame: (concept_id, concept_name, window, count, prevalence, reward_weight)
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
            "concept_name": concept_name_map.get(concept_id, ""),
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
    # Scatter plot: reward weight for user-specified concept IDs
    # -----------------------------------------------------------------------
    if not args.plot_concept_ids:
        print("No --plot_concept_ids specified; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for concept_id in args.plot_concept_ids:
        sub = df[df["concept_id"] == concept_id].sort_values("window_days")
        if sub.empty:
            print(f"Warning: concept_id {concept_id} not found in stats; skipping.")
            continue
        name = concept_name_map.get(concept_id, concept_id)
        label = f"{concept_id}: {name[:60]}"
        ax.scatter(sub["window_days"], sub["reward_weight"], label=label, s=80)
        ax.plot(sub["window_days"], sub["reward_weight"], alpha=0.5)
        plotted += 1

    if plotted == 0:
        print("No valid concept IDs found in stats; skipping plot.")
        return

    ax.set_xlabel("Prediction Window (days)")
    ax.set_ylabel("Reward Weight α(c, w)")
    ax.set_title("Reward Weight vs Prediction Horizon")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(args.output_dir, "reward_weight_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
