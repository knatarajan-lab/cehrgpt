"""
Visualize reward weights for a given list of concept IDs across prediction horizons.

Reads the CSV produced by summarize_rl_reward_stats.py.

When --include_descendants is set, the provided concept IDs are treated as ancestors
and expanded to all descendants. The reward weights are then aggregated (mean ± std)
across descendants per prediction window, with one band per ancestor.

Usage
-----
# Plot specific concept IDs directly
python -m cehrgpt.rl.plot_rl_reward_weights \
    --stats_csv /path/to/concept_window_reward_stats.csv \
    --concept_ids 4021791 443454 \
    [--output /path/to/reward_weight_plot.png]

# Provide ancestor IDs and plot aggregated reward weight across descendants
python -m cehrgpt.rl.plot_rl_reward_weights \
    --stats_csv /path/to/concept_window_reward_stats.csv \
    --concept_ids 4021791 \
    --include_descendants \
    --vocab_dir /path/to/athena_parquet \
    [--output /path/to/reward_weight_plot.png]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_csv", required=True,
                        help="Path to concept_window_reward_stats.csv from summarize_rl_reward_stats.py")
    parser.add_argument("--concept_ids", nargs="+", required=True,
                        help="One or more concept IDs (ancestors when --include_descendants is set)")
    parser.add_argument("--include_descendants", action="store_true",
                        help="Expand concept_ids to descendants and plot aggregated mean ± std per ancestor")
    parser.add_argument("--vocab_dir", default=None,
                        help="Path to Athena parquet folder; required when --include_descendants is set")
    parser.add_argument("--output", default="reward_weight_plot.png",
                        help="Output path for the plot (default: reward_weight_plot.png)")
    return parser.parse_args()


def get_descendants(ancestor_id, ca_df):
    """Return descendant concept ID strings for a single ancestor."""
    return (
        ca_df[ca_df["ancestor_concept_id"] == int(ancestor_id)]
        ["descendant_concept_id"]
        .unique()
        .astype(str)
        .tolist()
    )


def load_concept_ancestor(vocab_dir):
    concept_ancestor_path = os.path.join(os.path.expanduser(vocab_dir), "concept_ancestor")
    if not os.path.isdir(concept_ancestor_path):
        raise FileNotFoundError(
            f"concept_ancestor parquet not found at {concept_ancestor_path}. "
            "Check --vocab_dir points to the Athena parquet folder."
        )
    return pd.read_parquet(
        concept_ancestor_path,
        columns=["ancestor_concept_id", "descendant_concept_id"],
    )


def load_concept_names(vocab_dir):
    concept_path = os.path.join(os.path.expanduser(vocab_dir), "concept")
    if not os.path.isdir(concept_path):
        return {}
    df = pd.read_parquet(concept_path, columns=["concept_id", "concept_name"])
    return {str(row.concept_id): row.concept_name for row in df.itertuples(index=False)}


def plot_direct(ax, df, concept_ids):
    """One line per concept ID."""
    plotted = 0
    for concept_id in concept_ids:
        sub = df[df["concept_id"] == concept_id].sort_values("window_days")
        if sub.empty:
            print(f"Warning: concept_id {concept_id} not found in stats; skipping.")
            continue
        name = sub["concept_name"].iloc[0]
        label = f"{concept_id}: {str(name)[:60]}" if name else concept_id
        ax.scatter(sub["window_days"], sub["reward_weight"], s=60)
        ax.plot(sub["window_days"], sub["reward_weight"], alpha=0.7, label=label)
        plotted += 1
    return plotted


def plot_aggregated(ax, df, ancestor_ids, ca_df, concept_name_map):
    """Mean ± std band across descendants, one band per ancestor."""
    plotted = 0
    for ancestor_id in ancestor_ids:
        desc_ids = get_descendants(ancestor_id, ca_df)
        sub = df[df["concept_id"].isin(desc_ids)]
        if sub.empty:
            print(f"Warning: no descendants of {ancestor_id} found in stats; skipping.")
            continue

        n_desc = sub["concept_id"].nunique()
        agg = sub.groupby("window_days")["reward_weight"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)

        name = concept_name_map.get(str(ancestor_id), str(ancestor_id))
        label = f"{ancestor_id}: {name[:50]} (n={n_desc})"

        color = f"C{plotted}"
        ax.plot(agg["window_days"], agg["mean"], marker="o", label=label, color=color)
        ax.fill_between(
            agg["window_days"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.2,
            color=color,
        )
        print(f"  {ancestor_id}: {n_desc} descendants matched in stats.")
        plotted += 1
    return plotted


def main():
    args = parse_args()

    df = pd.read_csv(args.stats_csv, dtype={"concept_id": str})

    fig, ax = plt.subplots(figsize=(10, 5))

    if args.include_descendants:
        if not args.vocab_dir:
            raise ValueError("--vocab_dir is required when --include_descendants is set.")
        ca_df = load_concept_ancestor(args.vocab_dir)
        concept_name_map = load_concept_names(args.vocab_dir)
        plotted = plot_aggregated(ax, df, args.concept_ids, ca_df, concept_name_map)
    else:
        plotted = plot_direct(ax, df, args.concept_ids)

    if plotted == 0:
        print("No valid concept IDs found in stats; nothing to plot.")
        return

    print(f"Plotted {plotted} concept(s).")
    ax.set_xlabel("Prediction Window (days)")
    ax.set_ylabel("Reward Weight α(c, w)")
    ax.set_title("Reward Weight vs Prediction Horizon")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
