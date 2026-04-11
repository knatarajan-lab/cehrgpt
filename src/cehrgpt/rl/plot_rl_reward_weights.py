"""
Visualize reward weights for a given list of concept IDs across prediction horizons.

Reads the CSV produced by summarize_rl_reward_stats.py.

Usage
-----
python -m cehrgpt.rl.plot_rl_reward_weights \
    --stats_csv /path/to/concept_window_reward_stats.csv \
    --concept_ids 4021791 443454 \
    [--output /path/to/reward_weight_plot.png]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_csv", required=True,
                        help="Path to concept_window_reward_stats.csv from summarize_rl_reward_stats.py")
    parser.add_argument("--concept_ids", nargs="+", required=True,
                        help="One or more concept IDs to plot")
    parser.add_argument("--output", default="reward_weight_plot.png",
                        help="Output path for the plot (default: reward_weight_plot.png)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.stats_csv, dtype={"concept_id": str})

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for concept_id in args.concept_ids:
        sub = df[df["concept_id"] == concept_id].sort_values("window_days")
        if sub.empty:
            print(f"Warning: concept_id {concept_id} not found in stats; skipping.")
            continue
        name = sub["concept_name"].iloc[0]
        label = f"{concept_id}: {str(name)[:60]}" if name else concept_id
        ax.scatter(sub["window_days"], sub["reward_weight"], label=label, s=80)
        ax.plot(sub["window_days"], sub["reward_weight"], alpha=0.5)
        plotted += 1

    if plotted == 0:
        print("No valid concept IDs found in stats; nothing to plot.")
        return

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
