"""
Plot RL training and evaluation metrics from a HuggingFace trainer_state.json file.

Usage
-----
python scripts/plot_rl_training.py path/to/trainer_state.json
python scripts/plot_rl_training.py path/to/trainer_state.json --output plots.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Each tuple: (train_key, eval_key, subplot_title)
METRIC_PAIRS = [
    ("rl_reward_mean", "eval_reward",       "Reward (mean)"),
    ("rl_baseline",    "eval_rl_baseline",  "Reward baseline"),
    ("rl_pg_loss",     "eval_rl_pg_loss",   "PG loss"),
    ("rl_kl_loss",     "eval_rl_kl_loss",   "KL loss"),
]


def load_log_history(path: str):
    with open(path) as f:
        state = json.load(f)
    train, eval_ = [], []
    for entry in state["log_history"]:
        if "eval_loss" in entry or "eval_reward" in entry:
            eval_.append(entry)
        else:
            train.append(entry)
    return train, eval_


def _get(entries, key):
    xs, ys = [], []
    for e in entries:
        if key in e:
            xs.append(e["step"])
            ys.append(e[key])
    return xs, ys


def plot(trainer_state_path: str, output: str):
    train, eval_ = load_log_history(trainer_state_path)

    # Only include rows where at least one of train or eval has data
    active_pairs = [
        (tk, ek, title)
        for tk, ek, title in METRIC_PAIRS
        if _get(train, tk)[0] or _get(eval_, ek)[0]
    ]

    n_rows = len(active_pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]  # keep 2-D indexing consistent

    for row, (train_key, eval_key, title) in enumerate(active_pairs):
        # ---- train ----
        ax_tr = axes[row][0]
        xs, ys = _get(train, train_key)
        if xs:
            ax_tr.plot(xs, ys, linewidth=1, color="steelblue")
        ax_tr.set_title(f"{title} — train")
        ax_tr.set_xlabel("Step")
        ax_tr.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_tr.grid(True, alpha=0.3)

        # ---- eval ----
        ax_ev = axes[row][1]
        xs, ys = _get(eval_, eval_key)
        if xs:
            ax_ev.plot(xs, ys, marker="o", linewidth=1.5, color="darkorange")
        ax_ev.set_title(f"{title} — eval")
        ax_ev.set_xlabel("Step")
        ax_ev.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_ev.grid(True, alpha=0.3)

    fig.suptitle(Path(trainer_state_path).parent.name, fontsize=13)

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RL training metrics from trainer_state.json")
    parser.add_argument("trainer_state", help="Path to trainer_state.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Save figure to this path instead of showing it")
    args = parser.parse_args()
    plot(args.trainer_state, args.output)


if __name__ == "__main__":
    main()
