import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(output_dir: str):
    with open(os.path.join(output_dir, "time_embedding_metrics.json"), "r") as f:
        time_embedding_metrics = json.load(f)
    with open(os.path.join(output_dir, "time_token_metrics.json"), "r") as f:
        time_token_metrics = json.load(f)

    common_steps = list(
        set(time_embedding_metrics["steps"]) & set(time_token_metrics["steps"])
    )

    time_embedding_aucs = []
    time_embedding_accuracies = []
    for step, roc_auc, accuracy in zip(
        time_embedding_metrics["steps"],
        time_embedding_metrics["roc_auc"],
        time_embedding_metrics["accuracy"],
    ):
        if step in common_steps:
            time_embedding_aucs.append(roc_auc)
            time_embedding_accuracies.append(accuracy)

    time_token_aucs = []
    time_token_accuracies = []
    for step, roc_auc, accuracy in zip(
        time_token_metrics["steps"],
        time_token_metrics["roc_auc"],
        time_token_metrics["accuracy"],
    ):
        if step in common_steps:
            time_token_aucs.append(roc_auc)
            time_token_accuracies.append(accuracy)

    # Create the accuracy plot
    plt.figure(figsize=(8, 5))  # Define figure size
    plt.plot(
        common_steps,
        time_embedding_accuracies,
        linestyle="-",
        color="b",
        label="Time Embedding",
        lw=1,
    )
    plt.plot(
        common_steps,
        time_token_accuracies,
        linestyle="--",
        color="r",
        label="Time Token",
        lw=1,
    )
    plt.title("Accuracy Comparison Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))

    # Create the ROC AUC plot
    plt.figure(figsize=(8, 5))  # Define figure size
    plt.plot(
        common_steps,
        time_embedding_aucs,
        linestyle="-",
        color="b",
        label="Time Embedding",
        lw=1,
    )
    plt.plot(
        common_steps,
        time_token_aucs,
        linestyle="--",
        color="r",
        label="Time Token",
        lw=1,
    )
    plt.title("ROC AUC Comparison Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.grid(False)
    plt.savefig(
        os.path.join(output_dir, "roc_auc_comparison.png")
    )  # Save the plot as a PNG file


if __name__ == "__main__":
    main(sys.argv[1])
