#!/usr/bin/env python3
"""Plot training and validation histories from the saved benchmark runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = Path("/scratch/scratch26/22cs30008/res/outputs/benchmark_all")
OUTPUT = Path("/scratch/scratch26/22cs30008/res/outputs/learning_curves")
MASKS = ("dense", "local", "window_dilations", "circulant", "hypercube", "random_circulant", "bigbird")
METRICS = {
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "mrpc": "eval_f1",
    "qqp": "eval_f1",
    "stsb": "eval_pearson",
    "rte": "eval_accuracy",
    "qnli": "eval_accuracy",
    "mnli": "eval_accuracy",
}
COLORS = ("#222222", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")


def rolling_median(losses, epochs):
    span = max(epochs[-1] - epochs[0], 1)
    radius = round(0.025 * len(losses) / span)
    if radius == 0:
        return losses
    return [float(np.median(losses[max(0, i - radius):min(len(losses), i + radius + 1)]))
            for i in range(len(losses))]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for task, metric in METRICS.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for mask, color in zip(MASKS, COLORS):
            path = RUNS / task / mask / "trainer_state.json"
            history = json.loads(path.read_text(encoding="utf-8"))["log_history"]
            train = [entry for entry in history if "loss" in entry and "epoch" in entry]
            validation = {}
            for entry in history:
                if "eval_loss" in entry and "step" in entry:
                    validation.setdefault(entry["step"], entry)
            validation = list(validation.values())
            if not train or not validation or any(metric not in entry for entry in validation):
                raise ValueError(f"Incomplete learning history: {path}")

            epochs = [entry["epoch"] for entry in train]
            losses = [entry["loss"] for entry in train]
            label = mask.replace("_", " ")
            axes[0].plot(epochs, losses, color=color, alpha=0.12, linewidth=0.7)
            axes[0].plot(epochs, rolling_median(losses, epochs),
                         color=color, linewidth=2, label=label)
            axes[1].plot([entry["epoch"] for entry in validation],
                         [entry["eval_loss"] for entry in validation],
                         color=color, marker="o", markersize=4, linewidth=1.5)
            axes[2].plot([entry["epoch"] for entry in validation],
                         [entry[metric] for entry in validation],
                         color=color, marker="o", markersize=4, linewidth=1.5)

        score_title = "Matched accuracy" if task == "mnli" else metric.removeprefix("eval_")
        for axis, title in zip(axes, ("Training loss (0.05-epoch rolling median)",
                                      "Validation loss", score_title)):
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("Loss")
        fig.suptitle(task.upper())
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.12, 1, 0.96))
        path = OUTPUT / f"{task}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
