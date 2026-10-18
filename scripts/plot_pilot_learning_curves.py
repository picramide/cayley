#!/usr/bin/env python3
"""Plot training and validation histories from the saved benchmark runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for task, metric in METRICS.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for mask in MASKS:
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

            axes[0].plot([entry["epoch"] for entry in train],
                         [entry["loss"] for entry in train], label=mask)
            axes[1].plot([entry["epoch"] for entry in validation],
                         [entry["eval_loss"] for entry in validation], marker="o", label=mask)
            axes[2].plot([entry["epoch"] for entry in validation],
                         [entry[metric] for entry in validation], marker="o", label=mask)

        score_title = "Matched accuracy" if task == "mnli" else metric.removeprefix("eval_")
        for axis, title in zip(axes, ("Training loss", "Validation loss", score_title)):
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
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
