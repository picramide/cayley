#!/usr/bin/env python3
"""Print the training recipe used by the five-epoch pilot benchmarks."""

import json
from pathlib import Path
import sys


# Defaults of the Transformers/PyTorch versions recorded in benchmark_all.jsonl.
# These are used only when a run's saved training_args.bin is unavailable.
DEFAULTS = {
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
}

FIELDS = {
    "model_name": "Model",
    "optim": "Optimizer",
    "adam_beta1": "Adam beta1",
    "adam_beta2": "Adam beta2",
    "adam_epsilon": "Adam epsilon",
    "learning_rate": "Peak learning rate",
    "lr_scheduler_type": "LR schedule",
    "warmup_steps": "Warmup steps",
    "weight_decay": "Weight decay",
    "max_grad_norm": "Gradient clipping norm",
    "per_device_train_batch_size": "Train batch size per device",
    "gradient_accumulation_steps": "Gradient accumulation",
    "per_device_eval_batch_size": "Eval batch size per device",
    "num_train_epochs": "Epochs",
    "max_length": "Maximum token length",
    "seed": "Seed",
    "data_seed": "Data seed",
    "load_best_model_at_end": "Load best model at end",
    "attention_implementation": "Attention implementation",
}


def plain(value):
    return value.value if hasattr(value, "value") else value


def load_saved_args(row):
    path = Path(row["trainer_state_path"]).with_name("training_args.bin")
    if not path.is_file():
        return {}, f"missing {path}"
    try:
        import torch
        # TrainingArguments is pickled; only load your own trusted checkpoints.
        args = torch.load(path, map_location="cpu", weights_only=False)
        return {name: plain(getattr(args, name)) for name in (*FIELDS, "fp16", "bf16")
                if hasattr(args, name)}, None
    except Exception as exc:
        return {}, f"could not read {path}: {exc}"


def field_value(row, saved, name):
    if name in saved:
        return saved[name], "saved training_args.bin"
    if name in row:
        return row[name], "benchmark JSONL"
    if (row.get("transformers_version") == "5.16.1"
            and str(row.get("torch_version", "")).startswith("2.6.0")
            and name in DEFAULTS):
        return DEFAULTS[name], "versioned default"
    return None, "unknown"


def shared_value(records, name):
    entries = [field_value(row, saved, name) for row, saved in records]
    values = {repr(value) for value, _ in entries}
    if len(values) != 1 or entries[0][0] is None:
        return "unknown or differs across runs"
    value = entries[0][0]
    sources = ", ".join(sorted({source for _, source in entries}))
    return f"{value} ({sources})"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("benchmark_all.jsonl")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("num_train_epochs") != 5:
                continue
            key = (row["task_name"], row["mask_name"])
            if key in rows:
                raise SystemExit(f"Duplicate five-epoch run: {key}")
            rows[key] = row
    if not rows:
        raise SystemExit("No five-epoch runs found")

    records = []
    missing_files = []
    conflicts = []
    for key, row in sorted(rows.items()):
        saved, error = load_saved_args(row)
        records.append((row, saved))
        if error:
            missing_files.append(error)
        for name in FIELDS:
            if name in saved and name in row and saved[name] != row[name]:
                conflicts.append(f"{key}: {name}")

    print(f"Pilot: {len(rows)} five-epoch runs; "
          f"{len({task for task, _ in rows})} tasks, {len({mask for _, mask in rows})} masks")
    print("Tasks: " + ", ".join(sorted({task for task, _ in rows})))
    print("Masks: " + ", ".join(sorted({mask for _, mask in rows})))
    for name, label in FIELDS.items():
        print(f"{label}: {shared_value(records, name)}")

    optimizers = {field_value(row, saved, "optim")[0] for row, saved in records}
    correction = "yes (PyTorch AdamW)" if optimizers <= {"adamw_torch", "adamw_torch_fused"} else "unknown"
    print(f"Adam bias correction: {correction}")

    precision = set()
    for _, saved in records:
        if "fp16" not in saved or "bf16" not in saved:
            precision.add("unknown")
        elif saved["fp16"] and saved["bf16"]:
            precision.add("invalid: both fp16 and bf16 enabled")
        else:
            precision.add("fp16" if saved["fp16"] else "bf16" if saved["bf16"] else "fp32")
    print(f"Training precision: {next(iter(precision)) if len(precision) == 1 else 'unknown or differs across runs'}")
    print("Historical GPU model: not recorded here; check the pilot job log")
    if missing_files:
        print(f"Warning: training_args.bin unavailable for {len(missing_files)} runs; defaults are marked above")
    if conflicts:
        print("Warning: saved arguments differ from JSONL for: " + ", ".join(conflicts))


if __name__ == "__main__":
    main()
