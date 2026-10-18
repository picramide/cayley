#!/usr/bin/env python3
"""Recover the five-epoch benchmark recipe for the Experiment B prerequisite.

Reads the benchmark JSONL plus each run's saved training_args.bin and
trainer_state.json. It never trains or changes a checkpoint. Run it in the
same Transformers/PyTorch environment as the benchmarks if possible.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess


TASKS = ("cola", "sst2", "stsb", "rte", "mrpc", "qqp", "mnli", "qnli")
MASKS = ("dense", "local", "hypercube", "circulant", "window_dilations", "random_circulant", "bigbird")

# Defaults in Transformers 5.16.1 with PyTorch 2.6; use only when the saved
# training_args.bin is unavailable, and label them as inferred rather than saved.
VERSIONED_DEFAULTS = {
    "optim": "adamw_torch", "optim_args": None,
    "lr_scheduler_type": "linear", "lr_scheduler_kwargs": None,
    "warmup_steps": 0,
    "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
    "gradient_accumulation_steps": 1, "max_grad_norm": 1.0, "bf16": False,
}
FIELDS = tuple(VERSIONED_DEFAULTS) + (
    "fp16", "tf32", "half_precision_backend", "learning_rate", "weight_decay",
    "per_device_train_batch_size", "per_device_eval_batch_size", "num_train_epochs",
    "seed", "data_seed", "max_steps", "load_best_model_at_end",
)
JSONL_FIELDS = (
    "learning_rate", "weight_decay", "per_device_train_batch_size",
    "per_device_eval_batch_size", "num_train_epochs", "seed", "data_seed",
    "load_best_model_at_end",
)


def plain(value):
    if hasattr(value, "value"):
        return plain(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def read_rows(path):
    selected = {}
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if float(row.get("num_train_epochs", -1)) != 5.0:
                continue
            key = (row.get("task_name"), row.get("mask_name"))
            if key in selected:
                raise ValueError(f"Duplicate five-epoch run for {key} at line {line_number}; "
                                 "provide a results snapshot with one row per task/mask")
            selected[key] = row
    if not selected:
        raise ValueError("No num_train_epochs == 5 runs found")
    return selected, total


def read_training_args(path):
    if not path.is_file():
        return {}, "training_args.bin missing"
    try:
        import torch
        # These are the user's own Trainer checkpoints. TrainingArguments is a
        # pickled Python object, so weights_only=True cannot read this file.
        args = torch.load(path, map_location="cpu", weights_only=False)
        settings = {name: plain(getattr(args, name)) for name in FIELDS if hasattr(args, name)}
        settings["saved_n_gpu"] = plain(vars(args).get("_n_gpu"))
        settings["local_rank"] = plain(getattr(args, "local_rank", None))
        return settings, None
    except Exception as exc:
        return {}, f"Cannot read training_args.bin: {type(exc).__name__}: {exc}"


def read_trainer_state(path):
    if not path.is_file():
        return {}, "trainer_state.json missing"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        learning_rates = [entry["learning_rate"] for entry in state.get("log_history", [])
                          if "learning_rate" in entry]
        return {
            "global_step": state.get("global_step"),
            "max_steps": state.get("max_steps"),
            "best_global_step": state.get("best_global_step"),
            "first_logged_learning_rate": learning_rates[0] if learning_rates else None,
            "last_logged_learning_rate": learning_rates[-1] if learning_rates else None,
        }, None
    except Exception as exc:
        return {}, f"Cannot read trainer_state.json: {type(exc).__name__}: {exc}"


def current_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


def recover_run(row, pilot_world_size):
    state_path = Path(row["trainer_state_path"])
    args_path = state_path.parent / "training_args.bin"
    saved, args_error = read_training_args(args_path)
    state, state_error = read_trainer_state(state_path)
    warnings = [message for message in (args_error, state_error) if message]
    version_defaults_apply = (row.get("transformers_version") == "5.16.1"
                              and str(row.get("torch_version", "")).startswith("2.6.0"))

    settings = {}
    for field in FIELDS:
        if field in saved:
            value, source = saved[field], "saved training_args.bin"
            if field in JSONL_FIELDS and field in row and value != row[field]:
                warnings.append(f"{field} differs between training_args.bin and JSONL")
        elif field in JSONL_FIELDS and field in row:
            value, source = row[field], "benchmark JSONL"
        elif version_defaults_apply and field in VERSIONED_DEFAULTS:
            value, source = VERSIONED_DEFAULTS[field], "inferred versioned default"
        else:
            value, source = None, "unavailable"
        settings[field] = {"value": value, "source": source}

    optim = settings["optim"]["value"]
    if optim in ("adamw_torch", "adamw_torch_fused"):
        bias_correction = {"value": True, "source": "inferred from PyTorch AdamW implementation"}
    else:
        bias_correction = {"value": None, "source": "unavailable for this optimiser"}
    settings["bias_correction"] = bias_correction

    fp16, bf16 = settings["fp16"]["value"], settings["bf16"]["value"]
    precision = "fp16" if fp16 else "bf16" if bf16 else "fp32" if fp16 is False and bf16 is False else None
    settings["precision"] = {"value": precision,
                             "source": "saved training_args.bin" if "fp16" in saved and "bf16" in saved else "unavailable"}
    if precision is None:
        warnings.append("Pilot precision unavailable; benchmark code enabled FP16 when CUDA was available")

    planned_steps = state.get("max_steps")
    warmup = settings["warmup_steps"]["value"]
    if isinstance(warmup, (int, float)) and warmup >= 1:
        effective_warmup = int(warmup)
        warmup_source = "warmup_steps setting"
    elif isinstance(warmup, (int, float)) and isinstance(planned_steps, int) and planned_steps > 0:
        effective_warmup = math.ceil(planned_steps * warmup)
        warmup_source = "warmup_steps fraction + trainer_state.max_steps"
    elif warmup == 0:
        effective_warmup = 0
        warmup_source = "warmup_steps setting"
    else:
        effective_warmup = None
        warmup_source = "unavailable"
    settings["effective_warmup_updates"] = {"value": effective_warmup,
                                            "source": warmup_source}

    per_device = settings["per_device_train_batch_size"]["value"]
    accumulation = settings["gradient_accumulation_steps"]["value"]
    effective_batch = (per_device * accumulation * pilot_world_size
                       if pilot_world_size and isinstance(per_device, int)
                       and isinstance(accumulation, int) else None)
    if state.get("global_step") is not None and state.get("max_steps") is not None:
        if state["global_step"] != state["max_steps"]:
            warnings.append("Final global_step differs from planned max_steps")

    return {
        "task": row["task_name"], "mask": row["mask_name"],
        "versions": {"transformers": row.get("transformers_version"),
                     "torch": row.get("torch_version")},
        "model": row.get("model_name"), "dataset": row.get("dataset_name"),
        "max_length": row.get("max_length"),
        "max_train_samples": row.get("max_train_samples"),
        "max_eval_samples": row.get("max_eval_samples"),
        "attention_implementation": row.get("attention_implementation"),
        "mask_config": row.get("mask_config"),
        "metric_for_best_model": row.get("metric_for_best_model"),
        "saved_training_args_path": str(args_path), "trainer_state_path": str(state_path),
        "settings": settings, "trainer_state": state,
        "saved_n_gpu": saved.get("saved_n_gpu"), "saved_local_rank": saved.get("local_rank"),
        "effective_batch_size": effective_batch,
        "warnings": warnings,
    }


def summarize(runs):
    shared = {}
    for name in list(FIELDS) + ["bias_correction", "precision", "effective_warmup_updates"]:
        values = [run["settings"][name]["value"] for run in runs]
        unique = list(dict.fromkeys(json.dumps(value, sort_keys=True) for value in values))
        shared[name] = json.loads(unique[0]) if len(unique) == 1 else "VARIES BY RUN"

    by_task = defaultdict(list)
    for run in runs:
        by_task[run["task"]].append(run)
    targets = {}
    for task in TASKS:
        task_runs = by_task[task]
        values = {run["trainer_state"].get("max_steps") for run in task_runs}
        p = next(iter(values)) if len(task_runs) == len(MASKS) and len(values) == 1 and None not in values else None
        targets[task] = {"P_updates": p, "S_updates": p, "S_plus_updates": 3 * p if p else None,
                         "G_updates": p, "C_s_downstream_updates": p, "C_d_downstream_updates": p,
                         "verified_run_count": len(task_runs),
                         "completed_run_count": sum(run["trainer_state"].get("global_step") == p
                                                    for run in task_runs) if p is not None else None}
    return shared, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="benchmark_all.jsonl from the institute server")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report path")
    parser.add_argument("--pilot-gpu", help="Historical GPU model, if verified in the pilot job log")
    parser.add_argument("--pilot-world-size", type=int, help="Historical number of training processes, if verified")
    args = parser.parse_args()
    if args.pilot_world_size is not None and args.pilot_world_size < 1:
        parser.error("--pilot-world-size must be positive")

    selected, total = read_rows(args.results)
    runs = [recover_run(row, args.pilot_world_size) for _, row in sorted(selected.items())]
    shared, targets = summarize(runs)
    expected = {(task, mask) for task in TASKS for mask in MASKS}
    missing = sorted(expected - set(selected))
    report = {
        "selection": "num_train_epochs == 5; one row per task/mask",
        "results_file": str(args.results), "input_rows": total,
        "selected_rows": len(runs), "excluded_rows": total - len(runs),
        "missing_task_mask_pairs": missing,
        "shared_settings": shared,
        "per_task_update_targets": targets,
        "pilot_gpu_model": args.pilot_gpu,
        "pilot_gpu_model_source": "user-supplied; verify against historical job log" if args.pilot_gpu else "not recoverable from current artifacts",
        "pilot_world_size": args.pilot_world_size,
        "current_gpu_query": current_gpus(),
        "current_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "condition_definitions": {
            "P": "Five-epoch benchmark recipe; hard mask from the start",
            "S": "Strong recipe; same downstream optimizer-update count as P (settings still to specify)",
            "S_plus": "S with three times its downstream optimizer-update count",
            "G": "Gradual transition to the same hard mask; otherwise match S exactly",
            "C_s": "Continued sparse masked-LM pretraining, then S",
            "C_d": "Equal continued dense masked-LM pretraining, then S",
        },
        "runs": runs,
        "limitations": [
            "The current GPU query does not identify the GPU used by historical benchmark jobs.",
            "Trainer global_step counts training steps; FP16 skipped optimizer updates are not recoverable here.",
            "The values defining S, G's transition, and continued MLM step counts are not in the pilot artifacts.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"Selected {len(runs)} five-epoch runs from {total} rows; missing task/mask pairs: {len(missing)}")
    for name in ("optim", "adam_beta1", "adam_beta2", "adam_epsilon", "bias_correction",
                 "lr_scheduler_type", "learning_rate", "warmup_steps",
                 "effective_warmup_updates",
                 "weight_decay", "max_grad_norm", "precision", "per_device_train_batch_size",
                 "gradient_accumulation_steps", "num_train_epochs", "seed"):
        print(f"  {name}: {shared[name]}")
    print("  pilot GPU model:", args.pilot_gpu or "UNKNOWN (check historical job log)")
    print("  current GPU(s):", "; ".join(report["current_gpu_query"]) or "none visible")
    print("  per-task scheduled update targets (P/S, then S+):")
    for task in TASKS:
        print(f"    {task}: {targets[task]['P_updates']} / {targets[task]['S_plus_updates']}")
    print("  Experiment B conditions:")
    for name, description in report["condition_definitions"].items():
        print(f"    {name}: {description}")
    print("  S hyperparameters, G transition, and C_s/C_d pretraining budget: NOT in pilot artifacts")
    issues = Counter(message for run in runs for message in run["warnings"])
    for message, count in issues.items():
        print(f"  WARNING ({count} runs): {message}")
    print(f"JSON report: {args.report}")


if __name__ == "__main__":
    main()
