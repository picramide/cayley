#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nyu-mll/glue")
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--output_root", type=str, default="outputs/mrpc_dense_window")
    parser.add_argument("--results_file", type=str, default="results/mrpc_dense_window.jsonl")
    parser.add_argument("--masks_dir", type=str, default="masks")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def benchmark_cmd(args, mask_name: str, mask_path: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/benchmark_roberta_glue.py",
        "--task_name",
        "mrpc",
        "--dataset_name",
        args.dataset_name,
        "--model_name",
        args.model_name,
        "--do_train",
        "--do_eval",
        "--max_length",
        str(args.max_length),
        "--num_train_epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--mask_name",
        mask_name,
        "--per_device_train_batch_size",
        str(args.train_batch_size),
        "--per_device_eval_batch_size",
        str(args.eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--output_dir",
        str(Path(args.output_root) / mask_name),
        "--results_file",
        args.results_file,
        "--run_name",
        f"mrpc_{mask_name}",
    ]
    if mask_path is not None:
        cmd.extend(["--mask_path", str(mask_path)])
    return cmd


def main():
    args = parse_args()
    masks_dir = Path(args.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    window_mask = masks_dir / f"local_w{args.window}_{args.max_length}.pt"

    run(
        [
            sys.executable,
            "scripts/generate_masks.py",
            "--kind",
            "local",
            "--seq",
            str(args.max_length),
            "--window",
            str(args.window),
            "--output",
            str(window_mask),
        ],
        args.dry_run,
    )

    runs = [
        ("dense", None),
        (f"local_w{args.window}", window_mask),
    ]
    print("Running full MRPC dense/window benchmark: 2 runs, no sample caps.")
    for mask_name, mask_path in runs:
        run(benchmark_cmd(args, mask_name, mask_path), args.dry_run)


if __name__ == "__main__":
    main()
