#!/usr/bin/env python3
"""
Benchmark script for running all available masks on QQP, MNLI, and QNLI benchmarks.

Usage:
    python scripts/benchmark_all_masks.py [--output_dir BASE_DIR] [--results_dir RESULTS_DIR]

This script runs all mask types on each of the three benchmarks (QQP, MNLI, QNLI)
and saves results as JSONL files.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Available mask types with their arguments
MASK_TYPES = [
    ("dense", []),
    ("local", ["--window", "16"]),
    ("hypercube", []),
    ("circulant", ["--generators", "1,2,3"]),
    ("window_dilations", ["--window", "16", "--dilations", "16,32,64"]),
    ("random_circulant", ["--degree", "8", "--seed", "42"]),
    ("bigbird", ["--global_tokens", "2", "--block_size", "2"]),
    ("bipartite", ["--premise_len", "64", "--local_window", "3", "--cross_window", "2"]),
    ("nearly_dense", ["--drop_degree", "4"]),
]

# Benchmarks to run
BENCHMARKS = [
    {
        "name": "qqp",
        "task_name": "qqp",
        "dataset_name": "nyu-mll/glue",
        "model_name": "FacebookAI/roberta-base",
        "max_length": 128,
        "num_train_epochs": 5.0,
        "seed": 42,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
    },
    {
        "name": "mnli",
        "task_name": "mnli",
        "dataset_name": "nyu-mll/glue",
        "model_name": "FacebookAI/roberta-base",
        "max_length": 128,
        "num_train_epochs": 3.0,  # MNLI is larger, fewer epochs
        "seed": 42,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
    },
    {
        "name": "qnli",
        "task_name": "qnli",
        "dataset_name": "nyu-mll/glue",
        "model_name": "FacebookAI/roberta-base",
        "max_length": 128,
        "num_train_epochs": 5.0,
        "seed": 42,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
    },
]


def generate_mask(kind: str, mask_path: str, mask_args: list, cwd: Path) -> bool:
    """Generate a mask using the generate_masks.py script."""
    cmd = [
        "python",
        "-u",
        "scripts/generate_masks.py",
        "--kind",
        kind,
        "--seq",
        "128",
        "--output",
        mask_path,
    ] + mask_args

    print(f"[MASK] Generating {kind} mask -> {mask_path}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)

    if result.returncode != 0:
        print(f"[ERROR] Failed to generate mask {kind}")
        return False
    return True


def run_benchmark(
    task_name: str,
    dataset_name: str,
    model_name: str,
    mask_name: str,
    mask_path: str | None,
    output_dir: str,
    results_file: str,
    run_name: str,
    max_length: int,
    num_train_epochs: float,
    seed: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    cwd: Path,
) -> bool:
    """Run a single benchmark using the benchmark_roberta_glue.py script."""
    cmd = [
        "python",
        "-u",
        "scripts/benchmark_roberta_glue.py",
        "--task_name",
        task_name,
        "--dataset_name",
        dataset_name,
        "--model_name",
        model_name,
        "--do_train",
        "--do_eval",
        "--max_length",
        str(max_length),
        "--num_train_epochs",
        str(num_train_epochs),
        "--seed",
        str(seed),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--learning_rate",
        str(learning_rate),
        "--mask_name",
        mask_name,
        "--output_dir",
        output_dir,
        "--results_file",
        results_file,
        "--run_name",
        run_name,
    ]

    if mask_path:
        cmd.extend(["--mask_path", mask_path])

    print(f"[BENCHMARK] Running {task_name} with {mask_name}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)

    if result.returncode != 0:
        print(f"[ERROR] Failed benchmark: {task_name} with {mask_name}")
        return False
    return True


def load_existing_results(results_file: Path) -> set:
    """Load already completed benchmark results to skip them."""
    completed = set()
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    key = (data["task_name"], data["mask_name"])
                    completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all masks on QQP, MNLI, and QNLI"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benchmark_all",
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for results JSONL files",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip already completed benchmarks (uses results/benchmark_all.jsonl)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Determine project root (script is in scripts/, project root is parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # Setup directories
    output_base = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Results file for tracking completed runs
    results_file = results_dir / "benchmark_all.jsonl"

    # Load completed benchmarks if skipping
    completed = set()
    if args.skip_completed:
        completed = load_existing_results(results_file)
        print(f"Found {len(completed)} already completed benchmarks")

    print(f"Project root: {project_root}")
    print(f"Output base: {output_base}")
    print(f"Results dir: {results_dir}")
    print("-" * 60)

    total = len(MASK_TYPES) * len(BENCHMARKS)
    executed = 0
    succeeded = 0
    failed = 0

    for benchmark in BENCHMARKS:
        for mask_kind, mask_args in MASK_TYPES:
            executed += 1

            # Create unique run name
            run_name = f"{benchmark['name']}_{mask_kind}"
            mask_name = mask_kind

            # Generate mask path
            mask_path = None
            if mask_kind != "dense":
                mask_path = f"masks/{run_name}.pt"

            # Output directories
            output_dir = str(output_base / benchmark["name"] / mask_kind)
            results_file_path = str(results_dir / "benchmark_all.jsonl")

            # Skip if already completed
            if args.skip_completed:
                if (benchmark["name"], mask_kind) in completed:
                    print(f"[SKIP] {benchmark['name']} with {mask_kind}")
                    succeeded += 1
                    continue

            print(f"\n[{executed}/{total}] Starting: {run_name}")
            print(f"  Mask: {mask_kind}")
            print(f"  Task: {benchmark['name']}")

            # Generate mask if needed
            if mask_kind != "dense":
                if not generate_mask(mask_kind, mask_path, mask_args, project_root):
                    failed += 1
                    continue

            # Run benchmark
            if not run_benchmark(
                task_name=benchmark["task_name"],
                dataset_name=benchmark["dataset_name"],
                model_name=benchmark["model_name"],
                mask_name=mask_name,
                mask_path=mask_path,
                output_dir=output_dir,
                results_file=results_file_path,
                run_name=run_name,
                max_length=benchmark["max_length"],
                num_train_epochs=benchmark["num_train_epochs"],
                seed=benchmark["seed"],
                per_device_train_batch_size=benchmark["per_device_train_batch_size"],
                per_device_eval_batch_size=benchmark["per_device_eval_batch_size"],
                learning_rate=benchmark["learning_rate"],
                cwd=project_root,
            ):
                failed += 1
                continue

            succeeded += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {succeeded}/{executed} succeeded, {failed} failed")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
