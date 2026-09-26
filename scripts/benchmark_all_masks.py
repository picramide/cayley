#!/usr/bin/env python3
"""
Benchmark script for running all available masks on the supported GLUE tasks.

Usage:
    python scripts/benchmark_all_masks.py [--output_dir BASE_DIR] [--results_dir RESULTS_DIR]

This script trains and validates every task/mask combination and saves JSONL results.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Keep direct invocation usable before installing the package, including dry runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cayley.benchmarking import BENCHMARK_PROTOCOL_VERSION, run_key

# Available mask types with their arguments
MASK_TYPES = [
    ("dense", []),
    ("local", ["--window", "16"]),
    ("hypercube", []),
    ("circulant", ["--generators", "1,2,3"]),
    ("window_dilations", ["--window", "16", "--dilations", "16,32,64"]),
    ("random_circulant", ["--degree", "8", "--seed", "42"]),
    ("bigbird", [
        "--global_tokens", "2",
        "--block_size", "2",
        "--num_random_blocks", "3",
        "--window_block_left", "1",
        "--window_block_right", "1",
    ]),
]

# Benchmarks to run
# dataset_name and model_name can be either:
# - Remote: "nyu-mll/glue", "FacebookAI/roberta-base" (for online use)
# - Local paths: "/path/to/offline/glue/qnli", "/path/to/offline/models/roberta-base"
BENCHMARKS = [
    {
        "name": "cola",
        "task_name": "cola",
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
        "name": "sst2",
        "task_name": "sst2",
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
        "name": "stsb",
        "task_name": "stsb",
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
        "name": "rte",
        "task_name": "rte",
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
        "name": "mrpc",
        "task_name": "mrpc",
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
        "num_train_epochs": 5.0,
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


def generate_mask(kind: str, mask_path: str, mask_args: list, cwd: Path,
                  seq_len: int = 128, dry_run: bool = False) -> bool:
    """Generate a mask using the generate_masks.py script."""
    cmd = [
        sys.executable,
        "-u",
        "scripts/generate_masks.py",
        "--kind",
        kind,
        "--seq",
        str(seq_len),
        "--heads",
        "12",
        "--output",
        mask_path,
    ] + mask_args

    if dry_run:
        print(f"[DRY RUN] {shlex.join(cmd)}")
        return True

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)

    print(f"[MASK] Generating {kind} mask -> {mask_path}")
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=False)

    if result.returncode != 0:
        print(f"[ERROR] Failed to generate mask {kind}")
        return False
    return True


def run_benchmark(
    task_name: str,
    dataset_name: str,
    model_name: str,
    mask_name: str,
    mask_path: Optional[str],
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
    mask_config: Optional[dict] = None,
    dry_run: bool = False,
) -> bool:
    """Run a single benchmark using the benchmark_roberta_glue.py script."""
    cmd = [
        sys.executable,
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
    if mask_config is not None:
        cmd.extend(["--mask_config", json.dumps(mask_config, sort_keys=True)])

    if dry_run:
        print(f"[DRY RUN] {shlex.join(cmd)}")
        return True

    print(f"[BENCHMARK] Running {task_name} with {mask_name}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)

    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=False)

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
                    if (data.get("benchmark_protocol_version") == BENCHMARK_PROTOCOL_VERSION
                            and data.get("do_train") and data.get("do_eval") and data.get("metrics")):
                        completed.add(run_key(data))
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Train and validate all masks on GLUE"
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
        help="Skip matching runs from the current protocol in <results_dir>/benchmark_all.jsonl",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--offline_data_dir",
        type=str,
        default=None,
        help="Base directory for offline data (downloaded datasets and models)",
    )
    args = parser.parse_args()

    # Determine project root (script is in scripts/, project root is parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # Setup directories - accept absolute or relative paths
    output_base = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    # Convert relative paths to be relative to project root
    if not output_base.is_absolute():
        output_base = project_root / output_base
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    if not args.dry_run:
        output_base.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    # Results file for tracking completed runs
    results_file = results_dir / "benchmark_all.jsonl"

    # Load completed benchmarks if skipping
    completed = set()
    if args.skip_completed:
        completed = load_existing_results(results_file)
        print(f"Found {len(completed)} already completed benchmarks")

    # Handle offline data paths
    offline_data_dir = Path(args.offline_data_dir) if args.offline_data_dir else None
    if offline_data_dir:
        print(f"Offline data directory: {offline_data_dir}")

    benchmarks = [dict(benchmark) for benchmark in BENCHMARKS]
    # Update benchmarks to use offline paths if specified
    if offline_data_dir:
        for benchmark in benchmarks:
            # Local dataset path: offline_data_dir/glue/<task_name>
            benchmark["dataset_name"] = str(offline_data_dir / "glue" / benchmark["name"])
            # Local model path: offline_data_dir/models/roberta-base
            benchmark["model_name"] = str(offline_data_dir / "models" / "roberta-base")
        print("Using offline dataset and model paths")

    print(f"Project root: {project_root}")
    print(f"Output base: {output_base}")
    print(f"Results dir: {results_dir}")
    print("-" * 60)

    total = len(MASK_TYPES) * len(benchmarks)
    executed = 0
    succeeded = 0
    failed = 0

    for benchmark in benchmarks:
        for mask_kind, mask_args in MASK_TYPES:
            executed += 1

            # Create unique run name
            run_name = f"{benchmark['name']}_{mask_kind}"
            mask_name = mask_kind
            mask_config = {
                "kind": mask_kind, "seq_len": benchmark["max_length"],
                "heads": 12, "args": mask_args,
            }

            # Generate mask path
            mask_path = None
            if mask_kind != "dense":
                mask_path = str(output_base / "masks" / f"{run_name}.pt")

            # Output directories
            output_dir = str(output_base / benchmark["name"] / mask_kind)
            results_file_path = str(results_dir / "benchmark_all.jsonl")

            # Skip if already completed
            if args.skip_completed:
                expected = {
                    **benchmark, "mask_name": mask_kind, "mask_config": mask_config,
                    "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                    "weight_decay": 0.01,
                }
                if run_key(expected) in completed:
                    print(f"[SKIP] {benchmark['name']} with {mask_kind}")
                    succeeded += 1
                    continue

            print(f"\n[{executed}/{total}] Starting: {run_name}")
            print(f"  Mask: {mask_kind}")
            print(f"  Task: {benchmark['name']}")

            # Generate mask if needed
            if mask_kind != "dense":
                if not generate_mask(mask_kind, mask_path, mask_args, project_root,
                                     seq_len=benchmark["max_length"], dry_run=args.dry_run):
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
                mask_config=mask_config,
                dry_run=args.dry_run,
            ):
                failed += 1
                continue

            succeeded += 1

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN: previewed/skipped {executed} jobs; no benchmarks executed")
    else:
        print(f"SUMMARY: {succeeded}/{executed} succeeded, {failed} failed")
        print(f"Results saved to: {results_file}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
