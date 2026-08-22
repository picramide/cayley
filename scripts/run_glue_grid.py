#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = ["mrpc", "sst2", "mnli", "stsb", "qnli", "cola", "rte", "qqp"]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--dataset_name", type=str, default="nyu-mll/glue")
    parser.add_argument("--masks_dir", type=str, default="masks")
    parser.add_argument("--output_root", type=str, default="outputs/grid")
    parser.add_argument("--results_file", type=str, default="results/glue_grid.jsonl")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def maybe_sample_args(flag: str, value: int | None) -> list[str]:
    if value is None:
        return []
    return [flag, str(value)]


def generate_all_masks(args) -> list[tuple[str, Path | None]]:
    masks_dir = Path(args.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("dense_mask", ["--kind", "dense", "--seq", str(args.max_length)]),
        ("hypercube", ["--kind", "hypercube", "--seq", str(args.max_length)]),
        (
            "circulant_1_2_4_8_16",
            [
                "--kind",
                "circulant",
                "--seq",
                str(args.max_length),
                "--generators",
                "1,2,4,8,16",
            ],
        ),
        (
            "window_dilations",
            [
                "--kind",
                "window_dilations",
                "--seq",
                str(args.max_length),
                "--window",
                "4",
                "--dilations",
                "8,16,32,64",
            ],
        ),
        (
            "random_circulant_d8",
            [
                "--kind",
                "random_circulant",
                "--seq",
                str(args.max_length),
                "--degree",
                "8",
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "random_circulant_d16",
            [
                "--kind",
                "random_circulant",
                "--seq",
                str(args.max_length),
                "--degree",
                "16",
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "local_w16",
            ["--kind", "local", "--seq", str(args.max_length), "--window", "16"],
        ),
        (
            "bigbird_g2_b2_r1",
            [
                "--kind",
                "bigbird",
                "--seq",
                str(args.max_length),
                "--heads",
                str(args.heads),
                "--global_tokens",
                "2",
                "--block_size",
                "2",
                "--num_random_blocks",
                "1",
                "--window_block_left",
                "1",
                "--window_block_right",
                "1",
                "--seed",
                str(args.seed),
            ],
        ),
    ]

    if not is_power_of_two(args.max_length):
        specs = [(name, spec) for name, spec in specs if name != "hypercube"]
        print(f"Skipping hypercube mask because max_length={args.max_length} is not a power of two.")

    masks: list[tuple[str, Path | None]] = [("dense", None)]
    for name, spec in specs:
        out = masks_dir / f"{name}_{args.max_length}.pt"
        if args.skip_existing and out.exists():
            print(f"using existing mask {out}")
        else:
            run([sys.executable, "scripts/generate_masks.py", *spec, "--output", str(out)], args.dry_run)
        masks.append((name, out))
    return masks


def main():
    args = parse_args()
    masks = generate_all_masks(args)
    total_runs = len(args.tasks) * len(masks)
    print(f"Running {total_runs} benchmark jobs: {len(args.tasks)} tasks x {len(masks)} attention patterns.")

    for task in args.tasks:
        for mask_name, mask_path in masks:
            cmd = [
                sys.executable,
                "scripts/benchmark_roberta_glue.py",
                "--task_name",
                task,
                "--dataset_name",
                args.dataset_name,
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
                str(Path(args.output_root) / task / mask_name),
                "--results_file",
                args.results_file,
                "--run_name",
                f"{task}_{mask_name}",
            ]
            cmd.extend(maybe_sample_args("--max_train_samples", args.max_train_samples))
            cmd.extend(maybe_sample_args("--max_eval_samples", args.max_eval_samples))
            if mask_path is not None:
                cmd.extend(["--mask_path", str(mask_path)])
            run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
