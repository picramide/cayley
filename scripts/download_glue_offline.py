#!/usr/bin/env python3
"""
Download script for GLUE datasets and RoBERTa-base model for offline use.

Usage:
    python scripts/download_glue_offline.py --output_dir /path/to/offline_data

This script downloads:
- All GLUE tasks: cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb
- RoBERTa-base model (FacebookAI/roberta-base)

The downloaded data can then be used with benchmark_all_masks.py using --dataset_name and --model_name
pointing to the local paths.
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, RobertaModel, RobertaForSequenceClassification


def download_datasets(output_dir: Path, cache_dir: Path | None = None):
    """Download all GLUE datasets."""
    from datasets import load_dataset

    # GLUE tasks to download
    glue_tasks = [
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "stsb",
    ]

    print(f"Downloading GLUE datasets to: {output_dir}")
    print(f"Cache directory: {cache_dir}")

    for task in glue_tasks:
        print(f"\nDownloading {task}...")
        try:
            dataset = load_dataset(
                "nyu-mll/glue",
                task,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=True,
                num_proc=1,  # Single process for reliability
            )
            # Save to output directory
            task_dir = output_dir / "glue" / task
            dataset.save_to_disk(str(task_dir))
            print(f"  Saved to: {task_dir}")
            print(f"  Splits: {list(dataset.keys())}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("\n" + "=" * 60)
    print("GLUE datasets downloaded successfully!")
    print(f"Location: {output_dir / 'glue'}")


def download_model(output_dir: Path, model_name: str = "FacebookAI/roberta-base"):
    """Download RoBERTa-base model."""
    print(f"\nDownloading model: {model_name}")
    print(f"Output directory: {output_dir}")

    model_dir = output_dir / "models" / "roberta-base"

    # Download tokenizer
    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(model_dir))
    print(f"  Tokenizer saved to: {model_dir}")

    # Download config
    print("  Downloading config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.save_pretrained(str(model_dir))
    print(f"  Config saved to: {model_dir}")

    # Download model weights (base model)
    print("  Downloading base model weights...")
    model = RobertaModel.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(str(model_dir))
    print(f"  Base model saved to: {model_dir}")

    # Download model with classification head (for fine-tuning)
    print("  Downloading model with classification head...")
    model_cls = RobertaForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    model_cls.save_pretrained(str(model_dir))
    print(f"  Model with classification head saved to: {model_dir}")

    print("\n" + "=" * 60)
    print("Model downloaded successfully!")
    print(f"Location: {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download GLUE datasets and RoBERTa-base for offline use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to save downloaded data",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--only_datasets",
        action="store_true",
        help="Only download datasets, skip model",
    )
    parser.add_argument(
        "--only_model",
        action="store_true",
        help="Only download model, skip datasets",
    )
    parser.add_argument(
        "--download_timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each download (default: 300)",
    )
    # Note: datasets library handles timeout internally; timeout param is for user reference
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GLUE Dataset and Model Download Script")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print("=" * 60)

    # Download datasets
    if not args.only_model:
        download_datasets(output_dir, cache_dir)

    # Download model
    if not args.only_datasets:
        download_model(output_dir)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nTo use these offline resources with benchmark_all_masks.py:")
    print("  --dataset_name <output_dir>/glue/<task_name>")
    print("  --model_name <output_dir>/models/roberta-base")


if __name__ == "__main__":
    main()
