#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


class DataCollatorWithPaddingAndLabels(DataCollatorWithPadding):
    """Data collator that includes labels column."""

    def __call__(self, features):
        # Extract labels if present
        labels = None
        if "labels" in features[0]:
            labels = torch.tensor([f["labels"] for f in features])

        # Call parent to collate inputs
        batch = super().__call__(features)

        # Add labels back
        if labels is not None:
            batch["labels"] = labels

        return batch


from cayley.benchmarking import BENCHMARK_PROTOCOL_VERSION
from cayley.glue import (
    compute_metrics_fn,
    get_task_config,
    normalize_task_name,
    preprocess_examples,
)
from cayley.roberta_sparse_attention import configure_sparse_attention, load_mask


DEFAULT_GLUE_DATASET = "nyu-mll/glue"


def build_training_arguments(**kwargs) -> TrainingArguments:
    """Handle Transformers versions that renamed evaluation_strategy."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "evaluation_strategy" not in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_GLUE_DATASET)
    parser.add_argument("--output_dir", type=str, default="outputs/roberta_glue")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--mask_name", type=str, default="dense")
    parser.add_argument("--mask_config", type=json.loads, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--results_file", type=str, default=None)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--verbose_mask", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=1)
    return parser.parse_args()


def subset_dataset(dataset, max_samples):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def write_results(path: str | None, payload: dict) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def restore_best_checkpoint(model, checkpoint: str) -> None:
    """Apply checkpoint name conversions before copying all weights strictly.

    Transformers 5.3 saves RoBERTa LayerNorm gamma/beta aliases, but Trainer's
    raw best-checkpoint loader does not convert them back to weight/bias.
    Loading through from_pretrained also handles sharded checkpoints correctly.
    Keep the existing model object so Trainer's device/wrapper references remain valid.
    """
    best_model = RobertaForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, attn_implementation="eager",
    )
    model.load_state_dict(best_model.state_dict(), strict=True)


def main():
    args = parse_args()
    task_name = normalize_task_name(args.task_name)
    task_config = get_task_config(task_name)
    set_seed(args.seed)

    configure_sparse_attention(args.mask_path, verbose=args.verbose_mask)
    if args.mask_path:
        mask = load_mask(args.mask_path)
        print(f"Loaded sparse mask {args.mask_path} with shape {tuple(mask.shape)}")
    else:
        print("No mask_path supplied; running dense attention.")

    # Determine if using local or remote dataset
    dataset_path = Path(args.dataset_name)
    if dataset_path.is_dir():
        # Load locally saved dataset using load_from_disk
        dataset = load_from_disk(str(args.dataset_name))
        print(f"Loaded local dataset from {args.dataset_name}")
        print(f"Dataset splits: {list(dataset.keys())}")
        print(f"Column names: {dataset[list(dataset.keys())[0]].column_names}")
    else:
        # Load from Hugging Face hub
        dataset = load_dataset(args.dataset_name, task_name)
        print(f"Loaded remote dataset {args.dataset_name}/{task_name}")

    # Get column keys from task config
    sentence1_key = task_config["sentence1_key"]
    sentence2_key = task_config["sentence2_key"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)

    # Map to tokenize text and include labels
    def preprocess_with_labels(batch):
        """Preprocess text and include labels for model training."""
        result = preprocess_examples(
            batch,
            tokenizer,
            args.max_length,
            sentence1_key,
            sentence2_key,
        )
        # Add labels
        if "labels" in batch:
            result["labels"] = batch["labels"]
        elif "label" in batch:
            result["labels"] = batch["label"]
        return result

    encoded = dataset.map(
        preprocess_with_labels,
        batched=True,
    )

    train_dataset = subset_dataset(encoded["train"], args.max_train_samples)
    if task_name == "mnli":
        eval_dataset = subset_dataset(encoded["validation_matched"], args.max_eval_samples)
        eval_dataset_mismatched = subset_dataset(encoded["validation_mismatched"], args.max_eval_samples)
    else:
        eval_dataset = subset_dataset(encoded["validation"], args.max_eval_samples)
        eval_dataset_mismatched = None

    model_kwargs = {
        "num_labels": task_config["num_labels"],
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
        # Offline checkpoints have a two-label head; STS-B and MNLI need
        # different output sizes. Reinitialize only when training the new head.
        "ignore_mismatched_sizes": args.do_train,
        # Use the same backend for dense and sparse runs. In Transformers 4,
        # SDPA can bypass the patched class; Transformers 5 changes mask dtype.
        "attn_implementation": "eager",
    }
    if task_config["is_regression"]:
        model_kwargs["problem_type"] = "regression"

    model = RobertaForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)

    training_args = build_training_arguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        do_train=args.do_train,
        do_eval=args.do_eval,
        eval_strategy="epoch" if args.do_train and args.do_eval else "no",
        save_strategy="epoch" if args.do_train else "no",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.do_train and args.do_eval,
        metric_for_best_model=task_config["metric_for_best_model"],
        greater_is_better=True,
    )

    # processing_class replaced tokenizer in newer Transformers versions.
    tokenizer_arg = (
        "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        **{tokenizer_arg: tokenizer},
        data_collator=DataCollatorWithPaddingAndLabels(tokenizer=tokenizer),
        compute_metrics=compute_metrics_fn(task_name),
    )

    train_metrics = {}
    if args.do_train:
        train_metrics = trainer.train().metrics
        if training_args.load_best_model_at_end:
            restore_best_checkpoint(model, trainer.state.best_model_checkpoint)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    all_metrics = {}
    if args.do_eval:
        if task_name == "mnli":
            all_metrics.update(
                trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_matched")
            )
            all_metrics.update(
                trainer.evaluate(eval_dataset=eval_dataset_mismatched, metric_key_prefix="eval_mismatched")
            )
        else:
            all_metrics.update(trainer.evaluate())

    if args.do_train:
        # Persist loss/gradient history at the run root, independent of checkpoint rotation.
        trainer.save_state()

    best_checkpoint = trainer.state.best_model_checkpoint
    best_step = int(Path(best_checkpoint).name.rsplit("-", 1)[1]) if best_checkpoint else None
    best_epoch = next((
        entry["epoch"] for entry in trainer.state.log_history
        if entry.get("step") == best_step and "eval_loss" in entry
    ), None)
    payload = {
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "task_name": task_name,
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "mask_name": args.mask_name,
        "mask_path": args.mask_path,
        "mask_config": args.mask_config,
        "mask_sha256": hashlib.sha256(Path(args.mask_path).read_bytes()).hexdigest() if args.mask_path else None,
        "seed": args.seed,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "weight_decay": args.weight_decay,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "attention_implementation": model.config._attn_implementation,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "trainer_seed": training_args.seed,
        "data_seed": training_args.data_seed,
        "load_best_model_at_end": training_args.load_best_model_at_end,
        "metric_for_best_model": training_args.metric_for_best_model,
        "best_checkpoint": best_checkpoint,
        "best_metric": trainer.state.best_metric,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "trainer_state_path": str(Path(args.output_dir) / "trainer_state.json") if args.do_train else None,
        "metrics": all_metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    write_results(args.results_file, payload)


if __name__ == "__main__":
    main()
