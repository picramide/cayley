#!/usr/bin/env python3
import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from cayley.glue import (
    compute_metrics_fn,
    get_task_config,
    normalize_task_name,
    preprocess_examples,
)
from cayley.roberta_sparse_attention import configure_sparse_attention, load_mask


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
    parser.add_argument("--output_dir", type=str, default="outputs/roberta_glue")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--mask_name", type=str, default="dense")
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

    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)

    encoded = dataset.map(
        lambda batch: preprocess_examples(
            batch,
            tokenizer,
            args.max_length,
            task_config["sentence1_key"],
            task_config["sentence2_key"],
        ),
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
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_fn(task_name),
    )

    if args.do_train:
        trainer.train()
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

    payload = {
        "task_name": task_name,
        "model_name": args.model_name,
        "mask_name": args.mask_name,
        "mask_path": args.mask_path,
        "seed": args.seed,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "metrics": all_metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    write_results(args.results_file, payload)


if __name__ == "__main__":
    main()
