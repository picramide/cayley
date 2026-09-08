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

    # For local dataset paths, use 'default' config; for remote, use task_name
    from pathlib import Path
    dataset_path = Path(args.dataset_name)
    if dataset_path.is_dir():
        # Local dataset - load with default config
        dataset = load_dataset(args.dataset_name, "default")
        # For local datasets, we need to infer the column names from the schema
        # Get column names from the first split
        first_split = list(dataset.keys())[0]
        columns = dataset[first_split].column_names
        # Check for common column name patterns
        has_sentence1 = 'sentence1' in columns
        has_sentence2 = 'sentence2' in columns
        has_question1 = 'question1' in columns
        has_question2 = 'question2' in columns
        has_premise = 'premise' in columns
        has_hypothesis = 'hypothesis' in columns
        has_question = 'question' in columns
        has_sentence = 'sentence' in columns

        # Map task to column names based on what's available
        if task_name == "mrpc":
            sentence1_key = "sentence1" if has_sentence1 else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = "sentence2" if has_sentence2 else None
        elif task_name == "qqp":
            sentence1_key = "question1" if has_question1 else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = "question2" if has_question2 else None
        elif task_name == "qnli":
            sentence1_key = "question" if has_question else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = "sentence" if has_sentence else None
        elif task_name == "rte":
            sentence1_key = "sentence1" if has_sentence1 else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = "sentence2" if has_sentence2 else None
        elif task_name == "sst2":
            sentence1_key = "sentence" if has_sentence else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = None
        elif task_name == "cola":
            sentence1_key = "sentence" if has_sentence else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = None
        elif task_name == "mnli":
            sentence1_key = "premise" if has_premise else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = "hypothesis" if has_hypothesis else None
        elif task_name == "stsb":
            sentence1_key = "sentence1" if has_sentence1 else (columns[0] if len(columns) > 0 else "text")
            sentence2_key = "sentence2" if has_sentence2 else None
        else:
            sentence1_key = columns[0] if len(columns) > 0 else "text"
            sentence2_key = columns[1] if len(columns) > 1 else None
    else:
        # Remote dataset - use task_name as config
        dataset = load_dataset(args.dataset_name, task_name)
        # Get column keys from task config
        sentence1_key = task_config["sentence1_key"]
        sentence2_key = task_config["sentence2_key"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)

    def convert_to_string(batch):
        """Convert batch to proper string format for tokenizer."""
        result = {}
        if isinstance(batch[sentence1_key], str):
            # Single string - already correct
            result[sentence1_key] = batch[sentence1_key]
        elif isinstance(batch[sentence1_key], list):
            # List of strings - join them
            result[sentence1_key] = " ".join(str(x) for x in batch[sentence1_key])
        else:
            result[sentence1_key] = str(batch[sentence1_key])

        if sentence2_key and sentence2_key in batch:
            if isinstance(batch[sentence2_key], str):
                result[sentence2_key] = batch[sentence2_key]
            elif isinstance(batch[sentence2_key], list):
                result[sentence2_key] = " ".join(str(x) for x in batch[sentence2_key])
            else:
                result[sentence2_key] = str(batch[sentence2_key])

        return result

    # Convert data to proper string format
    dataset = dataset.map(convert_to_string, batched=False)

    encoded = dataset.map(
        lambda batch: preprocess_examples(
            batch,
            tokenizer,
            args.max_length,
            sentence1_key,
            sentence2_key,
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
        "trust_remote_code": True,
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
        "dataset_name": args.dataset_name,
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
