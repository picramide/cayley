#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MethodType

import numpy as np
import torch
import transformers
from datasets import load_from_disk
from filelock import FileLock
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments, set_seed
from transformers.models.deberta_v2.modeling_deberta_v2 import scaled_size_sqrt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cayley import roberta_sparse_attention as roberta_attention
from cayley.deberta_sparse_attention import CayleyDebertaV2ForSequenceClassification, load_mask
from cayley.glue import compute_metrics_fn, get_task_config, preprocess_examples
from student_package.scripts.masks import soft_mask_bias

LOG_STEPS = dict(cola=9, sst2=75, mrpc=4, stsb=6, rte=2, qnli=110, qqp=400)
PLACEMENTS = ["hypercube", "hypercube_gray", *[f"hypercube_hampath{i}" for i in range(3)]]
MASKS = ["dense", "local", "window_dilations", *PLACEMENTS]
TARGET = dict(cola="hypercube", qnli="window_dilations", rte="local")
MODELS = dict(roberta=RobertaForSequenceClassification, deberta=CayleyDebertaV2ForSequenceClassification)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def saved_split(dataset, task, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task}.json"
    metadata = dict(task=task, size=len(dataset), dataset_fingerprint=dataset._fingerprint,
                    seed=42, selection_fraction=0.1, stratified=task != "stsb")
    with FileLock(str(path) + ".lock"):
        if path.exists():
            split = json.loads(path.read_text())
            if any(split.get(key) != value for key, value in metadata.items()):
                raise ValueError(f"Saved split does not match the dataset: {path}")
        else:
            train, selection = train_test_split(
                np.arange(len(dataset)), test_size=0.1, random_state=42,
                stratify=None if task == "stsb" else dataset["label"],
            )
            split = dict(metadata, train_indices=sorted(train.tolist()), selection_indices=sorted(selection.tolist()))
            write_json(path, split)
    train, selection = split["train_indices"], split["selection_indices"]
    if sorted(train + selection) != list(range(len(dataset))) or len(selection) != math.ceil(len(dataset) / 10):
        raise ValueError(f"Invalid saved split: {path}")
    return train, selection, path


def audited_mask(model, task, name):
    report_path = ROOT / "data" / ("implementation_audit_fp16.json" if model == "roberta"
                                   else "implementation_audit_deberta_fp16.json")
    report = json.loads(report_path.read_text())
    if not report["passed"]:
        raise ValueError(f"Implementation audit failed: {report_path}")
    if name == "dense":
        return None, None
    path = ROOT / "masks" / (f"{name}.pt" if name in PLACEMENTS[1:] else f"{task}_{name}.pt")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    case = report["checks"]["masks_and_cross_task_equality"]["cases"][name]
    if not case["passed"] or case["file_sha256"].get(path.name) != digest:
        raise ValueError(f"Mask does not match its passed audit: {path}")
    mask = load_mask(path)
    if mask.shape[-2:] != (128, 128):
        raise ValueError(f"Expected a 128-token mask: {path}")
    return mask, digest


def roberta_soft_forward(self, hidden_states, *args, **kwargs):
    if self.soft_bias is not None:
        dtype = torch.get_autocast_dtype(hidden_states.device.type) if torch.is_autocast_enabled(hidden_states.device.type) else hidden_states.dtype
        bias = self.soft_bias[..., :hidden_states.shape[1], :hidden_states.shape[1]].to(dtype=dtype)
        padding = args[0] if args else kwargs.get("attention_mask")
        if padding is not None:
            padding = (torch.zeros_like(padding, dtype=dtype).masked_fill(~padding, float("-inf"))
                       if padding.dtype == torch.bool else padding.to(dtype=dtype))
            bias = padding + bias
        if args:
            args = (bias, *args[1:])
        else:
            kwargs["attention_mask"] = bias
    return self.hard_forward(hidden_states, *args, **kwargs)


def deberta_soft_forward(self, hidden_states, attention_mask, output_attentions=False,
                         query_states=None, relative_pos=None, rel_embeddings=None):
    if self.soft_bias is None:
        return self.hard_forward(hidden_states, attention_mask, output_attentions,
                                 query_states, relative_pos, rel_embeddings)
    query_states = hidden_states if query_states is None else query_states
    query = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    key = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    value = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
    scale = scaled_size_sqrt(query, factor).to(dtype=query.dtype)
    scores = torch.bmm(query, key.transpose(-1, -2) / scale)
    if self.relative_attention:
        scores = scores + self.disentangled_attention_bias(
            query, key, relative_pos, self.pos_dropout(rel_embeddings), factor,
        )
    scores = scores.view(-1, self.num_attention_heads, scores.size(-2), scores.size(-1))
    scores = scores + self.soft_bias[..., :scores.size(-2), :scores.size(-1)].to(dtype=scores.dtype)
    scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)
    probabilities = self.dropout(torch.softmax(scores, dim=-1))
    context = torch.bmm(probabilities.reshape(-1, scores.size(-2), scores.size(-1)), value)
    context = context.view(-1, self.num_attention_heads, context.size(-2), context.size(-1))
    context = context.permute(0, 2, 1, 3).contiguous().flatten(2)
    return context, probabilities if output_attentions else None


class MaskSchedule:
    def __init__(self, model, name, mask, condition):
        self.model, self.name, self.mask, self.condition = model, name, mask, condition
        self.unit_bias = None if mask is None else torch.from_numpy(soft_mask_bias(mask.numpy(), 1.0))
        if name == "roberta":
            roberta_attention.patch_roberta_attention()
        self.layers = getattr(model, name).encoder.layer
        for layer in self.layers:
            attention = layer.attention.self
            attention.soft_bias = None
            if condition == "G":
                attention.hard_forward = attention.forward
                attention.forward = MethodType(roberta_soft_forward if name == "roberta" else deberta_soft_forward, attention)

    def apply(self, training=False, step=0, total=1):
        device = next(self.model.parameters()).device
        mask = None if self.mask is None else self.mask.to(device)
        soft = training and self.condition == "G" and 2 * step < total
        bias = None
        if soft:
            beta = 20.0 * step / (total / 2)
            bias = self.unit_bias.to(device) * beta
            bias = bias[None, None] if bias.ndim == 2 else bias[None]
        for layer in self.layers:
            layer.attention.self.soft_bias = bias
        if self.name == "roberta":
            roberta_attention.GLOBAL_SPARSE_MASK = None if soft or mask is None else (mask[None, None] if mask.ndim == 2 else mask[None])
        else:
            self.model.set_structural_mask(None if soft else mask)


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.size, self.seed, self.epoch = len(dataset), seed, 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist())


class StudyTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        return EpochSampler(self.train_dataset if train_dataset is None else train_dataset, self.args.data_seed)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.mask_schedule.apply(model.training, self.state.global_step, self.state.max_steps)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if not torch.isfinite(loss).all() or not torch.isfinite(outputs.logits).all():
            raise FloatingPointError(f"Non-finite loss or logits at step {self.state.global_step}")
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, *args, **kwargs):
        self.mask_schedule.apply()
        return super().prediction_step(*args, **kwargs)

    def _load_best_model(self):
        best = type(self.model).from_pretrained(self.state.best_model_checkpoint, local_files_only=True,
                                               attn_implementation="eager", dtype=torch.float32)
        self.model.load_state_dict(best.state_dict(), strict=True)


def training_arguments(output, task, condition, precision, seed, train_size):
    epochs = 15 if condition == "S+" else 5
    steps = math.ceil(train_size / 8) * epochs
    return TrainingArguments(
        output_dir=str(output), num_train_epochs=epochs,
        learning_rate=2e-5, lr_scheduler_type="linear", warmup_steps=0 if condition == "P" else math.ceil(steps * 0.1),
        optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        weight_decay=0.01, max_grad_norm=1.0, per_device_train_batch_size=8,
        per_device_eval_batch_size=16, gradient_accumulation_steps=1,
        fp16=precision == "fp16", bf16=False, tf32=False, seed=seed, data_seed=seed,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model=get_task_config(task)["metric_for_best_model"],
        greater_is_better=True, logging_strategy="steps", logging_steps=LOG_STEPS[task],
        logging_first_step=True, logging_nan_inf_filter=False, report_to="none",
        dataloader_num_workers=0,
    )


def encode(dataset, tokenizer, task):
    config = get_task_config(task)
    return dataset.map(
        lambda examples: preprocess_examples(examples, tokenizer, 128, config["sentence1_key"], config["sentence2_key"]),
        batched=True, remove_columns=[column for column in dataset.column_names if column != "label"],
    )


def run_one(args, name, seed, tokenizer, dataset, split, mask, mask_hash, output):
    config = get_task_config(args.task)
    set_seed(seed)
    model = MODELS[args.model].from_pretrained(
        args.model_path, num_labels=config["num_labels"], ignore_mismatched_sizes=True,
        problem_type="regression" if config["is_regression"] else "single_label_classification",
        local_files_only=True, attn_implementation="eager", dtype=torch.float32,
    )
    head = hashlib.sha256()
    for key, value in model.state_dict().items():
        if key.startswith(("classifier.", "pooler.")):
            head.update(key.encode())
            head.update(value.cpu().contiguous().numpy().tobytes())
    train_indices, selection_indices, split_path = split
    training_args = training_arguments(output, args.task, args.condition, args.precision, seed, len(train_indices))
    if training_args.world_size != 1 or training_args.n_gpu > 1:
        raise ValueError("The protocol uses one GPU; set CUDA_VISIBLE_DEVICES to a single device")
    trainer = StudyTrainer(
        model=model, args=training_args, train_dataset=dataset["train"].select(train_indices),
        eval_dataset=dataset["train"].select(selection_indices),
        data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer,
        compute_metrics=compute_metrics_fn(args.task),
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.mask_schedule = MaskSchedule(model, args.model, mask, args.condition)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    write_json(output / "run_config.json", dict(
        model=args.model, model_path=str(args.model_path), task=args.task, mask=name, condition=args.condition,
        seed=seed, precision=args.precision, parameter_dtype=str(model.dtype), max_length=128,
        training_args=training_args.to_dict(),
        split_path=str(split_path), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        mask_sha256=mask_hash, initial_head_sha256=head.hexdigest(),
        git_commit=revision.stdout.strip(), git_dirty=bool(dirty.stdout.strip()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    ))
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        trainer.save_state()
        prediction = trainer.predict(dataset["validation"], metric_key_prefix="dev")
        values = prediction.predictions.squeeze(-1) if args.task == "stsb" else prediction.predictions.argmax(-1)
        cast = float if args.task == "stsb" else int
        temporary = output / "dev_predictions.jsonl.tmp"
        with temporary.open("w") as file:
            for idx, (label, pred, padding) in enumerate(zip(prediction.label_ids, values, dataset["validation"]["attention_mask"])):
                file.write(json.dumps(dict(idx=idx, label=cast(label), pred=cast(pred), length=int(sum(padding))), allow_nan=False) + "\n")
        temporary.replace(output / "dev_predictions.jsonl")
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        summary = dict(train_runtime_s=runtime, first_logged_loss=losses[0], last_logged_loss=losses[-1],
                       best_checkpoint=trainer.state.best_model_checkpoint, best_selection_metric=trainer.state.best_metric,
                       dev_metrics=prediction.metrics,
                       degenerate=None if args.task == "stsb" else len(np.unique(values)) == 1)
        write_json(output / "run_summary.json", summary)
        print(json.dumps(dict(model=args.model, task=args.task, mask=name, seed=seed, **summary)))
        return runtime, summary["degenerate"]
    except BaseException as error:
        trainer.save_state()
        write_json(output / "crash.json", dict(error=repr(error), traceback=traceback.format_exc(), step=trainer.state.global_step))
        raise
    finally:
        trainer.mask_schedule.apply()


def validate_request(args):
    if args.experiment == "exp1":
        if args.task not in TARGET:
            raise ValueError("Experiment 1 uses CoLA, QNLI and RTE")
        allowed = [TARGET[args.task], "dense"] if args.condition != "G" else [TARGET[args.task]]
        if args.task == "qnli" and args.condition == "P":
            allowed.append("local")
        maximum = 5 if args.task == "qnli" else 20
    else:
        if args.condition != "P" or args.precision != "fp16":
            raise ValueError("Experiment 2 uses P in fp16")
        allowed = ["dense", *PLACEMENTS]
        if args.task == "cola":
            allowed = PLACEMENTS[1:]
        elif args.task in ("rte", "qnli"):
            allowed = PLACEMENTS
        elif args.task == "qqp":
            allowed = ["dense", *PLACEMENTS[:2]]
        maximum = dict(cola=20, sst2=5, mrpc=10, stsb=10, rte=10, qnli=5, qqp=3)[args.task]
    if any(mask not in allowed for mask in args.masks) or any(seed not in range(1, maximum + 1) for seed in args.seeds):
        raise ValueError(f"This arm permits masks {allowed} and seeds 1–{maximum}; shared controls belong in exp1")
    if len(set(args.masks)) != len(args.masks) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Masks and seeds must not contain duplicates")
    if args.checkpoint1 and (args.model != "deberta" or args.task not in ("cola", "qnli") or
                            args.masks != ["dense"] or args.condition != "P" or args.precision != "fp16" or len(args.seeds) != 1):
        raise ValueError("Checkpoint 1 timing uses one dense DeBERTa P fp16 run on CoLA or QNLI")
    if args.precision == "fp32":
        if args.condition != "P" or args.masks != [TARGET[args.task]] or len(args.seeds) != 1:
            raise ValueError("The fp32 check uses P and the target sparse mask, for one degenerate seed")
        path = args.output_dir / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        candidates = []
        for row in rows:
            if all(row[key] == value for key, value in dict(model=args.model, experiment="exp1", task=args.task,
                    mask=args.masks[0], condition="P", precision="fp16").items()):
                predictions = [json.loads(line)["pred"] for line in (args.output_dir / row["run_dir"] / "dev_predictions.jsonl").read_text().splitlines()]
                if len(set(predictions)) == 1:
                    candidates.append(row["seed"])
        if not candidates or args.seeds[0] != min(candidates):
            raise ValueError("Select the first degenerate fp16 P seed for the fp32 check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=LOG_STEPS, required=True)
    parser.add_argument("--masks", nargs="+", choices=MASKS, required=True)
    parser.add_argument("--experiment", choices=("exp1", "exp2"), default="exp1")
    parser.add_argument("--condition", choices=("P", "S", "S+", "G"), default="P")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "offline" / "glue")
    parser.add_argument("--output-dir", type=Path, default=Path("/scratch/scratch26/22cs30008/res/outputs/runs"))
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--retry-crashed", action="store_true")
    parser.add_argument("--checkpoint1", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.model_path is None:
        args.model_path = Path.home() / "offline" / "models" / ("roberta-base" if args.model == "roberta" else "deberta-v3-base")
    validate_request(args)
    if args.precision == "fp16" and not torch.cuda.is_available():
        raise ValueError("fp16 training requires CUDA")
    masks = {name: audited_mask(args.model, args.task, name) for name in args.masks}
    raw = load_from_disk(str(args.data_root / args.task))
    split = saved_split(raw["train"], args.task, ROOT / "data" / "splits")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.padding_side != "right":
        raise ValueError("Position-zero readout requires right padding")
    dataset = {part: encode(raw[part], tokenizer, args.task) for part in ("train", "validation")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = args.output_dir / "results.jsonl"
    for name in args.masks:
        for seed in args.seeds:
            suffix = "_fp32" if args.precision == "fp32" else ""
            relative = Path(args.model) / args.experiment / args.task / name / args.condition / f"seed{seed}{suffix}"
            output = args.output_dir / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(output) + ".lock", timeout=0):
                if output.exists():
                    if (output / "result.json").exists() and args.skip_completed:
                        row = json.loads((output / "result.json").read_text())
                        if not all((output / file).is_file() for file in ("dev_predictions.jsonl", "trainer_state.json")):
                            raise ValueError(f"Incomplete completed run: {output}")
                        print(f"Keeping completed run: {relative}")
                    elif not (output / "result.json").exists() and args.retry_crashed:
                        archive = output.with_name(output.name + f"_crashed_{time.time_ns()}")
                        output.rename(archive)
                        print(f"Retrying seed {seed}; report this crash to the professor. Preserved: {archive}")
                    else:
                        raise FileExistsError(f"Keeping {output}; use --skip-completed or --retry-crashed")
                if not output.exists():
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    output.mkdir()
                    runtime, degenerate = run_one(args, name, seed, tokenizer, dataset, split, *masks[name], output)
                    row = dict(model=args.model, experiment=args.experiment, task=args.task, mask=name,
                               condition=args.condition, seed=seed, precision=args.precision,
                               run_dir=relative.as_posix(), train_runtime_s=runtime)
                    write_json(output / "result.json", row)
                with FileLock(str(results) + ".lock"):
                    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()] if results.exists() else []
                    if not any(previous["run_dir"] == row["run_dir"] for previous in rows):
                        with results.open("a") as file:
                            file.write(json.dumps(row, allow_nan=False) + "\n")
                if args.checkpoint1 and json.loads((output / "run_summary.json").read_text())["degenerate"]:
                    raise SystemExit("Dense DeBERTa collapsed. Results are preserved; stop and tell the professor.")


if __name__ == "__main__":
    main()
