#!/usr/bin/env python3
"""Check the released DeBERTa-v3 RTD head on WikiText-103 validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers import AutoTokenizer, DebertaV2Model
from transformers.activations import ACT2FN


class RTDHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        states = self.LayerNorm(hidden_states[:, :1] + hidden_states)
        return self.classifier(self.activation(self.dense(states))).squeeze(-1)


def load_model_and_head(model_path, device):
    model, info = DebertaV2Model.from_pretrained(
        model_path, local_files_only=True, attn_implementation="eager", output_loading_info=True
    )
    if info["missing_keys"]:
        raise ValueError(f"Missing encoder weights: {info['missing_keys']}")
    if model.config.position_biased_input is not False:
        raise ValueError("Expected DeBERTa-v3 relative-only position encoding")
    checkpoint = torch.load(model_path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    prefix = "mask_predictions."
    head_weights = {name[len(prefix):]: weight for name, weight in checkpoint.items() if name.startswith(prefix)}
    head = RTDHead(model.config)
    head.load_state_dict(head_weights, strict=True)
    return model.to(device=device, dtype=torch.float32).eval(), head.to(device=device, dtype=torch.float32).eval()


def token_batches(split, tokenizer, batch_size=256):
    texts = []
    for row in split:
        if row["text"].strip():
            texts.append(row["text"])
        if len(texts) == batch_size:
            yield tokenizer(texts, add_special_tokens=False)["input_ids"]
            texts.clear()
    if texts:
        yield tokenizer(texts, add_special_tokens=False)["input_ids"]


def count_unigrams(split, tokenizer, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.int64)
    special = set(tokenizer.all_special_ids)
    processed = 0
    for batch in token_batches(split, tokenizer):
        ids = np.fromiter((token for row in batch for token in row if token not in special), dtype=np.int64)
        if ids.size:
            if ids.min() < 0 or ids.max() >= vocab_size:
                raise ValueError("Tokenizer produced an ID outside the model vocabulary")
            counts += np.bincount(ids, minlength=vocab_size)
        processed += len(batch)
        if processed // 100000 > (processed - len(batch)) // 100000:
            print(f"Tokenized {processed} training rows", flush=True)
    if counts.sum() == 0:
        raise ValueError("No training tokens found")
    return counts


def make_validation_inputs(split, tokenizer, counts, seed, length=128):
    special = set(tokenizer.all_special_ids)
    tokens = [token for batch in token_batches(split, tokenizer) for row in batch
              for token in row if token not in special]
    content_length = length - tokenizer.num_special_tokens_to_add(pair=False)
    if not tokens or content_length <= 0:
        raise ValueError("No validation tokens or invalid window length")
    windows = [tokens[start:start + content_length] for start in range(0, len(tokens), content_length)]
    original = np.full((len(windows), length), tokenizer.pad_token_id, dtype=np.int64)
    padding = np.zeros(original.shape, dtype=bool)
    eligible = np.zeros(original.shape, dtype=bool)
    for index, content in enumerate(windows):
        ids = tokenizer.build_inputs_with_special_tokens(content)
        special_mask = tokenizer.get_special_tokens_mask(content, already_has_special_tokens=False)
        if len(ids) != len(special_mask) or len(ids) > length:
            raise ValueError("Unexpected tokenizer special-token layout")
        original[index, :len(ids)] = ids
        padding[index, :len(ids)] = True
        eligible[index, :len(ids)] = np.logical_not(special_mask)

    rng = np.random.default_rng(seed)
    positions = np.flatnonzero(eligible)
    chosen = rng.choice(positions, size=round(0.15 * len(positions)), replace=False)
    replacements = rng.choice(len(counts), size=len(chosen), p=counts / counts.sum())
    corrupted = original.copy()
    corrupted.flat[chosen] = replacements
    labels = eligible & (corrupted != original)
    if not labels.any() or labels.sum() == eligible.sum():
        raise ValueError("Replacement labels contain only one class")
    return corrupted, padding, eligible, labels, len(chosen)


def score(model, head, input_ids, padding, eligible, labels, device, batch_size):
    all_scores = []
    all_labels = []
    with torch.inference_mode():
        for start in range(0, len(input_ids), batch_size):
            end = start + batch_size
            ids = torch.from_numpy(input_ids[start:end]).to(device)
            mask = torch.from_numpy(padding[start:end]).to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            logits = head(hidden).float().cpu().numpy()
            valid = eligible[start:end]
            all_scores.append(logits[valid])
            all_labels.append(labels[start:end][valid])
    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_labels)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite RTD scores")
    return float(roc_auc_score(targets, scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wikitext", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "deberta_rtd_auroc.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    from datasets import load_from_disk

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data = load_from_disk(str(args.wikitext))
    model, head = load_model_and_head(args.model, torch.device(args.device))
    counts = count_unigrams(data["train"], tokenizer, model.config.vocab_size)
    inputs, padding, eligible, labels, selected = make_validation_inputs(data["validation"], tokenizer, counts, args.seed)
    auc = score(model, head, inputs, padding, eligible, labels, torch.device(args.device), args.batch_size)
    report = {
        "model": str(args.model), "wikitext": str(args.wikitext), "device": args.device,
        "precision": "fp32", "attention": "dense", "replacement_fraction": 0.15,
        "seed": args.seed, "window_length": 128, "windows": len(inputs),
        "training_tokens": int(counts.sum()), "eligible_validation_tokens": int(eligible.sum()),
        "selected_for_replacement": selected, "actually_replaced": int(labels.sum()), "auroc": auc,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
