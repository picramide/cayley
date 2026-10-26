#!/usr/bin/env python3
"""No-training DeBERTa-v3-base attention audit."""

import argparse
import json
from pathlib import Path
import sys
import traceback

import torch
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cayley.deberta_sparse_attention import (
    CayleyDebertaV2ForSequenceClassification,
    load_mask,
    pair_attention_mask,
)
from audit_roberta_implementation import (
    LENGTH, MASKS, autocast_for, check_pilot_masks, load_saved_tensor,
    max_abs_difference, run_check,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 audit requires CUDA")
    if args.report is None:
        args.report = Path(__file__).resolve().parent.parent / "data" / f"implementation_audit_deberta_{args.precision}.json"
    return args


def test_batch(tokenizer, device, length=60):
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if any(token is None for token in (cls, sep, pad)) or tokenizer.vocab_size <= 1200:
        raise ValueError("Unexpected DeBERTa tokenizer")
    if any(token in (cls, sep, pad) for token in range(100, 1200, 100)):
        raise ValueError("A probe token is special")
    ids = torch.full((2, length), pad, dtype=torch.long, device=device)
    ids[0, 0], ids[0, 1:length - 1], ids[0, length - 1] = cls, 100, sep
    ids[1, 0], ids[1, 1:31], ids[1, 31] = cls, 200, sep
    return ids, ids.ne(pad).long()


def check_all_ones(model, ids, padding, args):
    labels = torch.tensor([0, 1], device=ids.device)
    model.set_structural_mask(None)
    model.zero_grad(set_to_none=True)
    with autocast_for(args):
        dense = model(input_ids=ids, attention_mask=padding, labels=labels)
    dense.loss.backward()
    dense_logits, dense_loss = dense.logits.detach().clone(), dense.loss.detach().clone()
    dense_grads = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                   for name, parameter in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    model.set_structural_mask(torch.ones(LENGTH, LENGTH, dtype=torch.bool))
    with autocast_for(args):
        full = model(input_ids=ids, attention_mask=padding, labels=labels)
    full.loss.backward()

    mismatched = []
    max_grad_difference = 0.0
    for name, parameter in model.named_parameters():
        expected = dense_grads[name]
        actual = None if parameter.grad is None else parameter.grad.detach().cpu()
        if expected is None or actual is None:
            if expected is not None or actual is not None:
                mismatched.append(name)
        elif not torch.equal(expected, actual):
            mismatched.append(name)
            difference = max_abs_difference(expected, actual)
            if difference is None:
                max_grad_difference = None
            elif max_grad_difference is not None:
                max_grad_difference = max(max_grad_difference, difference)

    logits_exact = torch.equal(dense_logits, full.logits)
    loss_exact = torch.equal(dense_loss, full.loss)
    return {
        "passed": logits_exact and loss_exact and not mismatched,
        "logits_exact": logits_exact,
        "loss_exact": loss_exact,
        "max_abs_logit_difference": max_abs_difference(dense_logits, full.logits),
        "max_abs_loss_difference": max_abs_difference(dense_loss, full.loss),
        "gradient_parameters_compared": sum(grad is not None for grad in dense_grads.values()),
        "parameters_without_gradient": [name for name, grad in dense_grads.items() if grad is None],
        "mismatched_gradient_count": len(mismatched),
        "first_mismatched_gradients": mismatched[:10],
        "max_abs_gradient_difference": max_grad_difference,
    }


def check_blocked_probabilities(model, ids, padding, reference_paths, args):
    cases = {"all_ones": None, **reference_paths}
    results = {}
    for mask_name, path in cases.items():
        mask = torch.ones(LENGTH, LENGTH, dtype=torch.bool) if path is None else load_mask(path)
        model.set_structural_mask(mask)
        with torch.no_grad(), autocast_for(args):
            attentions = model(input_ids=ids, attention_mask=padding, output_attentions=True).attentions
        allowed = pair_attention_mask(padding, model.structural_mask)
        if allowed.ndim == 3:
            allowed = allowed[:, None]
        blocked_nonzero = 0
        nonfinite = 0
        max_row_sum_error = 0.0
        for probabilities in attentions:
            blocked = ~allowed.expand_as(probabilities)
            blocked_nonzero += int(torch.count_nonzero(probabilities[blocked]).item())
            nonfinite += int((~torch.isfinite(probabilities)).sum().item())
            valid_rows = padding[:, None, :probabilities.shape[-2]].bool().expand(probabilities.shape[:-1])
            error = (probabilities.sum(-1)[valid_rows].float() - 1.0).abs().max().item()
            max_row_sum_error = max(max_row_sum_error, float(error))
        tolerance = 0.005 if args.precision == "fp16" else 0.00001
        results[mask_name] = {
            "passed": len(attentions) == model.config.num_hidden_layers and blocked_nonzero == 0
                      and nonfinite == 0 and max_row_sum_error <= tolerance,
            "layers_captured": len(attentions),
            "blocked_probabilities_nonzero": blocked_nonzero,
            "nonfinite_probabilities": nonfinite,
            "max_valid_row_sum_error": max_row_sum_error,
        }
    missing = [name for name in MASKS if name not in results]
    return {"passed": not missing and all(case["passed"] for case in results.values()),
            "cases": results, "missing_mask_types": missing}


def check_circulant(model, ids, padding, path, args):
    mask = load_saved_tensor(path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D circulant mask, found {tuple(mask.shape)}")
    adjacency = mask[:60, :60].bool()
    reachable = torch.zeros(60, dtype=torch.bool)
    reachable[0] = True
    for _ in range(12):
        reachable = adjacency[reachable].any(dim=0)

    model.set_structural_mask(mask)
    one, one_padding = ids[:1].clone(), padding[:1]
    with torch.no_grad(), autocast_for(args):
        original = model(input_ids=one, attention_mask=one_padding).logits.detach().clone()
        far = one.clone()
        far[0, 45] = 200
        far_logits = model(input_ids=far, attention_mask=one_padding).logits.detach().clone()
        far_exact = torch.equal(original, far_logits)
        finite = bool(torch.isfinite(original).all() and torch.isfinite(far_logits).all())
        near_difference = 0.0
        near_changes = False
        for replacement in range(200, 1200, 100):
            near = one.clone()
            near[0, 20] = replacement
            changed = model(input_ids=near, attention_mask=one_padding).logits
            finite &= bool(torch.isfinite(changed).all())
            difference = max_abs_difference(original, changed)
            if difference is None:
                near_difference = None
            elif near_difference is not None:
                near_difference = max(near_difference, difference)
            near_changes |= not torch.equal(original, changed)

    return {
        "passed": finite and bool(reachable[20]) and not bool(reachable[45]) and far_exact and near_changes,
        "logits_finite": finite,
        "reachable_positions_after_12_layers": int(reachable.sum().item()),
        "position_20_reachable": bool(reachable[20]),
        "position_45_reachable": bool(reachable[45]),
        "position_45_logits_exact": far_exact,
        "position_45_max_abs_logit_difference": max_abs_difference(original, far_logits),
        "position_20_logits_changed": near_changes,
        "position_20_max_abs_logit_difference": near_difference,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    report = {
        "model": str(args.model), "mask_dir": str(args.mask_dir),
        "pilot_results": str(args.pilot_results), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": args.precision, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "checks": {},
    }
    checks = report["checks"]
    reference_paths = {}

    def mask_audit():
        result, paths = check_pilot_masks(args)
        reference_paths.update(paths)
        return result

    run_check(checks, "pilot_masks_and_cross_task_equality", mask_audit)
    try:
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = CayleyDebertaV2ForSequenceClassification.from_pretrained(
            args.model, local_files_only=True, attn_implementation="eager"
        ).to(device).eval()
        config = model.config
        if (config.num_hidden_layers != 12 or config.num_attention_heads != 12
                or config.hidden_size != 768 or config.intermediate_size != 3072):
            raise ValueError("The audit requires DeBERTa-v3-base")
        if config._attn_implementation != "eager" or config.position_biased_input is not False:
            raise ValueError("Expected eager attention and relative-only position encoding")
        if model.deberta.embeddings.position_embeddings is not None or config.num_labels < 2:
            raise ValueError("Unexpected embeddings or classification head")
        probe = torch.randn(2, 5, config.hidden_size, device=device)
        altered = probe.clone()
        altered[:, 1:] += 10
        position_zero_only = torch.equal(model.pooler(probe), model.pooler(altered))
        if not position_zero_only:
            raise ValueError("The classifier pooler reads positions beyond 0")
        ids, padding = test_batch(tokenizer, device)
        report["model_config"] = {
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_labels": config.num_labels,
            "attention_implementation": config._attn_implementation,
            "position_biased_input": config.position_biased_input,
            "classifier_position_zero_only": position_zero_only,
            "all_ones_test_lengths": [60, 32],
            "probability_test_lengths": [128, 32],
            "probe_token_ids": {"base": 100, "position_45_replacement": 200,
                                "position_20_replacements": list(range(200, 1200, 100))},
        }
        run_check(checks, "dense_equals_all_ones", lambda: check_all_ones(model, ids, padding, args))
        full_ids, full_padding = test_batch(tokenizer, device, length=LENGTH)
        run_check(checks, "blocked_probabilities", lambda: check_blocked_probabilities(
            model, full_ids, full_padding, reference_paths, args))
        if "circulant" in reference_paths:
            run_check(checks, "circulant_60_tokens", lambda: check_circulant(
                model, ids, padding, reference_paths["circulant"], args))
        else:
            checks["circulant_60_tokens"] = {"passed": False, "error": "No pilot circulant mask found"}
    except Exception as exc:
        checks["model_setup"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()}
        print("model_setup: FAIL", flush=True)

    report["passed"] = len(checks) == 4 and all(check["passed"] for check in checks.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}; report: {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
