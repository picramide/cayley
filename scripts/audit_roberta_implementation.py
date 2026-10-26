#!/usr/bin/env python3
"""No-training RoBERTa-base attention audit for Appendix B.1.

Checks the saved pilot and follow-up masks, dense/all-ones equivalence, blocked attention
probabilities, and the 60-token circulant reachability prediction. A frozen
pilot JSONL is required so replaced mask files cannot silently pass the audit.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch
import transformers
from transformers import RobertaForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cayley import roberta_sparse_attention as sparse


TASKS = ("cola", "sst2", "stsb", "rte", "mrpc", "qqp", "mnli", "qnli")
MASKS = ("local", "hypercube", "circulant", "window_dilations", "random_circulant", "bigbird")
NEW_MASKS = ("hypercube_gray", "hypercube_hampath0", "hypercube_hampath1", "hypercube_hampath2")
LENGTH = 128
MASK_DIR = Path(__file__).resolve().parent.parent / "masks"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved RoBERTa-base classification model")
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR, help="Saved mask directory")
    parser.add_argument("--pilot-results", type=Path, required=True, help="Frozen pilot JSONL (one row per task/mask)")
    parser.add_argument("--report", type=Path, help="Where to write the Appendix B.1 JSON report")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 audit requires CUDA")
    if args.report is None:
        args.report = Path(__file__).resolve().parent.parent / "data" / f"implementation_audit_{args.precision}.json"
    return args


def load_saved_tensor(path):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    return tensor


def check_pilot_masks(args):
    rows = {}
    duplicates = []
    with args.pilot_results.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("task_name"), row.get("mask_name"))
            if key in rows:
                duplicates.append({"line": line_number, "task": key[0], "mask": key[1]})
            else:
                rows[key] = row

    failures = []
    reference = {}
    reference_paths = {}
    checked_files = 0
    for task in TASKS:
        if (task, "dense") not in rows:
            failures.append(f"Missing pilot result for {task}/dense")
        for mask_name in MASKS:
            row = rows.get((task, mask_name))
            path = args.mask_dir / f"{task}_{mask_name}.pt"
            if row is None:
                failures.append(f"Missing pilot result for {task}/{mask_name}")
                continue
            if not path.is_file():
                failures.append(f"Missing saved mask: {path}")
                continue
            recorded_path = row.get("mask_path")
            if not recorded_path or Path(recorded_path).name != path.name:
                failures.append(f"Pilot mask path differs for {task}/{mask_name}: {recorded_path}")
            recorded_hash = row.get("mask_sha256")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if not recorded_hash or actual_hash != recorded_hash:
                failures.append(f"Pilot file SHA-256 differs for {task}/{mask_name}: "
                                f"recorded={recorded_hash}, saved={actual_hash}")
            try:
                tensor = load_saved_tensor(path)
            except Exception as exc:
                failures.append(f"Cannot load {path}: {exc}")
                continue
            checked_files += 1
            expected_shape = (12, LENGTH, LENGTH) if mask_name == "bigbird" else (LENGTH, LENGTH)
            if tensor.dtype != torch.bool or tuple(tensor.shape) != expected_shape:
                failures.append(f"Wrong dtype/shape for {task}/{mask_name}: {tensor.dtype}, {tuple(tensor.shape)}")
                continue
            if mask_name not in reference:
                reference[mask_name] = tensor
                reference_paths[mask_name] = path
            elif not torch.equal(reference[mask_name], tensor):
                different = int((reference[mask_name] != tensor).count_nonzero().item())
                failures.append(f"{task}/{mask_name} differs from {reference_paths[mask_name]} "
                                f"at {different} tensor entries")

    if duplicates:
        failures.append("Pilot JSONL has duplicate task/mask rows; use a frozen pilot snapshot")
    return {
        "passed": not failures,
        "checked_mask_files": checked_files,
        "reference_paths": {name: str(path) for name, path in reference_paths.items()},
        "tensor_sha256": {name: hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
                          for name, tensor in reference.items()},
        "duplicates": duplicates,
        "failures": failures,
    }, reference_paths


def check_new_masks(args, pilot_hypercube_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "student_package" / "scripts"))
    from followup_masks import followups

    expected = followups()
    failures = []
    paths = {}
    tensor_hashes = {}
    file_hashes = {}
    patterns = {}
    if pilot_hypercube_path is None:
        failures.append("No pilot hypercube mask found")
    else:
        patterns["hypercube"] = load_saved_tensor(pilot_hypercube_path)

    for name in NEW_MASKS:
        path = args.mask_dir / f"{name}.pt"
        if not path.is_file():
            failures.append(f"Missing new mask: {path}")
            continue
        try:
            tensor = load_saved_tensor(path)
        except Exception as exc:
            failures.append(f"Cannot load {path}: {exc}")
            continue
        if tensor.dtype != torch.bool or tuple(tensor.shape) != (12, LENGTH, LENGTH):
            failures.append(f"Wrong dtype/shape for {name}: {tensor.dtype}, {tuple(tensor.shape)}")
            continue
        if not torch.equal(tensor, tensor[:1].expand_as(tensor)):
            failures.append(f"{name} differs across heads")
        pattern = tensor[0]
        if not pattern.diagonal().all() or not torch.equal(pattern, pattern.T):
            failures.append(f"{name} lacks self-loops or symmetry")
        if not bool((pattern.sum(-1) == 8).all()) or not bool(pattern.diagonal(offset=1).all()):
            failures.append(f"{name} is not a degree-8 adjacent-token placement")
        if not torch.equal(pattern, torch.from_numpy(expected[name])):
            failures.append(f"{name} differs from the supplied generator")
        for other, previous in patterns.items():
            if torch.equal(pattern, previous):
                failures.append(f"{name} duplicates {other}")
        patterns[name] = pattern
        paths[name] = path
        tensor_hashes[name] = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        file_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "passed": not failures and len(paths) == len(NEW_MASKS),
        "checked_mask_files": len(paths),
        "reference_paths": {name: str(path) for name, path in paths.items()},
        "tensor_sha256": tensor_hashes,
        "file_sha256": file_hashes,
        "failures": failures,
    }, paths


def test_batch(config, device, length=60):
    bos, eos, pad = config.bos_token_id, config.eos_token_id, config.pad_token_id
    assert all(token is not None for token in (bos, eos, pad))
    assert config.vocab_size > 1200 and all(token not in (100, 200) for token in (bos, eos, pad))
    ids = torch.full((2, length), pad, dtype=torch.long, device=device)
    ids[0, 0], ids[0, 1:length - 1], ids[0, length - 1] = bos, 100, eos
    ids[1, 0], ids[1, 1:31], ids[1, 31] = bos, 200, eos
    return ids, ids.ne(pad).long()


def autocast_for(args):
    if args.precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def max_abs_difference(left, right):
    difference = (left.detach().float() - right.detach().float()).abs()
    if not torch.isfinite(difference).all():
        return None
    return float(difference.max().item()) if difference.numel() else 0.0


def check_all_ones(model, ids, padding, args):
    labels = torch.tensor([0, 1], device=ids.device)
    model.zero_grad(set_to_none=True)
    sparse.GLOBAL_SPARSE_MASK = None
    with autocast_for(args):
        dense = model(input_ids=ids, attention_mask=padding, labels=labels)
    dense.loss.backward()
    dense_logits = dense.logits.detach().clone()
    dense_loss = dense.loss.detach().clone()
    dense_grads = {name: None if param.grad is None else param.grad.detach().cpu().clone()
                   for name, param in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    sparse.patch_roberta_attention()
    sparse.GLOBAL_SPARSE_MASK = torch.ones(1, 1, LENGTH, LENGTH, dtype=torch.bool)
    with autocast_for(args):
        full = model(input_ids=ids, attention_mask=padding, labels=labels)
    full.loss.backward()

    mismatched = []
    max_grad_difference = 0.0
    for name, param in model.named_parameters():
        expected = dense_grads[name]
        actual = None if param.grad is None else param.grad.detach().cpu()
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
    sparse.patch_roberta_attention()
    layers = model.roberta.encoder.layer
    results = {}
    cases = {"all_ones": None, **reference_paths}
    for mask_name, path in cases.items():
        sparse.GLOBAL_SPARSE_MASK = (torch.ones(1, 1, LENGTH, LENGTH, dtype=torch.bool)
                                     if path is None else sparse.load_mask(str(path)))
        captured = {}
        handles = []
        for layer_number, layer in enumerate(layers):
            def capture(_module, inputs, number=layer_number):
                captured[number] = inputs[0].detach()
            handles.append(layer.attention.self.dropout.register_forward_pre_hook(capture))
        try:
            with torch.no_grad(), autocast_for(args):
                model(input_ids=ids, attention_mask=padding)
        finally:
            for handle in handles:
                handle.remove()

        blocked_nonzero = 0
        nonfinite = 0
        max_row_sum_error = 0.0
        for layer_number, probabilities in captured.items():
            batch, heads, query_length, key_length = probabilities.shape
            allowed = sparse.get_sliced_mask(sparse.GLOBAL_SPARSE_MASK, heads,
                                              query_length, key_length, probabilities.device)
            allowed = allowed & padding[:, None, None, :key_length].bool()
            blocked = ~allowed.expand(batch, heads, query_length, key_length)
            blocked_nonzero += int(torch.count_nonzero(probabilities[blocked]).item())
            nonfinite += int((~torch.isfinite(probabilities)).sum().item())
            valid_rows = padding[:, None, :query_length].bool().expand(batch, heads, query_length)
            row_sums = probabilities.sum(-1)
            if valid_rows.any():
                error = (row_sums[valid_rows].float() - 1.0).abs().max().item()
                max_row_sum_error = max(max_row_sum_error, float(error))

        row_tolerance = 0.005 if args.precision == "fp16" else 0.00001
        results[mask_name] = {
            "passed": len(captured) == len(layers) and blocked_nonzero == 0
                      and nonfinite == 0 and max_row_sum_error <= row_tolerance,
            "layers_captured": len(captured),
            "blocked_probabilities_nonzero": blocked_nonzero,
            "nonfinite_probabilities": nonfinite,
            "max_valid_row_sum_error": max_row_sum_error,
        }
    missing = [name for name in MASKS + NEW_MASKS if name not in results]
    return {"passed": not missing and all(case["passed"] for case in results.values()),
            "cases": results, "missing_mask_types": missing}


def check_circulant(model, ids, padding, path, args):
    sparse.patch_roberta_attention()
    mask = load_saved_tensor(path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D circulant mask, found {tuple(mask.shape)}")
    adjacency = mask[:60, :60].bool()
    reachable = torch.zeros(60, dtype=torch.bool)
    reachable[0] = True
    for _ in range(12):
        reachable = adjacency[reachable].any(dim=0)

    sparse.GLOBAL_SPARSE_MASK = sparse.load_mask(str(path))
    one = ids[:1].clone()
    one_padding = padding[:1]
    with torch.no_grad(), autocast_for(args):
        original = model(input_ids=one, attention_mask=one_padding).logits.detach().clone()
        far = one.clone()
        far[0, 45] = 200
        far_logits = model(input_ids=far, attention_mask=one_padding).logits.detach().clone()
        far_exact = torch.equal(original, far_logits)
        finite = bool(torch.isfinite(original).all() and torch.isfinite(far_logits).all())
        near_difference = 0.0
        near_changes = False
        for replacement in (200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100):
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


def run_check(report, name, function):
    try:
        report[name] = function()
    except Exception as exc:
        report[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}
    print(f"{name}: {'PASS' if report[name]['passed'] else 'FAIL'}", flush=True)


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
    new_reference_paths = {}
    def mask_audit():
        result, paths = check_pilot_masks(args)
        reference_paths.update(paths)
        return result
    run_check(checks, "pilot_masks_and_cross_task_equality", mask_audit)
    def new_mask_audit():
        result, paths = check_new_masks(args, reference_paths.get("hypercube"))
        new_reference_paths.update(paths)
        return result
    run_check(checks, "new_masks", new_mask_audit)

    try:
        model = RobertaForSequenceClassification.from_pretrained(
            args.model, local_files_only=True, attn_implementation="eager"
        ).to(device).eval()
        if (model.config.num_hidden_layers != 12 or model.config.num_attention_heads != 12
                or model.config.hidden_size != 768 or model.config.intermediate_size != 3072):
            raise ValueError("The audit requires RoBERTa-base: 12 layers, 12 heads, 768 hidden, 3072 FFN")
        if model.config._attn_implementation != "eager":
            raise ValueError("The audit requires eager attention, as in the benchmark")
        if model.config.num_labels < 2:
            raise ValueError("The audit requires a classification head with at least two labels")
        ids, padding = test_batch(model.config, device)
        report["model_config"] = {
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_labels": model.config.num_labels,
            "attention_implementation": model.config._attn_implementation,
            "all_ones_test_lengths": [60, 32],
            "probability_test_lengths": [128, 32],
            "probe_token_ids": {"base": 100, "position_45_replacement": 200,
                                "position_20_replacements": list(range(200, 1200, 100))},
        }
        run_check(checks, "dense_equals_all_ones", lambda: check_all_ones(model, ids, padding, args))
        full_ids, full_padding = test_batch(model.config, device, length=LENGTH)
        run_check(checks, "blocked_probabilities", lambda: check_blocked_probabilities(
            model, full_ids, full_padding, {**reference_paths, **new_reference_paths}, args))
        if "circulant" in reference_paths:
            run_check(checks, "circulant_60_tokens", lambda: check_circulant(
                model, ids, padding, reference_paths["circulant"], args))
        else:
            checks["circulant_60_tokens"] = {"passed": False, "error": "No pilot circulant mask found"}
    except Exception as exc:
        checks["model_setup"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()}
        print("model_setup: FAIL", flush=True)
    finally:
        sparse.GLOBAL_SPARSE_MASK = None

    report["passed"] = len(checks) == 5 and all(check["passed"] for check in checks.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}; report: {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
