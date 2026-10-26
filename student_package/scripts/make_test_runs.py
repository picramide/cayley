"""FABRICATED run outputs for testing the analysis pipeline. Not results.

  python3 scripts/make_test_runs.py OUTDIR

Writes OUTDIR/results.jsonl (every line marked "test_data": true), per-run directories with
dev_predictions.jsonl and trainer_state.json, OUTDIR/objective_loss.jsonl and OUTDIR/diagnostics.json,
in exactly the format described in RUN_OUTPUTS.md. The analysis scripts refuse to write
results computed from these files into the paper directory.
"""
import json, os, sys
import numpy as np

DEV = {"cola": (1043, 0.69), "sst2": (872, 0.51), "mrpc": (408, 0.68), "qqp": (40430, 0.37),
       "stsb": (1500, None), "rte": (277, 0.47), "qnli": (5463, 0.49)}          # size, share of label 1
TRAIN_STEPS_PER_EPOCH = {"cola": 962, "sst2": 7577, "mrpc": 413, "qqp": 40933, "stsb": 647, "rte": 280, "qnli": 11784}
H = {"cola": 0.607, "sst2": 0.686, "mrpc": 0.631, "qqp": 0.659, "stsb": 2.3, "rte": 0.693, "qnli": 0.693}
ACC = {"cola": 0.83, "sst2": 0.93, "mrpc": 0.87, "qqp": 0.90, "stsb": 0.9, "rte": 0.72, "qnli": 0.91}
RUNTIME_H = {"cola": 0.11, "sst2": 0.76, "mrpc": 0.05, "qqp": 3.93, "stsb": 0.074, "rte": 0.036, "qnli": 1.10}


def make_run(out, rng, rows, *, model="roberta", experiment, task, mask, condition, seed, p_deg, acc_drop, precision="fp16", esc_mean=0.2):
    name = f"{model}_{experiment}_{task}_{mask}_{condition}_s{seed}_{precision}"
    d = os.path.join(out, "runs", name); os.makedirs(d, exist_ok=True)
    n, p1 = DEV[task]
    r = np.random.default_rng(sum(map(ord, task)))               # same examples, labels, lengths for every run
    y = r.uniform(0, 5, n).round(2) if p1 is None else (r.random(n) < p1).astype(int)
    L = np.clip(r.gamma(3, 9 if task not in ("qnli", "rte", "mrpc") else 17, n).astype(int) + 4, 4, 128)
    if model == "deberta":                                         # different tokenizer, different lengths
        L = np.clip((L * 0.95).astype(int), 4, 128)
    deg = rng.random() < p_deg and p1 is not None
    if p1 is None:                                                 # regression: noisy predictions
        noise = 0.6 + 6 * acc_drop + abs(rng.normal(0, 0.05))
        p = (y + rng.normal(0, noise, n)).round(3)
    elif deg:
        p = np.full(n, int(p1 > 0.5))
    else:
        q = ACC[task] - acc_drop + rng.normal(0, 0.01)
        p = np.where(rng.random(n) < q, y, 1 - y)
    cast = float if p1 is None else int
    with open(os.path.join(d, "dev_predictions.jsonl"), "w") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, "label": cast(y[i]), "pred": cast(p[i]), "length": int(L[i])}) + "\n")
    mult = 3 if condition == "S+" else 1
    steps = 5 * TRAIN_STEPS_PER_EPOCH[task] * mult; log_every = max(1, TRAIN_STEPS_PER_EPOCH[task] // 100)
    esc = None if deg else max(0.01, rng.exponential(esc_mean)) * 5 * TRAIN_STEPS_PER_EPOCH[task]
    hist = []
    for st in range(log_every, steps + 1, log_every):
        base = H[task] if (esc is None or st < esc) else H[task] * (0.5 + 0.5 * np.exp(-(st - esc) / steps * 8))
        hist.append({"step": st, "epoch": st / TRAIN_STEPS_PER_EPOCH[task], "loss": float(base + rng.normal(0, 0.03))})
    json.dump({"log_history": hist, "global_step": steps}, open(os.path.join(d, "trainer_state.json"), "w"))
    rows.append({"model": model, "experiment": experiment, "task": task, "mask": mask, "condition": condition, "seed": seed,
                 "precision": precision, "run_dir": os.path.relpath(d, out),
                 "train_runtime_s": RUNTIME_H[task] * mult * 3600 * (2 if precision == "fp32" else 1) * (1.3 if model == "deberta" else 1), "test_data": True})


def main(out):
    os.makedirs(out, exist_ok=True); rng = np.random.default_rng(0); rows = []
    for model in ["roberta", "deberta"]:
        kw = dict(model=model)
        # Experiment 1
        spec = {("cola", "hypercube"): {"P": 0.6, "S": 0.45, "S+": 0.2, "G": 0.3},
                ("qnli", "window_dilations"): {"P": 0.8, "S": 0.6, "S+": 0.2, "G": 0.4},
                ("rte", "local"): {"P": 0.3, "S": 0.2, "S+": 0.1, "G": 0.2}}
        if model == "deberta":                                        # fabricated: weaker failures
            spec = {k: {c: p / 3 for c, p in v.items()} for k, v in spec.items()}
        for (task, mask), probs in spec.items():
            seeds = range(1, 21) if task in ("cola", "rte") else range(1, 6)
            for c, pd in probs.items():
                for s in seeds:
                    make_run(out, rng, rows, **kw, experiment="exp1", task=task, mask=mask, condition=c, seed=s, p_deg=pd,
                             acc_drop=0.08, esc_mean={"P": 0.5, "S": 0.35, "S+": 0.5, "G": 0.25}[c])
                    if c != "G":
                        make_run(out, rng, rows, **kw, experiment="exp1", task=task, mask="dense", condition=c, seed=s, p_deg=0.05, acc_drop=0.0, esc_mean=0.05)
            if task == "qnli":
                for s in seeds:
                    make_run(out, rng, rows, **kw, experiment="exp1", task=task, mask="local", condition="P", seed=s, p_deg=0.05, acc_drop=0.1)
            make_run(out, rng, rows, **kw, experiment="exp1", task=task, mask=mask, condition="P", seed=1, p_deg=0.6, acc_drop=0.08, precision="fp32")
        # Experiment 2
        drop = {"hypercube": 0.08, "hypercube_gray": 0.03, "hypercube_hampath0": 0.035, "hypercube_hampath1": 0.04, "hypercube_hampath2": 0.03}
        plan = [("cola", 20), ("sst2", 5), ("mrpc", 10), ("qqp", 3), ("stsb", 10), ("rte", 10), ("qnli", 5)]
        for task, ns in plan:
            seeds = range(1, ns + 1)
            for m, dr in drop.items():
                if task == "cola" and m == "hypercube":
                    continue                                          # shared with Experiment 1
                if task == "qqp" and m not in ("hypercube", "hypercube_gray"):
                    continue                                          # QQP: binary versus Gray only
                for s_ in seeds:
                    make_run(out, rng, rows, **kw, experiment="exp2", task=task, mask=m, condition="P", seed=s_,
                             p_deg=0.3 if m == "hypercube" else 0.05, acc_drop=dr)
            if task in ("sst2", "mrpc", "qqp", "stsb"):                # dense not already run in Experiment 1
                for s_ in seeds:
                    make_run(out, rng, rows, **kw, experiment="exp2", task=task, mask="dense", condition="P", seed=s_, p_deg=0.0, acc_drop=0.0)
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # pretraining-objective loss (2-i): MLM for RoBERTa, RTD for DeBERTa
    base = {"roberta": {"dense": 1.80, "hypercube": 3.10, "hypercube_gray": 2.70, "hypercube_hampath0": 2.75, "hypercube_hampath1": 2.78, "hypercube_hampath2": 2.72},
            "deberta": {"dense": 0.20, "hypercube": 0.42, "hypercube_gray": 0.33, "hypercube_hampath0": 0.34, "hypercube_hampath1": 0.35, "hypercube_hampath2": 0.33}}
    with open(os.path.join(out, "objective_loss.jsonl"), "w") as f:
        for model, bm in base.items():
            for w in range(2000):
                nt = int(rng.integers(15, 22)) if model == "roberta" else 126
                hard = rng.normal(0, 0.1 * bm["dense"])
                for m, b in bm.items():
                    f.write(json.dumps({"model": model, "window": w, "mask": m, "n_tokens": nt,
                                        "loss_sum": float(nt * max(0.01, b + hard + rng.normal(0, 0.1 * b)))}) + "\n")
    D = {model: {m: {t: {"retained": rng.uniform(0.2, 0.9, 12).tolist(), "single": rng.uniform(0.05, 0.6, 12).tolist(),
                         "accumulated": float(rng.uniform(0.2, 1.0)), "readout": float(rng.uniform(0.1, 1.0))} for t in DEV}
                 for m in ["local", "window_dilations", "hypercube", "hypercube_gray", "hypercube_hampath0", "hypercube_hampath1", "hypercube_hampath2"]}
         for model in ["roberta", "deberta"]}
    json.dump(D, open(os.path.join(out, "diagnostics.json"), "w"))
    print(f"FABRICATED test data: {len(rows)} runs in {out}")


if __name__ == "__main__":
    main(sys.argv[1])
