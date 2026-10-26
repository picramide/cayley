"""Shared loading and statistics for the analysis of Experiments 1 and 2.

Run outputs (produced on the training machine; see RUN_OUTPUTS.md):
  results.jsonl       one line per fine-tuning run:
      {"model": "roberta" | "deberta", "experiment": "exp1" | "exp2", "task": "cola" | "sst2" | "qnli",
       "mask": "dense" | "local" | "window_dilations" | "hypercube" | "hypercube_gray"
               | "hypercube_hampath0" | "hypercube_hampath1" | "hypercube_hampath2",
       "condition": "P" | "S" | "S+" | "G", "seed": int, "precision": "fp16" | "fp32",
       "run_dir": path (relative to results.jsonl or absolute),
       "train_runtime_s": float}
  <run_dir>/dev_predictions.jsonl   one line per GLUE development example:
      {"idx": int, "label": int, "pred": int, "length": int}   (length = non-padded tokens;
       for STS-B, label and pred are floats)
  <run_dir>/trainer_state.json      as written by the Hugging Face Trainer
  objective_loss.jsonl  Experiment 2 (2-i), one line per (model, text window, mask):
      {"model": str, "window": int, "mask": str, "loss_sum": float, "n_tokens": int}
      RoBERTa: masked-language-model loss summed over the masked tokens (n_tokens = masked tokens);
      DeBERTa-v3: replaced-token-detection loss summed over all scored tokens (n_tokens = scored tokens).
  diagnostics.json    {model: {mask: {task: {"retained": [12 floats], "single": [12 floats],
                                              "accumulated": float, "readout": float}}}}
"""
import json, math, os
import numpy as np

MODELS = ["roberta", "deberta"]
MODEL_NAME = {"roberta": "RoBERTa-base", "deberta": "DeBERTa-v3-base"}
OBJECTIVE = {"roberta": "MLM", "deberta": "RTD"}
TASKS = ["cola", "sst2", "mrpc", "qqp", "stsb", "rte", "qnli"]
TASK_NAME = {"cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC", "qqp": "QQP", "stsb": "STS-B", "rte": "RTE", "qnli": "QNLI"}
METRIC_NAME = {"cola": "MCC", "sst2": "accuracy", "mrpc": "F1", "qqp": "F1", "stsb": "Pearson", "rte": "accuracy", "qnli": "accuracy"}
REGRESSION = {"stsb"}
# entropy (nats) of the training-label distribution of the GLUE training sets (classification tasks)
LABEL_ENTROPY = {"cola": 0.607, "sst2": 0.686, "mrpc": 0.631, "qqp": 0.659, "rte": 0.693, "qnli": 0.693}
# score of the best constant predictor on the development set (Appendix A)
CONSTANT_BASELINE = {"cola": 0.0, "sst2": 50.92, "mrpc": 81.22, "qqp": 53.82, "stsb": 0.0, "rte": 52.71, "qnli": 50.54}
MASK_NAME = {"dense": "Dense", "local": r"Local ($r{=}16$)", "window_dilations": "Window + dilations",
             "hypercube": "Binary", "hypercube_gray": "Gray", "hypercube_hampath0": "Ham.\\ path 0",
             "hypercube_hampath1": "Ham.\\ path 1", "hypercube_hampath2": "Ham.\\ path 2"}
PLACEMENTS = ["hypercube", "hypercube_gray", "hypercube_hampath0", "hypercube_hampath1", "hypercube_hampath2"]
ADJ_PLACEMENTS = PLACEMENTS[1:]
CONDITIONS = ["P", "S", "S+", "G"]
COND_TEX = {"P": "(P)", "S": "(S)", "S+": "(S$^{+}$)", "G": "(G)"}
LENGTH_STRATA = [(1, 17), (18, 37), (38, 64), (65, 128)]


# ----------------------------------------------------------------------------- loading
def load_runs(results_path, need_logs=True):
    base = os.path.dirname(os.path.abspath(results_path))
    runs = []
    for line in open(results_path):
        if not line.strip():
            continue
        r = json.loads(line)
        d = r["run_dir"] if os.path.isabs(r["run_dir"]) else os.path.join(base, r["run_dir"])
        rows = sorted((json.loads(l) for l in open(os.path.join(d, "dev_predictions.jsonl")) if l.strip()),
                      key=lambda x: x["idx"])
        r["idx"] = np.array([x["idx"] for x in rows])
        cast = float if r.get("task") == "stsb" else int
        r["y"] = np.array([cast(x["label"]) for x in rows])
        r["p"] = np.array([cast(x["pred"]) for x in rows])
        r["length"] = np.array([x.get("length", -1) for x in rows])
        r.setdefault("precision", "fp16"); r.setdefault("model", "roberta")
        ts = os.path.join(d, "trainer_state.json")
        r["log"] = json.load(open(ts))["log_history"] if (need_logs and os.path.exists(ts)) else None
        runs.append(r)
    return runs


def guard_test_data(runs, out):
    """Refuse to write results computed from fabricated test runs into the paper directory."""
    if any(r.get("test_data") for r in runs) and os.path.exists(os.path.join(out, "main.tex")):
        raise SystemExit("results.jsonl contains test data; use --out with a scratch directory")


def select(runs, **kw):
    out = []
    for r in runs:
        if all((r.get(k) == v) if not isinstance(v, (list, tuple, set)) else (r.get(k) in v) for k, v in kw.items()):
            out.append(r)
    return out


def by_seed(runs):
    d = {}
    for r in runs:
        if r["seed"] in d:
            raise ValueError(f"duplicate seed {r['seed']} for {r.get('model')}/{r['task']}/{r['mask']}/{r['condition']}")
        d[r["seed"]] = r
    return d


def stack(runs_by_seed, seeds):
    """Predictions as [n_seeds, n_examples], checking that all runs share the same examples."""
    ref = runs_by_seed[seeds[0]]
    for s in seeds:
        if not np.array_equal(runs_by_seed[s]["idx"], ref["idx"]) or not np.array_equal(runs_by_seed[s]["y"], ref["y"]):
            raise ValueError("runs do not share the same development examples and labels")
    return np.stack([runs_by_seed[s]["p"] for s in seeds]), ref["y"]


# ----------------------------------------------------------------------------- metrics
def metric_batch(task, P, y):
    """Primary metric for each row of P (predictions [k, n]) against labels y [n]:
    MCC (CoLA), F1 of the positive class in % (MRPC, QQP), Pearson correlation (STS-B), accuracy in % (others)."""
    P = np.atleast_2d(P)
    if task == "stsb":
        Pc = P - P.mean(1, keepdims=True); yc = y - y.mean()
        den = np.sqrt((Pc ** 2).sum(1) * (yc ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (Pc * yc).sum(1) / den
        return np.where(den > 0, r, 0.0)          # constant predictions: reported as 0
    tp = ((P == 1) & (y == 1)).sum(1).astype(float); tn = ((P == 0) & (y == 0)).sum(1).astype(float)
    fp = ((P == 1) & (y == 0)).sum(1).astype(float); fn = ((P == 0) & (y == 1)).sum(1).astype(float)
    if task == "cola":
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (tp * tn - fp * fn) / den
        return np.where(den > 0, m, 0.0)
    if task in ("mrpc", "qqp"):
        den = 2 * tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, 100.0 * 2 * tp / den, 0.0)
    return 100.0 * (P == y).mean(1)


def is_degenerate(p):
    """A single predicted class (classification) or constant predictions (regression)."""
    return len(np.unique(np.round(np.asarray(p, float), 6))) == 1


# ----------------------------------------------------------------------------- intervals and tests
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_paired(a, b, z=1.96):
    """Score interval (Newcombe 1998, method 10) for p_a - p_b with paired binary outcomes."""
    a = np.asarray(a, bool); b = np.asarray(b, bool); n = len(a)
    A = int((a & b).sum()); B = int((a & ~b).sum()); C = int((~a & b).sum()); D = int((~a & ~b).sum())
    p1, p2 = (A + B) / n, (A + C) / n
    l1, u1 = wilson(A + B, n, z); l2, u2 = wilson(A + C, n, z)
    denom = (A + B) * (C + D) * (A + C) * (B + D)
    if denom == 0:
        phi = 0.0
    else:
        num = A * D - B * C
        if num > 0:                              # continuity term of Newcombe's method 10
            num = max(num - n / 2, 0.0)
        phi = num / math.sqrt(denom)
    d = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return d, max(-1.0, d - dl), min(1.0, d + du)


def mcnemar_exact(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_contrast(task, arms, weights, y, B=10000, seed=0, return_p=False):
    """Joint bootstrap over seeds (paired across arms) and examples.

    arms: list of prediction arrays [n_seeds, n_examples] with rows aligned by seed;
    weights: contrast coefficients, e.g. [1, -1] or [1, -1, -1, 1].
    Returns (estimate, low, high), or (estimate, low, high, p) with the two-sided bootstrap
    p-value 2 min(P(T<=0), P(T>=0)) if return_p is set."""
    rng = np.random.default_rng(seed)
    k, n = arms[0].shape
    est = sum(w * metric_batch(task, A, y).mean() for w, A in zip(weights, arms))
    out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, k, k); e = rng.integers(0, n, n)
        out[b] = sum(w * metric_batch(task, A[s][:, e], y[e]).mean() for w, A in zip(weights, arms))
    lo, hi = np.percentile(out, [2.5, 97.5])
    if return_p:
        p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
        return float(est), float(lo), float(hi), max(p, 1.0 / B)
    return float(est), float(lo), float(hi)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i])); adj[i] = running
    return adj


def classify(lo, hi, margin, direction=+1):
    """'supported' / 'practically absent' / 'inconclusive' (Appendix: statistical analysis plan)."""
    if direction > 0 and lo > 0 or direction < 0 and hi < 0:
        return "supported"
    if lo > -margin and hi < margin:
        return "practically absent"
    return "inconclusive"


# ----------------------------------------------------------------------------- escape step
def escape_step(log, H, epochs_window=0.05, delta=0.05):
    """First optimisation step at which the rolling-median training loss falls below H - delta.
    Returns (step, observed, last_step); observed=False means right-censored at last_step."""
    tr = [(e["step"], e["epoch"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
    if not tr:
        raise ValueError("no training-loss entries in log_history")
    steps = np.array([t[0] for t in tr]); ep = np.array([t[1] for t in tr]); loss = np.array([t[2] for t in tr])
    interval = np.median(np.diff(ep)) if len(ep) > 1 else epochs_window
    w = max(1, int(round(epochs_window / max(interval, 1e-9))))
    med = np.array([np.median(loss[max(0, i - w + 1):i + 1]) for i in range(len(loss))])
    hit = np.flatnonzero((med < H - delta) & (np.arange(len(med)) >= w - 1))
    if len(hit):
        return int(steps[hit[0]]), True, int(steps[-1])
    return int(steps[-1]), False, int(steps[-1])


def kaplan_meier(times, observed):
    t = np.asarray(times, float); o = np.asarray(observed, bool)
    xs, ys = [0.0], [1.0]; s = 1.0
    for u in np.unique(t[o]):
        at_risk = (t >= u).sum(); d = ((t == u) & o).sum()
        s *= 1 - d / at_risk; xs.append(u); ys.append(s)
    return np.array(xs), np.array(ys)


def logrank_stratified(groups):
    """groups: list of (time, observed, group in {0,1}, stratum). Two-sided p-value (chi-square, 1 df)."""
    from collections import defaultdict
    strata = defaultdict(list)
    for t, o, g, s in groups:
        strata[s].append((t, o, g))
    O_E, V = 0.0, 0.0
    for items in strata.values():
        t = np.array([x[0] for x in items], float); o = np.array([x[1] for x in items], bool)
        g = np.array([x[2] for x in items], int)
        for u in np.unique(t[o]):
            risk = t >= u; n = risk.sum(); n1 = (risk & (g == 1)).sum()
            d = ((t == u) & o).sum(); d1 = ((t == u) & o & (g == 1)).sum()
            O_E += d1 - d * n1 / n
            if n > 1:
                V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V <= 0:
        return 1.0
    chi2 = O_E ** 2 / V
    return float(math.erfc(math.sqrt(chi2 / 2)))


# ----------------------------------------------------------------------------- formatting
def fmt_score(task, x):
    return f"{x:.3f}" if task in ("cola", "stsb") else f"{x:.2f}"


def fmt_signed(task, x):
    return f"{x:+.3f}" if task in ("cola", "stsb") else f"{x:+.2f}"


def fmt_ci(task, est, lo, hi):
    return f"{fmt_score(task, est)} [{fmt_score(task, lo)}, {fmt_score(task, hi)}]"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(text)
    print("wrote", path)
