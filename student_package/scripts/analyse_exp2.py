"""Analysis of Experiment 2 (placements of one abstract graph) and of the perturbation diagnostics,
for each encoder.

  python3 scripts/analyse_exp2.py RUNS/results.jsonl RUNS/objective_loss.jsonl [RUNS/diagnostics.json] [--B 10000] [--out .]

Writes tables/exp2_objective.tex, tables/exp2_scores.tex, tables/exp2_tests.tex, tables/diag.tex,
figures/exp2.pdf and data/exp2_summary.json.
"""
import argparse, json, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import *

OBJ_MARGIN = 0.01
MARGIN = {"cola": 0.01, "sst2": 1.0, "mrpc": 1.0, "qqp": 1.0, "stsb": 0.01, "rte": 1.0, "qnli": 1.0}
ORDER = ["dense"] + PLACEMENTS


def sid(name):
    return zlib.crc32(name.encode()) % 2**31


# ----------------------------------------------------------------------------- (2-i) pretraining objective
def objective_analysis(rows, model, B):
    rows = [r for r in rows if r.get("model", "roberta") == model]
    if not rows:
        return None
    masks = sorted({r["mask"] for r in rows}, key=lambda m: ORDER.index(m) if m in ORDER else 99)
    wins = sorted({r["window"] for r in rows}); wi = {w: i for i, w in enumerate(wins)}
    L = {m: np.full(len(wins), np.nan) for m in masks}; N = {m: np.full(len(wins), np.nan) for m in masks}
    for r in rows:
        L[r["mask"]][wi[r["window"]]] = r["loss_sum"]; N[r["mask"]][wi[r["window"]]] = r.get("n_tokens", r.get("n_masked"))
    for m in masks:
        if np.isnan(L[m]).any():
            raise ValueError(f"{model}: mask {m} lacks some text windows")
        if not np.array_equal(N[m], N[masks[0]]):
            raise ValueError(f"{model}: the scored positions must be identical across masks")
    n = N[masks[0]]
    rng = np.random.default_rng(sid(model + "obj")); idx = rng.integers(0, len(wins), (B, len(wins)))
    tot = n[idx].sum(1); boot = {m: L[m][idx].sum(1) / tot for m in masks}
    est = {m: float(L[m].sum() / n.sum()) for m in masks}
    res = {"objective": OBJECTIVE[model], "n_windows": len(wins), "n_tokens": int(n.sum()), "loss": {},
           "diff_vs_dense": {}, "diff_vs_binary": {}}
    for m in masks:
        res["loss"][m] = (est[m], *np.percentile(boot[m], [2.5, 97.5]).tolist())
        for ref, key in [("dense", "diff_vs_dense"), ("hypercube", "diff_vs_binary")]:
            if ref in masks and m != ref:
                d = boot[m] - boot[ref]; lo, hi = np.percentile(d, [2.5, 97.5])
                res[key][m] = dict(est=est[m] - est[ref], lo=float(lo), hi=float(hi), classification=classify(lo, hi, OBJ_MARGIN))
    adj = [m for m in ADJ_PLACEMENTS if m in masks]
    if len(adj) >= 2:
        X = np.stack([L[m] / n for m in adj], 1)
        stat = lambda Z: float((((Z * n[:, None]).sum(0) / n.sum() - ((Z * n[:, None]).sum(0) / n.sum()).mean()) ** 2).sum())
        obs = stat(X); prng = np.random.default_rng(sid(model + "perm")); cnt = 0; NP = 10000
        for _ in range(NP):
            cnt += stat(np.take_along_axis(X, np.argsort(prng.random(X.shape), 1), 1)) >= obs - 1e-12
        res["omnibus_adjacent_p"] = (cnt + 1) / (NP + 1)
    res["masks"] = masks
    return res


# ----------------------------------------------------------------------------- (2-ii) fine-tuning
def ft_analysis(runs, model, B):
    res = {"cells": {}, "gray_minus_binary": {}, "pooled": {}}
    pr = select(runs, model=model, condition="P", precision="fp16", mask=PLACEMENTS + ["dense"])
    std = {}
    for task in TASKS:
        for m in PLACEMENTS + ["dense"]:
            d = by_seed(select(pr, task=task, mask=m))
            if not d:
                continue
            seeds = sorted(d); P, y = stack(d, seeds); sc = metric_batch(task, P, y)
            est, lo, hi = bootstrap_contrast(task, [P], [1], y, B=B, seed=sid(model + task + m))
            res["cells"][f"{task}/{m}"] = dict(mean=est, lo=lo, hi=hi, n=len(seeds),
                                               degenerate=int(sum(is_degenerate(d[s]["p"]) for s in seeds)),
                                               baseline=CONSTANT_BASELINE[task],
                                               scores=sc.tolist(), seeds=seeds)
        v = [np.var(res["cells"][f"{task}/{m}"]["scores"], ddof=1) for m in PLACEMENTS
             if f"{task}/{m}" in res["cells"] and len(res["cells"][f"{task}/{m}"]["scores"]) > 1]
        std[task] = float(np.sqrt(np.mean(v))) if v and np.mean(v) > 0 else None
    res["within_sd"] = std
    pv, done, gb = [], [], []
    for task in TASKS:
        dg, db = by_seed(select(pr, task=task, mask="hypercube_gray")), by_seed(select(pr, task=task, mask="hypercube"))
        seeds = sorted(set(dg) & set(db))
        if len(seeds) < 2:
            continue
        Pg, y = stack(dg, seeds); Pb, _ = stack(db, seeds)
        est, lo, hi, p = bootstrap_contrast(task, [Pg, Pb], [1, -1], y, B=B, seed=sid(model + "gb" + task), return_p=True)
        dif = metric_batch(task, Pg, y) - metric_batch(task, Pb, y)
        res["gray_minus_binary"][task] = dict(est=est, lo=lo, hi=hi, p=p, n_seeds=len(seeds), classification=classify(lo, hi, MARGIN[task]))
        pv.append(p); done.append(task)
        if std.get(task):
            gb.append(dif / std[task])
    for t, padj in zip(done, holm(np.array(pv)) if pv else []):
        res["gray_minus_binary"][t]["p_holm"] = float(padj)
    blocks = []
    for task in TASKS:
        if not std.get(task):
            continue
        ds = {m: by_seed(select(pr, task=task, mask=m)) for m in ADJ_PLACEMENTS}
        common = sorted(set.intersection(*[set(d) for d in ds.values()])) if all(ds.values()) else []
        for s in common:
            blocks.append((task, np.array([metric_batch(task, ds[m][s]["p"], ds[m][s]["y"])[0] / std[task] for m in ADJ_PLACEMENTS])))
    if blocks:
        X = np.stack([b[1] for b in blocks]); tk = np.array([b[0] for b in blocks])
        for t in TASKS:
            if (tk == t).any():
                X[tk == t] -= X[tk == t].mean()
        stat = lambda Z: float(((Z.mean(0) - Z.mean()) ** 2).sum())
        obs = stat(X); prng = np.random.default_rng(sid(model + "omni")); NP = 10000
        cnt = sum(stat(np.take_along_axis(X, np.argsort(prng.random(X.shape), 1), 1)) >= obs - 1e-12 for _ in range(NP))
        res["pooled"]["omnibus_adjacent_p"] = (cnt + 1) / (NP + 1); res["pooled"]["n_blocks"] = len(blocks)
    if gb:
        rng = np.random.default_rng(sid(model + "pooled-gb"))
        bs = [np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in gb]) for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res["pooled"]["gray_minus_binary_std"] = dict(est=float(np.mean([g.mean() for g in gb])), lo=float(lo), hi=float(hi))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("objective"); ap.add_argument("diagnostics", nargs="?")
    ap.add_argument("--B", type=int, default=10000); ap.add_argument("--out", default=".")
    a = ap.parse_args(); out = a.out
    runs = load_runs(a.results, need_logs=False); guard_test_data(runs, out)
    obj_rows = [json.loads(l) for l in open(a.objective) if l.strip()]
    models = [m for m in MODELS if select(runs, model=m) or any(r.get("model", "roberta") == m for r in obj_rows)]
    S = {m: {"objective": objective_analysis(obj_rows, m, a.B), "finetune": ft_analysis(runs, m, a.B)} for m in models}

    d3 = lambda x: "--" if x is None else f"{x['est']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
    # ---------------- (2-i) table: rows = masks, column blocks = models
    cols = []
    for m in models:
        o = S[m]["objective"]; cols.append({} if o is None else o)
    lines = []
    for mk in ORDER:
        cells = []
        for m, o in zip(models, cols):
            if not o or mk not in o["loss"]:
                cells += ["--", "--"]; continue
            e, lo, hi = o["loss"][mk]
            cells += [f"{e:.3f} [{lo:.3f}, {hi:.3f}]", d3(o["diff_vs_binary"].get(mk)) if mk not in ("dense", "hypercube") else "--"]
        lines.append(f"{MASK_NAME[mk]} & " + " & ".join(cells) + r" \\")
    foot = [r"\midrule", "Equality test, $p$ & " + " & ".join(
        (r"\multicolumn{2}{c}{" + (fmt_p(o["omnibus_adjacent_p"]) if o and "omnibus_adjacent_p" in o else "--") + "}") for o in cols) + r" \\"]
    hdr1 = " & ".join(r"\multicolumn{2}{c}{" + MODEL_NAME[m] + f" ({OBJECTIVE[m]} loss)" + "}" for m in models)
    hdr2 = " & ".join(r"loss (nats) & $\Delta$ vs binary" for _ in models)
    rules = " ".join(r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}" for i in range(len(models)))
    write(os.path.join(out, "tables/exp2_objective.tex"),
          r"\begin{tabular}{l" + "cc" * len(models) + "}\n" r"\toprule" "\n & " + hdr1 + r" \\" + "\n" + rules + "\n"
          + "Attention & " + hdr2 + r" \\" "\n" r"\midrule" "\n" + "\n".join(lines + foot) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- scores table: rows = tasks, columns = placements, blocks = models
    lines = []
    for m in models:
        lines.append(r"\multicolumn{7}{l}{\emph{" + MODEL_NAME[m] + r"}} \\")
        ft = S[m]["finetune"]
        for task in TASKS:
            cells = []
            for mk in PLACEMENTS + ["dense"]:
                c = ft["cells"].get(f"{task}/{mk}")
                cells.append("--" if not c else (f"{fmt_score(task, c['mean'])}" + (f" ({c['degenerate']}/{c['n']})" if task not in REGRESSION else f" ({c['n']})")))
            lines.append(f"{TASK_NAME[task]} ({METRIC_NAME[task]}) & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    write(os.path.join(out, "tables/exp2_scores.tex"),
          r"\begin{tabular}{lcccccc}" "\n" r"\toprule" "\n" r"Task & Binary & Gray & Ham.\ path 0 & Ham.\ path 1 & Ham.\ path 2 & Dense \\" "\n"
          r"\midrule" "\n" + "\n".join(lines[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- tests table: rows = contrasts, column blocks = models
    def gbcell(task, g):
        if not g:
            return ["--", "--"]
        return [f"{fmt_signed(task, g['est'])} [{fmt_signed(task, g['lo'])}, {fmt_signed(task, g['hi'])}]", fmt_p(g.get("p_holm", g["p"]))]
    lines = []
    for task in TASKS:
        cells = []
        for m in models:
            cells += gbcell(task, S[m]["finetune"]["gray_minus_binary"].get(task))
        lines.append(f"Gray $-$ binary, {TASK_NAME[task]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for m in models:
        pg = S[m]["finetune"]["pooled"].get("gray_minus_binary_std")
        cells += ["--" if not pg else f"{pg['est']:+.2f} [{pg['lo']:+.2f}, {pg['hi']:+.2f}]", "--"]
    lines.append("Gray $-$ binary, pooled (standardised) & " + " & ".join(cells) + r" \\")
    cells = []
    for m in models:
        po = S[m]["finetune"]["pooled"].get("omnibus_adjacent_p")
        cells += ["--", "--" if po is None else fmt_p(po)]
    lines.append("Equality of Gray and Ham.\\ paths 0--2, pooled & " + " & ".join(cells) + r" \\")
    hdr1 = " & ".join(r"\multicolumn{2}{c}{" + MODEL_NAME[m] + "}" for m in models)
    rules = " ".join(r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}" for i in range(len(models)))
    write(os.path.join(out, "tables/exp2_tests.tex"),
          r"\begin{tabular}{l" + "cc" * len(models) + "}\n" r"\toprule" "\n & " + hdr1 + r" \\" + "\n" + rules + "\n"
          + "Contrast & " + " & ".join(r"estimate [95\% interval] & $p$ (Holm)" for _ in models) + r" \\" "\n" r"\midrule" "\n"
          + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- diagnostics
    if a.diagnostics:
        D = json.load(open(a.diagnostics))
        if D and not set(D) <= set(MODELS):          # old format without the model level
            D = {"roberta": D}
        lines = []
        for m in [m for m in MODELS if m in D]:
            lines.append(r"\multicolumn{5}{l}{\emph{" + MODEL_NAME[m] + r"}} \\")
            for mk in [k for k in ["local", "window_dilations"] + PLACEMENTS if k in D[m]]:
                tv = [D[m][mk][t] for t in TASKS if t in D[m][mk]]
                vals = [np.mean([np.mean(x["retained"]) for x in tv]), np.mean([np.mean(x["single"]) for x in tv]),
                        np.mean([x["accumulated"] for x in tv]), np.mean([x["readout"] for x in tv])]
                lines.append(f"{MASK_NAME[mk]} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
            lines.append(r"\midrule")
        for m in D:
            S.setdefault(m, {})["diagnostics"] = D[m]
        write(os.path.join(out, "tables/diag.tex"),
              r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n" r"Mask & Retained mass & \multicolumn{3}{c}{Relative output perturbation} \\" "\n"
              r" & & single layer & accumulated & readout \\" "\n" r"\midrule" "\n" + "\n".join(lines[:-1]) + "\n"
              r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- figure: one row per model; (left) objective loss, (right) headroom per task and placement
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    short = {"dense": "dense", "hypercube": "binary", "hypercube_gray": "Gray", "hypercube_hampath0": "H0",
             "hypercube_hampath1": "H1", "hypercube_hampath2": "H2"}
    pcol = {"hypercube": "#d95f02", "hypercube_gray": "#2a78d6", "hypercube_hampath0": "#5aa0e6",
            "hypercube_hampath1": "#163f73", "hypercube_hampath2": "#7f7f7f"}
    fig, axes = plt.subplots(len(models), 2, figsize=(7.2, 2.7 * len(models)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 2.6]))
    letters = iter("abcdefgh")
    for r_, m in enumerate(models):
        ax = axes[r_]; o = S[m]["objective"]; ft = S[m]["finetune"]
        if o:
            pl = [k for k in o["masks"] if k != "dense"]
            for i, k in enumerate(pl):
                e, lo, hi = o["loss"][k]
                ax[0].errorbar(i, e, yerr=[[e - lo], [hi - e]], fmt="o", ms=3.5, capsize=2, color=pcol.get(k, "#2a78d6"))
            if "dense" in o["loss"]:
                ax[0].axhline(o["loss"]["dense"][0], color="#999999", ls="--", lw=1, label="dense"); ax[0].legend(frameon=False, fontsize=7)
            ax[0].set_xticks(range(len(pl)), [short[k] for k in pl])
        ax[0].set_ylabel(f"{MODEL_NAME[m]}\n{OBJECTIVE[m]} loss (nats)")
        ax[0].set_title(f"({next(letters)}) pretrained model, no training", fontsize=8)
        A = ax[1]
        for ti, task in enumerate(TASKS):
            cd = ft["cells"].get(f"{task}/dense")
            if not cd:
                continue
            b = CONSTANT_BASELINE[task]; den = cd["mean"] - b
            if den <= 0:
                continue
            for pi, k in enumerate(PLACEMENTS):
                c = ft["cells"].get(f"{task}/{k}")
                if not c:
                    continue
                x = ti + (pi - 2) * 0.14
                hr, lo, hi = (c["mean"] - b) / den, (c["lo"] - b) / den, (c["hi"] - b) / den
                A.errorbar(x, hr, yerr=[[hr - lo], [hi - hr]], fmt="o", ms=3, capsize=1.5, lw=0.9, color=pcol[k],
                           label=short[k] if ti == 0 else None)
                if task not in REGRESSION and c["degenerate"]:
                    A.text(x, 1.0, str(c["degenerate"]), fontsize=5.5, ha="center", va="bottom", color="#c0392b",
                           transform=A.get_xaxis_transform())
        A.axhline(1, color="#999999", ls="--", lw=1); A.axhline(0, color="#cccccc", lw=0.8)
        A.set_xticks(range(len(TASKS)), [TASK_NAME[t] for t in TASKS]); A.set_ylabel("headroom recovered")
        A.set_title(f"({next(letters)}) fine-tuning: headroom recovered (red: degenerate seeds)", fontsize=8, pad=9)
        if r_ == 0:
            A.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout(pad=0.4); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    plt.savefig(os.path.join(out, "figures/exp2.pdf"), bbox_inches="tight"); plt.close(); print("wrote figures/exp2.pdf")
    write(os.path.join(out, "data/exp2_summary.json"), json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
