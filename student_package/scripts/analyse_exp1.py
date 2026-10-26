"""Analysis of Experiment 1 (adaptation under a fixed final mask), for each encoder.

  python3 scripts/analyse_exp1.py RUNS/results.jsonl [--B 10000] [--out .]

Writes tables/exp1_main.tex, tables/exp1_tests.tex, tables/exp1_fp32.tex,
figures/exp1.pdf and data/exp1_summary.json (all numbers quoted in the text).
"""
import argparse, json, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import *

CONFIGS = [("cola", "hypercube"), ("qnli", "window_dilations"), ("rte", "local")]
CONFIG_TEX = {("cola", "hypercube"): "Hypercube (binary), CoLA", ("qnli", "window_dilations"): "Window + dilations, QNLI",
              ("rte", "local"): "Local ($r{=}16$), RTE"}
CONFIG_SHORT = {("cola", "hypercube"): "hypercube, CoLA", ("qnli", "window_dilations"): "window+dil., QNLI",
                ("rte", "local"): "local, RTE"}
MARGIN = {"cola": 0.01, "sst2": 1.0, "qnli": 1.0, "rte": 1.0}
DEG_MARGIN = 0.10


def sid(name):
    return zlib.crc32(name.encode()) % 2**31


def analyse_model(runs, model, B):
    R = select(runs, model=model)
    arm = lambda task, mask, cond: by_seed(select(R, task=task, mask=mask, condition=cond, precision="fp16"))
    deg_vector = lambda d, seeds: np.array([is_degenerate(d[s]["p"]) for s in seeds])
    steps_P = {}
    for task in {t for t, _ in CONFIGS}:
        lp = [escape_step(r["log"], LABEL_ENTROPY[task])[2] for r in select(R, task=task, condition="P") if r["log"]]
        steps_P[task] = max(lp) if lp else None

    def esc(r):
        st, obs, _ = escape_step(r["log"], LABEL_ENTROPY[r["task"]])
        return st / steps_P[r["task"]], obs

    cell, rows = {}, []
    for task, mask in CONFIGS:
        entries = [(mask, CONFIG_TEX[(task, mask)]), ("dense", f"Dense, {TASK_NAME[task]}")]
        if task == "qnli":
            entries.append(("local", "Local, QNLI (reference for O3)"))
        for m, label in entries:
            line = [label]
            for c in CONDITIONS:
                d = arm(task, m, c)
                if not d:
                    line.append("n/a" if (m == "dense" and c == "G") or (m == "local" and task == "qnli" and c != "P") else "--"); continue
                seeds = sorted(d); P, y = stack(d, seeds)
                sc = metric_batch(task, P, y); k = int(deg_vector(d, seeds).sum())
                x = dict(mean=float(sc.mean()), n=len(seeds), degenerate=k, degenerate_wilson=wilson(k, len(seeds)))
                if steps_P[task] and all(d[s]["log"] for s in seeds):
                    e = [esc(d[s]) for s in seeds]
                    tt = np.array([v[0] for v in e]); oo = np.array([v[1] for v in e])
                    xs, ys = kaplan_meier(tt, oo); below = xs[ys <= 0.5]
                    x["escape"] = dict(times=tt.tolist(), observed=oo.tolist())
                    x["median_escape"] = float(below[0]) if len(below) else None
                cell[(task, m, c)] = x
                line.append(f"{fmt_score(task, sc.mean())} ({k}/{len(seeds)})")
            rows.append(" & ".join(line) + r" \\")
    # ---------------- contrasts
    def contrast(task, specs, weights, name):
        ds = [arm(task, m, c) for m, c in specs]
        if not all(ds):
            return None
        seeds = sorted(set.intersection(*[set(d) for d in ds]))
        if len(seeds) < 2:
            return None
        arms = [stack(d, seeds)[0] for d in ds]; y = stack(ds[0], seeds)[1]
        est, lo, hi = bootstrap_contrast(task, arms, weights, y, B=B, seed=sid(model + name))
        return dict(est=est, lo=lo, hi=hi, n_seeds=len(seeds), classification=classify(lo, hi, MARGIN[task]))

    def deg_contrast(task, s1, s2):
        d1, d2 = arm(task, *s1), arm(task, *s2)
        if not d1 or not d2:
            return None
        seeds = sorted(set(d1) & set(d2)); v1, v2 = deg_vector(d1, seeds), deg_vector(d2, seeds)
        dd, lo, hi = newcombe_paired(v1, v2)
        return dict(est=dd, lo=lo, hi=hi, p_mcnemar=mcnemar_exact(v1, v2), n_seeds=len(seeds),
                    classification=classify(lo, hi, DEG_MARGIN))

    def esc_test(task, s1, s2):
        c1, c2 = cell.get((task, *s1)), cell.get((task, *s2))
        if not c1 or not c2 or "escape" not in c1 or "escape" not in c2:
            return None
        d1, d2 = arm(task, *s1), arm(task, *s2); seeds = sorted(set(d1) & set(d2)); g = []
        for grp, d in [(0, d1), (1, d2)]:
            for s in seeds:
                t, o = esc(d[s]); g.append((t, o, grp, s))
        return dict(p_logrank=logrank_stratified(g), n_seeds=len(seeds))

    tests = {}
    for task, mask in CONFIGS:
        T = {"deg_sparse_minus_dense_P": deg_contrast(task, (mask, "P"), ("dense", "P")),
             "score_sparse_minus_dense_P": contrast(task, [(mask, "P"), ("dense", "P")], [1, -1], f"sd{task}")}
        for c in ["S", "S+"]:
            T[f"score_{c}_minus_P_sparse"] = contrast(task, [(mask, c), (mask, "P")], [1, -1], f"sp{c}{task}")
            T[f"did_{c}_minus_P"] = contrast(task, [(mask, c), (mask, "P"), ("dense", c), ("dense", "P")], [1, -1, -1, 1], f"did{c}{task}")
            T[f"deg_{c}_minus_P_sparse"] = deg_contrast(task, (mask, c), (mask, "P"))
        T["score_G_minus_S_sparse"] = contrast(task, [(mask, "G"), (mask, "S")], [1, -1], f"gs{task}")
        T["deg_G_minus_S_sparse"] = deg_contrast(task, (mask, "G"), (mask, "S"))
        T["escape_S_vs_P"] = esc_test(task, (mask, "P"), (mask, "S"))
        T["escape_Splus_vs_S"] = esc_test(task, (mask, "S"), (mask, "S+"))
        T["escape_G_vs_S"] = esc_test(task, (mask, "S"), (mask, "G"))
        if task == "qnli":
            T["score_local_minus_window_P"] = contrast(task, [("local", "P"), (mask, "P")], [1, -1], "lw")
            T["deg_window_minus_local_P"] = deg_contrast(task, (mask, "P"), ("local", "P"))
        tests[f"{task}/{mask}"] = T
    fp32 = []
    for r in select(R, precision="fp32"):
        twin = select(R, task=r["task"], mask=r["mask"], condition=r["condition"], seed=r["seed"], precision="fp16")
        fp32.append(dict(task=r["task"], mask=r["mask"], condition=r["condition"], seed=r["seed"],
                         degenerate_fp16=is_degenerate(twin[0]["p"]) if twin else None, degenerate_fp32=is_degenerate(r["p"]),
                         score_fp16=float(metric_batch(r["task"], twin[0]["p"], twin[0]["y"])[0]) if twin else float("nan"),
                         score_fp32=float(metric_batch(r["task"], r["p"], r["y"])[0])))
    return dict(cell=cell, rows=rows, tests=tests, steps_P=steps_P, fp32=fp32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(); out = a.out
    runs = load_runs(a.results); guard_test_data(runs, out)
    models = [m for m in MODELS if select(runs, model=m, experiment="exp1")]
    res = {m: analyse_model(runs, m, a.B) for m in models}

    # ---------------- main table
    body = []
    for m in models:
        body.append(r"\multicolumn{5}{l}{\emph{" + MODEL_NAME[m] + r"}} \\")
        body += res[m]["rows"]; body.append(r"\midrule")
    write(os.path.join(out, "tables/exp1_main.tex"),
          r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n" r"Configuration & (P) & (S) & (S$^{+}$) & (G) \\" "\n"
          r"\midrule" "\n" + "\n".join(body[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- contrasts table: columns = model x configuration
    def ci(task, x, deg=False):
        if x is None:
            return "--"
        if deg:
            return f"{x['est']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]"
        f = (lambda v: f"{v:+.3f}") if task == "cola" else (lambda v: f"{v:+.2f}")
        return f"{f(x['est'])} [{f(x['lo'])}, {f(x['hi'])}]"
    spec = [("(1-i) Confirmation under (P)", None, None),
            (r"$\Delta$ P(deg.), sparse $-$ dense", "deg_sparse_minus_dense_P", "deg"),
            (r"$\Delta$ score, sparse $-$ dense", "score_sparse_minus_dense_P", "score"),
            (r"$\Delta$ score, local $-$ window + dil.", "score_local_minus_window_P", "score"),
            ("(1-ii) Protocol and budget, sparse model", None, None),
            (r"$\Delta$ score, (S) $-$ (P)", "score_S_minus_P_sparse", "score"),
            (r"$\Delta$ P(deg.), (S) $-$ (P)", "deg_S_minus_P_sparse", "deg"),
            (r"Diff.-in-diff., (S) $-$ (P)", "did_S_minus_P", "score"),
            (r"$\Delta$ score, (S$^{+}$) $-$ (P)", "score_S+_minus_P_sparse", "score"),
            (r"$\Delta$ P(deg.), (S$^{+}$) $-$ (P)", "deg_S+_minus_P_sparse", "deg"),
            (r"Diff.-in-diff., (S$^{+}$) $-$ (P)", "did_S+_minus_P", "score"),
            ("(1-iii) Gradual transition, sparse model", None, None),
            (r"$\Delta$ score, (G) $-$ (S)", "score_G_minus_S_sparse", "score"),
            (r"$\Delta$ P(deg.), (G) $-$ (S)", "deg_G_minus_S_sparse", "deg"),
            (r"Escape step, log-rank $p$", None, None),
            (r"(S) vs (P)", "escape_S_vs_P", "esc"), (r"(S$^{+}$) vs (S)", "escape_Splus_vs_S", "esc"),
            (r"(G) vs (S)", "escape_G_vs_S", "esc")]
    lines = []
    for m in models:
        lines.append(r"\multicolumn{4}{l}{\textbf{" + MODEL_NAME[m] + r"}} \\")
        for lab, key, kind in spec:
            if key is None:
                lines.append(r"\multicolumn{4}{l}{\emph{" + lab + r"}} \\"); continue
            cells = []
            for task, mask in CONFIGS:
                x = res[m]["tests"][f"{task}/{mask}"].get(key)
                cells.append(("--" if x is None else fmt_p(x["p_logrank"])) if kind == "esc" else ci(task, x, kind == "deg"))
            lines.append(r"\quad " + lab + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    write(os.path.join(out, "tables/exp1_tests.tex"),
          r"\begin{tabular}{lccc}" "\n" r"\toprule" "\n" r"Contrast & Hypercube, CoLA & Window + dil., QNLI & Local, RTE \\" "\n"
          r"\midrule" "\n" + "\n".join(lines[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- fp32 table
    yn = lambda v: "--" if v is None else ("yes" if v else "no")
    fl = [(m, x) for m in models for x in res[m]["fp32"]]
    write(os.path.join(out, "tables/exp1_fp32.tex"),
          r"\begin{tabular}{lllrcccc}" "\n" r"\toprule" "\n"
          r"Model & Task & Mask & Seed & Deg.\ (fp16) & Deg.\ (fp32) & Score (fp16) & Score (fp32) \\" "\n" r"\midrule" "\n"
          + "".join(f"{MODEL_NAME[m]} & {TASK_NAME[x['task']]} & {MASK_NAME.get(x['mask'], x['mask'])} & {x['seed']} & "
                    f"{yn(x['degenerate_fp16'])} & {yn(x['degenerate_fp32'])} & {fmt_score(x['task'], x['score_fp16'])} & "
                    f"{fmt_score(x['task'], x['score_fp32'])} \\\\\n" for m, x in fl)
          + r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- figure: one row per model
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    col = {"P": "#4d4d4d", "S": "#2a78d6", "S+": "#163f73", "G": "#d95f02"}
    fig, axes = plt.subplots(len(models), 4, figsize=(8.4, 2.5 * len(models)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.7, 1, 1, 1]))
    groups = [("cola", "hypercube", "Hyp.\nCoLA"), ("cola", "dense", "Dense\nCoLA"),
              ("qnli", "window_dilations", "W+d\nQNLI"), ("qnli", "dense", "Dense\nQNLI"),
              ("rte", "local", "Local\nRTE"), ("rte", "dense", "Dense\nRTE")]
    for r_, m in enumerate(models):
        ax = axes[r_]; cell = res[m]["cell"]
        for gi, (task, mm, lab) in enumerate(groups):
            for ci_, c in enumerate(CONDITIONS):
                x = cell.get((task, mm, c))
                if not x:
                    continue
                k, n = x["degenerate"], x["n"]; lo, hi = wilson(k, n); xp = gi + (ci_ - 1.5) * 0.18
                ax[0].errorbar(xp, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", ms=3.5, color=col[c], capsize=2, lw=1,
                               label=COND_TEX[c].replace("$^{+}$", "+") if gi == 0 else None)
        ax[0].set_xticks(range(len(groups)), [g[2] for g in groups], fontsize=6.5); ax[0].set_ylim(-0.03, 1.03)
        ax[0].set_ylabel(f"{MODEL_NAME[m]}\nprobability of degeneracy")
        ax[0].set_title(f"({'aeim'[r_ % 4]}) degeneracy", fontsize=8)
        if r_ == 0:
            ax[0].legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), handletextpad=0.2, columnspacing=0.8)
        for j, (task, mm) in enumerate(CONFIGS):
            A = ax[1 + j]
            for c in CONDITIONS + ["dense"]:
                x = cell.get((task, mm, c)) if c != "dense" else cell.get((task, "dense", "P"))
                if x and "escape" in x:
                    xs, ys = kaplan_meier(x["escape"]["times"], x["escape"]["observed"])
                    xs = np.append(xs, max(x["escape"]["times"])); ys = np.append(ys, ys[-1])
                    A.step(xs, 1 - ys, where="post", lw=1.2 if c != "dense" else 1, ls="-" if c != "dense" else "--",
                           color=col.get(c, "#999999"), label=(COND_TEX[c].replace("$^{+}$", "+") if c != "dense" else "dense (P)"))
            A.set_xlabel("training step / steps of (P)"); A.set_ylim(-0.03, 1.03)
            A.set_title(f"({'bcdfgh'[3 * r_ + j] if r_ < 2 else ''}) escape: {CONFIG_SHORT[(task, mm)]}", fontsize=8)
            if j == 0 and r_ == 0:
                A.set_ylabel("fraction escaped"); A.legend(frameon=False, fontsize=7)
    plt.tight_layout(pad=0.4); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    plt.savefig(os.path.join(out, "figures/exp1.pdf"), bbox_inches="tight"); plt.close(); print("wrote figures/exp1.pdf")

    S = {m: dict(cells={f"{t}/{mm}/{c}": v for (t, mm, c), v in res[m]["cell"].items()}, tests=res[m]["tests"],
                 steps_P=res[m]["steps_P"], fp32=res[m]["fp32"]) for m in models}
    write(os.path.join(out, "data/exp1_summary.json"), json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
