"""Length-stratified scores and measured compute of the controlled study, for each encoder.

  python3 scripts/analyse_extra.py RUNS/results.jsonl [--out .]

Writes tables/length_strata.tex and tables/compute_controlled.tex.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import *


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    runs = load_runs(a.results, need_logs=False)
    guard_test_data(runs, a.out)
    models = [m for m in MODELS if select(runs, model=m)]

    # ---------------- length strata, condition (P), fp16; models side by side
    ns = len(LENGTH_STRATA); ncol = 1 + ns * len(models)
    def strata_cells(model, task, mask):
        tr = select(runs, model=model, condition="P", precision="fp16", task=task)
        d = by_seed(select(tr, mask=mask))
        if not d:
            return ["--"] * ns, None, 0
        L = tr[0]["length"]
        if (L < 0).any():
            raise ValueError("dev_predictions.jsonl lacks the 'length' field")
        seeds = sorted(d); P, y = stack(d, seeds)
        counts = [int(((L >= lo) & (L <= hi)).sum()) for lo, hi in LENGTH_STRATA]
        cells = ["--" if c < 20 else fmt_score(task, metric_batch(task, P[:, (L >= lo) & (L <= hi)], y[(L >= lo) & (L <= hi)]).mean())
                 for (lo, hi), c in zip(LENGTH_STRATA, counts)]
        return cells, counts, len(seeds)
    lines = []
    for task in TASKS:
        masks = [m for m in ["dense", "local", "window_dilations"] + PLACEMENTS
                 if any(select(runs, model=mo, condition="P", precision="fp16", task=task, mask=m) for mo in models)]
        if not masks:
            continue
        rows, counts = [], {}
        for m in masks:
            cells = []
            for mo in models:
                c, cnt, k = strata_cells(mo, task, m)
                cells += c
                if cnt:
                    counts[mo] = cnt
            rows.append(f"{MASK_NAME[m]} & " + " & ".join(cells) + r" \\")
        info = "; ".join(f"{MODEL_NAME[mo].split('-')[0]}: " + "/".join(map(str, counts[mo])) for mo in models if mo in counts)
        lines.append(r"\multicolumn{" + str(ncol) + r"}{l}{\emph{" + TASK_NAME[task] + f" ({METRIC_NAME[task]}); examples per stratum, {info}" + r"}} \\")
        lines += rows
    hdr1 = " & ".join(r"\multicolumn{" + str(ns) + "}{c}{" + MODEL_NAME[mo] + "}" for mo in models)
    rules = " ".join(r"\cmidrule(lr){" + f"{2 + ns * i}-{1 + ns * (i + 1)}" + "}" for i in range(len(models)))
    hdr2 = " & ".join(f"$\\le${hi}" if lo == 1 else (f"$>${lo - 1}" if hi == 128 else f"{lo}--{hi}") for _ in models for lo, hi in LENGTH_STRATA)
    write(os.path.join(a.out, "tables/length_strata.tex"),
          r"\begin{tabular}{l" + "c" * (ns * len(models)) + "}\n" r"\toprule" "\n & " + hdr1 + r" \\" + "\n" + rules + "\n"
          + r"Mask, by input length $L$ & " + hdr2 + r" \\" "\n" r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # ---------------- measured compute
    comp = {}
    for r in runs:
        key = (r["model"], "Exp.~1" if r["experiment"] == "exp1" else "Exp.~2",
               r["condition"] if r["experiment"] == "exp1" else "P", r["precision"])
        comp.setdefault(key, [0, 0.0]); comp[key][0] += 1; comp[key][1] += r.get("train_runtime_s", 0.0) / 3600
    lines, tot = [], [0, 0.0]
    for model in models:
        sub = [0, 0.0]
        for (m, e, c, p), (n, h) in sorted(comp.items()):
            if m != model:
                continue
            lines.append(f"{MODEL_NAME[m]} & {e}, {COND_TEX.get(c, c)}" + (", fp32 check" if p == "fp32" else "") + f" & {n} & {h:.1f} \\\\")
            sub[0] += n; sub[1] += h
        lines.append(f"{MODEL_NAME[model]} & subtotal & {sub[0]} & {sub[1]:.1f} \\\\"); lines.append(r"\midrule")
        tot[0] += sub[0]; tot[1] += sub[1]
    write(os.path.join(a.out, "tables/compute_controlled.tex"),
          r"\begin{tabular}{llrr}" "\n" r"\toprule" "\n" r"Model & Component & Runs & GPU-hours \\" "\n" r"\midrule" "\n"
          + "\n".join(lines) + "\n" + f"Total & & {tot[0]} & {tot[1]:.1f} \\\\" + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


if __name__ == "__main__":
    main()
