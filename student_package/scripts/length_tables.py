"""Tables for Appendix 'Input-length statistics and realised density' from the
output of `diagnostics.py lengths` (data/length_stats.json).

Usage: python3 scripts/length_tables.py data/length_stats.json [--suffix _deberta]
Writes tables/lengths{suffix}.tex and tables/valid_density_tasks{suffix}.tex. With a suffix
(statistics computed with another tokenizer), only the masks of the controlled study are listed.
"""
import json, os, sys

TASKS = [("cola", "CoLA"), ("sst2", "SST-2"), ("mrpc", "MRPC"), ("qqp", "QQP"),
         ("stsb", "STS-B"), ("rte", "RTE"), ("qnli", "QNLI"), ("mnli", "MNLI-m")]
MASKS = [("local", r"Local ($r{=}16$)"), ("window_dilations", "Window + dilations"),
         ("circulant", r"Circulant $\{\pm1,\pm2,\pm3\}$"), ("dyadic", "Dyadic circulant"),
         ("hypercube", r"Hypercube $Q_7$, binary"), ("hypercube_gray", "Hypercube $Q_7$, Gray"),
         ("random_circulant", "Random circulant"), ("bigbird", r"BigBird-style$^\dagger$")]


def pct(x):
    return f"{100 * x:.1f}"


def main(path, suffix=""):
    S = json.load(open(path))
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "tables")
    rows = []
    for k, name in TASKS:
        r = S[k]
        seg = (f"{r['seg2_start_median']:.0f} & {pct(r['frac_seg2_starts_after_36'])}"
               if "seg2_start_median" in r else "n/a & n/a")
        rows.append(f"{name} & {r['n']} & {r['mean']:.1f} & {r['median']:.0f} & {r['p90']:.0f} & "
                    f"{pct(r['frac_len_le_17'])} & {pct(r['frac_len_gt_37'])} & {pct(r['frac_truncated'])} & {seg} \\\\")
    lhdr = ("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nTask & $n$ & Mean $L$ & Median & 90th pct. & $L\\le17$ (\\%) & $L>37$ (\\%) & "
            "Truncated (\\%) & Median start & Start $>36$ (\\%) \\\\\n\\midrule\n") if suffix else ""
    open(os.path.join(out, f"lengths{suffix}.tex"), "w").write(lhdr + "\n".join(rows) + "\n\\bottomrule\n" + ("\\end{tabular}\n" if suffix else ""))

    rows = []
    masks = MASKS if not suffix else [(m, n) for m, n in MASKS if m in ("local", "window_dilations", "hypercube")]
    if suffix:
        masks = [(m, n.replace(", binary", " (all placements)")) for m, n in masks]
    for m, name in masks:
        vals = " & ".join(f"{S[k]['valid_density'][m]:.3f}" for k, _ in TASKS)
        rows.append(f"{name} & {vals} \\\\")
    hdr = ("\\begin{tabular}{lcccccccc}\n\\toprule\nMask & " + " & ".join(n for _, n in TASKS) + " \\\\\n\\midrule\n") if suffix else ""
    open(os.path.join(out, f"valid_density_tasks{suffix}.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n" + ("\\end{tabular}\n" if suffix else ""))
    # consistency checks used in the text
    for k, _ in TASKS:
        v = S[k]["valid_density"]
        assert v["hypercube"] == v["hypercube_gray"], k          # Proposition 7
        assert v["window_dilations"] >= v["local"], k            # Corollary 1 (L <= 112)
    print(f"ok; tables/lengths{suffix}.tex and tables/valid_density_tasks{suffix}.tex written")


if __name__ == "__main__":
    args = sys.argv[1:]
    suffix = ""
    if "--suffix" in args:
        k = args.index("--suffix"); suffix = args[k + 1]; del args[k:k + 2]
    main(args[0] if args else "data/length_stats.json", suffix)
