"""Two diagnostics the paper asks for that need the model and data (not run by the
draft's author; run on the training machine). Requires torch, transformers, datasets.

1. lengths:  token-length statistics per GLUE dev set, incl. the position where the
             second segment starts, and each mask's realised density on valid tokens
             (rho_valid). Fills the [PENDING] numbers in Sections 3 and 5.
3. profile:  mean pretrained attention by relative offset and on <s>/separators, the
             input to select_generators.py (compatibility-guided Cayley masks).
2. retained: fraction of the *pretrained, dense* RoBERTa attention mass that falls on
             each mask's edges (per layer, averaged over heads, queries and examples).
             Tests the "mask/pretraining mismatch" explanation of degenerate runs.

Usage:
  python diagnostics.py lengths  --glue /home/.../offline/glue --model /home/.../roberta-base
  python diagnostics.py retained --glue ... --model ... --masks /path/to/masks --n 512
"""
import argparse, json, os, sys
import numpy as np

FIELDS = {"cola": ("sentence", None), "sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
          "qqp": ("question1", "question2"), "stsb": ("sentence1", "sentence2"), "rte": ("sentence1", "sentence2"),
          "qnli": ("question", "sentence"), "mnli": ("premise", "hypothesis")}
DEV = {"mnli": "validation_matched"}


def load_dev(root, task):
    from datasets import load_from_disk, load_dataset
    path = os.path.join(root, task)
    try:
        ds = load_from_disk(path)
    except Exception:
        ds = load_dataset(path)
    return ds[DEV.get(task, "validation")]


def encode(tok, ex, task, max_length=128):
    a, b = FIELDS[task]
    return tok(ex[a], ex[b] if b else None, truncation=True, max_length=max_length)


def lengths(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    out = {}
    for task in FIELDS:
        dev = load_dev(args.glue, task)
        L, seg2 = [], []
        for ex in dev:
            ids = encode(tok, ex, task)["input_ids"]
            L.append(len(ids))
            if FIELDS[task][1]:
                # RoBERTa pair format: <s> A </s></s> B </s>; B starts after the 2nd </s>
                sep = [i for i, t in enumerate(ids) if t == tok.sep_token_id]
                seg2.append(sep[1] + 1 if len(sep) >= 2 else len(ids))
        L = np.array(L)
        rec = dict(n=len(L), mean=float(L.mean()), median=float(np.median(L)), p90=float(np.percentile(L, 90)),
                   frac_len_gt_37=float((L > 37).mean()), frac_truncated=float((L >= 128).mean()))
        if seg2:
            s = np.array(seg2)
            rec.update(seg2_start_median=float(np.median(s)), frac_seg2_starts_after_36=float((s > 36).mean()))
        # realised (valid-token) attention density per mask: sum_x |A_L(x)| / sum_x L(x)^2
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from masks import STUDIED, hypercube, cayley_Zn
        ms = dict(STUDIED, hypercube_gray=hypercube(labeling="gray"), dyadic=cayley_Zn([1, 2, 4, 8, 16, 32, 64]))
        rec["valid_density"] = {k: float(sum(A[:l, :l].sum() for l in L) / float((L.astype(float) ** 2).sum()))
                                for k, A in ms.items()}
        rec["frac_len_le_17"] = float((L <= 17).mean())
        out[task] = rec
        print(task, json.dumps(rec))
    json.dump(out, open("length_stats.json", "w"), indent=1)


def retained(args):
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager").eval()
    masks = {}
    for f in sorted(os.listdir(args.masks)):
        if f.endswith(".pt"):
            m = torch.load(os.path.join(args.masks, f), map_location="cpu")
            m = m if m.dtype == torch.bool else (m > -1e4)
            while m.dim() > 2:
                m = m[0]
            masks[f[:-3]] = m
    res = {}
    for task in args.tasks.split(","):
        dev = load_dev(args.glue, task).select(range(args.n))
        acc = {k: np.zeros(12) for k in masks}
        cnt = 0
        for ex in dev:
            enc = encode(tok, ex, task)
            ids = torch.tensor([enc["input_ids"]])
            att = model(input_ids=ids, output_attentions=True).attentions  # 12 x [1, H, L, L]
            L = ids.shape[1]
            for k, m in masks.items():
                mk = m[:L, :L].float()
                for layer, a in enumerate(att):
                    acc[k][layer] += float((a[0] * mk).sum(-1).mean())  # mean over heads & queries
            cnt += 1
        res[task] = {k: (v / cnt).tolist() for k, v in acc.items()}
        print(task, {k: round(float(np.mean(v)), 3) for k, v in res[task].items()})
    json.dump(res, open("retained_mass.json", "w"), indent=1)


def profile(args):
    """Mean dense attention probability by signed offset j - i (per layer), plus the mean
    probability on position 0 (<s>) and on separator tokens. Input to select_generators.py."""
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager").eval()
    n = 128
    off = np.zeros((12, 2 * n - 1)); cnt = np.zeros((12, 2 * n - 1)); to0 = np.zeros(12); tosep = np.zeros(12); m = 0
    for task in args.tasks.split(","):
        for ex in load_dev(args.glue, task).select(range(args.n)):
            enc = encode(tok, ex, task); ids = torch.tensor([enc["input_ids"]]); L = ids.shape[1]
            att = model(input_ids=ids, output_attentions=True).attentions
            sep = (ids[0] == tok.sep_token_id).numpy()
            i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); d = (j - i + n - 1).ravel()
            for layer, a in enumerate(att):
                a = a[0].mean(0).numpy()                       # heads averaged: [L, L]
                np.add.at(off[layer], d, a.ravel()); np.add.at(cnt[layer], d, 1.0)
                to0[layer] += a[:, 0].mean(); tosep[layer] += a[:, sep].sum(1).mean()
            m += 1
    prof = {"offsets": list(range(-(n - 1), n)), "per_layer": (off / np.maximum(cnt, 1)).tolist(),
            "to_position0": (to0 / m).tolist(), "to_separators": (tosep / m).tolist()}
    json.dump(prof, open("offset_profile.json", "w"))
    print("wrote offset_profile.json; mean mass on <s> per layer:", np.round(to0 / m, 3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["lengths", "retained", "profile"])
    p.add_argument("--glue", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--masks")
    p.add_argument("--tasks", default="cola,sst2,mrpc,qqp,stsb,rte,qnli,mnli")
    p.add_argument("--n", type=int, default=512)
    a = p.parse_args()
    {"lengths": lengths, "retained": retained, "profile": profile}[a.cmd](a)
