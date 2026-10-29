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
The lengths command defaults to the saved pilot mask directory on the institute server.
Results are written to /scratch/scratch26/22cs30008/res/outputs.
"""
import argparse, json, os
import numpy as np

PILOT_MASK_DIR = "/scratch/scratch26/22cs30008/res/outputs/benchmark_all/masks"
OUTPUT_DIR = "/scratch/scratch26/22cs30008/res/outputs"
PILOT_MASKS = ("local", "window_dilations", "circulant", "hypercube", "random_circulant", "bigbird")

FIELDS = {"cola": ("sentence", None), "sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
          "qqp": ("question1", "question2"), "stsb": ("sentence1", "sentence2"), "rte": ("sentence1", "sentence2"),
          "qnli": ("question", "sentence"), "mnli": ("premise", "hypothesis")}
DEV = {"mnli": "validation_matched"}


def write_result(filename, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w") as handle:
        json.dump(data, handle, indent=1)


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


def pilot_masks(mask_dir, task):
    import torch
    masks = {"dense": np.ones((128, 128), dtype=bool)}
    for name in PILOT_MASKS:
        path = os.path.join(mask_dir, f"{task}_{name}.pt")
        mask = torch.load(path, map_location="cpu", weights_only=True).numpy()
        expected = (12, 128, 128) if name == "bigbird" else (128, 128)
        if mask.dtype != bool or mask.shape != expected:
            raise ValueError(f"Unexpected pilot mask in {path}: {mask.dtype}, {mask.shape}")
        masks[name] = mask
    return masks


def valid_density(mask, lengths):
    if mask.ndim == 2:
        mask = mask[None, :, :]
    counts = np.bincount(lengths)
    allowed = sum(count * mask[:, :length, :length].sum()
                  for length, count in enumerate(counts) if count)
    possible = mask.shape[0] * sum(count * length**2
                                   for length, count in enumerate(counts) if count)
    return float(allowed / possible)


def lengths(args):
    from transformers import AutoTokenizer
    from masks import hypercube, cayley_Zn
    from followup_masks import followups
    tok = AutoTokenizer.from_pretrained(args.model)
    tokenizer_name = type(tok).__name__.lower()
    if tokenizer_name.startswith("roberta"):
        model_name, separator_index = "roberta", 1
    elif tokenizer_name.startswith("deberta"):
        model_name, separator_index = "deberta", 0
    else:
        raise ValueError(f"Unsupported tokenizer: {type(tok).__name__}")
    new_masks = followups()
    out = {}
    for task in FIELDS:
        dev = load_dev(args.glue, task)
        L, seg2 = [], []
        for ex in dev:
            ids = encode(tok, ex, task)["input_ids"]
            L.append(len(ids))
            if FIELDS[task][1]:
                sep = [i for i, t in enumerate(ids) if t == tok.sep_token_id]
                seg2.append(sep[separator_index] + 1 if len(sep) > separator_index else len(ids))
        L = np.array(L)
        rec = dict(n=len(L), mean=float(L.mean()), median=float(np.median(L)), p90=float(np.percentile(L, 90)),
                   frac_len_gt_37=float((L > 37).mean()), frac_truncated=float((L >= 128).mean()))
        if seg2:
            s = np.array(seg2)
            rec.update(seg2_start_median=float(np.median(s)), frac_seg2_starts_after_36=float((s > 36).mean()))
        # Count only valid-token pairs; BigBird is averaged over its 12 saved heads.
        ms = pilot_masks(args.masks or PILOT_MASK_DIR, task)
        ms["hypercube_gray"] = hypercube(labeling="gray")
        ms.update({name: new_masks[name] for name in ("hypercube_hampath0", "hypercube_hampath1", "hypercube_hampath2")})
        ms["dyadic"] = cayley_Zn([1, 2, 4, 8, 16, 32, 64])
        rec["valid_density"] = {k: valid_density(A, L) for k, A in ms.items()}
        rec["frac_len_le_17"] = float((L <= 17).mean())
        out[task] = rec
        print(task, json.dumps(rec))
    write_result(f"length_stats_{model_name}.json", out)


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
    write_result("retained_mass.json", res)


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
    write_result("offset_profile.json", prof)
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
