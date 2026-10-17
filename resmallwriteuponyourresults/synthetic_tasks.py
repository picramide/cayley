"""Controlled communication tasks for Experiment 5 (Section 9 of the paper).

Each task is written as GLUE-style JSONL ({"sentence": ...} or {"sentence1", "sentence2"},
plus "label") so it runs through the same fine-tuning pipeline as the GLUE tasks.
Words are drawn from lists of words that the RoBERTa tokenizer encodes as ONE token
(with a leading space), so word w_k sits at token position k+1 (<s> is position 0).
Pass --tokenizer to filter the word lists automatically; otherwise the built-in lists
are used as they are and you should check them once with the tokenizer.

Tasks
  order      Local order. Two marked words u, v are placed next to each other at a random
             position; label = 1 if the word from list A comes first. Only an adjacent-pair
             relation carries the label.
  retrieve   Distant retrieval. The word at token position d is drawn from list A (label 1)
             or list B (label 0); every other word is filler, independent of the label.
             If dist_G(0, d) > K the population accuracy is 50%. The dev set is built from
             counterfactual pairs (same filler, opposite labels), so a model that cannot see
             position d scores exactly 50% per example and 0% paired accuracy.
  compare    Cross-segment comparison. Segment 1 and segment 2 each contain one marked
             word, at controlled positions; label = 1 if the two marked words are equal.

Usage:
  python synthetic_tasks.py OUTDIR [--n_train 20000 --n_dev 2000 --length 120
                                    --distances 4,8,16,24,32,48,64,96 --tokenizer PATH]
"""
import argparse, json, os, random

FILLER = ("the of and to in is was for on that with as by at from his her it an were are which this "
          "be or has had but not they their one all she he we been more who will would there some "
          "other so new can only also when into time after year its first two may could most over "
          "many such these than him them then any about up out what if just like no made people").split()
LIST_A = "red green blue black white yellow brown pink purple orange gray gold silver dark light".split()
LIST_B = "cat dog horse bird fish cow sheep pig duck mouse lion bear wolf fox deer".split()
MARK = "@"   # marker word preceding a designated token (order / compare tasks)


def single_token_filter(words, tokenizer):
    keep = []
    for w in words:
        ids = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            keep.append(w)
    return keep


def filler(rng, k):
    return [rng.choice(FILLER) for _ in range(k)]


def task_order(rng, length):
    words = filler(rng, length)
    p = rng.randrange(0, length - 1)
    a, b = rng.choice(LIST_A), rng.choice(LIST_B)
    label = rng.randrange(2)
    words[p], words[p + 1] = (a, b) if label else (b, a)
    return {"sentence": " ".join(words), "label": label, "pos": p + 1}


def task_retrieve(rng, length, d):
    """Target at token position d (word index d-1). Requires 1 <= d <= length."""
    words = filler(rng, length)
    label = rng.randrange(2)
    words[d - 1] = rng.choice(LIST_A if label else LIST_B)
    return {"sentence": " ".join(words), "label": label, "pos": d}


def task_retrieve_pair(rng, length, d, pair_id):
    """Counterfactual pair: identical filler, opposite labels, differing only at position d.
    If position d is outside the readout's receptive field, a deterministic classifier must
    give both members the same prediction, so exactly one is correct: per-example accuracy is
    exactly 50% and paired accuracy (both correct) is exactly 0, on any finite set."""
    words = filler(rng, length)
    a = dict(sentence=None, label=1, pos=d, pair_id=pair_id)
    b = dict(sentence=None, label=0, pos=d, pair_id=pair_id)
    wa, wb = list(words), list(words)
    wa[d - 1], wb[d - 1] = rng.choice(LIST_A), rng.choice(LIST_B)
    a["sentence"], b["sentence"] = " ".join(wa), " ".join(wb)
    return [a, b]


def task_compare(rng, len1, len2, p1, p2):
    """Marked word at word index p1 of segment 1 and p2 of segment 2 (each preceded by MARK)."""
    s1, s2 = filler(rng, len1), filler(rng, len2)
    x = rng.choice(LIST_A)
    label = rng.randrange(2)
    y = x if label else rng.choice([w for w in LIST_A if w != x])
    s1[p1 - 1], s1[p1] = MARK, x
    s2[p2 - 1], s2[p2] = MARK, y
    return {"sentence1": " ".join(s1), "sentence2": " ".join(s2), "label": label}


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_dev", type=int, default=2000)
    ap.add_argument("--length", type=int, default=120, help="words per input (<= 126 for max_length 128)")
    ap.add_argument("--distances", default="3,7,12,15,24,37,48,63,85,96,111,119",
                    help="token positions d; choose them with boundary_positions.py so that they straddle each mask's horizon")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    global FILLER, LIST_A, LIST_B
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        FILLER, LIST_A, LIST_B = (single_token_filter(w, tok) for w in (FILLER, LIST_A, LIST_B))
        assert len(single_token_filter([MARK], tok)) == 1, "marker must be a single token"
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for split, n in (("train", a.n_train), ("dev", a.n_dev)):
        write(os.path.join(a.out, f"order_{split}.jsonl"), [task_order(rng, a.length) for _ in range(n)])
        for d in map(int, a.distances.split(",")):
            if split == "dev":   # counterfactual pairs: the theorem becomes an exact data check
                rows = [r for k in range(n // 2) for r in task_retrieve_pair(rng, a.length, d, k)]
            else:
                rows = [task_retrieve(rng, a.length, d) for _ in range(n)]
            write(os.path.join(a.out, f"retrieve_d{d}_{split}.jsonl"), rows)
        half = (a.length - 3) // 2
        for p1, p2 in ((2, 2), (half // 2, half // 2), (half - 1, half - 1)):
            write(os.path.join(a.out, f"compare_p{p1}_{p2}_{split}.jsonl"),
                  [task_compare(rng, half, half, p1, p2) for _ in range(n)])
    print("wrote tasks to", a.out)


if __name__ == "__main__":
    main()
