"""Export the masks for the follow-up experiments proposed in Section 7 of the paper.

Writes one boolean tensor per mask, shape [heads, n, n] (same pattern on every head,
matching mask_config["heads"] = 12 in the original runs). Adapt the save format to
whatever your training script expects (e.g. an additive -inf mask).

Usage:  python followup_masks.py OUTDIR
"""
import os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from masks import (circular_window, window_dilations, circulant, hypercube, with_hub,
                   random_circulant, cayley_Zn, random_hamiltonian_path,
                   placement_mask_from_path, N)

HEADS = 12


def followups():
    m = {}
    # degree sweep inside one Cayley family, Cay(Z_128, {±1..±r}); r = 1, 2 violate the
    # all-prefix reach constraint (K = 12) and are kept only as reach-limited references
    for r in (1, 2, 4, 8, 16, 32):
        m[f"cwin{r}"] = circular_window(r)
    # E2b: dyadic circulant, degree 14, adjacent tokens linked, reach <= 7 hops
    m["dyadic"] = cayley_Zn([1, 2, 4, 8, 16, 32, 64])
    # E3: same abstract graph (Q_7), different placements along the sequence.
    # Gray placement = attention pattern of the Hypercube Transformer (Wang et al., 2022).
    m["hypercube_gray"] = hypercube(labeling="gray")
    # Distinct Hamiltonian-path placements: every neighbouring pair linked, but different
    # connections at other offsets. (Composing a placement with an automorphism of Q_7 gives
    # the IDENTICAL mask, so it is not a control; masks are deduplicated below.)
    for s in range(5):
        m[f"hypercube_hampath{s}"] = placement_mask_from_path(random_hamiltonian_path(seed=s))
    prng = np.random.default_rng(1)
    for s in range(3):
        m[f"hypercube_randplace{s}"] = hypercube(labeling=prng.permutation(N))
    # E4: hub ablation - add <s> (position 0) as a global token to each Cayley mask
    m["circulant+hub"] = with_hub(circulant())
    m["hypercube+hub"] = with_hub(hypercube())
    m["window_dilations+hub"] = with_hub(window_dilations())
    # E4 control: same number of extra edges as the hub, but placed at random
    rng = np.random.default_rng(0)
    for name, base in [("circulant", circulant()), ("hypercube", hypercube())]:
        extra = int(with_hub(base).sum() - base.sum())
        A = base.copy()
        cand = np.argwhere(~A)
        pick = cand[rng.choice(len(cand), size=extra // 2, replace=False)]
        A[pick[:, 0], pick[:, 1]] = True
        A[pick[:, 1], pick[:, 0]] = True
        m[f"{name}+random_edges"] = A
    # E5: several random-circulant draws (distribution, not one sample)
    for s in range(5):
        m[f"random_circulant_s{s}"] = random_circulant(seed=s)[0]
    return m


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    try:
        import torch
    except ImportError:
        torch = None
    seen = {}
    for name, A in followups().items():
        h = hashlib.sha256(A.tobytes()).hexdigest()
        if h in seen:
            print(f"SKIP {name}: identical to {seen[h]}"); continue
        seen[h] = name
        assert A.shape == (N, N) and A.diagonal().all()
        T = np.broadcast_to(A, (HEADS, N, N)).copy()
        path = os.path.join(out, f"{name}.pt" if torch else f"{name}.npy")
        if torch:
            torch.save(torch.from_numpy(T), path)
        else:
            np.save(path, T)
        print(f"{name:26s} degree={A.sum(1).mean():6.1f} "
              f"sha256(pattern)={hashlib.sha256(A.tobytes()).hexdigest()[:16]}  -> {path}")
