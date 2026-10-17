"""Compatibility-guided Cayley masks (Section 10 of the paper).

Choose a generator set S for Cay(Z_128, S) under a budget of B generator pairs that keeps
as much of the pretrained attention as possible, subject to structural constraints:
  * reach: position 0 sees every valid token within K layers at every length L
  * adjacency (optional): +-1 is a generator, so neighbouring tokens are linked
The score is a cheap proxy computed from an offset profile (diagnostics.py profile):
expected retained attention mass, summed over layers. Proposition 6 of the paper shows that
retained mass is not the whole perturbation, so the selected candidates should be re-ranked
with the measured output drift (diagnostics.py retained) before training.

Usage:
  python select_generators.py offset_profile.json --pairs 7 [--K 12 --adjacency --top 5]
  python select_generators.py --demo        # synthetic profile, for testing only
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from masks import cayley_Zn, N
from graph_analysis import hops_to_cover

CANDIDATES = list(range(1, N // 2 + 1))     # generator pairs {+-g}; 64 = -64 is a single element


def retained_mass(gens, prof):
    """prof: [layers, 2N-1] mean attention by offset j-i (index offset + N-1). Self-loop included."""
    offs = {0} | {g for g in gens} | {-g for g in gens}
    idx = [o + N - 1 for o in offs] + [o + N - 1 for o in (o - N if o > 0 else o + N for o in offs) if -N < o < N]
    return float(prof[:, sorted(set(idx))].sum())


def max_hops(gens):
    A = cayley_Zn(gens)
    return max(hops_to_cover(A, L) for L in range(2, N + 1))


def select(prof, pairs, K=12, adjacency=True, beam=8):
    start = [1] if adjacency else []
    frontier = [tuple(start)]
    for _ in range(pairs - len(start)):
        cand = {}
        for S in frontier:
            for g in CANDIDATES:
                if g in S: continue
                T = tuple(sorted(S + (g,)))
                cand[T] = retained_mass(T, prof)
        frontier = [T for T, _ in sorted(cand.items(), key=lambda kv: -kv[1])[:beam]]
    ranked = sorted(frontier, key=lambda T: -retained_mass(T, prof))
    ok = [T for T in ranked if max_hops(T) <= K]
    if ok:
        return ok, ranked
    # reach repair: replace the least valuable generator with the dyadic generator that most reduces hops
    T = list(ranked[0])
    while max_hops(T) > K:
        worst = min((g for g in T if not (adjacency and g == 1)), key=lambda g: retained_mass([x for x in T if x != g], prof))
        best = min((d for d in (2, 4, 8, 16, 32, 64) if d not in T), key=lambda d: max_hops([x for x in T if x != worst] + [d]))
        T = sorted([x for x in T if x != worst] + [best])
    return [tuple(T)], ranked


def demo_profile(layers=12):
    o = np.arange(-(N - 1), N)
    prof = np.exp(-np.abs(o) / 3.0) + 0.3 * (np.abs(o) == 1)
    prof = np.tile(prof / prof.sum(), (layers, 1))
    return prof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--pairs", type=int, default=7)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--adjacency", action="store_true", default=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    prof = demo_profile() if a.demo else np.array(json.load(open(a.profile))["per_layer"])
    ok, ranked = select(prof, a.pairs, a.K, a.adjacency)
    print("selected (reach-feasible), best first:")
    for T in ok[:a.top]:
        print(f"  S = {{±{', ±'.join(map(str, T))}}}  degree={cayley_Zn(list(T)).sum(1).mean():.0f}  "
              f"retained-mass proxy={retained_mass(T, prof):.3f}  max hops={max_hops(T)}")
    print("comparison: dyadic", round(retained_mass((1, 2, 4, 8, 16, 32, 64), prof), 3),
          "| circulant {1,2,3}", round(retained_mass((1, 2, 3), prof), 3), "hops", max_hops((1, 2, 3)))
