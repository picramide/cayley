"""Pick retrieval positions that straddle each mask's receptive-field boundary (Experiment 5).
For every mask and depth K, list positions d whose graph distance from the readout is exactly K
(just inside) and K+1 (just outside) at the given input length, and show that sequence distance
and graph distance are decoupled across placements.
Usage: python boundary_positions.py [--L 122 --K 2,3,4,6,12]"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from masks import *

def dist0(A, L):
    A = A[:L, :L]; d = np.full(L, np.inf); d[0] = 0; fr = np.zeros(L, bool); fr[0] = True; seen = fr.copy(); k = 0
    while fr.any():
        k += 1; nxt = A[fr].any(0) & ~seen; d[nxt] = k; seen |= nxt; fr = nxt
    return d

MASKS = {'local16': local(16), 'window_dil': window_dilations(), 'dyadic': dyadic(), 'circulant123': circulant(),
         'hypercube_binary': hypercube(), 'hypercube_gray': hypercube(labeling='gray'),
         'hypercube_ham0': placement_mask_from_path(random_hamiltonian_path(seed=0))}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--L', type=int, default=122); ap.add_argument('--K', default='2,3,4,6,12')
    a = ap.parse_args(); rng = np.random.default_rng(0)
    for name, A in MASKS.items():
        d = dist0(A, a.L)
        out = []
        for K in map(int, a.K.split(',')):
            inside = np.nonzero(d == K)[0]; outside = np.nonzero(d == K + 1)[0]
            pick = lambda xs: sorted(rng.choice(xs, size=min(3, len(xs)), replace=False).tolist()) if len(xs) else []
            out.append(f'K={K}: in {pick(inside)} out {pick(outside)}')
        print(f'{name:17s} max dist {int(np.nanmax(d[np.isfinite(d)]))}; ' + '; '.join(out))
