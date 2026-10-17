"""Appendix table: placements of the same hypercube Q_7 (Experiment on placement)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from masks import *
from graph_analysis import hops_to_cover, adjacent_coverage
from boundary_positions import dist0
P = [('Binary', hypercube()), ('Gray (Wang et al.)', hypercube(labeling='gray'))]
P += [(f'Hamiltonian path {s}', placement_mask_from_path(random_hamiltonian_path(seed=s))) for s in range(3)]
P += [('Random permutation 0', hypercube(labeling=np.random.default_rng(1).permutation(N)))]
def off(M, k):
    return int(np.triu(M, 1)[np.arange(N - k), np.arange(k, N)].sum())
rows = []
for name, M in P:
    d = dist0(M, 122)
    rows.append(' & '.join([name, f'{adjacent_coverage(M, 32):.2f}'] + [f'{M[:L, :L].mean():.3f}' for L in (24, 48, 100)]
                + [str(off(M, k)) for k in (2, 3, 5)] + [f'{d[1:].mean():.2f}', (lambda h: '$\\infty$' if not np.isfinite(h) else str(int(h)))(hops_to_cover(M, 24)),
                (lambda h: '$\\infty$' if not np.isfinite(h) else str(int(h)))(max(hops_to_cover(M, L) for L in range(2, 129)))]) + r' \\')
open('tables/placements.tex', 'w').write('\n'.join(rows) + '\n\\bottomrule\n')
print('\n'.join(rows))
