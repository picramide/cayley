"""Theory figure for Experiment 5: layers needed before position 0 can depend on position d
(Proposition on receptive fields). Cells above the layer budget are provably at chance on the
retrieval task, for any parameters."""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(__file__))
from masks import *
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
L = 122  # 120 words + <s> + </s>
D = [3, 7, 12, 15, 24, 37, 48, 63, 85, 96, 111, 119]   # chosen to decouple sequence and graph distance
M = [('Local ($r{=}16$)', local(16)), ('Window + dil.', window_dilations()), ('Dyadic circulant', dyadic()),
     ('Circulant $\\{\\pm1,\\pm2,\\pm3\\}$', circulant()), ('Circular window $r{=}1$', circular_window(1)),
     ('Hypercube, binary', hypercube()), ('Hypercube, Gray', hypercube(labeling='gray')),
     ('Hypercube, Ham. path 0', placement_mask_from_path(random_hamiltonian_path(seed=0))),
     ('Random circulant$^\\ast$', random_circulant()[0])]
def dist0(A, L):
    A = A[:L, :L]; d = np.full(L, np.inf); d[0] = 0; fr = np.zeros(L, bool); fr[0] = True; seen = fr.copy(); k = 0
    while fr.any():
        k += 1; nxt = A[fr].any(0) & ~seen; d[nxt] = k; seen |= nxt; fr = nxt
    return d
H = np.array([[dist0(A, L)[d] for d in D] for _, A in M])
fig, a = plt.subplots(figsize=(6.4, 2.7))
cmap = LinearSegmentedColormap.from_list('s', ['#f4f3ef', '#a9c8f0', '#2a78d6', '#163f73'])
a.imshow(np.clip(np.where(np.isfinite(H), H, 13), 0, 12), cmap=cmap, vmin=0, vmax=12, aspect='auto')
from matplotlib.patches import Rectangle
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if (not np.isfinite(H[i, j])) or H[i, j] > 12:
            a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color='#fbe3e1', zorder=2))
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H[i, j]; over = (not np.isfinite(v)) or v > 12
        a.text(j, i, '$\\infty$' if not np.isfinite(v) else f'{int(v)}', ha='center', va='center', fontsize=7, zorder=3,
               color='#c0392b' if over else ('white' if v > 7 else '#0b0b0b'), fontweight='bold' if over else 'normal')
a.set_xticks(range(len(D)), D); a.set_yticks(range(len(M)), [m for m, _ in M]); a.tick_params(length=0)
a.set_xlabel('position $d$ of the label-carrying token'); [s.set_visible(False) for s in a.spines.values()]
plt.tight_layout(pad=0.3); plt.savefig('figures/retrieval_horizon.pdf', bbox_inches='tight'); plt.close()
print(H)
