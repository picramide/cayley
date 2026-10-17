"""Graph statistics of attention masks under padding.
Usage: python graph_analysis.py OUTDIR [--saved MASKDIR]   (MASKDIR holds <task>_<mask>.pt)"""
import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from masks import *

K = 12  # RoBERTa-base layers, same mask in every layer

def receptive_field(A, L, src=0, K=K):
    """Positions j in [0,L) whose input embedding can influence position `src` after K layers.
    Information flows j -> i when query i attends to key j (A[i,j])."""
    A = A[:L, :L]
    seen = np.zeros(L, bool); seen[src] = True; frontier = seen.copy()
    for _ in range(K):
        nxt = A[frontier].any(0) & ~seen
        if not nxt.any(): break
        seen |= nxt; frontier = nxt
    return seen

def hops_to_cover(A, L, src=0):
    A = A[:L, :L]; seen = np.zeros(L, bool); seen[src] = True; fr = seen.copy(); h = 0
    while not seen.all():
        nxt = A[fr].any(0) & ~seen
        if not nxt.any(): return np.inf
        seen |= nxt; fr = nxt; h += 1
    return h

def n_components(A, L):
    S = A[:L, :L] | A[:L, :L].T; lab = -np.ones(L, int); c = 0
    for s in range(L):
        if lab[s] >= 0: continue
        st = [s]; lab[s] = c
        while st:
            u = st.pop()
            for v in np.nonzero(S[u] & (lab < 0))[0]: lab[v] = c; st.append(v)
        c += 1
    return c

def adjacent_coverage(A, L):
    """Fraction of ordered neighbouring pairs (i, i±1) inside [0,L) that are directly attended."""
    i = np.arange(L - 1)
    return (A[i, i + 1].sum() + A[i + 1, i].sum()) / (2 * (L - 1))

def local_coverage(A, L, r=4):
    tot = hit = 0
    for i in range(L):
        js = [j for j in range(max(0, i - r), min(L, i + r + 1)) if j != i]
        tot += len(js); hit += A[i, js].sum()
    return hit / tot

def spectral_gap(A):
    S = (A | A.T).astype(float); np.fill_diagonal(S, 0)
    d = S.sum(1); Dm = np.diag(1 / np.sqrt(d))
    ev = np.sort(np.linalg.eigvalsh(np.eye(len(d)) - Dm @ S @ Dm))
    return ev[1]

def summarize(masks):
    res = {}
    for name, A in masks.items():
        Ls = list(range(2, 129))
        res[name] = dict(
            degree=float(A.sum(1).mean()), density=float(A.mean()),
            spectral_gap_full=float(spectral_gap(A)),
            reach=[float(receptive_field(A, L).mean()) for L in Ls],
            hops=[float(hops_to_cover(A, L)) for L in Ls],
            components=[n_components(A, L) for L in Ls],
            adj_cov=[float(adjacent_coverage(A, L)) for L in Ls],
            loc4_cov=[float(local_coverage(A, L, 4)) for L in Ls],
            Ls=Ls)
    return res

if __name__ == '__main__':
    out = sys.argv[1]
    res = summarize(STUDIED)
    # distribution over random circulant draws (the original sampler is unknown to us)
    rc = []
    for s in range(200):
        A, g = random_circulant(seed=s)
        rc.append(dict(gens=g, reach16=float(receptive_field(A, 16).mean()),
                       reach32=float(receptive_field(A, 32).mean()),
                       reach64=float(receptive_field(A, 64).mean()),
                       comp16=n_components(A, 16), comp32=n_components(A, 32),
                       gap=float(spectral_gap(A))))
    res['_random_circulant_draws'] = rc
    # proposed follow-ups
    extra = {'hypercube_gray': hypercube(labeling='gray'),
             'circulant+hub': with_hub(circulant()), 'hypercube+hub': with_hub(hypercube()),
             'window_dilations+hub': with_hub(window_dilations())}
    for r in [1, 2, 4, 8, 16, 32]: extra[f'cwin{r}'] = circular_window(r)
    res['_proposed'] = summarize(extra)
    json.dump(res, open(f'{out}/graph_stats.json', 'w'))
    for name, d in list(res.items()):
        if name.startswith('_'): continue
        L = d['Ls']; ix = {l: L.index(l) for l in (16, 32, 64, 128)}
        print(f"{name:17s} deg={d['degree']:6.1f} dens={d['density']:.3f} gap={d['spectral_gap_full']:.4f} "
              + ' '.join(f"L{l}:reach={d['reach'][i]:.2f},hops={d['hops'][i]:.0f},comp={d['components'][i]},adj={d['adj_cov'][i]:.2f},loc4={d['loc4_cov'][i]:.2f}" for l, i in ix.items()))
    for name, d in res['_proposed'].items():
        L = d['Ls']; i64 = L.index(64); i128 = L.index(128)
        print(f"  {name:22s} deg={d['degree']:6.1f} gap={d['spectral_gap_full']:.4f} reach64={d['reach'][i64]:.2f} hops128={d['hops'][i128]} adj64={d['adj_cov'][i64]:.2f}")
    import statistics as st
    print('random circulant draws: P(disconnected at L=16)=%.2f, P(at L=32)=%.2f, mean reach16=%.2f reach64=%.2f, gap median=%.3f' % (
        np.mean([d['comp16'] > 1 for d in rc]), np.mean([d['comp32'] > 1 for d in rc]),
        np.mean([d['reach16'] for d in rc]), np.mean([d['reach64'] for d in rc]), st.median([d['gap'] for d in rc])))
