"""Exhaustive numerical checks of the propositions in Section 3. Writes tables/verify.tex."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from masks import *
from graph_analysis import receptive_field, hops_to_cover, n_components

n = N
rows = []
def add(stmt, scope, ok, total):
    rows.append((stmt, scope, ok, total)); print(f'{stmt:40s} {ok}/{total}')

def distance_graph(Shat, L):
    d = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    return np.isin(d, [0] + [abs(s) for s in Shat])

def apsp(A):
    """All-pairs shortest-path lengths by repeated boolean matrix products."""
    L = len(A); D = np.full((L, L), np.inf); D[np.eye(L, dtype=bool)] = 0
    R = np.eye(L, dtype=bool); k = 0
    Ai = A.astype(np.int32)
    while True:
        k += 1; Rn = (R.astype(np.int32) @ Ai) > 0
        new = Rn & ~R
        if not new.any(): return D
        D[new] = k; R = Rn

# Prop 1 (distance-graph reduction) and non-vertex-transitivity
fams = {'circulant {1,2,3}': [1, 2, 3], 'window+dilations': list(range(1, 17)) + [32, 64]}
for r in (1, 2, 4, 8, 16, 32): fams[f'circular window r={r}'] = list(range(1, r + 1))
ok = tot = 0; ok2 = tot2 = 0
for name, S in fams.items():
    A = cayley_Zn(S); m = max(S)
    for L in range(2, n - m + 1):
        tot += 1; ok += (A[:L, :L] == distance_graph(S, L)).all()
        if L > m + 1:
            deg = A[:L, :L].sum(1); tot2 += 1; ok2 += deg.min() != deg.max()
add('Distance graph: $G_L = D_L(\\hat S)$', f'{len(fams)} circulant masks, all $2\\le L\\le n-m$', ok, tot)
add('Distance graph: $G_L$ not regular', 'same masks, all $m+1<L\\le n-m$', ok2, tot2)

# Cor 1
loc, cw, wd = local(16), circular_window(16), window_dilations()
add('Local $=$ circular window', '$2\\le L\\le112$', sum((loc[:L, :L] == cw[:L, :L]).all() for L in range(2, 113)), 111)
add('Local $\\ne$ circular window at $L=113$', '$L=113$', int(not (loc[:113, :113] == cw[:113, :113]).all()), 1)
add('Window + dil.\\ $\\supseteq$ local', '$2\\le L\\le112$', sum((wd[:L, :L] >= loc[:L, :L]).all() for L in range(2, 113)), 111)

# Cor 2
C = circulant()
add('Circulant horizon: 0 sees exactly $[0,36]$', '$37\\le L\\le125$, $K=12$',
    sum((np.nonzero(receptive_field(C, L))[0] == np.arange(37)).all() for L in range(37, 126)), 89)

# Prop 3 hypercube prefix isometry
H = hypercube(); ham = np.vectorize(lambda a: bin(a).count('1'))(np.bitwise_xor.outer(np.arange(n), np.arange(n)))
ok = 0
for L in range(1, n + 1):
    D = apsp(H[:L, :L]); ok += (D == ham[:L, :L]).all()
add('Hypercube prefixes: $\\mathrm{dist}=\\mathrm{Ham}$', 'all pairs, all $1\\le L\\le128$', ok, n)
add('Hypercube prefixes: diam.\\ $\\le\\lceil\\log_2 L\\rceil$', 'all $2\\le L\\le128$',
    sum(apsp(H[:L, :L]).max() <= int(np.ceil(np.log2(L))) for L in range(2, n + 1)), n - 1)

# Prop 4
G = hypercube(labeling='gray')
i = np.arange(n - 1)
add('Placement: $i\\sim i{+}1$ iff $i$ even (binary)', '$0\\le i\\le126$', int(((H[i, i + 1]) == (i % 2 == 0)).sum()), n - 1)
add('Placement: all neighbours adjacent (Gray)', '$0\\le i\\le126$', int(G[i, i + 1].sum()), n - 1)
g = np.arange(n) ^ (np.arange(n) >> 1)
P = np.zeros((n, n), int); P[np.arange(n), g] = 1          # position p -> vertex g(p)
add('Placement: Gray mask $=P Q_7 P^\\top$', 'explicit permutation', int((P @ H.astype(int) @ P.T > 0).astype(bool).__eq__(G).all()), 1)
ev = lambda A: np.sort(np.linalg.eigvalsh(A.astype(float)))
add('Placement: identical spectra', 'adjacency matrices', int(np.allclose(ev(H), ev(G))), 1)
add('Gray: position 0 sees all within 7 layers', 'all $2\\le L\\le128$', sum(hops_to_cover(G, L) <= 7 for L in range(2, n + 1)), n - 1)

# Wang et al. (2022), Algorithm 1: sequence-to-hypercube labels; mask must equal the Gray mask
X = [0, 1]
while len(X) < n:
    X = [v << 1 for v in X] + [(v << 1) + 1 for v in reversed(X)]
X = np.array(X[:n]); W = np.vectorize(lambda a: bin(a).count('1'))(np.bitwise_xor.outer(X, X)) <= 1
add('Wang et al.\\ mapping gives the Gray mask', 'all $128\\times128$ entries', int((W == G).all()), 1)

# dyadic circulant: position 0 sees every valid token within 7 layers
Dy = cayley_Zn([1, 2, 4, 8, 16, 32, 64])
add('Dyadic circulant: 0 sees all within 7 layers', 'all $2\\le L\\le128$', sum(hops_to_cover(Dy, L) <= 7 for L in range(2, n + 1)), n - 1)

# Automorphisms: composing a placement with an automorphism of Q_7 gives the identical mask
perm = np.random.default_rng(0).permutation(7)
aut = lambda v: sum((((v >> perm[k]) & 1) << k) for k in range(7)) ^ 37
Ga = hypercube(labeling=[aut(v) for v in (np.arange(n) ^ (np.arange(n) >> 1))])
add('Automorphism-composed placement: same mask', 'Gray placement, one random automorphism', int((Ga == G).all()), 1)

# Recursive placements: identical realised edge counts at every prefix length
Hb = hypercube(); recs = [G] + [placement_mask_from_path(random_hamiltonian_path(seed=s)) for s in range(20)]
add('Recursive placements: equal $\\lVert A_L\\rVert_0$', 'Gray + 20 Hamiltonian paths, all $L$',
    sum(all(M[:L, :L].sum() == Hb[:L, :L].sum() for L in range(1, n + 1)) for M in recs), len(recs))

# Strengthened recursive-placement result: automorphism image of the binary initial segment
from recursive_placements import check as _rec_check
ok_r, tot_r = _rec_check(7, range(20))
from recursive_placements import check_gray as _gray_check
ok_g, tot_g = _gray_check(7); ok_r += ok_g; tot_r += tot_g
add('Recursive placements: prefix $=\\sigma(\\{0,\\dots,L-1\\})$, $\\sigma\\in\\mathrm{Aut}(Q_7)$', 'Gray + 20 Hamiltonian paths, all $L$', ok_r, tot_r)
pc = lambda j: bin(j).count('1')
add('Edge count $|E(G_L)|=\\sum_{j<L}\\mathrm{popcount}(j)$', 'binary, Gray + 20 paths, all $L$',
    sum(all(int(np.triu(M[:L, :L], 1).sum()) == sum(pc(j) for j in range(L)) for L in range(1, n + 1)) for M in [Hb] + recs), 1 + len(recs))

# Prop 5 over random circulant draws
ok = tot = 0
for s in range(200):
    A, gens = random_circulant(seed=s)
    hat = [x if x <= n // 2 else x - n for x in gens]
    for L in range(2, n // 2 + 1):
        small = [abs(x) for x in hat if abs(x) < L]
        q = np.gcd.reduce(small) if small else 0
        if q >= 2:
            tot += 1
            comp_ok = n_components(A, L) >= min(q, L)
            reach0 = np.nonzero(receptive_field(A, L, K=10 ** 6))[0]
            ok += comp_ok and (reach0 % q == 0).all()
add('Padding disconnection', f'200 draws, all $L\\le64$ with $q\\ge2$ ({tot} cases)', ok, tot)

# Prop 6 / Prop 7: hard and soft masking of one head (random instances)
rng = np.random.default_rng(0); ok1 = ok2 = ok3 = 0; T = 2000
for _ in range(T):
    k = rng.integers(2, 40); d = rng.integers(1, 16)
    p = rng.dirichlet(np.ones(k) * rng.uniform(0.1, 3)); V = rng.normal(size=(k, d))
    keep = rng.random(k) < rng.uniform(0.1, 0.9); keep[rng.integers(k)] = True
    R = p[keep].sum(); pA = np.where(keep, p, 0) / R
    o, oA = p @ V, pA @ V
    muS = (p[keep] @ V[keep]) / R
    muN = (p[~keep] @ V[~keep]) / (1 - R) if (~keep).any() else np.zeros(d)
    ok1 += np.isclose(np.abs(p - pA).sum(), 2 * (1 - R))
    ok2 += np.allclose(oA - o, (1 - R) * (muS - muN))
    beta = rng.uniform(0, 8); w = np.where(keep, p, p * np.exp(-beta)); pb = w / w.sum()
    Z = R + (1 - R) * np.exp(-beta); f = R * (1 - np.exp(-beta)) / Z
    ok3 += np.allclose(pb @ V - o, f * (oA - o)) and np.isclose(np.abs(p - pb).sum(), 2 * (1 - R) * f)
add('Hard masking: $\\lVert p-p^A\\rVert_1=2(1-R)$', f'{T} random heads', int(ok1), T)
add('Hard masking: $o_A-o=(1-R)(\\mu_S-\\mu_{\\bar S})$', f'{T} random heads', int(ok2), T)
add('Soft masking: $o_\\beta\\in[o,o_A]$', f'{T} random heads and $\\beta$', int(ok3), T)

# Prop 8: dyadic circulant, dist(j,0) <= popcount(j) in every prefix
Dy = dyadic(); okd = 0
for L in range(2, n + 1):
    D = apsp(Dy[:L, :L]); okd += all(D[0, j] <= bin(j).count('1') for j in range(L))
add('Dyadic reach: $\\mathrm{dist}(0,j)\\le\\mathrm{popcount}(j)$', 'all $j<L$, all $2\\le L\\le128$', okd, n - 1)

with open('tables/verify.tex', 'w') as f:
    for st, sc, o, t in rows:
        f.write(f'{st} & {sc} & {o}/{t} \\\\\n')
    f.write('\\bottomrule\n')
assert all(o == t for _, _, o, t in rows), 'a check failed'
print('all checks pass')
