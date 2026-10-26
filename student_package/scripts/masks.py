"""Reference constructions of the attention masks studied in the paper, plus the
follow-up masks proposed in Section 7. A[i, j] = True means query i may attend to key j.
All masks include self-loops. n = 128 throughout.

NOTE: random_circulant and bigbird depend on the sampler used in the original runs.
For exact numbers, load the saved .pt masks instead (see load_saved_mask)."""
import numpy as np

N = 128

def cayley_Zn(gens, n=N):
    """Cay(Z_n, S) with S = gens ∪ -gens, plus identity (self-loop)."""
    A = np.eye(n, dtype=bool)
    idx = np.arange(n)
    for g in gens:
        A[idx, (idx + g) % n] = True
        A[idx, (idx - g) % n] = True
    return A

def dense(n=N):              return np.ones((n, n), dtype=bool)
def circulant(gens=(1, 2, 3), n=N): return cayley_Zn(gens, n)
def circular_window(r, n=N): return cayley_Zn(range(1, r + 1), n)
def window_dilations(r=16, dil=(16, 32, 64), n=N):
    return cayley_Zn(list(range(1, r + 1)) + list(dil), n)

def local(r=16, n=N):
    """Non-circular band |i-j| <= r. Not a Cayley graph on Z_n (boundary), but under
    padding it coincides with Cay(Z_n, {±1..±r}) restricted to [0, L) whenever L <= n - r."""
    i = np.arange(n)
    return np.abs(i[:, None] - i[None, :]) <= r

def hypercube(n=N, labeling=None):
    """Cay(Z_2^d, {e_1..e_d}). labeling[p] = hypercube vertex placed at sequence position p.
    labeling=None -> identity (the original run); 'gray' -> binary-reflected Gray code,
    which makes consecutive positions adjacent (a Hamiltonian path along the sequence)."""
    d = int(np.log2(n)); assert 2 ** d == n
    if labeling is None: lab = np.arange(n)
    elif isinstance(labeling, str) and labeling == 'gray': lab = np.arange(n) ^ (np.arange(n) >> 1)
    else: lab = np.asarray(labeling)
    x = lab[:, None] ^ lab[None, :]
    pc = np.vectorize(lambda v: bin(v).count('1'))(x)
    return pc <= 1

def random_circulant(degree=8, seed=42, n=N):
    """Illustrative sampler: `degree` distinct offsets in 1..n-1 with distinct inverses."""
    rng = np.random.default_rng(seed)
    gens = set()
    while len(gens) < degree:
        g = int(rng.integers(1, n))
        if g in gens or (n - g) in gens or 2 * g % n == 0: continue
        gens.add(g)
    return cayley_Zn(sorted(gens), n), sorted(gens)

# Exact generator pairs of the random circulant used in the exploratory benchmark, recovered
# from its saved tensor (spectral gap 0.454, adjacent-token coverage 0).
BENCHMARK_RANDOM_CIRCULANT = [4, 18, 25, 41, 43, 44, 52, 60]

def bigbird(global_tokens=2, block=2, n_random=3, wl=1, wr=1, seed=42, n=N):
    rng = np.random.default_rng(seed)
    nb = n // block
    B = np.zeros((nb, nb), dtype=bool)
    for b in range(nb):
        B[b, max(0, b - wl):min(nb, b + wr + 1)] = True
        cand = [c for c in range(nb) if not B[b, c]]
        B[b, rng.choice(cand, size=n_random, replace=False)] = True
    A = np.kron(B, np.ones((block, block), dtype=bool))
    A[:global_tokens, :] = True; A[:, :global_tokens] = True
    return A | np.eye(n, dtype=bool)

def with_hub(A, hubs=(0,), mode='both'):
    """Add hub tokens to any mask (Experiment 5).
    mode='aggregate': the hub reads every token      (rows    A[h, :] = 1)
    mode='broadcast': every token reads the hub      (columns A[:, h] = 1)
    mode='both':      bidirectional global token, as in BigBird/ETC."""
    A = A.copy(); h = list(hubs)
    if mode in ('aggregate', 'both'): A[h, :] = True
    if mode in ('broadcast', 'both'): A[:, h] = True
    return A

def random_hamiltonian_path(d=7, seed=0, start=0):
    """A random Hamiltonian path of Q_d starting at `start`, built recursively: split on a
    random coordinate c, traverse the half containing `start` by a random Hamiltonian path,
    cross coordinate c at its end, and traverse the other half by an independent random path.
    Different seeds give genuinely different paths, not automorphic images of one placement;
    always deduplicate the resulting mask tensors before use."""
    rng = np.random.default_rng(seed)
    def rec(coords, s):
        if len(coords) == 1:
            return [s, s ^ (1 << coords[0])]
        c = coords[rng.integers(len(coords))]
        rest = [x for x in coords if x != c]
        first = rec(rest, s)
        return first + rec(rest, first[-1] ^ (1 << c))
    path = rec(list(range(d)), start)
    assert len(set(path)) == 2 ** d and all(bin(a ^ b).count('1') == 1 for a, b in zip(path, path[1:]))
    return np.array(path)

def placement_mask_from_path(path):
    """Hypercube mask whose placement is a Hamiltonian path: position i -> vertex path[i]."""
    return hypercube(labeling=path)

def dyadic(n=N):
    """Cay(Z_n, {±1, ±2, ±4, ..., n/2}): degree 2*log2(n), adjacent tokens linked,
    every prefix position reaches 0 in <= popcount hops."""
    return cayley_Zn([2 ** k for k in range(int(np.log2(n)))], n)

def soft_mask_bias(A, beta):
    """Additive logit bias for a gradual dense-to-sparse transition: -beta on removed pairs.
    beta = 0 is dense attention, beta -> inf is the hard mask A. Padding must stay hard-masked."""
    return -beta * (~A).astype(np.float32)

def load_saved_mask(path):
    import torch
    m = torch.load(path, map_location='cpu')
    m = m.numpy() if hasattr(m, 'numpy') else np.asarray(m)
    while m.ndim > 2: m = m[0]      # [heads, n, n] -> [n, n]; same pattern on all heads
    return m.astype(bool) if m.dtype == bool else (m > -1e4)   # bool or additive mask

STUDIED = {
    'dense': dense(), 'local': local(16), 'window_dilations': window_dilations(),
    'circulant': circulant(), 'hypercube': hypercube(),
    'random_circulant': random_circulant()[0], 'bigbird': bigbird(),
}
