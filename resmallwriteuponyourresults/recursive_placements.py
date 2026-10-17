"""Constructive check of the strengthened recursive-placement proposition.
For a recursive ordering of Q_d (built exactly as random_hamiltonian_path in masks.py), construct,
for every prefix length L, the cube automorphism sigma of the proof and verify that
(i) sigma preserves adjacency (it is an automorphism of Q_d), and
(ii) sigma maps the binary initial segment {0, ..., L-1} onto the first L vertices of the ordering.
Hence the deployed graphs of all recursive placements are isomorphic (unrooted) at every L."""
import numpy as np

def recursive_path_with_structure(d=7, seed=0, start=0):
    """Same random choices as masks.random_hamiltonian_path; also returns the split structure."""
    rng = np.random.default_rng(seed)
    def rec(coords, s):
        if len(coords) == 1:
            return [s, s ^ (1 << coords[0])], ('leaf', coords[0], s)
        c = coords[rng.integers(len(coords))]
        rest = [x for x in coords if x != c]
        p1, t1 = rec(rest, s)
        p2, t2 = rec(rest, p1[-1] ^ (1 << c))
        a = (s >> c) & 1
        return p1 + p2, ('node', c, a, rest, t1, t2)
    return rec(list(range(d)), start)

def sigma_for_prefix(tree, k, L):
    """Return sigma: binary label x in Q_k (bit k-1 = top) -> vertex, with sigma({0..L-1}) = prefix."""
    if tree[0] == 'leaf':
        _, c, s = tree
        return lambda x: s ^ ((x & 1) << c)
    _, c, a, rest, t1, t2 = tree
    half = 1 << (k - 1)
    tau = sigma_for_prefix(t1, k - 1, L) if L <= half else sigma_for_prefix(t2, k - 1, L - half)
    mask_low = half - 1
    def sigma(x):
        top = (x >> (k - 1)) & 1
        v = tau(x & mask_low)
        v &= ~(1 << c)
        return v | ((a ^ top) << c)
    return sigma

def check(d=7, seeds=range(20)):
    n = 1 << d; ok = 0; tot = 0
    pc = lambda v: bin(v).count('1')
    for s in seeds:
        path, tree = recursive_path_with_structure(d, seed=s)
        for L in range(1, n + 1):
            sig = sigma_for_prefix(tree, d, L)
            img = [sig(x) for x in range(n)]
            auto = len(set(img)) == n and all(pc(img[x] ^ img[x ^ (1 << b)]) == 1 for x in range(n) for b in range(d))
            onto = set(img[:L]) == set(path[:L])
            tot += 1; ok += auto and onto
    return ok, tot

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from masks import random_hamiltonian_path
    assert all((recursive_path_with_structure(seed=s)[0] == random_hamiltonian_path(seed=s).tolist()) for s in range(5))
    print('constructive automorphism check: %d/%d' % check())


# ---- reflected Gray code as a recursive ordering (reversal and translation of trees) ----
def _path(t): return [t[2], t[2] ^ (1 << t[1])] if t[0] == 'leaf' else _path(t[4]) + _path(t[5])
def _rev(t): return ('leaf', t[1], t[2] ^ (1 << t[1])) if t[0] == 'leaf' else ('node', t[1], 1 - t[2], t[3], _rev(t[5]), _rev(t[4]))
def _xor(t, m): return ('leaf', t[1], t[2] ^ m) if t[0] == 'leaf' else ('node', t[1], t[2] ^ ((m >> t[1]) & 1), t[3], _xor(t[4], m), _xor(t[5], m))
def gray_structure(k):
    return ('leaf', 0, 0) if k == 1 else ('node', k - 1, 0, list(range(k - 1)), gray_structure(k - 1),
                                          _xor(_rev(gray_structure(k - 1)), 1 << (k - 1)))

def check_gray(d=7):
    n = 1 << d; T = gray_structure(d); P = _path(T)
    assert P == [i ^ (i >> 1) for i in range(n)]
    pc = lambda v: bin(v).count('1'); ok = 0
    for L in range(1, n + 1):
        sig = sigma_for_prefix(T, d, L); img = [sig(x) for x in range(n)]
        ok += len(set(img)) == n and all(pc(img[x] ^ img[x ^ (1 << b)]) == 1 for x in range(n) for b in range(d)) \
              and set(img[:L]) == set(P[:L])
    return ok, n
