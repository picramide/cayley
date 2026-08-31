import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def _as_unique_offsets(generators: Iterable[int], seq_len: int) -> list[int]:
    offsets = sorted({int(g) % seq_len for g in generators})
    return [g for g in offsets if g != 0]


def mask_stats(mask: torch.Tensor) -> dict[str, float | int | tuple[int, ...]]:
    keep = mask.bool()
    return {
        "shape": tuple(keep.shape),
        "kept_edges": int(keep.sum().item()),
        "total_edges": int(keep.numel()),
        "density": float(keep.float().mean().item()),
    }


def build_dense_mask(seq_len: int) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool)


def build_local_mask(seq_len: int, window: int, include_self: bool = True) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        left = max(0, i - window)
        right = min(seq_len, i + window + 1)
        mask[i, left:right] = True
    if include_self:
        mask.fill_diagonal_(True)
    return mask


def build_circulant_cayley_mask(
    seq_len: int,
    generators: Sequence[int],
    include_inverse: bool = True,
    include_self: bool = True,
) -> torch.Tensor:
    """Build a Cayley graph mask over Z_n.

    Rows are query positions and columns are key positions. True means the query
    may attend to the key. With include_inverse=True, every generator g also
    adds -g, producing an undirected circulant graph.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    gens = set(_as_unique_offsets(generators, seq_len))
    if include_inverse:
        gens.update((-g) % seq_len for g in list(gens))

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    idx = torch.arange(seq_len)
    for g in sorted(gens):
        mask[idx, (idx + g) % seq_len] = True

    if include_self:
        mask.fill_diagonal_(True)
    return mask


def build_hypercube_mask(seq_len: int, include_self: bool = True) -> torch.Tensor:
    if seq_len <= 0 or (seq_len & (seq_len - 1)) != 0:
        raise ValueError(f"seq_len must be a power of 2 for hypercube attention, got {seq_len}")

    dim = int(math.log2(seq_len))
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        for bit in range(dim):
            mask[i, i ^ (1 << bit)] = True

    if include_self:
        mask.fill_diagonal_(True)
    return mask


def build_window_dilation_mask(
    seq_len: int,
    window: int,
    dilations: Sequence[int],
    include_inverse: bool = True,
    include_self: bool = True,
) -> torch.Tensor:
    gens = list(range(1, window + 1)) + [int(d) for d in dilations]
    return build_circulant_cayley_mask(
        seq_len=seq_len,
        generators=gens,
        include_inverse=include_inverse,
        include_self=include_self,
    )


def build_random_circulant_mask(
    seq_len: int,
    degree: int,
    seed: int,
    include_inverse: bool = True,
    include_self: bool = True,
) -> torch.Tensor:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    candidates = torch.randperm(seq_len - 1, generator=gen) + 1
    generators = candidates[:degree].tolist()
    return build_circulant_cayley_mask(
        seq_len=seq_len,
        generators=generators,
        include_inverse=include_inverse,
        include_self=include_self,
    )


@dataclass(frozen=True)
class BigBirdMaskConfig:
    global_tokens: int = 2
    block_size: int = 2
    num_random_blocks: int = 1
    window_block_left: int = 1
    window_block_right: int = 1
    seed: int = 42


def _block_bounds(block_idx: int, block_size: int, seq_len: int) -> tuple[int, int]:
    start = block_idx * block_size
    return start, min(seq_len, start + block_size)


def build_bigbird_mask(seq_len: int, n_heads: int, config: BigBirdMaskConfig) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")

    global_tokens = min(config.global_tokens, seq_len)
    block_size = max(1, config.block_size)
    num_blocks = math.ceil(seq_len / block_size)
    global_blocks = math.ceil(global_tokens / block_size) if global_tokens else 0

    base = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if global_tokens:
        base[:, :global_tokens] = True
        base[:global_tokens, :] = True

    for q_block in range(num_blocks):
        qs, qe = _block_bounds(q_block, block_size, seq_len)
        left = max(0, q_block - config.window_block_left)
        right = min(num_blocks - 1, q_block + config.window_block_right)
        for k_block in range(left, right + 1):
            ks, ke = _block_bounds(k_block, block_size, seq_len)
            base[qs:qe, ks:ke] = True
    base.fill_diagonal_(True)

    mask = base.unsqueeze(0).repeat(n_heads, 1, 1)
    if config.num_random_blocks <= 0:
        return mask

    all_blocks = list(range(num_blocks))
    for head_idx in range(n_heads):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(config.seed + head_idx)
        for q_block in range(num_blocks):
            excluded = set(range(global_blocks))
            excluded.update(
                range(
                    max(0, q_block - config.window_block_left),
                    min(num_blocks - 1, q_block + config.window_block_right) + 1,
                )
            )
            legal = [b for b in all_blocks if b not in excluded]
            if not legal:
                continue
            order = torch.randperm(len(legal), generator=rng).tolist()
            for idx in order[: config.num_random_blocks]:
                ks, ke = _block_bounds(legal[idx], block_size, seq_len)
                qs, qe = _block_bounds(q_block, block_size, seq_len)
                mask[head_idx, qs:qe, ks:ke] = True

    return mask

def build_bipartite_cayley_mask(
    seq_len: int,
    premise_len: int,
    local_window: int = 3,
    cross_window: int = 2,
    global_tokens: int = 1,
    include_self: bool = True,
) -> torch.Tensor:
    """Build a Z_2 x Z_m Direct Product Cayley Graph mask for NLI tasks.
    
    The premise spans indices [0, premise_len - 1].
    The hypothesis spans indices [premise_len, seq_len - 1].
    The first `global_tokens` act as global hubs (e.g., [CLS] tokens).
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if premise_len < 0 or premise_len > seq_len:
        raise ValueError("premise_len must be between 0 and seq_len")
    if global_tokens < 0:
        raise ValueError("global_tokens must be non-negative")

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    hypo_len = seq_len - premise_len
    
    # Calculate the max length for the Z_m dimension
    m = max(premise_len - global_tokens, hypo_len)

    # 1. Global Hub Assignment
    if global_tokens > 0:
        bound = min(global_tokens, seq_len)
        mask[:, :bound] = True
        mask[:bound, :] = True

    # 2. Intra-Sentence Parsing Subgraph
    for i in range(global_tokens, seq_len):
        left = max(global_tokens, i - local_window)
        right = min(seq_len, i + local_window + 1)
        
        # Restrict edges to remain within their respective segments
        if i < premise_len:
            right = min(right, premise_len)
        else:
            left = max(left, premise_len)
            
        mask[i, left:right] = True

    # 3. Inter-Sentence Cross Subgraph
    if m > 0:
        # Generate the dyadic chords based on the maximum segment length
        dyadic_chords = {2**k for k in range(2, int(math.log2(max(1, m))) + 1)}
        
        for i in range(global_tokens, premise_len):
            x = i - global_tokens
            for j in range(max(global_tokens, premise_len), seq_len):
                y = j - premise_len
                
                dist = abs(x - y)
                if dist <= cross_window or dist in dyadic_chords:
                    mask[i, j] = True
                    mask[j, i] = True

    # 4. Self-Attention Alignment
    if include_self:
        mask.fill_diagonal_(True)
        
    return mask

def save_mask(mask: torch.Tensor, output_path: str) -> None:
    torch.save(mask.bool().contiguous(), output_path)
