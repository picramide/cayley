#!/usr/bin/env python3
import argparse
from pathlib import Path

from cayley.masks import (
    BigBirdMaskConfig,
    build_bigbird_mask,
    build_circulant_cayley_mask,
    build_dense_mask,
    build_hypercube_mask,
    build_local_mask,
    build_random_circulant_mask,
    build_window_dilation_mask,
    mask_stats,
    save_mask,
)


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=[
        "dense",
        "local",
        "hypercube",
        "circulant",
        "window_dilations",
        "random_circulant",
        "bigbird",
    ])
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--generators", type=parse_int_list, default=[])
    parser.add_argument("--dilations", type=parse_int_list, default=[16, 32, 64])
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--no_self", action="store_true")
    parser.add_argument("--global_tokens", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--num_random_blocks", type=int, default=1)
    parser.add_argument("--window_block_left", type=int, default=1)
    parser.add_argument("--window_block_right", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    include_self = not args.no_self
    include_inverse = not args.directed

    if args.kind == "dense":
        mask = build_dense_mask(args.seq)
    elif args.kind == "local":
        mask = build_local_mask(args.seq, window=args.window, include_self=include_self)
    elif args.kind == "hypercube":
        mask = build_hypercube_mask(args.seq, include_self=include_self)
    elif args.kind == "circulant":
        mask = build_circulant_cayley_mask(
            args.seq,
            generators=args.generators,
            include_inverse=include_inverse,
            include_self=include_self,
        )
    elif args.kind == "window_dilations":
        mask = build_window_dilation_mask(
            args.seq,
            window=args.window,
            dilations=args.dilations,
            include_inverse=include_inverse,
            include_self=include_self,
        )
    elif args.kind == "random_circulant":
        mask = build_random_circulant_mask(
            args.seq,
            degree=args.degree,
            seed=args.seed,
            include_inverse=include_inverse,
            include_self=include_self,
        )
    elif args.kind == "bigbird":
        config = BigBirdMaskConfig(
            global_tokens=args.global_tokens,
            block_size=args.block_size,
            num_random_blocks=args.num_random_blocks,
            window_block_left=args.window_block_left,
            window_block_right=args.window_block_right,
            seed=args.seed,
        )
        mask = build_bigbird_mask(args.seq, args.heads, config)
    else:
        raise AssertionError(args.kind)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_mask(mask, str(output))
    print(f"saved {args.kind} mask to {output}")
    print(mask_stats(mask))


if __name__ == "__main__":
    main()
