"""Emit machine-readable XLA/Splash attention latency measurements."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from flaxchat.gpt import exact_attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-left", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "xla", "splash"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    shape = (1, args.sequence_length, args.heads, args.head_dim)
    q, k, v = jax.random.normal(jax.random.key(0), (3, *shape), dtype=jnp.float32)
    window = args.window_left or args.sequence_length
    run = jax.jit(lambda: exact_attention(q, k, v, window_left=window, backend=args.backend))
    for _ in range(args.warmup):
        run().block_until_ready()
    started = time.perf_counter()
    for _ in range(args.iterations):
        run().block_until_ready()
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "backend": args.backend,
        "jax_backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "sequence_length": args.sequence_length,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "window_left": window,
        "iterations": args.iterations,
        "mean_seconds": elapsed / args.iterations,
    }, indent=2))


if __name__ == "__main__":
    main()
