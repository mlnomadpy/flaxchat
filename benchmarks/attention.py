"""Emit machine-readable XLA/Splash latency and peak-memory measurements."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from flaxchat.gpt import exact_attention


def _memory_stats(compiled) -> dict[str, int | None]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"argument_bytes": None, "output_bytes": None, "temporary_bytes": None,
                "peak_bytes_estimate": None}
    arguments = int(stats.argument_size_in_bytes)
    outputs = int(stats.output_size_in_bytes)
    temporary = int(stats.temp_size_in_bytes)
    return {
        "argument_bytes": arguments,
        "output_bytes": outputs,
        "temporary_bytes": temporary,
        "peak_bytes_estimate": arguments + outputs + temporary,
    }


def benchmark(sequence_length: int, args) -> dict:
    shape = (1, sequence_length, args.heads, args.head_dim)
    q, k, v = jax.random.normal(jax.random.key(0), (3, *shape), dtype=jnp.float32)
    window = args.window_left or sequence_length
    run = jax.jit(
        lambda: exact_attention(q, k, v, window_left=window, backend=args.backend)
    )
    compile_started = time.perf_counter()
    compiled = run.lower().compile()
    compile_seconds = time.perf_counter() - compile_started
    for _ in range(args.warmup):
        compiled().block_until_ready()
    started = time.perf_counter()
    for _ in range(args.iterations):
        compiled().block_until_ready()
    elapsed = time.perf_counter() - started
    return {
        "sequence_length": sequence_length,
        "window_left": window,
        "compile_seconds": compile_seconds,
        "mean_seconds": elapsed / args.iterations,
        "tokens_per_second": sequence_length * args.iterations / elapsed,
        "memory": _memory_stats(compiled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        help="benchmark multiple lengths in one process (for example 1024 2048 4096 8192)",
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-left", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "xla", "splash"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    lengths = args.sequence_lengths or [args.sequence_length]
    print(json.dumps({
        "backend": args.backend,
        "jax_backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": "float32",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "measurements": [benchmark(length, args) for length in lengths],
    }, indent=2))


if __name__ == "__main__":
    main()
