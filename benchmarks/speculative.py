"""Characterize greedy speculative decoding correctness and throughput."""

from __future__ import annotations

import argparse
import json
import time

from flax import nnx
import jax

from flaxchat.config import GPTConfig
from flaxchat.engine import generate_speculative, generate_with_cache
from flaxchat.gpt import GPT


def _timed(function, iterations: int):
    started = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = function()
    return result, (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--draft-steps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    sequence_length = 64
    common = dict(
        sequence_len=sequence_length,
        vocab_size=320,
        n_head=2,
        n_kv_head=1,
        n_embd=32,
        window_pattern="L",
    )
    main_model = GPT(GPTConfig(n_layer=4, **common), rngs=nnx.Rngs(0))
    draft_model = GPT(GPTConfig(n_layer=1, **common), rngs=nnx.Rngs(1))
    prompt = [0, 10, 20, 30]
    baseline = lambda: generate_with_cache(
        main_model, prompt, max_tokens=args.max_tokens, temperature=0
    )
    speculative = lambda: generate_speculative(
        main_model,
        draft_model,
        prompt,
        max_tokens=args.max_tokens,
        temperature=0,
        draft_steps=args.draft_steps,
        return_stats=True,
    )
    baseline()  # compile warmup
    speculative()
    baseline_tokens, baseline_seconds = _timed(baseline, args.iterations)
    (speculative_tokens, stats), speculative_seconds = _timed(
        speculative, args.iterations
    )
    print(json.dumps({
        "jax_backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "max_tokens": args.max_tokens,
        "draft_steps": args.draft_steps,
        "iterations": args.iterations,
        "greedy_exact_match": speculative_tokens == baseline_tokens,
        "baseline_seconds": baseline_seconds,
        "speculative_seconds": speculative_seconds,
        "speedup": baseline_seconds / speculative_seconds,
        "baseline_tokens_per_second": args.max_tokens / baseline_seconds,
        "speculative_tokens_per_second": args.max_tokens / speculative_seconds,
        "stats": stats,
    }, indent=2))


if __name__ == "__main__":
    main()
