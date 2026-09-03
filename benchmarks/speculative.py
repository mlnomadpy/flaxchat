"""Characterize greedy speculative decoding correctness and throughput."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import TypeVar

from flax import nnx
import jax

from flaxchat.config import GPTConfig
from flaxchat.engine import generate_speculative, generate_with_cache
from flaxchat.gpt import GPT


T = TypeVar("T")


def _timed(function: Callable[[], T], iterations: int) -> tuple[T, float]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    started = time.perf_counter()
    result = function()
    for _ in range(iterations - 1):
        result = function()
    return result, (time.perf_counter() - started) / iterations


def _align_synthetic_draft(main_model: GPT, draft_model: GPT) -> None:
    """Align the cheap draft's token/logit path for an acceptance upper bound.

    Fresh transformer residual projections are zero, so copying the shared
    embedding, smear, and output parameters makes the one-layer model predict
    the same greedy tokens as the deeper model. This is deliberately labeled
    synthetic; it measures mechanism overhead, not real draft quality.
    """
    assert main_model.lm_head is not None
    assert draft_model.lm_head is not None
    draft_model.wte.embedding[...] = main_model.wte.embedding[...]
    draft_model.lm_head.kernel[...] = main_model.lm_head.kernel[...]
    draft_model.smear_gate.kernel[...] = main_model.smear_gate.kernel[...]
    draft_model.smear_lambda[...] = main_model.smear_lambda[...]
    draft_model.backout_lambda[...] = main_model.backout_lambda[...]


def _measure_case(
    main_model: GPT,
    draft_model: GPT,
    prompt: list[int],
    baseline_tokens: list[int],
    baseline_seconds: float,
    *,
    max_tokens: int,
    draft_steps: int,
    iterations: int,
) -> dict[str, object]:
    def speculative():
        return generate_speculative(
            main_model,
            draft_model,
            prompt,
            max_tokens=max_tokens,
            temperature=0,
            draft_steps=draft_steps,
            return_stats=True,
        )

    speculative()
    (tokens, stats), seconds = _timed(speculative, iterations)
    return {
        "greedy_exact_match": tokens == baseline_tokens,
        "seconds": seconds,
        "tokens_per_second": max_tokens / seconds,
        "speedup": baseline_seconds / seconds,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--draft-steps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    sequence_length = 64
    def model_config(n_layer: int) -> GPTConfig:
        return GPTConfig(
            sequence_len=sequence_length,
            vocab_size=320,
            n_layer=n_layer,
            n_head=2,
            n_kv_head=1,
            n_embd=32,
            window_pattern="L",
        )

    main_model = GPT(model_config(4), rngs=nnx.Rngs(0))
    independent_draft = GPT(model_config(1), rngs=nnx.Rngs(1))
    aligned_draft = GPT(model_config(1), rngs=nnx.Rngs(2))
    _align_synthetic_draft(main_model, aligned_draft)
    prompt = [0, 10, 20, 30]
    baseline = lambda: generate_with_cache(
        main_model, prompt, max_tokens=args.max_tokens, temperature=0
    )
    baseline()  # compile warmup
    baseline_tokens, baseline_seconds = _timed(baseline, args.iterations)
    print(json.dumps({
        "jax_backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "max_tokens": args.max_tokens,
        "draft_steps": args.draft_steps,
        "iterations": args.iterations,
        "baseline": {
            "seconds": baseline_seconds,
            "tokens_per_second": args.max_tokens / baseline_seconds,
        },
        "cases": {
            "aligned_synthetic": _measure_case(
                main_model,
                aligned_draft,
                prompt,
                baseline_tokens,
                baseline_seconds,
                max_tokens=args.max_tokens,
                draft_steps=args.draft_steps,
                iterations=args.iterations,
            ),
            "independent_random": _measure_case(
                main_model,
                independent_draft,
                prompt,
                baseline_tokens,
                baseline_seconds,
                max_tokens=args.max_tokens,
                draft_steps=args.draft_steps,
                iterations=args.iterations,
            ),
        },
        "interpretation": (
            "aligned_synthetic is an acceptance upper bound; independent_random "
            "measures the rejection path, not a trained production draft"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
