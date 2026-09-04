"""Validate pinned native model construction and parameter budgets without an accelerator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from benchmarks.matched.common import PARAMETER_TOLERANCE, SEED, TARGET_PARAMETERS


def within_budget(parameters: int) -> bool:
    return abs(parameters - TARGET_PARAMETERS) / TARGET_PARAMETERS <= PARAMETER_TOLERANCE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanochat-source", required=True, type=Path)
    parser.add_argument("--maxtext-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["NANOCHAT_DTYPE"] = "float32"
    import jax
    from flax import nnx
    from flaxchat.config import GPTConfig as FlaxConfig
    from flaxchat.gpt import GPT as FlaxGPT

    flax_model = FlaxGPT(
        FlaxConfig(
            sequence_len=256,
            vocab_size=512,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
            window_pattern="L",
            attention_backend="xla",
        ),
        rngs=nnx.Rngs(SEED),
    )
    counts = {
        "flaxchat": sum(
            int(value.size) for value in jax.tree.leaves(nnx.state(flax_model, nnx.Param))
        )
    }
    sys.path.insert(0, str(args.nanochat_source))
    from nanochat.gpt import GPT as NanoGPT  # pyright: ignore[reportMissingImports]
    from nanochat.gpt import GPTConfig as NanoConfig  # pyright: ignore[reportMissingImports]

    nano_model = NanoGPT(NanoConfig(
        sequence_len=256,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
    ))
    counts["nanochat"] = sum(parameter.numel() for parameter in nano_model.parameters())
    sys.path.insert(0, str(args.maxtext_source / "src"))
    from maxtext.configs import pyconfig  # pyright: ignore[reportMissingImports]
    from maxtext.utils import model_creation_utils  # pyright: ignore[reportMissingImports]

    config = pyconfig.initialize([
        "matched",
        str(args.maxtext_source / "src" / "maxtext" / "configs" / "base.yml"),
        "run_name=matched-preflight", "hardware=cpu", "model_name=default",
        "override_model_config=true", "attention=dot_product", "base_emb_dim=128",
        "base_num_query_heads=4", "base_num_kv_heads=4", "head_dim=32",
        "base_mlp_dim=448", "base_num_decoder_layers=2", "vocab_size=512",
        "max_target_length=256", "per_device_batch_size=8", "dtype=float32",
        "weight_dtype=float32", "grad_dtype=float32", "scan_layers=false",
        "pure_nnx_decoder=false", "ici_fsdp_parallelism=1", "dcn_data_parallelism=1",
        "enable_checkpointing=false", "skip_jax_distributed_system=true",
    ])
    maxtext_model, _mesh = model_creation_utils.create_nnx_model(
        config, devices=[jax.devices()[0]], rng_key=jax.random.PRNGKey(SEED)
    )
    counts["MaxText"] = sum(
        int(value.size) for value in jax.tree.leaves(nnx.state(maxtext_model, nnx.Param))
    )
    outside = {name: count for name, count in counts.items() if not within_budget(count)}
    if outside:
        raise RuntimeError(f"models outside parameter budget: {outside}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"parameter_counts": counts, "passed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
