"""Run the matched trainer protocol with MaxText's native NNX model."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import platform
from pathlib import Path
import statistics
import sys
import time

from benchmarks.compare import protocol_sha256
from benchmarks.matched.common import (
    MEASURED_STEPS,
    SEED,
    WARMUP_STEPS,
    file_sha256,
    load_batches,
    make_record,
    validate_hardware,
    write_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "src"))

    import flax
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax
    from orbax import checkpoint as ocp
    from maxtext.configs import pyconfig  # pyright: ignore[reportMissingImports]
    from maxtext.utils import model_creation_utils  # pyright: ignore[reportMissingImports]

    device = jax.devices()[0]
    validate_hardware(device.device_kind)
    config_path = args.source / "src" / "maxtext" / "configs" / "base.yml"
    config = pyconfig.initialize([
        "matched",
        str(config_path),
        "run_name=matched",
        "hardware=gpu",
        "model_name=default",
        "override_model_config=true",
        "attention=dot_product",
        "base_emb_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=32",
        "base_mlp_dim=448",
        "base_num_decoder_layers=2",
        "vocab_size=512",
        "max_target_length=256",
        "per_device_batch_size=8",
        "dtype=float32",
        "weight_dtype=float32",
        "grad_dtype=float32",
        "scan_layers=false",
        "pure_nnx_decoder=false",
        "ici_fsdp_parallelism=1",
        "dcn_data_parallelism=1",
        "enable_checkpointing=false",
        "skip_jax_distributed_system=true",
    ])
    model, _mesh = model_creation_utils.create_nnx_model(
        config, devices=[device], rng_key=jax.random.PRNGKey(SEED)
    )
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    parameters = sum(int(value.size) for value in jax.tree.leaves(nnx.state(model, nnx.Param)))
    batches = load_batches(args.data)
    positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (8, 256))
    segments = jnp.ones((8, 256), dtype=jnp.int32)

    @nnx.jit
    def train_step(current_model, current_optimizer, inputs, targets):
        def loss_fn(candidate):
            logits = candidate(
                inputs,
                positions,
                decoder_segment_ids=segments,
                enable_dropout=False,
            )
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        loss, gradients = nnx.value_and_grad(loss_fn)(current_model)
        current_optimizer.update(current_model, gradients)
        return loss

    @nnx.jit
    def validation_step(current_model, inputs, targets):
        logits = current_model(
            inputs,
            positions,
            decoder_segment_ids=segments,
            enable_dropout=False,
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    first_inputs = jax.device_put(batches["train_inputs"][0], device)
    first_targets = jax.device_put(batches["train_targets"][0], device)
    started = time.perf_counter()
    compiled = train_step.lower(model, optimizer, first_inputs, first_targets).compile()
    compile_seconds = time.perf_counter() - started
    jax.block_until_ready(compiled(model, optimizer, first_inputs, first_targets))
    for index in range(1, WARMUP_STEPS + 1):
        jax.block_until_ready(compiled(
            model,
            optimizer,
            jax.device_put(batches["train_inputs"][index], device),
            jax.device_put(batches["train_targets"][index], device),
        ))
    durations = []
    for index in range(WARMUP_STEPS + 1, WARMUP_STEPS + MEASURED_STEPS + 1):
        inputs = jax.device_put(batches["train_inputs"][index], device)
        targets = jax.device_put(batches["train_targets"][index], device)
        started = time.perf_counter()
        jax.block_until_ready(compiled(model, optimizer, inputs, targets))
        durations.append(time.perf_counter() - started)
    validation_value = float(jax.block_until_ready(validation_step(
        model,
        jax.device_put(batches["validation_inputs"], device),
        jax.device_put(batches["validation_targets"], device),
    )))
    args.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    started = time.perf_counter()
    checkpointer.save(
        args.checkpoint_dir.resolve(),
        {
            "model": nnx.to_pure_dict(nnx.state(model)),
            "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
        },
        force=True,
    )
    close = getattr(checkpointer, "close", None)
    if close:
        close()
    checkpoint_seconds = time.perf_counter() - started
    stats = compiled.memory_analysis()
    if stats is None:
        raise RuntimeError("JAX executable did not provide memory analysis")
    peak_memory = (
        int(stats.argument_size_in_bytes)
        + int(stats.output_size_in_bytes)
        + int(stats.temp_size_in_bytes)
        - int(stats.alias_size_in_bytes)
    )
    record = make_record(
        framework="MaxText",
        source_revision=args.revision,
        model_parameters=parameters,
        steady_tokens_per_second=(8 * 256) / statistics.median(durations),
        compile_seconds=compile_seconds,
        peak_memory_bytes=peak_memory,
        checkpoint_seconds=checkpoint_seconds,
        validation_value=validation_value,
        protocol_sha256=protocol_sha256(args.protocol),
        data_sha256=file_sha256(args.data),
        software={
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": version("jaxlib"),
            "flax": flax.__version__,
            "optax": optax.__version__,
        },
        limitations="Uses MaxText's native Transformer with an isolated protocol AdamW step; peak memory is executable analysis.",
    )
    write_record(args.output, record)


if __name__ == "__main__":
    main()
