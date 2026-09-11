"""Run the matched trainer protocol with flaxchat's native model."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import platform
from pathlib import Path
import statistics
import time

import flax
from flax import nnx
import jax
import optax

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
from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from flaxchat.config import GPTConfig
from flaxchat.gpt import GPT


@nnx.jit
def train_step(model, optimizer, inputs, targets):
    loss, gradients = nnx.value_and_grad(lambda current: current(inputs, targets))(model)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def validation_step(model, inputs, targets):
    return model(inputs, targets)


def compiled_peak_bytes(compiled) -> int:
    stats = compiled.memory_analysis()
    if stats is None:
        raise RuntimeError("JAX executable did not provide memory analysis")
    return max(
        1,
        int(stats.argument_size_in_bytes)
        + int(stats.output_size_in_bytes)
        + int(stats.temp_size_in_bytes)
        - int(stats.alias_size_in_bytes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    batches = load_batches(args.data)
    device = jax.devices()[0]
    validate_hardware(device.device_kind)
    config = GPTConfig(
        sequence_len=256,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
        attention_backend="xla",
    )
    model = GPT(config, rngs=nnx.Rngs(SEED))
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    parameters = sum(int(value.size) for value in jax.tree.leaves(nnx.state(model, nnx.Param)))
    first_inputs = jax.device_put(batches["train_inputs"][0])
    first_targets = jax.device_put(batches["train_targets"][0])
    started = time.perf_counter()
    compiled = train_step.lower(model, optimizer, first_inputs, first_targets).compile()
    compile_seconds = time.perf_counter() - started
    jax.block_until_ready(compiled(model, optimizer, first_inputs, first_targets))
    for index in range(1, WARMUP_STEPS + 1):
        inputs = jax.device_put(batches["train_inputs"][index])
        targets = jax.device_put(batches["train_targets"][index])
        jax.block_until_ready(compiled(model, optimizer, inputs, targets))
    durations = []
    for index in range(WARMUP_STEPS + 1, WARMUP_STEPS + MEASURED_STEPS + 1):
        inputs = jax.device_put(batches["train_inputs"][index])
        targets = jax.device_put(batches["train_targets"][index])
        started = time.perf_counter()
        jax.block_until_ready(compiled(model, optimizer, inputs, targets))
        durations.append(time.perf_counter() - started)
    validation = validation_step(
        model,
        jax.device_put(batches["validation_inputs"]),
        jax.device_put(batches["validation_targets"]),
    )
    validation_value = float(jax.block_until_ready(validation))
    manager = create_checkpoint_manager(args.checkpoint_dir, async_checkpointing=False)
    started = time.perf_counter()
    save_checkpoint(manager, MEASURED_STEPS, model, optimizer, {"benchmark": "matched-v1"})
    manager.close()
    checkpoint_seconds = time.perf_counter() - started
    median_seconds = statistics.median(durations)
    record = make_record(
        framework="flaxchat",
        source_revision=args.revision,
        model_parameters=parameters,
        steady_tokens_per_second=(8 * 256) / median_seconds,
        compile_seconds=compile_seconds,
        peak_memory_bytes=compiled_peak_bytes(compiled),
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
        limitations="Peak memory is the executable memory-analysis estimate, not a process-wide HBM watermark.",
    )
    write_record(args.output, record)


if __name__ == "__main__":
    main()
