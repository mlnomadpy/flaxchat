"""Measure flaxchat single-host strong scaling with reproducible controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
import statistics
import platform
from pathlib import Path
import subprocess
import tempfile
import time

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
import flax
import jaxlib

from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from flaxchat.common import (
    COMPUTE_DTYPE,
    get_peak_flops,
    replicate_on_mesh,
    replicate_optimizer_state,
)
from flaxchat.config import GPTConfig
from flaxchat.gpt import GPT


@partial(nnx.jit, donate_argnames=("optimizer",))
def _train_step(model, optimizer, inputs, targets):
    loss, gradients = nnx.value_and_grad(lambda current: current(inputs, targets))(model)
    optimizer.update(model, gradients)
    return loss


def _source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def software_metadata() -> dict[str, str]:
    """Capture compiler-facing package versions in every benchmark record."""
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "flax": flax.__version__,
        "optax": optax.__version__,
    }


def validate_device_counts(requested: list[int], available: int) -> list[int]:
    counts = sorted(set(requested))
    if not counts or counts[0] != 1:
        raise ValueError("device counts must include the one-device baseline")
    if any(count <= 0 or count > available for count in counts):
        raise ValueError(f"device counts must be between 1 and {available}")
    return counts


def add_scaling_efficiency(
    measurements: list[dict], *, minimum_efficiency: float = 0.5
) -> list[dict]:
    if not 0.0 <= minimum_efficiency <= 1.0:
        raise ValueError("minimum efficiency must be between 0 and 1")
    baseline = measurements[0]
    if baseline["device_count"] != 1:
        raise ValueError("first measurement must be the one-device baseline")
    baseline_throughput = baseline["steady_tokens_per_second"]
    for measurement in measurements:
        ideal = baseline_throughput * measurement["device_count"]
        measurement["scaling_efficiency"] = (
            measurement["steady_tokens_per_second"] / ideal
        )
        measurement["meets_efficiency_threshold"] = (
            measurement["scaling_efficiency"] >= minimum_efficiency
        )
    return measurements


def _memory_dict(stats) -> dict[str, int]:
    result = {
        "argument_bytes": int(stats.argument_size_in_bytes),
        "output_bytes": int(stats.output_size_in_bytes),
        "temporary_bytes": int(stats.temp_size_in_bytes),
        "alias_bytes": int(stats.alias_size_in_bytes),
    }
    result["peak_bytes_estimate"] = (
        result["argument_bytes"]
        + result["output_bytes"]
        + result["temporary_bytes"]
        - result["alias_bytes"]
    )
    return result


def _benchmark_count(
    device_count: int,
    config: GPTConfig,
    *,
    global_batch_size: int,
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
    mode: str,
) -> dict:
    devices = np.asarray(jax.devices()[:device_count], dtype=object)
    mesh = Mesh(devices, axis_names=("data",))
    data_sharding = NamedSharding(mesh, P("data"))

    model = GPT(config, rngs=nnx.Rngs(seed))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    replicate_optimizer_state(optimizer, mesh)

    key = jax.random.key(seed + 1)
    inputs = jax.random.randint(
        key, (global_batch_size, config.sequence_len), 0, config.vocab_size
    )
    targets = jnp.roll(inputs, -1, axis=1)
    inputs = jax.device_put(inputs, data_sharding)
    targets = jax.device_put(targets, data_sharding)

    compile_started = time.perf_counter()
    compiled = _train_step.lower(model, optimizer, inputs, targets).compile()
    compile_seconds = time.perf_counter() - compile_started
    memory = _memory_dict(compiled.memory_analysis())

    losses: list[float] = []
    for _ in range(warmup):
        jax.block_until_ready(compiled(model, optimizer, inputs, targets))
    trial_step_seconds = []
    for _trial in range(trials):
        measured_started = time.perf_counter()
        for _ in range(iterations):
            loss = compiled(model, optimizer, inputs, targets)
            losses.append(float(jax.block_until_ready(loss)))
        trial_step_seconds.append(
            (time.perf_counter() - measured_started) / iterations
        )
    steady_seconds = statistics.median(trial_step_seconds)
    tokens_per_step = global_batch_size * config.sequence_len
    throughput = tokens_per_step / steady_seconds

    with tempfile.TemporaryDirectory(prefix="flaxchat-scaling-checkpoint-") as directory:
        manager = create_checkpoint_manager(directory, async_checkpointing=False)
        checkpoint_started = time.perf_counter()
        save_checkpoint(
            manager,
            iterations,
            model,
            optimizer,
            {"benchmark": f"single-host-{mode}-scaling", "device_count": device_count},
        )
        manager.close()
        checkpoint_seconds = time.perf_counter() - checkpoint_started

    parameters = sum(
        int(value.size) for value in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    peak_flops = get_peak_flops(jax.devices()[0].device_kind) * device_count
    model_flops_utilization = (
        6.0 * parameters * throughput / peak_flops
        if np.isfinite(peak_flops)
        else 0.0
    )
    return {
        "device_count": device_count,
        "model_parameters": parameters,
        "global_batch_size": global_batch_size,
        "input_shape": [global_batch_size, config.sequence_len],
        "tokens_per_step": tokens_per_step,
        "compile_seconds": compile_seconds,
        "steady_step_seconds": steady_seconds,
        "trial_step_seconds": trial_step_seconds,
        "step_seconds_population_stddev": statistics.pstdev(trial_step_seconds),
        "steady_tokens_per_second": throughput,
        "model_flops_utilization_estimate": model_flops_utilization,
        "compiled_memory": memory,
        "checkpoint_seconds": checkpoint_seconds,
        "losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=("strong", "weak"), default="strong")
    parser.add_argument(
        "--per-device-batch-size", type=int, default=8,
        help="used only for weak scaling",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--minimum-efficiency",
        type=float,
        default=0.5,
        help="flag non-baseline points below this scaling-efficiency threshold",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.global_batch_size <= 0
        or args.per_device_batch_size <= 0
        or args.warmup < 0
        or args.iterations < 1
        or args.trials < 2
        or not 0.0 <= args.minimum_efficiency <= 1.0
    ):
        parser.error("batch size and iterations must be positive; warmup cannot be negative")

    counts = validate_device_counts(args.device_counts, jax.device_count())
    if args.mode == "strong" and any(args.global_batch_size % count for count in counts):
        parser.error("global batch size must be divisible by every requested device count")
    config = GPTConfig(
        sequence_len=args.sequence_length,
        vocab_size=args.vocab_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.heads,
        n_embd=args.embedding_dim,
        window_pattern="L",
        attention_backend="xla",
    )
    measurements = add_scaling_efficiency([
        _benchmark_count(
            count,
            config,
            global_batch_size=(
                args.global_batch_size
                if args.mode == "strong"
                else args.per_device_batch_size * count
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            trials=args.trials,
            seed=args.seed,
            mode=args.mode,
        )
        for count in counts
    ], minimum_efficiency=args.minimum_efficiency)
    result = {
        "format_version": 1,
        "benchmark": f"single-host-{args.mode}-scaling",
        "source_revision": _source_revision(),
        "hardware": {
            "backend": jax.default_backend(),
            "device_kind": jax.devices()[0].device_kind,
            "available_device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "host_count": jax.process_count(),
            "topology": [str(device) for device in jax.devices()],
        },
        "software": software_metadata(),
        "model_config": asdict(config),
        "controls": {
            "global_batch_size": args.global_batch_size,
            "per_device_batch_size": (
                args.per_device_batch_size if args.mode == "weak" else None
            ),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "seed": args.seed,
            "precision": jnp.dtype(COMPUTE_DTYPE).name,
            "minimum_efficiency": args.minimum_efficiency,
            "synchronization": "jax.block_until_ready after every measured step",
        },
        "measurements": measurements,
        "efficiency_regressions": [
            measurement["device_count"]
            for measurement in measurements[1:]
            if not measurement["meets_efficiency_threshold"]
        ],
        "limitations": [
            "The MFU value uses the conventional 6*N*tokens approximation.",
            "Compiled memory is argument plus output plus temporary bytes minus "
            "aliases, not a device-wide HBM watermark.",
            f"Device subsets share one physical host, so this measures {args.mode} "
            "device scaling but does not establish inter-host efficiency.",
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
