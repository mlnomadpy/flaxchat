"""Replay saved MLM numerical tensors; never treat CPU replay as TPU qualification."""

import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from jax.sharding import Mesh

from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.runtime import runtime_identity
from flaxchat.profiling import TrainingTrace
from flaxchat.training import (
    gradients_for_mlm_microbatches,
    gradients_for_local_mlm_microbatches,
)
from scripts.numerical_evidence import read_numerical_evidence
from scripts.diagnostic_boundaries import MODES, with_boundaries


def restore_parameters(model, arrays, paths):
    """Restore by verified tree paths and shapes, never by evaluating path strings."""
    state = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree_util.tree_flatten_with_path(state)
    expected = {f"parameter_{i}" for i in range(len(leaves))}
    if {k for k in paths if k.startswith("parameter_")} != expected:
        raise ValueError("Incomplete or extra captured parameter leaves")
    values = []
    for i, (path, value) in enumerate(leaves):
        name = f"parameter_{i}"
        a = arrays.get(name)
        if (
            paths[name] != jax.tree_util.keystr(path)
            or a is None
            or a.shape != value.shape
            or a.dtype != np.dtype(value.dtype)
        ):
            raise ValueError(
                f"Captured parameter path, shape or dtype mismatch: {name}"
            )
        values.append(jnp.asarray(a))
    nnx.update(model, jax.tree_util.tree_unflatten(tree, values))


def flatten(tree):
    return {
        jax.tree_util.keystr(k): np.asarray(v, dtype=np.float32)
        for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def compare_gradients(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("Gradient paths differ")
    rows = []
    for path, a in reference.items():
        b = candidate[path]
        if a.shape != b.shape:
            raise ValueError("Gradient shapes differ")
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        aa, bb = a.astype(float), b.astype(float)
        rows.append(
            dict(
                path=path,
                finite=finite,
                exact=bool(np.array_equal(a, b)),
                absolute_max=float(np.max(np.abs(aa - bb))) if finite else None,
                relative_l2=float(
                    np.linalg.norm(aa - bb) / max(np.linalg.norm(aa), 1e-30)
                )
                if finite
                else None,
            )
        )
    return rows


def per_example_reference(model, x, y):
    """Independent eager sum of per-example derivatives; no accumulation helper.

    Run on the same backend as the candidate. This preserves model arithmetic
    and sums gradients in FP32; it is not an FP64 derivative oracle.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 3 or x.shape != y.shape or not all(x.shape):
        raise ValueError("Require nonempty matching microstep/batch/sequence inputs")
    # A separate module graph keeps execution flags out of the source model.
    graph, state = nnx.split(model)
    local = nnx.merge(graph, state)
    local.config = replace(model.config, yat_local_shards=False)
    local._local_projection_loss = True
    for layer in local.layers:
        layer.config = local.config
    fn = nnx.jit(nnx.value_and_grad(lambda m, xx, yy: m(xx, yy, loss_reduction="sum")))
    sums = {
        k: np.zeros_like(v) for k, v in flatten(nnx.state(model, nnx.Param)).items()
    }
    total, count = 0.0, 0
    for xx, yy in zip(
        x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), strict=True
    ):
        selected = int(((yy >= 0) & (xx != model.config.pad_token_id)).sum())
        if not selected:
            continue
        loss, grads = fn(local, jnp.asarray(xx[None]), jnp.asarray(yy[None]))
        total += float(loss)
        count += selected
        for key, value in flatten(grads).items():
            sums[key] += value
    per_example = {k: v / max(count, 1) for k, v in sums.items()}
    return total / max(count, 1), per_example, count


def replay(directory, *, per_example_only=False, profile_directory=None, boundary_mode="baseline"):
    if per_example_only and profile_directory is not None:
        raise ValueError("Profiling requires both distributed execution paths")
    arrays, receipt = read_numerical_evidence(directory)
    metadata = receipt["metadata"]
    config = EncoderConfig(**metadata["config"])
    x, y = arrays["inputs"], arrays["targets"]
    if x.ndim != 3 or x.shape != y.shape or not all(x.shape):
        raise ValueError("Require nonempty matching microstep/batch/sequence inputs")
    if not per_example_only and len(metadata["devices"]) != jax.device_count():
        raise ValueError("Distributed replay requires the captured device count")
    model = ModernBert(config, rngs=nnx.Rngs(0))
    restore_parameters(model, arrays, metadata["leaf_paths"])
    model = with_boundaries(model, boundary_mode)
    replayed = {}
    captured = {}
    for prefix in ("reference", "candidate"):
        captured[prefix] = {
            path: arrays[name]
            for name, path in metadata["leaf_paths"].items()
            if name.startswith(prefix + "_")
        }
        names = [
            name for name in metadata["leaf_paths"] if name.startswith(prefix + "_")
        ]
        if (
            len(names) != len(captured[prefix])
            or captured[prefix].keys() != flatten(nnx.state(model, nnx.Param)).keys()
        ):
            raise ValueError("Captured gradient paths do not match model")
    result = dict(
        boundary_mode=boundary_mode,
        capture_backend=metadata["backend"],
        capture_reference_kind=metadata.get("reference_kind", "legacy_batched"),
        backend=jax.default_backend(),
        capture_runtime=metadata["runtime"],
        replay_runtime=runtime_identity(),
        evidence_archive_sha256=receipt["archive_sha256"],
        comparisons={},
        losses={},
        production_qualified=False,
        scope="Saved parameters and inputs replayed exactly; compiled arithmetic can differ by platform/version. Per-example BF16 gradients summed in FP32 are a reduction reference, not an FP64 oracle or TPU qualification.",
        effective_matmul_precision=jax.config.values["jax_default_matmul_precision"],
    )
    if not per_example_only:
        mesh = Mesh(np.asarray(jax.devices()), ("data",))
        for name, fn in [
            ("reference", gradients_for_mlm_microbatches),
            ("candidate", partial(gradients_for_local_mlm_microbatches, mesh=mesh)),
        ]:
            compiled = nnx.jit(fn)
            xx, yy = jnp.asarray(x), jnp.asarray(y)
            loss, grads = compiled(model, xx, yy)
            jax.block_until_ready((loss, grads))
            if profile_directory is not None:
                trace = TrainingTrace(Path(profile_directory) / name, skip=1, steps=3)
                try:
                    for step in range(4):
                        with trace.step(step):
                            jax.block_until_ready(compiled(model, xx, yy))
                finally:
                    trace.close()
            replayed[name] = flatten(grads)
            result["losses"][name] = float(loss)
            result["comparisons"][name + "_capture_vs_replay"] = compare_gradients(
                captured[name], flatten(grads)
            )
    mean_loss, per_example, count = per_example_reference(model, x, y)
    result["losses"]["per_example"] = mean_loss
    result["masked_targets"] = count
    for name in captured:
        result["comparisons"][name + "_vs_per_example"] = compare_gradients(
            per_example, captured[name]
        )
    result["replayed_reference_gates"] = {}
    for name, gradients in replayed.items():
        result["comparisons"][name + "_replay_vs_per_example"] = compare_gradients(
            per_example, gradients
        )
        result["replayed_reference_gates"][name] = dict(
            gradients_passed=all(np.isfinite(gradients[k]).all() and np.isfinite(v).all()
                                 and np.allclose(gradients[k], v, atol=.003, rtol=.04)
                                 for k, v in per_example.items()),
            loss_passed=bool(np.isfinite(result["losses"][name]) and np.isfinite(mean_loss)
                             and np.isclose(result["losses"][name], mean_loss,
                                       atol=2e-5, rtol=2e-4)),
            gradient_atol=.003, gradient_rtol=.04, loss_atol=2e-5, loss_rtol=2e-4,
        )
    result["loss_finite"] = {k: bool(np.isfinite(v)) for k, v in result["losses"].items()}
    result["losses"] = {k: v if result["loss_finite"][k] else None
                        for k, v in result["losses"].items()}
    result["replay_source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*Path("flaxchat").glob("*.py"), Path(__file__), Path("scripts/diagnostic_boundaries.py")]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-example-only", action="store_true")
    parser.add_argument("--profile-directory", type=Path,
                        help="Capture three warmed steps per distributed path; not a throughput benchmark")
    parser.add_argument("--boundary-mode", choices=MODES, default="baseline",
                        help="Experimental boundaries; reference arithmetic changes too")
    parser.add_argument("--require-reference-gates", action="store_true",
                        help="Write evidence then fail if a fresh distributed reference gate fails")
    args = parser.parse_args()
    if args.require_reference_gates and args.per_example_only:
        parser.error("--require-reference-gates needs distributed replay")
    result = replay(args.evidence, per_example_only=args.per_example_only,
                    profile_directory=args.profile_directory, boundary_mode=args.boundary_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.require_reference_gates:
        gates = result["replayed_reference_gates"]
        if set(gates) != {"reference", "candidate"} or not all(
            g["gradients_passed"] and g["loss_passed"] for g in gates.values()
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
