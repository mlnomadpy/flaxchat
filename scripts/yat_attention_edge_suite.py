"""Reusable, single-client numerical gate for experimental YAT attention.

The supplied callable takes (q, k, v, segments, alpha, radius, query_tile).
Run in the benchmark process: a child process must not acquire the same TPU.
Passing this primitive gate does not qualify a model or precision policy.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.numerical_evidence import read_numerical_evidence, write_numerical_evidence
from tests.yat_attention_oracle import attention_and_gradients_tiled


FIELDS = ("output", "q", "k", "v", "alpha")
KINDS = (
    "random", "near", "identical_keys", "identical_values",
    "segment_keys", "segment_values", "padding",
)


def check_values(actual, expected, alpha_scale, *, kind):
    """Evaluate fixed oracle limits and exact softmax null-space invariants."""
    null_fields = (
        ("q", "alpha") if kind in ("identical_keys", "segment_keys") else
        ("q", "k", "alpha") if kind in ("identical_values", "segment_values") else
        ()
    )
    checks = {}
    for field, value, reference in zip(FIELDS, actual, expected, strict=True):
        value, reference = np.asarray(value, float), np.asarray(reference, float)
        if value.shape != reference.shape:
            raise ValueError(f"Unexpected {field} shape: {value.shape}")
        norm = float(np.linalg.norm(reference))
        error = float(np.linalg.norm(value - reference))
        limit = (.03 * norm if field == "output" else
                 .01 * alpha_scale + 1e-8 if field == "alpha" else
                 .08 * norm + 1e-8)
        finite = bool(np.isfinite(value).all() and np.isfinite(reference).all())
        null_exact = field not in null_fields or np.count_nonzero(value) == 0
        checks[field] = dict(
            error=error if np.isfinite(error) else None,
            limit=limit, null_required=field in null_fields,
            null_exact=bool(null_exact), passed=bool(finite and error <= limit and null_exact),
        )
    return checks


def validate_attention_edges(attention, output):
    """Save inputs and every result before rejecting; use the caller's JAX client.

126 cases cover packed documents, NaN padding, nearby coordinates, signed
trainable alpha, two query tiles, and global/local/self-only attention.
Inputs preserve device-rounded BF16 values for independent oracle replay.
"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(883)
    shape = (1, 65, 2, 64)
    x, random_k, v, g = [
        jnp.asarray(rng.normal(0, .2, shape), jnp.bfloat16) for _ in range(4)
    ]
    rows = []
    for radius in (None, 0, 8):
        for tile in (32, 64):
            def run(q, k, v, s, a, g, *, radius=radius, tile=tile):
                o, pull = jax.vjp(
                    lambda q, k, v, a: attention(q, k, v, s, a, radius, tile),
                    q, k, v, a,
                )
                return o, pull(g)

            fn = jax.jit(run)
            for alpha in (-.1, 0., .1):
                for kind in KINDS:
                    q, k, vv = x, random_k, v
                    s = jnp.asarray([[-1]*2 + [0]*29 + [1]*31 + [-1]*3], jnp.int32)
                    if kind == "padding":
                        s = jnp.full(shape[:2], -1, jnp.int32)
                    elif kind == "near":
                        k = x + jnp.bfloat16(.0002)
                    elif kind == "identical_keys":
                        k = jnp.broadcast_to(x[:, :1], shape)
                    elif kind == "identical_values":
                        vv = jnp.broadcast_to(v[:, :1], shape)
                    elif kind == "segment_keys":
                        k = jnp.where((s == 0)[..., None, None], random_k[:, :1], random_k[:, 1:2])
                    elif kind == "segment_values":
                        vv = jnp.where((s == 0)[..., None, None], v[:, :1], v[:, 1:2])
                    valid = (s >= 0)[..., None, None]
                    q, k, vv = [jnp.where(valid, z, jnp.nan) for z in (q, k, vv)]
                    a = jnp.float32(alpha)
                    o, grads = fn(q, k, vv, s, a, g)
                    clean = [np.where(np.asarray(valid), np.asarray(z, float), 0) for z in (q, k, vv)]
                    truth, tg, scale = attention_and_gradients_tiled(
                        clean[0], clean[1], clean[2], s, float(a), g, radius=radius,
                    )
                    checks = check_values((o, *grads), (truth, *tg), scale, kind=kind)
                    row = dict(kind=kind, radius=radius, tile=tile, alpha=alpha,
                               alpha_scale=scale, checks=checks,
                               passed=all(c["passed"] for c in checks.values()),
                               backend=jax.default_backend())
                    arrays = {f"input_{n}": z for n, z in zip(
                        ("q", "k", "v", "segments", "alpha", "go"), (q, k, vv, s, a, g), strict=True,
                    )}
                    for field, value, reference in zip(FIELDS, (o, *grads), (truth, *tg), strict=True):
                        arrays[f"actual_{field}"] = value
                        arrays[f"oracle_{field}"] = reference
                    write_numerical_evidence(output / f"case-{len(rows):03d}", arrays, metadata=row)
                    rows.append(row)
    if not all(row["passed"] for row in rows):
        raise ValueError(f"YAT edge validation failed; inspect {output}")
    return rows


def replay_attention_edges(output):
    """Recompute the independent oracle from saved inputs; distrust saved limits."""
    cases = sorted(Path(output).glob("case-*/manifest.json"))
    if len(cases) != 126:
        raise ValueError(f"Expected 126 completed edge cases, found {len(cases)}")
    results = []
    identities = set()
    for manifest in cases:
        arrays, receipt = read_numerical_evidence(manifest.parent)
        meta = receipt["metadata"]
        identity = (meta["kind"], meta["radius"], meta["tile"], meta["alpha"])
        identities.add(identity)
        s = arrays["input_segments"]
        valid = (s >= 0)[..., None, None]
        q, k, v = [np.where(valid, arrays[f"input_{n}"], 0) for n in ("q", "k", "v")]
        truth, grads, scale = attention_and_gradients_tiled(
            q, k, v, s, float(arrays["input_alpha"]), arrays["input_go"],
            radius=meta["radius"],
        )
        expected = (truth, *grads)
        for field, reference in zip(FIELDS, expected, strict=True):
            np.testing.assert_allclose(arrays[f"oracle_{field}"], reference, rtol=1e-12, atol=1e-12)
        checks = check_values(
            [arrays[f"actual_{field}"] for field in FIELDS], expected, scale, kind=meta["kind"],
        )
        if not all(c["passed"] for c in checks.values()):
            raise ValueError(f"Replayed edge gate failed: {manifest.parent}")
        results.append(dict(case=manifest.parent.name, checks=checks))
    required = {(kind, radius, tile, alpha) for kind in KINDS
                for radius in (None, 0, 8) for tile in (32, 64) for alpha in (-.1, 0., .1)}
    if identities != required:
        raise ValueError("Edge evidence does not cover the required case matrix")
    return results
