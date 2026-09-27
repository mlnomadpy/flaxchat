"""Experimental centered YAT attention with an explicit FP32 score boundary.

Bias is fixed at 1 and epsilon at .01; alpha remains differentiable. Geometry,
coordinate differences and weighted coordinate reductions are BF16. Scores,
softmax and coefficient state are FP32; TPU matmuls accumulate internally in
FP32. This backend is not strict all-BF16 and is not a fused FlashAttention.
"""

from functools import partial
from typing import cast
import jax
import jax.numpy as jnp
from flaxchat.yat import squared_distance_bf16
from flaxchat.yat_attention import windowed_yat_attention


def _compensated_add(total, correction, value):
    """Neumaier accumulation in the operands' dtype, including BF16."""
    # Reassociation must not erase the rounded sum's correction.
    summed = jax.lax.optimization_barrier(total + value)
    error = jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - summed) + value,
        (value - summed) + total,
    )
    return summed, correction + error


def _block(q, k, v, qs, ks, alpha, cotangent, qpositions, kpositions, *, radius=None):
    valid = qs >= 0
    key_valid = ks >= 0
    q, g = [
        cast(jax.Array, jnp.where(valid[..., None, None], x, 0)).transpose(0, 2, 1, 3)
        for x in (q, cotangent)
    ]
    k, v = [
        cast(jax.Array, jnp.where(key_valid[..., None, None], x, 0)).transpose(0, 2, 1, 3)
        for x in (k, v)
    ]
    z = jnp.matmul(
        q, jnp.swapaxes(k, -1, -2), preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    distance = squared_distance_bf16(q, k, block=32)
    den = distance.astype(jnp.float32) + 0.01
    zf = z.astype(jnp.float32)
    aa = jnp.asarray(alpha, jnp.float32)
    mask = (
        (qs[:, :, None] == ks[:, None, :]) & valid[:, :, None] & key_valid[:, None, :]
    )
    if radius is not None:
        mask &= jnp.abs(qpositions[:, None] - kpositions[None, :]) <= radius
    mask |= (~valid)[:, :, None] & (jnp.arange(k.shape[2]) == 0)[None, None, :]
    scores = jnp.where(mask[:, None], aa * jnp.square(zf + 1) / den, -jnp.inf)
    p = jax.nn.softmax(scores, axis=-1)
    gp = jnp.matmul(g, jnp.swapaxes(v, -1, -2), preferred_element_type=jnp.float32)
    index = jnp.argmax(scores, -1)
    selected = jnp.arange(k.shape[2]) == index[..., None]

    def reference_value(value):
        # Exactly one selected finite element per row; retain FP32 scalar state.
        # Avoid TPU scalar gathers, observed to dominate reference selection.
        return jnp.sum(cast(jax.Array, jnp.where(selected, value, 0)), axis=-1, keepdims=True)

    gp = gp - reference_value(gp)
    gs = p * (gp - jnp.sum(p * gp, -1, keepdims=True))
    a = 2 * aa * (zf + 1) / den
    b = -2 * aa * jnp.square(zf + 1) / jnp.square(den)
    # A detached maximum-probability key provides a nearby reference without
    # changing the real derivative: sum_j dL/ds_j = 0 for a softmax row.
    # A one-hot BF16 MXU projection selects the same key without a scalar-index
    # gather. Only the internal matmul accumulator is FP32.
    anchor = jnp.matmul(
        selected.astype(jnp.bfloat16), k, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    a0 = reference_value(a)
    b0 = reference_value(b)
    w1 = (gs * (a - b)).astype(jnp.bfloat16)
    w2 = (gs * (a - a0)).astype(jnp.bfloat16)
    w3 = (gs * (b - b0)).astype(jnp.bfloat16)
    # Keep subtraction of nearby coordinates in BF16. Dense scratch is bounded
    # by feature blocks here, but still quadratic in sequence length.
    parts = []
    for start in range(0, q.shape[-1], 32):
        ref = anchor[..., start : start + 32]
        dk = k[..., None, :, start : start + 32] - ref[..., None, :]
        dq = q[..., start : start + 32] - ref
        value = jnp.sum(w1[..., None] * dk, axis=-2, dtype=jnp.bfloat16)
        value += jnp.sum(w2, -1, dtype=jnp.bfloat16)[..., None] * ref
        value += jnp.sum(w3, -1, dtype=jnp.bfloat16)[..., None] * dq
        parts.append(value)
    gq = jnp.concatenate(parts, -1).transpose(0, 2, 1, 3)
    gz = (gs * a).astype(jnp.bfloat16)
    gb = (gs * b).astype(jnp.bfloat16)
    gk_dot = jnp.matmul(
        jnp.swapaxes(gz, -1, -2), q, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    key_parts = []
    for start in range(0, q.shape[-1], 32):
        delta = (
            q[..., :, None, start : start + 32] - k[..., None, :, start : start + 32]
        )
        weighted = gb[..., None] * delta
        key_parts.append(
            gk_dot[..., start : start + 32]
            - jnp.sum(weighted, axis=-3, dtype=jnp.bfloat16)
        )
    gk = jnp.concatenate(key_parts, -1).transpose(0, 2, 1, 3)
    gv = (
        jnp.matmul(
            jnp.swapaxes(p.astype(jnp.bfloat16), -1, -2),
            g,
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .transpose(0, 2, 1, 3)
    )
    kernel = jnp.square(zf + 1) / den
    reference = reference_value(kernel)
    ga = jnp.sum(gs * (kernel - reference), dtype=jnp.float32)
    return (
        jnp.where(valid[..., None, None], gq, 0),
        jnp.where(key_valid[..., None, None], gk, 0),
        jnp.where(key_valid[..., None, None], gv, 0),
        ga,
    )


def centered_backward(
    q, k, v, segments, alpha, cotangent, *, radius=None, query_tile=64,
    compensate_kv=False,
):
    """Query/window-tiled version; no full sequence-squared coordinate scratch."""
    if query_tile < 1 or (radius is not None and radius < 0):
        raise ValueError("Positive query tile and nonnegative radius required")
    length = q.shape[1]
    blocks = (length + query_tile - 1) // query_tile
    padded = blocks * query_tile
    halo = radius if radius is not None else 0
    keys = query_tile + 2 * halo if radius is not None else length
    qp, gp = [
        jnp.pad(x, ((0, 0), (0, padded - length), (0, 0), (0, 0)))
        for x in (q, cotangent)
    ]
    kp, vp = [
        jnp.pad(
            x,
            (
                (0, 0),
                (halo, padded - length + halo if radius is not None else 0),
                (0, 0),
                (0, 0),
            ),
        )
        for x in (k, v)
    ]
    qs = jnp.pad(segments, ((0, 0), (0, padded - length)), constant_values=-1)
    ks = jnp.pad(
        segments,
        ((0, 0), (halo, padded - length + halo if radius is not None else 0)),
        constant_values=-1,
    )

    def step(i, result):
        start = i * query_tile
        key_start = start if radius is not None else 0
        qq, gg = [
            jax.lax.dynamic_slice_in_dim(x, start, query_tile, axis=1) for x in (qp, gp)
        ]
        kk, vv = [
            jax.lax.dynamic_slice_in_dim(x, key_start, keys, axis=1) for x in (kp, vp)
        ]
        sq = jax.lax.dynamic_slice_in_dim(qs, start, query_tile, axis=1)
        sk = jax.lax.dynamic_slice_in_dim(ks, key_start, keys, axis=1)
        qpos = start + jnp.arange(query_tile)
        kpos = (start - halo if radius is not None else 0) + jnp.arange(keys)
        gq, gk, gv, ga = _block(
            qq, kk, vv, sq, sk, alpha, gg, qpos, kpos, radius=radius
        )
        rq, rk, rv, ra = result[:4]
        rq = jax.lax.dynamic_update_slice_in_dim(rq, gq, start, axis=1)
        oldk = jax.lax.dynamic_slice_in_dim(rk, key_start, keys, axis=1)
        oldv = jax.lax.dynamic_slice_in_dim(rv, key_start, keys, axis=1)
        if compensate_kv:
            ck, cv = result[4:]
            oldck = jax.lax.dynamic_slice_in_dim(ck, key_start, keys, axis=1)
            oldcv = jax.lax.dynamic_slice_in_dim(cv, key_start, keys, axis=1)
            nextk, nextck = _compensated_add(oldk, oldck, gk)
            nextv, nextcv = _compensated_add(oldv, oldcv, gv)
            rk = jax.lax.dynamic_update_slice_in_dim(rk, nextk, key_start, axis=1)
            rv = jax.lax.dynamic_update_slice_in_dim(rv, nextv, key_start, axis=1)
            ck = jax.lax.dynamic_update_slice_in_dim(ck, nextck, key_start, axis=1)
            cv = jax.lax.dynamic_update_slice_in_dim(cv, nextcv, key_start, axis=1)
            return rq, rk, rv, ra + ga, ck, cv
        rk = jax.lax.dynamic_update_slice_in_dim(rk, oldk + gk, key_start, axis=1)
        rv = jax.lax.dynamic_update_slice_in_dim(rv, oldv + gv, key_start, axis=1)
        return rq, rk, rv, ra + ga

    initial = (jnp.zeros_like(qp), jnp.zeros_like(kp), jnp.zeros_like(vp), jnp.float32(0))
    if compensate_kv:
        initial += (jnp.zeros_like(kp), jnp.zeros_like(vp))
    result = jax.lax.fori_loop(0, blocks, step, initial)
    gq, gk, gv, ga = result[:4]
    if compensate_kv:
        gk, gv = gk + result[4], gv + result[5]
    return (
        gq[:, :length],
        gk[:, halo : halo + length],
        gv[:, halo : halo + length],
        ga.astype(jnp.asarray(alpha).dtype),
    )


def _baseline(q, k, v, segments, alpha, radius, tile):
    return windowed_yat_attention(
        q,
        k,
        v,
        segments,
        alpha=alpha,
        radius=radius,
        block_size=tile,
        compute_mode="bf16",
        precision="fp32_scores",
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _attention(q, k, v, segments, alpha, radius, tile, compensate_kv):
    return _baseline(q, k, v, segments, alpha, radius, tile)


def _forward(q, k, v, segments, alpha, radius, tile, compensate_kv):
    return _baseline(q, k, v, segments, alpha, radius, tile), (q, k, v, segments, alpha)


def _backward(radius, tile, compensate_kv, residual, gradient):
    q, k, v, segments, alpha = residual
    gq, gk, gv, ga = centered_backward(
        q,
        k,
        v,
        segments,
        alpha,
        gradient,
        radius=radius,
        query_tile=tile,
        compensate_kv=compensate_kv,
    )
    return gq, gk, gv, None, ga


_attention.defvjp(_forward, _backward)


def centered_yat_attention(q, k, v, segments, *, alpha: float | jax.Array=1.0, radius=None, block_size=64, compensate_kv=False):
    """Opt-in first-order centered backward; BF16 geometry, FP32 score state.

    Supports packed documents, padding and arbitrary positive head widths.
    Global attention holds a query tile against the sequence; local attention
    restricts it to the tile's window. Physical full-model qualification is
    separate from primitive correctness. Alpha is a differentiable scalar.
    compensate_kv is an unqualified experimental BF16 accumulation correction;
    it leaves forward arithmetic unchanged and adds two gradient-sized buffers.
    """
    if (
        q.ndim != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or segments.shape != q.shape[:2]
    ):
        raise ValueError(
            "Require matching [batch, sequence, heads, features] Q/K/V and segments"
        )
    if any(size < 1 for size in q.shape):
        raise ValueError(
            "Positive batch, sequence, heads and feature dimensions required"
        )
    if any(x.dtype != jnp.bfloat16 for x in (q, k, v)):
        raise ValueError("Centered YAT requires BF16 operands")
    if not jnp.issubdtype(segments.dtype, jnp.integer):
        raise ValueError("Integer document segments required")
    if (
        type(block_size) is not int
        or block_size < 1
        or (radius is not None and (type(radius) is not int or radius < 0))
    ):
        raise ValueError(
            "Positive integer block size and nonnegative integer radius required"
        )
    alpha = jnp.asarray(alpha)
    if alpha.ndim != 0 or not jnp.issubdtype(alpha.dtype, jnp.floating):
        raise ValueError("Floating-point scalar alpha required")
    if type(compensate_kv) is not bool:
        raise ValueError("compensate_kv must be a static boolean")
    return _attention(q, k, v, segments, alpha, radius, block_size, compensate_kv)
