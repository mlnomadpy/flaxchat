"""Windowed YAT attention that never forms a full sequence score matrix."""
from typing import cast
import jax
import jax.numpy as jnp

from flaxchat.yat import adaptive_squared_distance_bf16, squared_distance_bf16, softmax_bf16


def windowed_yat_attention(q, k, v, segments, *, radius, alpha: float | jax.Array=1.,
                           compute_mode='bf16_adaptive', block_size=64,
                           softmax_backward_mode='factored', precision='native'):
    """Query tiles see only the keys in their local window's union.

    With radius=None each query tile sees all keys (global attention).
    This is a tiled XLA implementation, not a fused FlashAttention kernel.
    Each row still performs a single softmax over all eligible keys. Unlike
    streaming BF16 softmax, this avoids a new recurrent rounding error.
    """
    if q.shape != k.shape or q.shape != v.shape or segments.shape != q.shape[:2]:
        raise ValueError('Require matching [batch, sequence, heads, features] Q/K/V and segments')
    if (radius is not None and radius < 0) or block_size < 1 or q.shape[1] < 1:
        raise ValueError('Positive sequence/block size and nonnegative radius required')
    if compute_mode not in ('mixed', 'bf16', 'bf16_adaptive'):
        raise ValueError('Unknown YAT compute mode')
    if precision not in ('native', 'fp32_scores'):
        raise ValueError('Unknown YAT attention precision')
    if precision == 'fp32_scores' and (compute_mode != 'bf16' or softmax_backward_mode != 'factored'):
        raise ValueError('fp32_scores requires direct BF16 geometry and factored softmax')
    strict = compute_mode != 'mixed'
    if softmax_backward_mode not in ('factored', 'max_centered'):
        raise ValueError('Unknown softmax_backward_mode')
    if not strict and softmax_backward_mode != 'factored':
        raise ValueError('max_centered softmax backward requires BF16 YAT')
    dtype = jnp.bfloat16 if strict else jnp.float32
    if strict and any(a.dtype != jnp.bfloat16 for a in (q, k, v)):
        raise ValueError('BF16 YAT requires BF16 operands')
    batch, length, heads, width = q.shape
    tile = min(block_size, length)
    count = (length + tile - 1) // tile
    padded = count * tile
    halo = radius if radius is not None else 0
    keys = tile + 2 * halo if radius is not None else length
    valid = segments >= 0
    q, k, v = [cast(jax.Array, jnp.where(valid[:, :, None, None], a, 0)) for a in (q, k, v)]
    qp = jnp.pad(q, ((0, 0), (0, padded - length), (0, 0), (0, 0)))
    kp, vp = [jnp.pad(a, ((0, 0), (halo, padded - length + halo if radius is not None else 0), (0, 0), (0, 0)))
              for a in (k, v)]
    qs = jnp.pad(segments, ((0, 0), (0, padded - length)), constant_values=-1)
    ks = jnp.pad(segments, ((0, 0), (halo, padded - length + halo if radius is not None else 0)), constant_values=-1)
    local_mask = (jnp.abs(jnp.arange(tile)[:, None] - (jnp.arange(keys)[None, :] - halo)) <= halo
                  if radius is not None else jnp.ones((tile, keys), dtype=bool))

    def block(i, output):
        start = i * tile
        qq = jax.lax.dynamic_slice_in_dim(qp, start, tile, axis=1)
        key_start = start if radius is not None else 0
        kk = jax.lax.dynamic_slice_in_dim(kp, key_start, keys, axis=1)
        vv = jax.lax.dynamic_slice_in_dim(vp, key_start, keys, axis=1)
        sq = jax.lax.dynamic_slice_in_dim(qs, start, tile, axis=1)
        sk = jax.lax.dynamic_slice_in_dim(ks, key_start, keys, axis=1)
        dots = jnp.einsum('bqhd,bkhd->bhqk', qq, kk, preferred_element_type=dtype)
        a, b = qq.transpose(0, 2, 1, 3), kk.transpose(0, 2, 1, 3)
        if strict:
            distance = (adaptive_squared_distance_bf16(a, b, dots) if compute_mode == 'bf16_adaptive'
                        else squared_distance_bf16(a, b, block=32) if precision == 'fp32_scores'
                        else squared_distance_bf16(a, b))
        else:
            an = jnp.sum(a.astype(dtype) ** 2, -1)
            bn = jnp.sum(b.astype(dtype) ** 2, -1)
            distance = jnp.maximum(an[..., :, None] + bn[..., None, :] - 2 * dots, 0)
        if precision == 'fp32_scores':
            logits = jnp.asarray(alpha, jnp.float32) * jnp.square(dots.astype(jnp.float32) + 1.) / (distance.astype(jnp.float32) + .01)
        else:
            logits = jnp.asarray(alpha, dtype) * jnp.square(dots + 1.) / (distance + .01)
        mask = (sq[:, :, None] == sk[:, None, :]) & (sq[:, :, None] >= 0) & (sk[:, None, :] >= 0)
        mask &= local_mask[None]
        # Invalid queries use one dummy key solely to keep softmax finite.
        mask |= (sq < 0)[:, :, None] & (jnp.arange(keys) == 0)[None, None, :]
        logits = jnp.where(mask[:, None], logits, jnp.asarray(-jnp.inf, dtype))
        probs = (softmax_bf16(logits, backward_mode=softmax_backward_mode)
                 if strict and precision == 'native' else jax.nn.softmax(logits, axis=-1))
        result = jnp.einsum('bhqk,bkhd->bqhd', probs.astype(v.dtype), vv,
                            preferred_element_type=dtype).astype(v.dtype)
        result = jnp.where((sq >= 0)[:, :, None, None], result, 0)
        return jax.lax.dynamic_update_slice_in_dim(output, result, start, axis=1)

    out = jax.lax.fori_loop(0, count, jax.checkpoint(block, prevent_cse=False),
                            jnp.zeros((batch, padded, heads, width), v.dtype))
    return out[:, :length]
