"""Native BF16 YAT primitives; no external NMN dependency.

Direct squared differences avoid cancellation between large squared norms.
Only a small coordinate block is broadcast, never the full feature dimension.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import cast


_PAIR_CAPACITY = 32


def _prefer_dense_attention_repair(heuristic, width):
    """Select TPU repair at lowering time, including cross-platform export.

    Dense strips avoid costly sparse traversal for narrow attention geometry.
    Wide FFNs and other backends retain their existing dispatch heuristics.
    """
    if width > 128:
        return heuristic
    return jax.lax.platform_dependent(
        heuristic, tpu=lambda _: jnp.asarray(True), default=lambda decision: decision
    )


def _squared_distance_bf16(x, y, *, block=8):
    """Pairwise distances for [..., rows, features], entirely BF16 array ops."""
    if x.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
        raise ValueError("BF16 distance requires BF16 operands")
    if x.shape[-1] != y.shape[-1] or block < 1:
        raise ValueError("Matching feature dimensions and positive block required")
    features = x.shape[-1]
    padding = (-features) % block
    x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, padding)])
    y = jnp.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, padding)])
    shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2]) + (
        x.shape[-2],
        y.shape[-2],
    )

    def accumulate(index, total):
        a = jax.lax.dynamic_slice_in_dim(x, index * block, block, axis=-1)
        b = jax.lax.dynamic_slice_in_dim(y, index * block, block, axis=-1)
        delta = a[..., :, None, :] - b[..., None, :, :]
        return total + jnp.sum(delta * delta, axis=-1, dtype=jnp.bfloat16)

    return jax.lax.fori_loop(
        0,
        (features + padding) // block,
        jax.checkpoint(accumulate, prevent_cse=False),
        jnp.zeros(shape, dtype=jnp.bfloat16),
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _direct_distance(x, y, block, backward_block):
    return _squared_distance_bf16(x, y, block=block)


def _direct_forward(x, y, block, backward_block):
    return _squared_distance_bf16(x, y, block=block), (x, y)


def _direct_backward(block, residuals, cotangent):
    x, y = residuals
    features = x.shape[-1]
    padding = (-features) % block
    batch = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    a = jnp.pad(
        jnp.broadcast_to(x, batch + x.shape[-2:]),
        [(0, 0)] * (len(batch) + 1) + [(0, padding)],
    )
    b = jnp.pad(
        jnp.broadcast_to(y, batch + y.shape[-2:]),
        [(0, 0)] * (len(batch) + 1) + [(0, padding)],
    )

    def coordinate(index, gradients):
        gx, gy = gradients
        xx = jax.lax.dynamic_slice_in_dim(a, index * block, block, axis=-1)
        yy = jax.lax.dynamic_slice_in_dim(b, index * block, block, axis=-1)
        delta = xx[..., :, None, :] - yy[..., None, :, :]
        weighted = cotangent[..., None] * (2 * delta)
        dx = jnp.sum(weighted, axis=-2, dtype=jnp.bfloat16)
        dy = -jnp.sum(weighted, axis=-3, dtype=jnp.bfloat16)
        return (
            jax.lax.dynamic_update_slice_in_dim(gx, dx, index * block, axis=-1),
            jax.lax.dynamic_update_slice_in_dim(gy, dy, index * block, axis=-1),
        )

    gx, gy = jax.lax.fori_loop(
        0,
        (features + padding) // block,
        coordinate,
        (jnp.zeros_like(a), jnp.zeros_like(b)),
    )

    def unbroadcast(gradient, operand):
        gradient = gradient[..., :features]
        shape = (1,) * (gradient.ndim - operand.ndim) + operand.shape
        axes = tuple(
            i
            for i, (a, b) in enumerate(zip(gradient.shape, shape, strict=True))
            if a != b
        )
        return jnp.sum(gradient, axis=axes, keepdims=True, dtype=jnp.bfloat16).reshape(
            operand.shape
        )

    return unbroadcast(gx, x), unbroadcast(gy, y)


def _direct_tuned_backward(block, backward_block, residuals, cotangent):
    del block
    return _direct_backward(backward_block, residuals, cotangent)


_direct_distance.defvjp(_direct_forward, _direct_tuned_backward)


def squared_distance_bf16(x, y, *, block=8, custom_backward=True,
                          backward_block=None) -> jax.Array:
    """Cancellation-safe BF16 distances with a memory-bounded custom backward.

    Reverse-mode training is supported; set custom_backward=False for JVPs.
    The backward computes 2 * (x - y) * cotangent in coordinate blocks rather
    than transposing the forward accumulation loop. Do not rewrite this as a
    difference of matrix products: that reintroduces near-collision cancellation.
    backward_block independently tiles coordinate derivatives, leaving the
    forward summation order unchanged. Larger tiles trade temporary memory for
    fewer loop iterations; callers must qualify them on their target hardware.
    """
    backward_block = block if backward_block is None else backward_block
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (block, backward_block)):
        raise ValueError("Distance block sizes must be positive integers")
    if not custom_backward and backward_block != block:
        raise ValueError("backward_block requires custom_backward=True")
    if not custom_backward:
        return _squared_distance_bf16(x, y, block=block)
    return cast(jax.Array, _direct_distance(x, y, block, backward_block))


def _softmax_bf16_reference(logits):
    """BF16 normalization without jnp.sum's implicit FP32 accumulation."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    return weights / jnp.sum(weights, axis=-1, keepdims=True, dtype=jnp.bfloat16)


@jax.custom_vjp
def _softmax_bf16_explicit(logits):
    return _softmax_bf16_reference(logits)


def _softmax_forward(logits):
    probabilities = _softmax_bf16_reference(logits)
    return probabilities, probabilities


def _softmax_backward(probabilities, cotangent):
    # Factor the Jacobian before evaluating in BF16. Differentiating exp and
    # division independently introduces separate rounded reciprocal terms.
    mean = jnp.sum(probabilities * cotangent, axis=-1, keepdims=True,
                   dtype=jnp.bfloat16)
    return (probabilities * (cotangent - mean),)


_softmax_bf16_explicit.defvjp(_softmax_forward, _softmax_backward)


@jax.custom_vjp
def _softmax_bf16_max_centered(logits):
    return _softmax_bf16_reference(logits)


def _softmax_max_centered_backward(probabilities, cotangent):
    # Anchor to an eligible, high-probability key before subtracting a mean.
    # This avoids subtracting nearly equal large cotangents after reduction.
    # Stored BF16 probabilities need not sum exactly to one.
    index = jnp.argmax(probabilities, axis=-1)[..., None]
    anchor = jnp.take_along_axis(cotangent, index, axis=-1)
    centered = cotangent - anchor
    mass = jnp.sum(probabilities, axis=-1, keepdims=True, dtype=jnp.bfloat16)
    mean = jnp.sum(probabilities * centered, axis=-1, keepdims=True,
                   dtype=jnp.bfloat16) / mass
    return (probabilities * (centered - mean),)


_softmax_bf16_max_centered.defvjp(_softmax_forward, _softmax_max_centered_backward)


def softmax_bf16(logits, *, custom_backward=True, backward_mode='factored'):
    """BF16 softmax with factored reverse-mode Jacobian and unchanged forward.

    Set custom_backward=False for the automatic reference or forward-mode AD.
    The opt-in backward_mode='max_centered' reduces cancellation at additional
    reduction/gather cost; it leaves forward values unchanged. It has isolated
    TPU validation, not full-model quality qualification. Every variant requires
    a finite eligible logit in each row.
    """
    if backward_mode not in ('factored', 'max_centered'):
        raise ValueError('Unknown softmax backward_mode')
    if backward_mode == 'max_centered':
        if not custom_backward:
            raise ValueError('max_centered requires custom_backward=True')
        return _softmax_bf16_max_centered(logits)
    return (_softmax_bf16_explicit(logits) if custom_backward
            else _softmax_bf16_reference(logits))


def _pair_indices(mask):
    """Bounded pair list; callers must check capacity before selecting this path."""
    indices = jnp.nonzero(mask.reshape(-1), size=_PAIR_CAPACITY, fill_value=0)[0]
    valid = jnp.arange(_PAIR_CAPACITY) < jnp.sum(mask, dtype=jnp.int32)
    return indices // mask.shape[-1], indices % mask.shape[-1], valid


def _paired_distance(x, y):
    """Aligned pairs with the same block-32 BF16 summation as tiled repair."""
    padding = (-x.shape[-1]) % 32
    x = jnp.pad(x, ((0, 0), (0, padding)))
    y = jnp.pad(y, ((0, 0), (0, padding)))

    def accumulate(i, total):
        delta = jax.lax.dynamic_slice_in_dim(x, i * 32, 32, axis=-1) - (
            jax.lax.dynamic_slice_in_dim(y, i * 32, 32, axis=-1)
        )
        return total + jnp.sum(delta * delta, axis=-1, dtype=jnp.bfloat16)

    return jax.lax.fori_loop(
        0, x.shape[-1] // 32, accumulate, jnp.zeros(x.shape[0], jnp.bfloat16)
    )


def _adaptive_squared_distance_bf16(x, y, dots, *, row_tile=16, column_tile=64):
    """Reuse BF16 dots, recomputing cancellation-sensitive tiles directly.

    The 1/8 relative-distance threshold is a heuristic, not an error bound.
    Both branches use BF16 array arithmetic. Avoid vmap over conditional tiles:
    batching a conditional can evaluate both branches, defeating the fast path.
    """
    if any(a.dtype != jnp.bfloat16 for a in (x, y, dots)):
        raise ValueError("Adaptive distance requires BF16 operands and dots")
    if x.shape[-1] != y.shape[-1] or min(row_tile, column_tile) < 1:
        raise ValueError("Matching features and positive tile sizes required")
    batch = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    n, m, d = x.shape[-2], y.shape[-2], x.shape[-1]
    if dots.shape != batch + (n, m) or min(n, m, d) < 1:
        raise ValueError("Invalid dot shape or empty inputs")
    normx = jnp.sum(x * x, axis=-1, dtype=jnp.bfloat16)
    normy = jnp.sum(y * y, axis=-1, dtype=jnp.bfloat16)
    total_norm = normx[..., :, None] + normy[..., None, :]
    raw = total_norm - 2 * dots
    # Attention padding produces exact zero vectors. Their distance and all
    # distance derivatives are zero; do not route them through tile repair.
    # Test coordinates, not norms: BF16 squared norms can underflow for nonzero
    # vectors, which must retain the ordinary near-collision treatment.
    zero_pairs = jnp.all(x == 0, axis=-1)[..., :, None] & jnp.all(y == 0, axis=-1)[..., None, :]
    distance = jnp.where(zero_pairs, 0, jnp.maximum(raw, 0))
    sensitive = (raw <= total_norm * 0.125) & ~zero_pairs

    def repair(distance):
        r, c = min(row_tile, n), min(column_tile, m)
        nr, nc = (n + r - 1) // r, (m + c - 1) // c
        a = jnp.broadcast_to(x, batch + (n, d)).reshape(-1, n, d)
        b = jnp.broadcast_to(y, batch + (m, d)).reshape(-1, m, d)
        a = jnp.pad(a, ((0, 0), (0, nr * r - n), (0, 0)))
        b = jnp.pad(b, ((0, 0), (0, nc * c - m), (0, 0)))
        mask = jnp.pad(
            sensitive.reshape(-1, n, m), ((0, 0), (0, nr * r - n), (0, nc * c - m))
        )
        result = jnp.pad(
            distance.reshape(-1, n, m), ((0, 0), (0, nr * r - n), (0, nc * c - m))
        )

        def tile(index, result):
            batch_index = index // (nr * nc)
            row = (index // nc % nr) * r
            column = (index % nc) * c
            origin = (batch_index, row, column)
            selected = jax.lax.dynamic_slice(mask, origin, (1, r, c))

            def recompute(result):
                old = jax.lax.dynamic_slice(result, origin, (1, r, c))
                xx = jax.lax.dynamic_slice(a, (batch_index, row, 0), (1, r, d))
                yy = jax.lax.dynamic_slice(b, (batch_index, column, 0), (1, c, d))

                # Compact sparse pairs instead of broadcasting all r*c pairs.
                # Overflow retains the complete tile, never truncates repairs.
                def pairs(_):
                    ri, ci, valid = _pair_indices(selected)
                    values = _paired_distance(xx[0, ri], yy[0, ci])
                    # Padded indices are dropped, never overwrite a real pair.
                    return old.at[0, jnp.where(valid, ri, r), ci].set(
                        values, mode="drop"
                    )

                value = cast(
                    jax.Array,
                    jax.lax.cond(
                        jnp.sum(selected, dtype=jnp.int32) <= _PAIR_CAPACITY,
                        pairs,
                        lambda _: jnp.where(
                            selected, _squared_distance_bf16(xx, yy, block=32), old
                        ),
                        operand=None,
                    )
                    if d > 128
                    else jnp.where(
                        selected, _squared_distance_bf16(xx, yy, block=32), old
                    ),
                )
                return jax.lax.dynamic_update_slice(result, value, origin)

            # Keep slicing and updating inside the active branch. Otherwise AD
            # scatters full-sized cotangents even for tiles needing no repair.
            return jax.lax.cond(
                jnp.any(selected), recompute, lambda result: result, result
            )

        def sparse_repair(result):
            result = jax.lax.fori_loop(
                0, a.shape[0] * nr * nc, jax.checkpoint(tile, prevent_cse=False), result
            )
            return result[:, :n, :m].reshape(distance.shape)

        # Keep wide FFN geometry tiled: compiling its dense backward branch
        # reserves substantially more temporary memory even on the fast path.
        if d > 128:
            return sparse_repair(result)
        # Dense cancellation patterns (e.g. equal Q/K) make per-tile dispatch
        # more expensive than one direct pass. Count affected tiles in integers.
        affected = jnp.any(mask.reshape(a.shape[0], nr, r, nc, c), axis=(2, 4))
        dense = _prefer_dense_attention_repair(
            jnp.sum(affected, dtype=jnp.int32) * 4 >= affected.size, d
        )
        return jax.lax.cond(
            dense,
            # Match tiled/compacted repair arithmetic. Otherwise unrelated
            # sensitive pairs can change this pair's result by switching the
            # global dispatch between block-8 and block-32 BF16 reductions.
            lambda _: jnp.where(sensitive, squared_distance_bf16(x, y, block=32), distance),
            sparse_repair,
            result,
        )

    return jax.lax.cond(jnp.any(sensitive), repair, lambda distance: distance, distance)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _adaptive_distance(x, y, dots, row_tile, column_tile):
    return _adaptive_squared_distance_bf16(
        x, y, dots, row_tile=row_tile, column_tile=column_tile
    )


def _adaptive_forward(x, y, dots, row_tile, column_tile):
    result = _adaptive_squared_distance_bf16(
        x, y, dots, row_tile=row_tile, column_tile=column_tile
    )
    return result, (x, y, dots)


def _adaptive_backward(row_tile, column_tile, residuals, cotangent):
    x, y, dots = residuals
    batch = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    n, m, d = x.shape[-2], y.shape[-2], x.shape[-1]
    a = jnp.broadcast_to(x, batch + (n, d)).reshape(-1, n, d)
    b = jnp.broadcast_to(y, batch + (m, d)).reshape(-1, m, d)
    total = (
        jnp.sum(a * a, -1, dtype=jnp.bfloat16)[..., :, None]
        + jnp.sum(b * b, -1, dtype=jnp.bfloat16)[..., None, :]
    )
    raw = total - 2 * dots.reshape(-1, n, m)
    zero_pairs = jnp.all(a == 0, axis=-1)[..., :, None] & jnp.all(b == 0, axis=-1)[..., None, :]
    selected = (raw <= total * 0.125) & ~zero_pairs
    g = cotangent.reshape(-1, n, m)
    # Skip wholly inactive repairs, but preserve geometry-based dispatch.
    # TPU measurements show that pruning individual zero cotangents can turn
    # efficient dense strips into a slower sparse tile traversal.
    active_repair = jnp.any(selected & (g != 0))
    # Outside repaired/zero pairs the distance is positive and unclamped.
    fast = cast(jax.Array, jnp.where(selected | zero_pairs, 0, g))
    gx = (2 * a) * jnp.sum(fast, -1, dtype=jnp.bfloat16)[..., None]
    gy = (2 * b) * jnp.sum(fast, -2, dtype=jnp.bfloat16)[..., None]
    gdots = (-2 * fast).reshape(dots.shape)
    r, c = min(row_tile, n), min(column_tile, m)
    nr, nc = (n + r - 1) // r, (m + c - 1) // c
    a = jnp.pad(a, ((0, 0), (0, nr * r - n), (0, 0)))
    b = jnp.pad(b, ((0, 0), (0, nc * c - m), (0, 0)))
    gx = jnp.pad(gx, ((0, 0), (0, nr * r - n), (0, 0)))
    gy = jnp.pad(gy, ((0, 0), (0, nc * c - m), (0, 0)))
    selected = jnp.pad(selected, ((0, 0), (0, nr * r - n), (0, nc * c - m)))
    g = jnp.pad(g, ((0, 0), (0, nr * r - n), (0, nc * c - m)))

    def tile(i, grads):
        bi, row, col = i // (nr * nc), (i // nc % nr) * r, (i % nc) * c
        mask = jax.lax.dynamic_slice(selected, (bi, row, col), (1, r, c))

        def add(grads):
            gx, gy = grads
            xx = jax.lax.dynamic_slice(a, (bi, row, 0), (1, r, d))
            yy = jax.lax.dynamic_slice(b, (bi, col, 0), (1, c, d))
            gg = jnp.where(mask, jax.lax.dynamic_slice(g, (bi, row, col), (1, r, c)), 0)

            # Match the sparse forward reduction block; never subtract GEMMs.
            def pairs(_):
                ri, ci, valid = _pair_indices(mask)
                delta = xx[0, ri] - yy[0, ci]
                weighted = jnp.where(valid, gg[0, ri, ci], 0)[:, None] * (2 * delta)
                return (
                    jnp.zeros_like(xx).at[0, ri].add(weighted),
                    jnp.zeros_like(yy).at[0, ci].add(-weighted),
                )

            dx, dy = jax.lax.cond(
                jnp.sum(mask, dtype=jnp.int32) <= _PAIR_CAPACITY,
                pairs,
                lambda _: _direct_backward(32, (xx, yy), gg),
                operand=None,
            )
            oldx = jax.lax.dynamic_slice(gx, (bi, row, 0), (1, r, d))
            oldy = jax.lax.dynamic_slice(gy, (bi, col, 0), (1, c, d))
            return (
                jax.lax.dynamic_update_slice(gx, oldx + dx, (bi, row, 0)),
                jax.lax.dynamic_update_slice(gy, oldy + dy, (bi, col, 0)),
            )

        return jax.lax.cond(jnp.any(mask), add, lambda grads: grads, grads)

    def repair(grads):
        affected = jnp.any(selected.reshape(a.shape[0], nr, r, nc, c), axis=(2, 4))

        def dense(grads):
            # Bound dense scratch to at most 128 rows, even for long sequences.
            dr = min(128, a.shape[1])
            dn = (a.shape[1] + dr - 1) // dr
            pad_rows = dn * dr - a.shape[1]
            aa = jnp.pad(a, ((0, 0), (0, pad_rows), (0, 0)))
            mask_all = jnp.pad(selected, ((0, 0), (0, pad_rows), (0, 0)))
            grad_all = jnp.pad(g, ((0, 0), (0, pad_rows), (0, 0)))
            initial = (jnp.pad(grads[0], ((0, 0), (0, pad_rows), (0, 0))), grads[1])

            def row_gradient(i, grads):
                bi, row = i // dn, (i % dn) * dr
                xx = jax.lax.dynamic_slice(aa, (bi, row, 0), (1, dr, d))
                yy = jax.lax.dynamic_slice(b, (bi, 0, 0), (1, nc * c, d))
                mask = jax.lax.dynamic_slice(mask_all, (bi, row, 0), (1, dr, nc * c))
                gg = jnp.where(
                    mask,
                    jax.lax.dynamic_slice(grad_all, (bi, row, 0), (1, dr, nc * c)),
                    0,
                )
                dx, dy = _direct_backward(8, (xx, yy), gg)
                gx, gy = grads
                oldx = jax.lax.dynamic_slice(gx, (bi, row, 0), (1, dr, d))
                oldy = jax.lax.dynamic_slice(gy, (bi, 0, 0), (1, nc * c, d))
                return (
                    jax.lax.dynamic_update_slice(gx, oldx + dx, (bi, row, 0)),
                    jax.lax.dynamic_update_slice(gy, oldy + dy, (bi, 0, 0)),
                )

            dx, dy = jax.lax.fori_loop(0, a.shape[0] * dn, row_gradient, initial)
            return dx[:, : a.shape[1], :], dy

        # Many affected tiles can still contain very few selected pairs each.
        # Other backends use dense strips only when compaction is crowded;
        # narrow TPU geometry uses the physically validated dense path.
        use_dense = _prefer_dense_attention_repair(
            (jnp.sum(affected, dtype=jnp.int32) * 4 >= affected.size)
            & (
                jnp.sum(selected, dtype=jnp.int32)
                > jnp.sum(affected, dtype=jnp.int32)
                * min(_PAIR_CAPACITY, max(1, r * c // 4))
            ), d,
        )
        return jax.lax.cond(
            use_dense,
            dense,
            lambda grads: jax.lax.fori_loop(0, a.shape[0] * nr * nc, tile, grads),
            grads,
        )

    gx, gy = jax.lax.cond(
        active_repair,
        repair,
        lambda grads: grads,
        (gx, gy),
    )

    def unbroadcast(g, operand, rows):
        g = g[:, :rows, :].reshape(batch + (rows, d))
        shape = (1,) * (g.ndim - operand.ndim) + operand.shape
        axes = tuple(
            i for i, (aa, bb) in enumerate(zip(g.shape, shape, strict=True)) if aa != bb
        )
        return jnp.sum(g, axis=axes, keepdims=True, dtype=jnp.bfloat16).reshape(
            operand.shape
        )

    return unbroadcast(gx, x, n), unbroadcast(gy, y, m), gdots


_adaptive_distance.defvjp(_adaptive_forward, _adaptive_backward)


def adaptive_squared_distance_bf16(
    x, y, dots, *, row_tile=16, column_tile=64, custom_backward=True
) -> jax.Array:
    """Adaptive BF16 distance with sparse direct-gradient repair for wide inputs.

    The forward is unchanged. For wide FFNs, reverse mode accumulates repaired
    input/prototype gradients without transposing full distance-matrix updates.
    The norm-identity branch returns its dot cotangent separately; autodiff adds
    it through the caller's projection. Near-collision derivatives subtract
    coordinates before multiplication. Both attention and wide FFNs use this
    backward. Set custom_backward=False to use the reference differentiation.
    """
    if not custom_backward:
        return _adaptive_squared_distance_bf16(
            x, y, dots, row_tile=row_tile, column_tile=column_tile
        )
    return cast(jax.Array, _adaptive_distance(x, y, dots, row_tile, column_tile))
