"""Experimental exact-objective TPU projection/cross-entropy with custom VJP.

Logit tiles live inside Pallas kernels. Only per-tile normalizers/target scores
cross HBM; hidden gradients accumulate in VMEM. No [tokens, vocabulary] logits are stored.
No sampled vocabulary or gradient filtering. BF16 projection rounding matches
ModernBert.project; reduction order can differ. Use interpret=True only in tests.
"""
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _dot(a, b):
    return jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())),
        precision=(jax.lax.Precision.HIGHEST if a.dtype == jnp.float32 else jax.lax.Precision.DEFAULT),
        preferred_element_type=jnp.float32)


def _logits(h, w, bias):
    return _dot(h, w.T).astype(h.dtype).astype(jnp.float32) + bias[None, :]


def _forward(h, w, bias, labels, tile, interpret):
    if h.shape[0] > 512:
        return _multi_forward(h, w, bias, labels, tile, interpret)
    m, dim = h.shape
    vocab = w.shape[0]
    padding = (-vocab) % tile
    wp = jnp.pad(w, ((0, padding), (0, 0)))
    bp = jnp.pad(bias, (0, padding))
    tiles = wp.shape[0] // tile

    def kernel(hr, wr, br, yr, lr, tr):
        columns = pl.program_id(0) * tile + jnp.arange(tile)
        z = _logits(hr[...], wr[...], br[...])
        z = cast(jax.Array, jnp.where(columns[None, :] < vocab, z, -jnp.inf))
        lr[0, 0, :] = jax.nn.logsumexp(z, axis=1)
        tr[0, 0, :] = jnp.where(columns[None, :] == yr[...][:, None], z, 0).sum(axis=1)

    stats = pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((tiles, 1, m), jnp.float32),) * 2,
        grid=(tiles,),
        in_specs=(pl.BlockSpec((m, dim), lambda i: (0, 0)),
                  pl.BlockSpec((tile, dim), lambda i: (i, 0)),
                  pl.BlockSpec((tile,), lambda i: (i,)),
                  pl.BlockSpec((m,), lambda i: (0,))),
        out_specs=(pl.BlockSpec((1, 1, m), lambda i: (i, 0, 0)),) * 2,
        interpret=interpret)(h, wp, bp, labels)
    lse = jax.nn.logsumexp(stats[0][:, 0, :], axis=0)
    losses = jnp.where(labels >= 0, lse - stats[1][:, 0, :].sum(axis=0), 0)
    return losses, (h, w, bias, labels, lse)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _ce(h, w, bias, labels, tile, interpret):
    return _forward(h, w, bias, labels, tile, interpret)[0]


def _backward(tile, interpret, residual, cotangent):
    if residual[0].shape[0] > 512:
        return _multi_backward(tile, interpret, residual, cotangent)
    h, w, bias, labels, lse = residual
    vocab = w.shape[0]
    padding = (-vocab) % tile
    w = jnp.pad(w, ((0, padding), (0, 0)))
    bias = jnp.pad(bias, (0, padding))
    m, dim = h.shape
    tiles = w.shape[0] // tile
    cotangent = jnp.where(labels >= 0, cotangent, 0)

    def kernel(hr, wr, br, yr, lr, gr, dhr, dwr, dbr):
        columns = pl.program_id(0) * tile + jnp.arange(tile)
        z = _logits(hr[...], wr[...], br[...])
        probs = jnp.exp(z - lr[...][:, None])
        dz = (probs - (columns[None, :] == yr[...][:, None])) * gr[...][:, None]
        dz = cast(jax.Array, jnp.where(columns[None, :] < vocab, dz, 0))
        # Bias receives FP32 gradients; the BF16 projection operands receive
        # gradients through the FP32-logit cast in the dense reference.
        dbr[...] = dz.sum(axis=0)
        dz = dz.astype(h.dtype)
        @pl.when(pl.program_id(0) == 0)
        def initialize_hidden_gradient():
            dhr[...] = jnp.zeros((m, dim), jnp.float32)
        dhr[...] += _dot(dz, wr[...])
        dwr[...] = _dot(dz.T, hr[...]).astype(w.dtype)

    dh, dw, db = pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((m, dim), jnp.float32),
                   jax.ShapeDtypeStruct(w.shape, w.dtype),
                   jax.ShapeDtypeStruct(bias.shape, bias.dtype)),
        grid=(tiles,),
        in_specs=(pl.BlockSpec((m, dim), lambda i: (0, 0)),
                  pl.BlockSpec((tile, dim), lambda i: (i, 0)),
                  pl.BlockSpec((tile,), lambda i: (i,)),
                  pl.BlockSpec((m,), lambda i: (0,)),
                  pl.BlockSpec((m,), lambda i: (0,)),
                  pl.BlockSpec((m,), lambda i: (0,))),
        out_specs=(pl.BlockSpec((m, dim), lambda i: (0, 0)),
                   pl.BlockSpec((tile, dim), lambda i: (i, 0)),
                   pl.BlockSpec((tile,), lambda i: (i,))),
        interpret=interpret)(h, w, bias, labels, lse, cotangent)
    return dh.astype(h.dtype), dw[:vocab], db[:vocab], None


_ce.defvjp(_forward, _backward)


def fused_cross_entropy(h, w, bias, labels, *, tile=1024, interpret=False) -> jax.Array:
    """Per-token full-vocabulary loss, with negative labels ignored.

    Inputs are one local token tile and replicated decoder weights; call inside
    shard_map for distributed training. The caller owns global normalization.
    """
    if not interpret and jax.default_backend() != 'tpu':
        raise ValueError('Fused cross entropy requires TPU (interpret=True for tests)')
    if h.ndim != 2 or w.ndim != 2 or h.shape[1] != w.shape[1]:
        raise ValueError('Expected hidden [tokens, hidden] and weights [vocab, hidden]')
    if labels.shape != (h.shape[0],) or bias.shape != (w.shape[0],):
        raise ValueError('Labels or bias shape mismatch')
    if min(*h.shape, w.shape[0]) <= 0 or not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError('Nonempty operands and integer labels required')
    if h.dtype != w.dtype or h.dtype not in (jnp.bfloat16, jnp.float32) or bias.dtype != jnp.float32:
        raise ValueError('Matching BF16/FP32 operands and FP32 bias required')
    if tile <= 0 or tile % 128 or h.shape[0] % 128 or h.shape[1] % 128:
        raise ValueError('Token/hidden dimensions and vocabulary tile must be multiples of 128')
    return cast(jax.Array, _ce(h, w, bias, labels, tile, interpret))


def sharded_fused_loss(hidden, weight, bias, labels, *, tile=1024, interpret=False, backend='pallas'):
    """Data-parallel loss with explicit local kernels and global token weighting."""
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec as P
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    if hidden.shape[0] % mesh.size:
        raise ValueError('Fused loss batch must divide across all data devices')

    @partial(jax.shard_map, mesh=mesh, in_specs=(P('data'), P(), P(), P('data')),
             out_specs=P(), check_vma=False)
    def local(h, w, b, y):
        n, dim = y.size, h.shape[-1]
        pad = (-n) % 128
        h = jnp.pad(h.reshape(n, dim), ((0, pad), (0, 0)))
        y = jnp.pad(y.reshape(-1), (0, pad), constant_values=-1)
        if backend == 'xla_full':
            logits = (h @ w.T).astype(jnp.float32) + b
            target = jnp.take_along_axis(logits, jnp.maximum(y, 0)[:, None], axis=-1)[:, 0]
            total = jnp.where(y >= 0, jax.nn.logsumexp(logits, axis=-1) - target, 0).sum()
        else:
            total = fused_cross_entropy(h, w, b, y, tile=tile, interpret=interpret).sum()
        total = jax.lax.psum(total, 'data')
        count = jax.lax.psum((y >= 0).sum(), 'data')
        return total / jnp.maximum(count, 1)
    return local(hidden, weight, bias, labels)


def _multi_forward(h, w, bias, labels, tile, interpret):
    """Tile both axes, keeping decoder tiles resident across token tiles."""
    m, dim = h.shape
    block = 512 if m % 512 == 0 else 128
    vocab = w.shape[0]
    pad = (-vocab) % tile
    wp, bp = jnp.pad(w, ((0, pad), (0, 0))), jnp.pad(bias, (0, pad))
    nv, nm = wp.shape[0] // tile, m // block
    def kernel(hr, wr, br, yr, lr, tr):
        columns = pl.program_id(0) * tile + jnp.arange(tile)
        z = _logits(hr[...], wr[...], br[...])
        z = cast(jax.Array, jnp.where(columns[None, :] < vocab, z, -jnp.inf))
        lr[0, 0, 0, :] = jax.nn.logsumexp(z, axis=1)
        tr[0, 0, 0, :] = jnp.where(columns[None, :] == yr[...][:, None], z, 0).sum(axis=1)
    normalizers, targets = pl.pallas_call(kernel, grid=(nv, nm),
        in_specs=(pl.BlockSpec((block, dim), lambda v, m: (m, 0)),
                  pl.BlockSpec((tile, dim), lambda v, m: (v, 0)),
                  pl.BlockSpec((tile,), lambda v, m: (v,)),
                  pl.BlockSpec((block,), lambda v, m: (m,))),
        out_specs=(pl.BlockSpec((1, 1, 1, block), lambda v, m: (v, m, 0, 0)),)*2,
        out_shape=(jax.ShapeDtypeStruct((nv, nm, 1, block), jnp.float32),)*2,
        interpret=interpret)(h, wp, bp, labels)
    lse = jax.nn.logsumexp(normalizers, axis=0).reshape(m)
    loss = jnp.where(labels >= 0, lse - targets.sum(axis=0).reshape(m), 0)
    return loss, (h, w, bias, labels, lse)


def _multi_backward(tile, interpret, residual, cotangent):
    from jax.experimental.pallas import tpu as pltpu
    h, w, bias, labels, lse = residual
    m, dim = h.shape
    block = 512 if m % 512 == 0 else 128
    vocab = w.shape[0]
    pad = (-vocab) % tile
    wp, bp = jnp.pad(w, ((0, pad), (0, 0))), jnp.pad(bias, (0, pad))
    nv, nm = wp.shape[0] // tile, m // block
    cotangent = jnp.where(labels >= 0, cotangent, 0)
    def dz(hr, wr, br, yr, lr, gr, v):
        columns = v * tile + jnp.arange(tile)
        z = _logits(hr[...], wr[...], br[...])
        grad = (jnp.exp(z - lr[...][:, None]) - (columns[None, :] == yr[...][:, None])) * gr[...][:, None]
        return cast(jax.Array, jnp.where(columns[None, :] < vocab, grad, 0))
    def dh_kernel(hr, wr, br, yr, lr, gr, out):
        grad = dz(hr, wr, br, yr, lr, gr, pl.program_id(1)).astype(h.dtype)
        @pl.when(pl.program_id(1) == 0)
        def init():
            out[...] = jnp.zeros((block, dim), jnp.float32)
        out[...] += _dot(grad, wr[...])
    dh = pl.pallas_call(dh_kernel, grid=(nm, nv),
        in_specs=(pl.BlockSpec((block, dim), lambda m, v: (m, 0)),
                  pl.BlockSpec((tile, dim), lambda m, v: (v, 0)),
                  pl.BlockSpec((tile,), lambda m, v: (v,)),
                  pl.BlockSpec((block,), lambda m, v: (m,)),
                  pl.BlockSpec((block,), lambda m, v: (m,)),
                  pl.BlockSpec((block,), lambda m, v: (m,))),
        out_specs=pl.BlockSpec((block, dim), lambda m, v: (m, 0)),
        out_shape=jax.ShapeDtypeStruct(h.shape, jnp.float32),
        interpret=interpret)(h, wp, bp, labels, lse, cotangent)
    def dw_kernel(hr, wr, br, yr, lr, gr, outw, outb, accum):
        grad = dz(hr, wr, br, yr, lr, gr, pl.program_id(0))
        @pl.when(pl.program_id(1) == 0)
        def init():
            accum[...] = jnp.zeros((tile, dim), jnp.float32)
            outb[...] = jnp.zeros((tile,), jnp.float32)
        accum[...] += _dot(grad.astype(h.dtype).T, hr[...])
        outb[...] += grad.sum(axis=0)
        @pl.when(pl.program_id(1) == nm - 1)
        def finish():
            outw[...] = accum[...].astype(w.dtype)
    dw, db = pl.pallas_call(dw_kernel, grid=(nv, nm),
        in_specs=(pl.BlockSpec((block, dim), lambda v, m: (m, 0)),
                  pl.BlockSpec((tile, dim), lambda v, m: (v, 0)),
                  pl.BlockSpec((tile,), lambda v, m: (v,)),
                  pl.BlockSpec((block,), lambda v, m: (m,)),
                  pl.BlockSpec((block,), lambda v, m: (m,)),
                  pl.BlockSpec((block,), lambda v, m: (m,))),
        out_specs=(pl.BlockSpec((tile, dim), lambda v, m: (v, 0)),
                   pl.BlockSpec((tile,), lambda v, m: (v,))),
        out_shape=(jax.ShapeDtypeStruct(wp.shape, w.dtype),jax.ShapeDtypeStruct(bp.shape, bias.dtype)),
        scratch_shapes=(pltpu.VMEM((tile, dim), jnp.float32),),
        interpret=interpret)(h, wp, bp, labels, lse, cotangent)
    return dh.astype(h.dtype), dw[:vocab], db[:vocab], None
