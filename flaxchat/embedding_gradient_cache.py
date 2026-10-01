"""Exact encoder gradient caching for a single global contrastive denominator.

The encoder is a pure function of parameters, frozen state, tokens and per-row
PRNG keys. Keys are replayed verbatim; no mutable RNG or optimizer operation is
permitted inside it. Forward stores embeddings, not encoder activation tapes.
Backward re-encodes bounded chunks and sums their parameter VJPs in FP32.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def make_cached_encoder(encode, chunk_size):
    """Wrap ``encode(params,frozen,tokens,row_keys)`` with an exact custom VJP.

    All inputs describe globally shaped arrays. Sharding remains JAX's job;
    this function does not stop remote gradients or form per-chunk losses.
    The caller must ensure chunk_size is divisible by its data mesh width.
    Integer tokens and PRNG keys have no cotangent; frozen state intentionally
    has no gradient. Chunked stochastic baseline must use identical row keys.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Positive encoder cache chunk size required')

    def validate(tokens, keys):
        if tokens.ndim != 2 or tokens.shape[0] % chunk_size or keys.shape[0] != tokens.shape[0]:
            raise ValueError('Global token rows must align with chunks and replay keys')

    def forward(params, frozen, tokens, keys):
        validate(tokens, keys)
        shape = (tokens.shape[0] // chunk_size, chunk_size)
        chunks = tokens.reshape(shape + tokens.shape[1:])
        key_chunks = keys.reshape(shape + keys.shape[1:])
        encoded = jax.lax.map(lambda item: encode(params, frozen, item[0], item[1]),
                              (chunks, key_chunks))
        return encoded.reshape((tokens.shape[0],) + encoded.shape[2:])

    @jax.custom_vjp
    def cached(params, frozen, tokens, keys):
        return forward(params, frozen, tokens, keys)

    def cache_forward(params, frozen, tokens, keys):
        return forward(params, frozen, tokens, keys), (params, frozen, tokens, keys)

    def cache_backward(residual, embedding_cotangent):
        params, frozen, tokens, keys = residual
        zeros = jax.tree.map(lambda value: jnp.zeros(value.shape, jnp.float32), params)
        def chunk_vjp(index, accumulated):
            first = index * chunk_size
            batch = jax.lax.dynamic_slice_in_dim(tokens, first, chunk_size)
            row_keys = jax.lax.dynamic_slice_in_dim(keys, first, chunk_size)
            cotangent = jax.lax.dynamic_slice_in_dim(embedding_cotangent, first, chunk_size)
            _, pullback = jax.vjp(lambda p: encode(p, frozen, batch, row_keys), params)
            gradient, = pullback(cotangent)
            return jax.tree.map(lambda total, value: total + value.astype(jnp.float32), accumulated, gradient)
        gradients = jax.lax.fori_loop(0, tokens.shape[0] // chunk_size, chunk_vjp, zeros)
        gradients = jax.tree.map(lambda value, parameter: value.astype(parameter.dtype), gradients, params)
        return gradients, None, None, None

    cached.defvjp(cache_forward, cache_backward)
    return cached


def nnx_cached_pool(model, tokens, chunk_size):
    """Zero-dropout ModernBert adapter; live NNX state is never mutated."""
    from flax import nnx
    from flaxchat.encoder import ModernBert
    if type(model) is not ModernBert:
        raise ValueError("NNX cache adapter requires the qualified zero-dropout ModernBert")
    graph, params, frozen = nnx.split(model, nnx.Param, ...)
    def encode(parameters, constants, rows, row_keys):
        del row_keys  # ModernBert explicitly has zero dropout.
        return nnx.merge(graph, parameters, constants).pool(rows)
    row_keys = jax.random.split(jax.random.key(0), tokens.shape[0])
    return make_cached_encoder(encode, chunk_size)(params, frozen, tokens, row_keys)
