"""INT8 weight-only QAT; FP32 masters, exact export rounding and identity STE.

This simulates storage quantization, not INT8 arithmetic. Scales are recomputed
from current masters and are not trainable. Activations/norms/alpha stay intact.
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def fake_quantize_int8(weight, axis):
    """Reduce input axis for native linear kernels, feature axis for gathered rows."""
    if weight.dtype != jnp.float32:
        raise ValueError('INT8 QAT requires FP32 master weights')
    maximum = jnp.max(jnp.abs(weight), axis=axis, keepdims=True)
    scale = jnp.maximum(maximum / jnp.float32(127), jnp.finfo(jnp.float32).tiny)
    scale = jnp.where(maximum == 0, jnp.float32(1), scale)
    quantized = jnp.clip(jnp.rint(weight / scale), -127, 127)
    return quantized * scale


@fake_quantize_int8.defjvp
def _fake_quantize_jvp(axis, primals, tangents):
    (weight,), (tangent,) = primals, tangents
    # Custom JVP keeps the rounded forward exactly (x + stop(dq-x) does not)
    # and applies an identity STE, including through scale selection.
    return fake_quantize_int8(weight, axis), tangent


def qat_linear(layer, x, config):
    """Keep the original nnx.Linear path bit-identical when QAT is disabled."""
    if config.weight_quantization == 'none':
        return layer(x)
    weight = fake_quantize_int8(layer.kernel[...], 0)
    dtype = getattr(jnp, config.compute_dtype)
    return jnp.matmul(x.astype(dtype), weight.astype(dtype))
