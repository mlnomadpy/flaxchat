"""Opt-in compiler boundaries for numerical experiments, never training defaults."""

import jax
from flax import nnx

MODES = ("baseline", "qkv", "attention_output", "ffn_output", "head", "all_dense")


class BoundaryLinear(nnx.Linear):
    def __call__(self, inputs):
        return jax.lax.optimization_barrier(super().__call__(inputs))


def _replace_linear(old):
    # Clone the full graph/state, preserving custom dot functions, precision,
    # dtype promotion and parameter metadata without random reinitialization.
    if type(old) is not nnx.Linear:
        raise ValueError("Diagnostic boundaries require ordinary nnx.Linear modules")
    current = nnx.clone(old)
    current.__class__ = BoundaryLinear
    return current


def with_boundaries(model, mode="baseline"):
    """Return an independent module graph; keep parameters and source unchanged."""
    if mode not in MODES:
        raise ValueError(f"Unknown boundary mode: {mode}")
    current = nnx.clone(model)
    for layer in current.layers:
        for label, name in (("qkv", "qkv"), ("attention_output", "attn_out"),
                            ("ffn_output", "wo")):
            if mode in (label, "all_dense"):
                setattr(layer, name, _replace_linear(getattr(layer, name)))
    if mode in ("head", "all_dense"):
        current.head_dense = _replace_linear(current.head_dense)
    return current
