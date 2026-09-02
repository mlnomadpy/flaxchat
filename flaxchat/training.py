"""Pure, testable helpers for numerically safe optimizer updates."""

import jax
import jax.numpy as jnp
from flax import nnx


def accumulation_dtype(name: str):
    """Resolve the configured gradient accumulation dtype."""
    aliases = {
        "float32": jnp.float32,
        "fp32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported gradient accumulation dtype: {name!r}") from exc


def tree_all_finite(tree):
    return jax.tree.reduce(
        lambda finite, leaf: finite & jnp.all(jnp.isfinite(leaf)),
        tree,
        initializer=jnp.array(True),
    )


def apply_gradients_if_finite(model, optimizer, grads, loss):
    """Atomically update model and optimizer, or preserve both on nonfinite input."""
    finite = jnp.isfinite(loss) & tree_all_finite(grads)

    def update(model, optimizer, grads):
        optimizer.update(model, grads)
        return jnp.array(True)

    def skip(model, optimizer, grads):
        del model, optimizer, grads
        return jnp.array(False)

    updated = nnx.cond(finite, update, skip, model, optimizer, grads)
    return updated


def gradients_for_microbatches(model, all_inputs, all_targets, dtype=jnp.float32):
    """Average microbatch gradients in an explicit accumulation dtype."""
    params = nnx.state(model, nnx.Param)
    zeros = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=dtype), params)

    @nnx.scan(
        in_axes=(nnx.Carry, None, 0, 0),
        out_axes=(nnx.Carry, 0),
    )
    def micro_step(accumulated, current_model, inputs, targets):
        def loss_fn(current_model):
            return current_model(inputs, targets)

        loss, grads = nnx.value_and_grad(loss_fn)(current_model)
        grads = jax.tree.map(lambda grad: grad.astype(dtype), grads)
        accumulated = jax.tree.map(jnp.add, accumulated, grads)
        return accumulated, loss

    accumulated, losses = micro_step(zeros, model, all_inputs, all_targets)
    count = all_inputs.shape[0]
    averaged = jax.tree.map(lambda grad: grad / count, accumulated)
    return jnp.mean(losses), averaged
