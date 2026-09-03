"""Reusable preference-policy objectives."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp


def centered_advantages(rewards):
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    return rewards - rewards.mean()


def preference_loss(logits, targets, advantages):
    safe_targets = jnp.maximum(targets, 0)
    token_log_probs = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1
    )[..., 0]
    valid = (targets >= 0).astype(jnp.float32)
    objective = jnp.sum(token_log_probs * advantages[:, None] * valid)
    return -objective / jnp.maximum(jnp.sum(valid), 1.0)


@nnx.jit
def train_step(model, optimizer, inputs, targets, advantages):
    loss, gradients = nnx.value_and_grad(
        lambda current: preference_loss(current(inputs), targets, advantages)
    )(model)
    optimizer.update(model, gradients)
    return loss
