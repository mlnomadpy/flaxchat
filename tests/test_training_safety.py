"""Regression tests for issue #6 and pretraining initialization order."""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.training import apply_gradients_if_finite, gradients_for_microbatches


def _arrays(tree):
    return [jnp.asarray(leaf) for leaf in jax.tree.leaves(tree)]


def test_nonfinite_step_preserves_model_and_optimizer(tiny_config, random_batch):
    model = GPT(tiny_config, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=0.1), wrt=nnx.Param)
    before_model = nnx.to_pure_dict(nnx.state(model))
    before_optimizer = optimizer.opt_state
    grads = nnx.state(model, nnx.Param)
    grads = jax.tree.map(lambda value: jnp.full_like(value, jnp.nan), grads)

    updated = apply_gradients_if_finite(model, optimizer, grads, jnp.array(jnp.nan))

    assert not bool(updated)
    for before, after in zip(_arrays(before_model), _arrays(nnx.to_pure_dict(nnx.state(model)))):
        assert jnp.array_equal(before, after)
    for before, after in zip(_arrays(before_optimizer), _arrays(optimizer.opt_state)):
        assert jnp.array_equal(before, after)


def test_fp32_microbatch_accumulation_matches_large_batch(tiny_config):
    model_micro = GPT(tiny_config, rngs=nnx.Rngs(0))
    model_full = GPT(tiny_config, rngs=nnx.Rngs(0))
    inputs = jax.random.randint(jax.random.key(1), (4, 16), 0, tiny_config.vocab_size)
    targets = jax.random.randint(jax.random.key(2), (4, 16), 0, tiny_config.vocab_size)

    _, micro_grads = gradients_for_microbatches(
        model_micro, inputs.reshape(2, 2, 16), targets.reshape(2, 2, 16),
        dtype=jnp.float32,
    )

    def loss_fn(model):
        return model(inputs, targets)

    _, full_grads = nnx.value_and_grad(loss_fn)(model_full)
    for micro, full in zip(_arrays(micro_grads), _arrays(full_grads)):
        assert micro.dtype == jnp.float32
        assert jnp.allclose(micro, full.astype(jnp.float32), atol=2e-5, rtol=2e-4)


def test_single_microbatch_path_is_supported(tiny_model, random_batch):
    inputs, targets = random_batch
    loss, grads = gradients_for_microbatches(
        tiny_model, inputs[None], targets[None], dtype=jnp.float32
    )
    assert jnp.isfinite(loss)
    assert all(leaf.dtype == jnp.float32 for leaf in _arrays(grads))


def test_schedule_is_defined_before_optimizer_construction():
    source = (Path(__file__).parents[1] / 'scripts' / 'pretrain.py').read_text()
    schedule = source.index('lr_schedule = make_lr_schedule(')
    optimizer = source.index('optimizer = setup_optimizer(')
    assert schedule < optimizer
