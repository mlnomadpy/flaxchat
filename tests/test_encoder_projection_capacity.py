"""Exercise actual short-context projection capacity and overflow boundaries."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from flaxchat.encoder import EncoderConfig, ModernBert


@pytest.mark.parametrize('capacity', [None, 64])
@pytest.mark.parametrize('selected_count', [0, 32, 33])
def test_short_context_compaction_preserves_targets_and_gradients(selected_count, capacity):
    if selected_count and capacity is not None:
        selected_count += capacity - 32
    # At length128/chunk32, one row holds32 targets;33 must use dense fallback.
    cfg = EncoderConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=1, num_attention_heads=2,
                        local_attention=16, loss_chunk_size=32,
                        compute_dtype='float32', use_remat=False,
                        mlm_loss_backend='xla_full')
    dense = ModernBert(cfg, rngs=nnx.Rngs(4))
    compact = ModernBert(replace(cfg, mlm_projection='masked', mlm_projection_capacity=capacity), rngs=nnx.Rngs(4))
    ids = (jnp.arange(256).reshape(2, 128) % 59 + 5).at[:, 127].set(0)
    targets = jnp.full_like(ids, -1)
    if selected_count:
        # Include late positions to detect truncation or incorrect gather indices.
        positions = np.linspace(1, 126, selected_count, dtype=int)
        targets = targets.at[0, positions].set(ids[0, positions])
        targets = targets.at[1, 126].set(9)
    targets = targets.at[:, 127].set(8)  # Padding is never a real target.
    call = nnx.jit(nnx.value_and_grad(lambda model: model(ids, targets)))
    expected, expected_grads = call(dense)
    actual, actual_grads = call(compact)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    for a, b in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)
    if not selected_count:
        assert float(actual) == 0
        assert all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(actual_grads))


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_capacity_rejects_invalid_values(capacity):
    with pytest.raises(ValueError, match='positive integer'):
        EncoderConfig(mlm_projection='masked', mlm_projection_capacity=capacity)


def test_capacity_is_independent_and_default_is_preserved():
    cfg = EncoderConfig(mlm_projection='masked', loss_chunk_size=32)
    assert cfg.projection_capacity(128) == 32
    assert replace(cfg, mlm_projection_capacity=64).projection_capacity(128) == 64
    assert replace(cfg, mlm_projection_capacity=64).projection_capacity(16) == 16
    with pytest.raises(ValueError, match='requires masked'):
        EncoderConfig(mlm_projection_capacity=64)
