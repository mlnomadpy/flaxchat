"""Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flaxchat.encoder import bidirectional_attention, yat_glu

pytestmark = pytest.mark.skipif(jax.device_count() != 4, reason='requires four devices')


def values(seed, shape):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape)*.2, jnp.bfloat16)


@pytest.mark.parametrize('repair', [False, True])
@pytest.mark.parametrize('width', [16, 160])
def test_local_ffn_forward_and_gradients(repair, width):
    x, w = values(1, (4, 4, width)), values(2, (width, 24))
    if repair:
        # Only one chip has cancellation-sensitive pairs; branch predicates
        # must be local without losing the globally shared weight gradient.
        x = x.at[0, 0].set(w[:, 0])
    alpha = jnp.asarray(1., jnp.float32)
    cotangent = values(3, (4, 4, 12)).astype(jnp.float32)
    results = []
    for local in [False, True]:
        def loss(x, w, a, local=local):
            y = yat_glu(x, w, alpha=a, compute_mode='bf16_adaptive', local_shards=local)
            return (y.astype(jnp.float32)*cotangent).sum(), y
        results.append(jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(x, w, alpha))
    for a, b in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        assert np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=.03, atol=.02)


@pytest.mark.parametrize('radius', [None, 2])
@pytest.mark.parametrize('block_size', [0, 4])
@pytest.mark.parametrize('backward_mode', ['factored', 'max_centered'])
def test_local_attention_preserves_packing_padding_and_gradients(radius, block_size, backward_mode):
    q, k, v = [values(seed, (4, 8, 2, 8)) for seed in range(3)]
    k = k.at[0].set(q[0])  # local dense repair on exactly one shard
    segments = jnp.asarray([[0,0,0,1,1,1,-1,-1]]*4)
    alpha = jnp.asarray(1., jnp.float32)
    cotangent = values(7, q.shape).astype(jnp.float32)
    results = []
    for local in [False, True]:
        def loss(q, k, v, alpha, local=local):
            out = bidirectional_attention(q, k, v, segments, radius=radius,
                score='yat_softmax', alpha=alpha, yat_compute_mode='bf16_adaptive', yat_local_shards=local,
                yat_attention_block_size=block_size, yat_softmax_backward=backward_mode)
            return (out.astype(jnp.float32)*cotangent).sum(), out
        results.append(jax.jit(jax.value_and_grad(loss, argnums=(0,1,2,3), has_aux=True))(q,k,v,alpha))
    for a,b in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
        a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        assert np.isfinite(b).all()
        np.testing.assert_allclose(a,b,rtol=.03,atol=.02)
    np.testing.assert_array_equal(np.asarray(results[1][0][1])[:, -2:], 0)


def test_local_batch_divisibility_is_explicit():
    with pytest.raises(ValueError, match='divide across'):
        yat_glu(values(0, (3,4,16)), values(1, (16,24)), local_shards=True)
    fn = partial(bidirectional_attention, score='yat_softmax', yat_local_shards=True)
    with pytest.raises(ValueError, match='divide across'):
        q=values(0, (3,4,2,8))
        fn(q,q,q,jnp.zeros((3,4),jnp.int32))


def test_local_full_encoder_training_and_exact_resume(tmp_path, monkeypatch):
    from tests.test_encoder_training import test_preparation_training_and_exact_resume
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16', 'bfloat16',
                                              'dense', 'xla', 'yat_adaptive_local')


@pytest.mark.parametrize('global_block_size', [0, 2])
@pytest.mark.parametrize('backward_mode', ['factored', 'max_centered'])
def test_windowed_local_training_and_exact_resume(tmp_path, monkeypatch, global_block_size, backward_mode):
    import tests.test_encoder_training as helper
    from flaxchat.encoder import EncoderConfig
    monkeypatch.setattr(helper, 'EncoderConfig', lambda **kw: EncoderConfig(**(kw | dict(
        num_hidden_layers=2, local_attention=2, yat_attention_block_size=2,
        yat_global_attention_block_size=global_block_size, yat_softmax_backward=backward_mode))))
    helper.test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16', 'bfloat16',
                                                     'dense', 'xla', 'yat_adaptive_local')
