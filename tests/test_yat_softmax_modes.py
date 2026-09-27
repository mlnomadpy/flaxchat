"""Regression tests for the opt-in, TPU-measured BF16 backward method."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flaxchat.yat import softmax_bf16


@pytest.mark.parametrize('length', [17, 512, 8192])
@pytest.mark.parametrize('seed', [3, 91])
def test_max_centered_backward_preserves_forward_and_null_direction(length, seed):
    rng = np.random.default_rng(seed)
    logits = jnp.asarray(rng.normal(size=(8, length)) * 3, jnp.bfloat16)
    logits = logits.at[:, :3].set(-jnp.inf)
    def fn(x):
        return softmax_bf16(x, backward_mode='max_centered')
    probabilities, backward = jax.vjp(fn, logits)
    np.testing.assert_array_equal(probabilities, softmax_bf16(logits))
    for value in [0., 1., 32., -32.]:
        np.testing.assert_array_equal(backward(jnp.full_like(logits, value))[0], 0)
    exact_logits = np.asarray(logits, dtype=np.float64)
    p = np.exp(exact_logits - exact_logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    for offset in [0., 1., -1.]:
        g = jnp.asarray(rng.normal(size=logits.shape) * .01 + offset, jnp.bfloat16)
        exact_g = np.asarray(g, dtype=np.float64)
        oracle = p * (exact_g - (p * exact_g).sum(-1, keepdims=True))
        actual = np.asarray(backward(g)[0], dtype=np.float64)
        assert np.linalg.norm(actual - oracle) < .04 * np.linalg.norm(oracle)
        # The anchor must belong to an eligible key, never a masked key.
        np.testing.assert_array_equal(backward(g.at[:, :3].set(1000))[0], actual)
    hlo = str(jax.jit(lambda x, g: jax.vjp(fn, x)[1](g)[0])
              .lower(logits, jnp.ones_like(logits)).compiler_ir())
    assert 'xf32' not in hlo


def test_backward_mode_validation_and_reference_forward_mode():
    logits = jnp.array([[1., 2., 3.]], jnp.bfloat16)
    with pytest.raises(ValueError, match='backward_mode'):
        softmax_bf16(logits, backward_mode='unknown')
    with pytest.raises(ValueError, match='custom_backward'):
        softmax_bf16(logits, backward_mode='max_centered', custom_backward=False)
    value, tangent = jax.jvp(lambda x: softmax_bf16(x, custom_backward=False),
                            (logits,), (jnp.ones_like(logits),))
    np.testing.assert_array_equal(value, softmax_bf16(logits))
    assert np.isfinite(tangent.astype(jnp.float32)).all()


@pytest.mark.parametrize('radius', [None, 7])
def test_max_centered_windowed_attention_near_collision(radius):
    from flaxchat.yat_attention import windowed_yat_attention
    from tests.yat_attention_oracle import attention_and_gradients

    q, k, v, go = [
        (jax.random.normal(jax.random.key(i), (1, 33, 2, 64)) *
         (.05 if i in (1, 2) else .2)).astype(jnp.bfloat16)
        for i in (1, 2, 3, 4)
    ]
    k = k.at[:, 3].set(q[:, 3] + jnp.bfloat16(.01))
    segments = np.array([[-1] * 3 + [0] * 13 + [1] * 13 + [-1] * 4])
    expected, grads, scale = attention_and_gradients(q, k, v, segments, .1, go, radius=radius)
    def fn(q, k, v, alpha):
        return windowed_yat_attention(
            q, k, v, jnp.asarray(segments), radius=radius, alpha=alpha,
            block_size=32, softmax_backward_mode='max_centered',
        )
    output, pullback = jax.vjp(fn, q, k, v, jnp.float32(.1))
    actual = jax.jit(pullback)(go)
    default = windowed_yat_attention(q, k, v, jnp.asarray(segments), radius=radius,
                                    alpha=.1, block_size=32)
    np.testing.assert_array_equal(output, default)
    for measured, exact in zip(actual[:3], grads[:3], strict=True):
        assert np.linalg.norm(np.asarray(measured, dtype=float) - exact) < .05 * np.linalg.norm(exact)
    assert abs(float(actual[-1]) - grads[-1]) < .01 * scale
    with pytest.raises(ValueError, match='requires BF16'):
        windowed_yat_attention(q, k, v, jnp.asarray(segments), radius=radius,
                               compute_mode='mixed', softmax_backward_mode='max_centered')
