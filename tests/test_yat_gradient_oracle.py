import numpy as np
import pytest
from tests.yat_attention_oracle import attention_and_gradients


@pytest.mark.parametrize('radius', [None, 1])
@pytest.mark.parametrize('alpha', [-.1, 0., .1])
def test_independent_analytic_gradients_match_directional_finite_differences(radius, alpha):
    rng = np.random.default_rng(182)
    q, k, v, go = [rng.normal(size=(1, 5, 2, 3)) * .2 for _ in range(4)]
    k[:, 0] = q[:, 0] + .03
    segments = np.array([[0, 0, 1, 1, -1]])
    output, gradients, _ = attention_and_gradients(q, k, v, segments, alpha, go, radius=radius)
    np.testing.assert_array_equal(output[:, -1], 0)
    args = [q, k, v, alpha]
    for i, analytic in enumerate(gradients):
        direction = rng.normal(size=np.shape(args[i]))
        plus, minus = list(args), list(args)
        step = 1e-6
        plus[i] = args[i] + step * direction
        minus[i] = args[i] - step * direction
        def loss(values):
            out, _, _ = attention_and_gradients(*values[:3], segments, values[3], go, radius=radius)
            return np.sum(out * go)
        numeric = (loss(plus) - loss(minus)) / (2 * step)
        np.testing.assert_allclose(np.sum(analytic * direction), numeric, rtol=2e-5, atol=2e-8)


def test_oracle_constant_values_and_packed_isolation():
    rng = np.random.default_rng(24)
    q, k, go = [rng.normal(size=(1, 7, 2, 3)) * .2 for _ in range(3)]
    v = np.broadcast_to(rng.normal(size=(1, 1, 2, 3)), q.shape)
    segments = np.array([[-1, 0, 0, 1, 1, 1, -1]])
    out, (gq, gk, gv, ga), _ = attention_and_gradients(q, k, v, segments, .1, go)
    np.testing.assert_allclose(out[:, 1:6], v[:, 1:6], atol=1e-14)
    for gradient in (gq, gk, ga):
        np.testing.assert_allclose(gradient, 0, atol=1e-13)
    np.testing.assert_array_equal(gv[:, [0, -1]], 0)


@pytest.mark.parametrize('radius', [None, 3])
@pytest.mark.parametrize('alpha', [-.1, 0., .1])
def test_bf16_attention_against_independent_gradient_oracle(radius, alpha):
    import jax
    import jax.numpy as jnp
    from flaxchat.yat_attention import windowed_yat_attention

    q, k, v, go = [
        (jax.random.normal(jax.random.key(i), (1, 17, 2, 64)) * .2).astype(jnp.bfloat16)
        for i in (1, 2, 3, 4)
    ]
    k = k.at[:, 0].set(q[:, 0] + jnp.bfloat16(.01))
    segments = np.array([[0] * 7 + [1] * 7 + [-1] * 3])
    expected, expected_grads, contribution_scale = attention_and_gradients(
        q, k, v, segments, alpha, go, radius=radius,
    )
    def forward(q, k, v, a):
        return windowed_yat_attention(
            q, k, v, jnp.asarray(segments), radius=radius, alpha=a, block_size=8,
        )
    actual, pullback = jax.vjp(forward, q, k, v, jnp.float32(alpha))
    gradients = jax.jit(pullback)(go)
    assert np.linalg.norm(np.asarray(actual, dtype=float) - expected) <= .03 * np.linalg.norm(expected)
    for actual_grad, expected_grad in zip(gradients[:3], expected_grads[:3], strict=True):
        error = np.linalg.norm(np.asarray(actual_grad, dtype=float) - expected_grad)
        assert error <= .08 * np.linalg.norm(expected_grad) + 1e-8
    # A scalar near zero can be ill-conditioned: normalize by the sum of the
    # absolute analytic contributions, not by the nearly cancelled final sum.
    assert abs(float(gradients[3]) - expected_grads[3]) <= .01 * contribution_scale + 1e-8
