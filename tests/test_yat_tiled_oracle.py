import numpy as np
import pytest

from tests.yat_attention_oracle import attention_and_gradients, attention_and_gradients_tiled


@pytest.mark.parametrize('radius', [None, 0, 3, 32])
@pytest.mark.parametrize('tile', [1, 7, 32])
@pytest.mark.parametrize('alpha', [-.2, 0., .3])
def test_tiled_oracle_matches_full_direct_oracle(radius, tile, alpha):
    rng = np.random.default_rng(91)
    q, k, v, g = [rng.normal(size=(2, 19, 3, 5)) * .3 for _ in range(4)]
    k[:, 3] = q[:, 3] + .01
    segments = np.array([[-1]*2 + [0]*8 + [1]*7 + [-1]*2, [-1]*19])
    expected = attention_and_gradients(q, k, v, segments, alpha, g, radius=radius)
    actual = attention_and_gradients_tiled(q, k, v, segments, alpha, g,
                                         radius=radius, query_tile=tile)
    for x, y in zip([actual[0], *actual[1], actual[2]],
                    [expected[0], *expected[1], expected[2]], strict=True):
        np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-12)


def test_tiled_oracle_gradient_by_finite_difference():
    rng = np.random.default_rng(17)
    q, k, v, g = [rng.normal(size=(1, 5, 1, 3)) * .2 for _ in range(4)]
    segments = np.array([[0, 0, 1, 1, -1]])
    alpha = .07
    _, gradients, _ = attention_and_gradients_tiled(q, k, v, segments, alpha, g, query_tile=2)
    def loss(q, k, v, alpha):
        return np.sum(attention_and_gradients_tiled(q, k, v, segments, alpha, g,
                                                   query_tile=2)[0] * g)
    epsilon = 1e-6
    for index in range(3):
        inputs = [q.copy(), k.copy(), v.copy()]
        pos = (0, 1, 0, 2)
        inputs[index][pos] += epsilon
        plus = loss(*inputs, alpha)
        inputs[index][pos] -= 2*epsilon
        minus = loss(*inputs, alpha)
        np.testing.assert_allclose((plus-minus)/(2*epsilon), gradients[index][pos], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose((loss(q,k,v,alpha+epsilon)-loss(q,k,v,alpha-epsilon))/(2*epsilon),
                               gradients[3], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('radius', [None, 0, 3])
@pytest.mark.parametrize('alpha', [-.3, .3])
@pytest.mark.parametrize('tile', [1, 7])
@pytest.mark.parametrize('constant', ['keys', 'values'])
def test_tiled_oracle_softmax_invariances(radius, alpha, tile, constant):
    """Null gradients must stay null even with packed, noncontiguous segments."""
    rng = np.random.default_rng(883)
    q, k, v, g = [rng.normal(size=(1, 9, 2, 4)) * .2 for _ in range(4)]
    segments = np.array([[-1, 0, 1, 0, 1, 0, 1, 1, -1]])
    operand = k if constant == 'keys' else v
    for segment in (0, 1):
        operand[0, segments[0] == segment] = rng.normal(size=(2, 4)) * .2
    output, gradients, _ = attention_and_gradients_tiled(
        q, k, v, segments, alpha, g, radius=radius, query_tile=tile,
    )
    changed_q = q + rng.normal(size=q.shape)
    changed_k = k if constant == 'keys' else k + rng.normal(size=k.shape)
    changed_output, _, _ = attention_and_gradients_tiled(
        changed_q, changed_k, v, segments, alpha * 1.7, g,
        radius=radius, query_tile=tile,
    )
    np.testing.assert_allclose(output, changed_output, rtol=0, atol=1e-12)
    np.testing.assert_allclose(gradients[0], 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(gradients[3], 0, rtol=0, atol=1e-12)
    if constant == 'values':
        np.testing.assert_allclose(gradients[1], 0, rtol=0, atol=1e-12)
