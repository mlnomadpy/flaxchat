import numpy as np
import pytest
from scripts.yat_ffn_oracle import yat_ffn_value_and_grad


@pytest.mark.parametrize('collision', [False, True])
def test_analytic_vjp_matches_finite_differences(collision):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 5))
    kernel = rng.normal(size=(5, 6)) * .2
    if collision:
        x[0] = 1.
        kernel[:, 0] = x[0]
        kernel[:, 1] = x[0] + 1e-5
    alpha = np.array(.7)
    weight = rng.normal(size=(2, 3))
    actual = yat_ffn_value_and_grad(x, kernel, alpha, weight)
    for operand, name in ((x, 'dx'), (kernel, 'dk'), (alpha, 'dalpha')):
        expected = np.empty_like(operand)
        for index in np.ndindex(operand.shape):
            old = operand[index].copy()
            step = 1e-6
            operand[index] = old + step
            plus = yat_ffn_value_and_grad(x, kernel, alpha, weight)['loss']
            operand[index] = old - step
            minus = yat_ffn_value_and_grad(x, kernel, alpha, weight)['loss']
            operand[index] = old
            expected[index] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(actual[name], expected, rtol=2e-5, atol=2e-6)


def test_zero_alpha_preserves_its_nonzero_derivative():
    result = yat_ffn_value_and_grad(np.ones((1, 3)), np.ones((3, 2)), 0., np.ones((1, 1)))
    for key in ('loss', 'output', 'dx', 'dk'):
        np.testing.assert_array_equal(result[key], 0.)
    assert result['dalpha'] == 4800.


def test_rejects_cotangent_broadcast_and_nonfinite():
    x, k = np.ones((2, 3)), np.ones((3, 4))
    with pytest.raises(ValueError, match='Cotangent'):
        yat_ffn_value_and_grad(x, k, 1., np.ones((1, 2)))
    with pytest.raises(ValueError, match='finite'):
        yat_ffn_value_and_grad(x, k, np.nan, np.ones((2, 2)))
