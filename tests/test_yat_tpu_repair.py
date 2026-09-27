"""Numerical coverage of TPU dense dispatch without requiring local TPU hardware."""
import jax
import jax.numpy as jnp
from jax import export
import numpy as np
import pytest

import flaxchat.yat as yat


@pytest.mark.parametrize('width', [33, 64, 128, 129])
@pytest.mark.parametrize('count', [0, 1, 31, 32, 33, 131])
def test_dense_dispatch_values_and_gradients(monkeypatch, width, count):
    rng = np.random.default_rng(173)
    x = jnp.asarray(rng.normal(size=(2, 17, width)), jnp.bfloat16)
    y = jnp.asarray(rng.normal(size=(2, 65, width)), jnp.bfloat16)
    if count:
        y = y.at[:, :min(count, 65)].set(x[:, :1] + jnp.bfloat16(.01))
    g = jnp.asarray(rng.normal(size=(2, 17, 65)), jnp.bfloat16)
    results = []
    for force_dense in (False, True):
        # Compare both dispatch choices on the same backend, avoiding an
        # unsupported claim that CPU arithmetic reproduces TPU bit patterns.
        monkeypatch.setattr(yat, '_prefer_dense_attention_repair',
                            lambda decision, d, force=force_dense: jnp.asarray(True) if force and d <= 128 else decision)

        def loss(x, y):
            dots = jnp.einsum('bnd,bmd->bnm', x, y, preferred_element_type=jnp.bfloat16)
            out = yat.adaptive_squared_distance_bf16(x, y, dots)
            return jnp.sum(out*g, dtype=jnp.bfloat16), out

        results.append(jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(x, y))
    np.testing.assert_array_equal(np.asarray(results[0][0][1]), np.asarray(results[1][0][1]))
    for a, b in zip(results[0][1], results[1][1], strict=True):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        assert np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-30) < .02


@pytest.mark.parametrize('radius', [None, 64])
@pytest.mark.parametrize('mode', ['factored', 'max_centered'])
def test_dense_dispatch_preserves_packed_isolation(monkeypatch, radius, mode):
    from tests.test_yat_windowed import test_near_collision_dispatch_preserves_packed_document_isolation
    monkeypatch.setattr(yat, '_prefer_dense_attention_repair',
                        lambda decision, width: jnp.asarray(True) if width <= 128 else decision)
    test_near_collision_dispatch_preserves_packed_document_isolation(radius, mode)


def test_dense_dispatch_preserves_independent_oracles(monkeypatch):
    from tests import test_yat_bf16 as suite
    monkeypatch.setattr(yat, '_prefer_dense_attention_repair',
                        lambda decision, width: jnp.asarray(True) if width <= 128 else decision)
    suite.test_adaptive_near_collision_gradient_matches_direct()
    for width in (33, 64, 128):
        suite.test_sparse_fallback_gradient_matches_direct_selected_pair(width)
        for rows in (17, 131):
            suite.test_adaptive_dense_backward_nonzero_distances_and_zero_vectors(rows, width)
    suite.test_underflowed_norms_are_not_treated_as_exact_zeros()


def test_dispatch_exports_for_target_backend_and_preserves_wide_geometry():
    fn = jax.jit(lambda decision: yat._prefer_dense_attention_repair(decision, 64))
    arg = jax.ShapeDtypeStruct((), jnp.bool_)
    for platform in ('cpu', 'tpu'):
        module = export.export(fn, platforms=(platform,))(arg)
        assert module.platforms == (platform,)
        if platform == jax.default_backend():
            assert bool(module.call(jnp.asarray(False))) == (platform == 'tpu')
    for decision in (False, True):
        assert bool(jax.jit(lambda x: yat._prefer_dense_attention_repair(x, 129))(
            jnp.asarray(decision))) == decision
