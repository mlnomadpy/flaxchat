import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import pytest

from scripts.diagnostic_boundaries import MODES, with_boundaries, BoundaryLinear
from tests.test_mlm_accumulation_replay import fixture


@pytest.mark.parametrize('mode', MODES)
def test_boundaries_preserve_source_parameters_and_forward(mode):
    _, model, _, _ = fixture()
    current = with_boundaries(model, mode)
    source = nnx.state(model, nnx.Param)
    assert jax.tree.structure(source) == jax.tree.structure(nnx.state(current, nnx.Param))
    for a, b in zip(jax.tree.leaves(source), jax.tree.leaves(nnx.state(current, nnx.Param)), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not isinstance(model.layers[0].attn_out, BoundaryLinear)
    x = jnp.array([[5, 6, 0]])
    fn = nnx.jit(lambda m, ids: m(ids))
    np.testing.assert_allclose(fn(model, x), fn(current, x), atol=2e-6)
    expected = 0 if mode == 'baseline' else 4 if mode == 'all_dense' else 1
    text = str(fn.lower(current, x).compiler_ir())
    assert text.count('stablehlo.optimization_barrier') == expected


def test_boundary_supports_grad_vmap_and_remat():
    layer = BoundaryLinear(4, 3, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    graph, state = nnx.split(layer)
    def loss(x):
        return nnx.merge(graph, state)(x).astype(jnp.float32).sum()
    x = jnp.ones((2, 4), jnp.bfloat16)
    direct = jax.jit(jax.grad(loss))(x)
    remat = jax.jit(jax.grad(jax.checkpoint(loss)))(x)
    mapped = jax.jit(jax.vmap(jax.grad(loss)))(x)
    np.testing.assert_array_equal(direct, remat)
    np.testing.assert_array_equal(direct, mapped)


def test_unknown_mode_rejected():
    _, model, _, _ = fixture()
    with pytest.raises(ValueError, match='Unknown boundary'):
        with_boundaries(model, 'typo')
