"""Independent derivative tiling must preserve forward and collision behavior."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flaxchat.yat import squared_distance_bf16


@pytest.mark.parametrize('tile', [16, 32, 64])
@pytest.mark.parametrize('width', [33, 128, 768])
def test_coordinate_tiling_preserves_forward_and_derivative(tile, width):
    x = jax.random.normal(jax.random.key(1), (2, 3, width)).astype(jnp.bfloat16)
    y = jax.random.normal(jax.random.key(2), (1, 5, width)).astype(jnp.bfloat16)
    y = y.at[0, 0].set(x[0, 0])
    weights = jax.random.normal(jax.random.key(3), (2, 3, 5)).astype(jnp.bfloat16)

    def run(tile):
        def loss(x, y):
            distance = squared_distance_bf16(x, y, backward_block=tile)
            return jnp.sum(distance * weights, dtype=jnp.bfloat16), distance
        fn = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))
        assert 'f32' not in str(fn.lower(x, y).compiler_ir(dialect='stablehlo'))
        return fn(x, y)

    baseline, candidate = run(8), run(tile)
    for actual, expected in zip(jax.tree.leaves(candidate), jax.tree.leaves(baseline), strict=True):
        np.testing.assert_array_equal(actual, expected)
    # Reference is evaluated from the actual device-rounded inputs, independently
    # of the implementation's backward and with both broadcast batches included.
    a, b, w = [np.asarray(v, dtype=np.float64) for v in (x, y, weights)]
    pair = 2 * (a[..., :, None, :] - b[..., None, :, :]) * w[..., None]
    expected = (pair.sum(axis=-2), -pair.sum(axis=-3).sum(axis=0, keepdims=True))
    for actual, reference in zip(candidate[1], expected, strict=True):
        error = np.linalg.norm(np.asarray(actual, dtype=float) - reference)
        assert error <= .02 * np.linalg.norm(reference) + 1e-8


@pytest.mark.parametrize('tile', [16, 32, 64])
def test_collision_and_tiny_weight_do_not_get_pruned(tile):
    x = jnp.full((2, 33), 64, jnp.bfloat16)
    y = jnp.full((3, 33), 64, jnp.bfloat16).at[2].set(64.5)
    w = jnp.zeros((2, 3), jnp.bfloat16).at[1, 2].set(2**-16)
    fn = jax.jit(jax.grad(lambda x, y: jnp.sum(
        squared_distance_bf16(x, y, backward_block=tile) * w,
        dtype=jnp.bfloat16), (0, 1)))
    dx, dy = fn(x, y)
    np.testing.assert_array_equal(dx[0], 0)
    np.testing.assert_array_equal(dx[1], -(2**-16))
    np.testing.assert_array_equal(dy[:2], 0)
    np.testing.assert_array_equal(dy[2], 2**-16)


@pytest.mark.parametrize('tile', [0, -1, True, 2.5])
def test_reject_invalid_tile(tile):
    x = jnp.ones((1, 1), jnp.bfloat16)
    with pytest.raises(ValueError, match='positive integers'):
        squared_distance_bf16(x, x, backward_block=tile)


@pytest.mark.parametrize('tile', [16, 32, 64])
@pytest.mark.parametrize('parameter_dtype', [jnp.bfloat16, jnp.float32])
def test_ffn_output_and_all_gradients_match_original(tile, parameter_dtype):
    from flaxchat.encoder import yat_glu
    x = jax.random.normal(jax.random.key(5), (2, 3, 33)).astype(jnp.bfloat16)
    kernel = (jax.random.normal(jax.random.key(6), (33, 34)) * .1).astype(parameter_dtype)
    alpha = jnp.array(.7, jnp.float32)
    weight = jax.random.normal(jax.random.key(7), (2, 3, 17))

    def run(tile):
        def loss(x, kernel, alpha):
            out = yat_glu(x, kernel, alpha=alpha, compute_mode='bf16',
                          distance_backward_block=tile)
            return (out.astype(jnp.float32) * weight).sum(), out
        return jax.jit(jax.value_and_grad(loss, (0, 1, 2), has_aux=True))(x, kernel, alpha)

    for actual, expected in zip(jax.tree.leaves(run(tile)), jax.tree.leaves(run(None)), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('tile', [0, -1, True, 2.5])
def test_encoder_rejects_invalid_backward_tile(tile):
    from flaxchat.encoder import EncoderConfig
    with pytest.raises(ValueError, match='positive integer'):
        EncoderConfig(yat_ffn_backward_block=tile)


@pytest.mark.parametrize('ffn,mode', [('geglu', 'bf16'), ('yat_glu', 'mixed'), ('yat_glu', 'bf16_adaptive')])
def test_encoder_rejects_unused_tile(ffn, mode):
    from flaxchat.encoder import EncoderConfig
    with pytest.raises(ValueError, match='requires direct BF16 YAT FFN'):
        EncoderConfig(ffn_type=ffn, yat_compute_mode=mode, yat_ffn_backward_block=32)
