import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flaxchat.fused_cross_entropy import fused_cross_entropy


def reference(h, w, b, y):
    logits = (h @ w.T).astype(jnp.float32) + b
    losses = jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, jnp.maximum(y, 0)[:, None], axis=-1)[:, 0]
    return jnp.where(y >= 0, losses, 0)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('rows', [128, 256, 1024])
def test_fused_loss_and_custom_vjp(dtype, empty, rows):
    # Odd vocabulary exercises padding and targets on either side of tile edges.
    h = jax.random.normal(jax.random.key(1), (rows, 128)).astype(dtype) * .1
    w = jax.random.normal(jax.random.key(2), (257, 128)).astype(dtype) * .1
    b = jnp.linspace(-2, 2, 257)
    y = jnp.arange(rows) % 128 * 2
    y = y.at[0].set(-1).at[-1].set(256)
    if empty:
        y = jnp.full_like(y, -1)
    scale = jnp.linspace(.01, 1, rows)
    def objective(h, w, b, fused):
        loss = fused_cross_entropy(h, w, b, y, tile=128, interpret=True) if fused else reference(h, w, b, y)
        return (loss * scale).sum()
    fn = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)), static_argnums=3)
    a, ga = fn(h, w, b, False)
    c, gc = fn(h, w, b, True)
    np.testing.assert_allclose(a, c, atol=1e-4, rtol=2e-5 if dtype == jnp.bfloat16 else 1e-6)
    for x, z in zip(ga, gc, strict=True):
        tol = 3e-3 if dtype == jnp.bfloat16 else 2e-5
        np.testing.assert_allclose(np.asarray(x, dtype=float), np.asarray(z, dtype=float), atol=tol, rtol=tol)
    if empty:
        assert float(c) == 0
        assert all(np.all(np.asarray(x) == 0) for x in gc)


def test_fused_sharded_global_normalization_and_gradients():
    if jax.device_count() < 2:
        pytest.skip('Requires multiple devices, e.g. JAX_NUM_CPU_DEVICES=4')
    from flaxchat.fused_cross_entropy import sharded_fused_loss
    devices = jax.device_count()
    h = jax.random.normal(jax.random.key(3), (devices, 128, 128)) * .1
    w = jax.random.normal(jax.random.key(4), (257, 128)) * .1
    b = jnp.zeros(257)
    y = jnp.full((devices, 128), -1, dtype=jnp.int32).at[0, :5].set(256).at[-1, :3].set(1)
    def dense(h, w, b):
        return reference(h.reshape(-1, 128), w, b, y.reshape(-1)).sum() / (y >= 0).sum()
    def fused(h, w, b):
        return sharded_fused_loss(h, w, b, y, tile=128, interpret=True)
    a, ga = jax.jit(jax.value_and_grad(dense, argnums=(0, 1, 2)))(h, w, b)
    z, gz = jax.jit(jax.value_and_grad(fused, argnums=(0, 1, 2)))(h, w, b)
    np.testing.assert_allclose(z, a, atol=2e-5, rtol=2e-5)
    for x, v in zip(ga, gz, strict=True):
        np.testing.assert_allclose(v, x, atol=2e-5, rtol=2e-5)


def test_fused_contract_rejects_unsupported_layout_and_label_type():
    h = jnp.zeros((128, 128))
    w = jnp.zeros((257, 128))
    b = jnp.zeros(257)
    labels = jnp.zeros(128, dtype=jnp.int32)
    with pytest.raises(ValueError, match='multiples of 128'):
        fused_cross_entropy(h, w, b, labels, tile=127, interpret=True)
    with pytest.raises(ValueError, match='integer labels'):
        fused_cross_entropy(h, w, b, labels.astype(jnp.float32), interpret=True)
    with pytest.raises(ValueError, match='shape mismatch'):
        fused_cross_entropy(h, w, b, labels[:1], interpret=True)
