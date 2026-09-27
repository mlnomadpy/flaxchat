import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flaxchat.encoder import bidirectional_attention
from flaxchat.yat_attention import windowed_yat_attention


@pytest.mark.parametrize('radius', [None, 64])
@pytest.mark.parametrize('backward', ['factored', 'max_centered'])
def test_near_collision_dispatch_preserves_packed_document_isolation(radius, backward):
    # Unrelated documents switch repair between dense and sparse execution.
    # Their dispatch must not change the target document's reduction arithmetic.
    a = (1 + jax.random.normal(jax.random.key(1), (64,)) * .1).astype(jnp.bfloat16)
    q = jnp.broadcast_to(a, (1, 256, 1, 64))
    k = (q + jax.random.normal(jax.random.key(41), q.shape) * .04).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.key(42), q.shape).astype(jnp.bfloat16)
    segments = jnp.array([[0] * 8 + [1] * 248])
    weights = jax.random.normal(jax.random.key(43), q.shape).astype(jnp.bfloat16).at[:, 8:].set(0)

    def loss(q, k, v, alpha):
        out = windowed_yat_attention(q, k, v, segments, radius=radius,
            alpha=alpha, block_size=64, softmax_backward_mode=backward)
        return jnp.sum(out * weights, dtype=jnp.bfloat16), out[:, :8]

    compiled = jax.jit(jax.value_and_grad(loss, (0, 1, 2, 3), has_aux=True))
    before = compiled(q, k, v, jnp.float32(.0001))
    after = compiled(q.at[:, 8:].set(-1), k.at[:, 8:].set(-3),
                     v.at[:, 8:].set(9), jnp.float32(.0001))
    for x, y in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(x, y)
        assert np.isfinite(np.asarray(y, dtype=np.float32)).all()


@pytest.mark.parametrize('length,radius,tile', [(17, 3, 8), (8, 0, 3), (9, 12, 4), (17, None, 8)])
@pytest.mark.parametrize('mode', ['mixed', 'bf16_adaptive'])
def test_windowed_attention_values_and_all_gradients(length, radius, tile, mode):
    dtype = jnp.float32 if mode == 'mixed' else jnp.bfloat16
    q, k, v = [(jax.random.normal(jax.random.key(i), (2, length, 2, 8))*.15).astype(dtype) for i in (1,2,3)]
    # Exercise near-collision distance repairs and document/sequence boundaries.
    k = k.at[:, 0].set(q[:, 0] + jnp.asarray(.01, dtype))
    segments = jnp.zeros((2, length), jnp.int32).at[:, length//2:].set(1).at[:, -2:].set(-1)
    segments = segments.at[1].set(-1)
    weights = jax.random.normal(jax.random.key(4), q.shape)
    def fn(q,k,v,alpha,blocked):
        y = (windowed_yat_attention(q,k,v,segments,radius=radius,alpha=alpha,compute_mode=mode,block_size=tile)
             if blocked else bidirectional_attention(q,k,v,segments,radius=radius,score='yat_softmax',alpha=alpha,yat_compute_mode=mode))
        return (y.astype(jnp.float32)*weights).sum(), y
    actual = jax.jit(jax.value_and_grad(lambda q,k,v,a:fn(q,k,v,a,True),(0,1,2,3),has_aux=True))(q,k,v,jnp.float32(.1))
    expected = jax.jit(jax.value_and_grad(lambda q,k,v,a:fn(q,k,v,a,False),(0,1,2,3),has_aux=True))(q,k,v,jnp.float32(.1))
    tolerance = 3e-5 if mode == 'mixed' else .035
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
        a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        assert np.isfinite(a).all()
        assert np.linalg.norm(a-b) <= tolerance*max(np.linalg.norm(b), 1e-3)+1e-5
    np.testing.assert_array_equal(actual[0][1][1], 0)


@pytest.mark.parametrize('radius', [3, None])
@pytest.mark.parametrize('mode', ['mixed', 'bf16_adaptive'])
def test_windowed_attention_isolates_packed_documents_and_padding(radius, mode):
    """Global centering/normalization must not leak masked values between docs."""
    dtype = jnp.float32 if mode == 'mixed' else jnp.bfloat16
    q, k, v = [
        (jax.random.normal(jax.random.key(i), (1, 33, 2, 8)) * .2).astype(dtype)
        for i in (1, 2, 3)
    ]
    segments = jnp.array([[-1] * 3 + [0] * 13 + [1] * 13 + [-1] * 4])
    target = segments == 1
    def forward(q, k, v):
        return windowed_yat_attention(
            q, k, v, segments, radius=radius, alpha=.1,
            compute_mode=mode, block_size=8,
        )
    compiled = jax.jit(forward)
    expected = compiled(q, k, v)
    # Perturb all three operands outside the target document, including padding.
    changed = [jnp.where(target[:, :, None, None], x, x + 10) for x in (q, k, v)]
    actual = compiled(*changed)
    np.testing.assert_array_equal(actual[:, 16:29], expected[:, 16:29])
    np.testing.assert_array_equal(actual[:, :3], 0)
    np.testing.assert_array_equal(actual[:, 29:], 0)
    _, pullback = jax.vjp(forward, q, k, v)
    cotangent = jnp.where(target[:, :, None, None], jnp.ones_like(v), 0)
    for gradient in jax.jit(pullback)(cotangent):
        np.testing.assert_array_equal(gradient[:, :16], 0)
        np.testing.assert_array_equal(gradient[:, 29:], 0)
