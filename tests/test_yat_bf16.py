import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flaxchat.yat import squared_distance_bf16,softmax_bf16
from flaxchat.encoder import yat_glu,bidirectional_attention


@pytest.mark.parametrize('width', [33, 64, 128])
def test_repair_dispatch_does_not_change_an_unchanged_pair(width):
    from flaxchat.yat import adaptive_squared_distance_bf16
    a = (1 + jax.random.normal(jax.random.key(1), (width,)) * .1).astype(jnp.bfloat16)
    b = (a + jax.random.normal(jax.random.key(41), (width,)) * .04).astype(jnp.bfloat16)
    x = jnp.broadcast_to(a, (64, width))
    y = jnp.broadcast_to(b, (256, width))

    @jax.jit
    def distance(x, y):
        dots = jnp.matmul(x, y.T, preferred_element_type=jnp.bfloat16)
        return adaptive_squared_distance_bf16(x, y, dots)

    # All tiles sensitive versus only one of sixteen: force both dispatches.
    dense = distance(x, y)[0, 0]
    sparse = distance(x.at[1:].set(-1), y.at[1:].set(-3))[0, 0]
    np.testing.assert_array_equal(dense, sparse)
    direct = np.sum((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)
    np.testing.assert_allclose(float(dense), direct, rtol=.02)


@pytest.mark.parametrize('weight', [0., 2**-16, .25])
def test_sensitive_distance_cotangents_preserve_zeros_and_small_nonzeros(weight):
    from flaxchat.yat import adaptive_squared_distance_bf16
    # Every pair is sensitive, but only one may carry a nonzero cotangent.
    # This catches tolerance pruning and wrong indexing after compaction.
    x = jnp.full((17, 33), 64., jnp.bfloat16)
    y = jnp.full((65, 33), 64.5, jnp.bfloat16)
    weights = jnp.zeros((17, 65), jnp.bfloat16).at[16, 64].set(weight)

    def loss(x, y):
        dots = jnp.matmul(x, y.T, preferred_element_type=jnp.bfloat16)
        return jnp.sum(adaptive_squared_distance_bf16(x, y, dots) * weights,
                       dtype=jnp.bfloat16)

    dx, dy = jax.jit(jax.grad(loss, (0, 1)))(x, y)
    expected_x = np.zeros(x.shape, np.float32)
    expected_y = np.zeros(y.shape, np.float32)
    expected_x[16] = -weight
    expected_y[64] = weight
    np.testing.assert_array_equal(dx.astype(jnp.float32), expected_x)
    np.testing.assert_array_equal(dy.astype(jnp.float32), expected_y)


@pytest.mark.parametrize('count', [1, 31, 32, 33])
def test_compact_pair_capacity_and_overflow_preserve_distances_and_gradients(count):
    """Repeated row indices, padding and overflow must neither lose nor duplicate pairs."""
    from flaxchat.yat import adaptive_squared_distance_bf16
    x = jnp.ones((3, 129), jnp.bfloat16).at[1:].set(-1)
    y = jnp.zeros((37, 129), jnp.bfloat16).at[:count].set(1.03125)
    weights = jnp.zeros((3, 37), jnp.bfloat16).at[0, :count].set(.25)

    def loss(a, b):
        dots = jnp.matmul(a, b.T, preferred_element_type=jnp.bfloat16)
        distances = adaptive_squared_distance_bf16(a, b, dots, row_tile=3, column_tile=37)
        return jnp.sum(distances * weights, dtype=jnp.bfloat16), distances

    (_, distances), gradients = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(x, y)
    delta = np.asarray(x.astype(jnp.float32))[:, None, :] - np.asarray(y.astype(jnp.float32))[None, :, :]
    np.testing.assert_allclose(distances.astype(jnp.float32), (delta * delta).sum(-1), rtol=.01)
    weighted = 2 * delta * np.asarray(weights.astype(jnp.float32))[..., None]
    for actual, expected in zip(gradients, (weighted.sum(1), -weighted.sum(0)), strict=True):
        np.testing.assert_allclose(actual.astype(jnp.float32), expected, rtol=.01, atol=.0001)
    ir = str(jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True)).lower(x, y).compiler_ir(dialect='stablehlo'))
    assert 'f32' not in ir


def test_direct_distance_preserves_near_collision_and_odd_width():
    x=jnp.array([[64,64,1]],dtype=jnp.bfloat16)
    y=jnp.array([[64.5,64,1],[64,64,1]],dtype=jnp.bfloat16)
    actual=squared_distance_bf16(x,y)
    np.testing.assert_array_equal(actual.astype(jnp.float32),[[.25,0]])
    assert actual.dtype==jnp.bfloat16
    grad=jax.grad(lambda z:squared_distance_bf16(z,y).sum(dtype=jnp.bfloat16))(x)
    assert np.isfinite(grad.astype(jnp.float32)).all()


@pytest.mark.parametrize('mode', ['bf16', 'bf16_adaptive'])
def test_yat_forward_ir_has_no_fp32_distance_or_reduction(mode):
    x=jnp.ones((2,8),jnp.bfloat16)
    w=jnp.ones((8,12),jnp.bfloat16)*.125
    fn=jax.jit(lambda x,w:yat_glu(x,w,compute_mode=mode))
    ir=str(fn.lower(x,w).compiler_ir(dialect='stablehlo'))
    assert 'f32' not in ir
    assert fn(x,w).dtype==jnp.bfloat16
    q=x.reshape(1,2,1,8)
    segments=jnp.array([[0,-1]])
    attention=jax.jit(lambda q:bidirectional_attention(q,q,q,segments,score='yat_softmax',yat_compute_mode=mode))
    assert 'f32' not in str(attention.lower(q).compiler_ir(dialect='stablehlo'))
    result=attention(q)
    assert np.isfinite(result.astype(jnp.float32)).all()
    assert np.array_equal(result[:,1],jnp.zeros_like(result[:,1]))


def test_bf16_softmax_and_random_forward_drift():
    logits=jnp.array([[1,2,-jnp.inf]],jnp.bfloat16)
    weights=softmax_bf16(logits)
    assert weights.dtype==jnp.bfloat16 and float(weights[0,2])==0
    x=jax.random.normal(jax.random.key(1),(8,32)).astype(jnp.bfloat16)
    w=(jax.random.normal(jax.random.key(2),(32,48))*.1).astype(jnp.bfloat16)
    actual=yat_glu(x,w,compute_mode='bf16').astype(jnp.float32)
    reference=yat_glu(x,w).astype(jnp.float32)
    relative=float(jnp.linalg.norm(actual-reference)/jnp.linalg.norm(reference))
    assert relative<.04
    for grad in jax.grad(lambda x,w:yat_glu(x,w,compute_mode='bf16').sum(dtype=jnp.bfloat16),(0,1))(x,w):
        assert np.isfinite(grad.astype(jnp.float32)).all()


def test_adaptive_distances_repair_collisions_in_batched_partial_tiles():
    from flaxchat.yat import adaptive_squared_distance_bf16
    x=jnp.array([[[64,64,1],[1,2,3],[7,8,9]],[[2,3,4],[5,6,7],[8,9,10]]],jnp.bfloat16)
    y=jnp.array([[[64.5,64,1],[64,64,1],[0,0,0],[1,2,3],[7,8,9]]],jnp.bfloat16)
    def distance(a,b):
        dots=jnp.matmul(a,jnp.swapaxes(b,-1,-2),preferred_element_type=jnp.bfloat16)
        return adaptive_squared_distance_bf16(a,b,dots,row_tile=2,column_tile=3)
    fn=jax.jit(distance)
    actual=fn(x,y)
    reference=jnp.sum((x.astype(jnp.float32)[..., :,None,:]-y.astype(jnp.float32)[...,None,:,:])**2,-1)
    np.testing.assert_allclose(actual.astype(jnp.float32),reference,rtol=.02,atol=.01)
    assert float(actual[0,0,0])==.25 and float(actual[0,0,1])==0
    for grad in jax.grad(lambda x,y:distance(x,y).sum(dtype=jnp.bfloat16),(0,1))(x,y):
        assert np.isfinite(grad.astype(jnp.float32)).all()


@pytest.mark.parametrize('width', [33, 129])
def test_adaptive_ffn_forward_and_gradients_against_mixed(width):
    x=jax.random.normal(jax.random.key(31),(5,width)).astype(jnp.bfloat16)
    w=(jax.random.normal(jax.random.key(32),(width,46))*.1).astype(jnp.bfloat16)
    def evaluate(mode):
        return jax.value_and_grad(lambda x,w,a:yat_glu(x,w,alpha=a,compute_mode=mode).astype(jnp.float32).sum(),(0,1,2))(x,w,jnp.float32(1))
    actual=evaluate('bf16_adaptive')
    expected=evaluate('mixed')
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
        a,b=a.astype(jnp.float32),b.astype(jnp.float32)
        assert float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(b),1e-8))<.04


def test_adaptive_near_collision_gradient_matches_direct():
    from flaxchat.yat import adaptive_squared_distance_bf16
    x=jnp.array([[2,3,4],[2.125,3,4]],jnp.bfloat16)
    y=jnp.array([[2,3.125,4],[2,3,4.125],[2.25,3,4]],jnp.bfloat16)
    def adaptive(a,b):
        return adaptive_squared_distance_bf16(a,b,jnp.matmul(a,b.T,preferred_element_type=jnp.bfloat16),row_tile=1,column_tile=2)
    for fn in (adaptive,squared_distance_bf16):
        output=jax.value_and_grad(lambda a,b,fn=fn:fn(a,b).sum(dtype=jnp.bfloat16),(0,1))(x,y)
        if fn is adaptive:
            actual=output
        else:
            for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(output),strict=True):
                np.testing.assert_array_equal(a.astype(jnp.float32),b.astype(jnp.float32))


@pytest.mark.parametrize('width', [16, 129])
def test_sparse_fallback_gradient_matches_direct_selected_pair(width):
    from flaxchat.yat import adaptive_squared_distance_bf16
    x=jax.random.normal(jax.random.key(91),(5,width)).astype(jnp.bfloat16)
    y=(jax.random.normal(jax.random.key(92),(7,width))*.01).astype(jnp.bfloat16)
    y=y.at[0].set(x[0]+jnp.bfloat16(.03125))
    def adaptive(a,b):
        dots=jnp.matmul(a,b.T,preferred_element_type=jnp.bfloat16)
        return adaptive_squared_distance_bf16(a,b,dots,row_tile=1,column_tile=1)[0,0]
    def reference(a,b):
        return squared_distance_bf16(a,b)[0,0]
    actual=jax.value_and_grad(adaptive,(0,1))(x,y)
    expected=jax.value_and_grad(reference,(0,1))(x,y)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
        np.testing.assert_array_equal(a.astype(jnp.float32),b.astype(jnp.float32))


def test_wide_sparse_repairs_match_float32_distances_and_weighted_gradients():
    from flaxchat.yat import adaptive_squared_distance_bf16
    x = jax.random.normal(jax.random.key(101), (3, 257)).astype(jnp.bfloat16)
    y = jax.random.normal(jax.random.key(102), (5, 257)).astype(jnp.bfloat16)
    y = y.at[0].set(x[0] + jnp.bfloat16(.03125))
    y = y.at[4].set(x[2] - jnp.bfloat16(.0625))
    weights = jnp.array([[1, -.5, .25, 0, 0], [0, 1, 0, .5, 0], [0, 0, -.5, 0, 2]], jnp.bfloat16)

    def loss(a, b):
        dots = jnp.matmul(a, b.T, preferred_element_type=jnp.bfloat16)
        distances = adaptive_squared_distance_bf16(a, b, dots, row_tile=2, column_tile=2)
        return jnp.sum(distances * weights, dtype=jnp.bfloat16), distances

    def reference(a, b):
        distances = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return jnp.sum(distances * weights.astype(jnp.float32)), distances

    actual = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(x, y)
    expected = jax.value_and_grad(reference, (0, 1), has_aux=True)(
        x.astype(jnp.float32), y.astype(jnp.float32))
    np.testing.assert_allclose(actual[0][1].astype(jnp.float32), expected[0][1], rtol=.025, atol=.001)
    for a, b in zip(actual[1], expected[1], strict=True):
        assert float(jnp.linalg.norm(a.astype(jnp.float32) - b) / jnp.linalg.norm(b)) < .025


@pytest.mark.parametrize('shapes', [((2, 1, 3, 35), (1, 4, 5, 35)),
                                   ((3, 35), (2, 5, 35))])
def test_direct_backward_broadcast_weighted_reference(shapes):
    x = jax.random.normal(jax.random.key(201), shapes[0]).astype(jnp.bfloat16)
    y = jax.random.normal(jax.random.key(202), shapes[1]).astype(jnp.bfloat16)
    output_shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2]) + (3, 5)
    weights = jax.random.normal(jax.random.key(203), output_shape).astype(jnp.bfloat16)

    def actual(a, b):
        return jnp.sum(squared_distance_bf16(a, b) * weights, dtype=jnp.bfloat16)

    def reference(a, b):
        distance = jnp.sum((a[..., :, None, :] - b[..., None, :, :]) ** 2, axis=-1)
        return jnp.sum(distance * weights.astype(jnp.float32))

    grads = jax.jit(jax.grad(actual, (0, 1)))(x, y)
    expected = jax.grad(reference, (0, 1))(x.astype(jnp.float32), y.astype(jnp.float32))
    for a, b in zip(grads, expected, strict=True):
        assert a.shape == b.shape
        assert a.dtype == jnp.bfloat16
        assert float(jnp.linalg.norm(a.astype(jnp.float32) - b) / jnp.linalg.norm(b)) < .02
    ir = str(jax.jit(jax.grad(actual, (0, 1))).lower(x, y).compiler_ir(dialect='stablehlo'))
    assert 'f32' not in ir


def test_direct_distance_reference_preserves_jvp_and_forward_values():
    x = jnp.array([[64, 64, 1]], jnp.bfloat16)
    y = jnp.array([[64.5, 64, 1]], jnp.bfloat16)
    def fn(a):
        return squared_distance_bf16(a, y, custom_backward=False)
    value, tangent = jax.jvp(fn, (x,), (jnp.ones_like(x),))
    np.testing.assert_array_equal(value, squared_distance_bf16(x, y))
    np.testing.assert_array_equal(tangent.astype(jnp.float32), [[-1.]])


@pytest.mark.parametrize('shapes', [((2, 1, 3, 64), (1, 2, 5, 64)),
                                   ((3, 16), (2, 5, 16)),
                                   ((2, 1, 3, 129), (1, 2, 5, 129)),
                                   ((3, 257), (2, 5, 257))])
def test_adaptive_sparse_backward_broadcast_and_independent_dot_gradient(shapes):
    from flaxchat.yat import adaptive_squared_distance_bf16
    x = jax.random.normal(jax.random.key(301), shapes[0]).astype(jnp.bfloat16)
    y = jax.random.normal(jax.random.key(302), shapes[1]).astype(jnp.bfloat16)
    # Include cancellation-sensitive entries in every broadcast batch.
    y = y.at[..., 0, :].set(x.reshape(-1, 3, shapes[0][-1])[0, 0] + jnp.bfloat16(.03125))
    dots = jnp.matmul(x, jnp.swapaxes(y, -1, -2), preferred_element_type=jnp.bfloat16)
    weights = jax.random.normal(jax.random.key(303), dots.shape).astype(jnp.bfloat16)

    def loss(a, b, dots, custom):
        distances = adaptive_squared_distance_bf16(a, b, dots, row_tile=2,
                                                  column_tile=3, custom_backward=custom)
        return jnp.sum(distances * weights, dtype=jnp.bfloat16), distances

    actual = jax.jit(jax.value_and_grad(lambda a,b,d:loss(a,b,d,True), (0,1,2), has_aux=True))(x,y,dots)
    reference = jax.jit(jax.value_and_grad(lambda a,b,d:loss(a,b,d,False), (0,1,2), has_aux=True))(x,y,dots)
    np.testing.assert_array_equal(actual[0][1], reference[0][1])
    for a,b in zip(actual[1],reference[1],strict=True):
        assert a.dtype == jnp.bfloat16 and a.shape == b.shape
        assert float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32)) /
                     jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)),1e-8)) < .025
    ir = str(jax.jit(jax.grad(lambda a,b,d:loss(a,b,d,True)[0], (0,1,2)))
             .lower(x,y,dots).compiler_ir(dialect='stablehlo'))
    assert 'f32' not in ir


@pytest.mark.parametrize("rows", [5, 131])
@pytest.mark.parametrize("width", [64, 129])
def test_adaptive_dense_backward_nonzero_distances_and_zero_vectors(rows, width):
    from flaxchat.yat import adaptive_squared_distance_bf16
    x = jnp.ones((rows, width), jnp.bfloat16)
    y = jnp.full((7, width), 1.03125, jnp.bfloat16)
    weights = (jnp.arange(rows * 7, dtype=jnp.bfloat16).reshape(rows,7) % 32) / 32
    def loss(a, b, custom):
        dots = jnp.matmul(a, b.T, preferred_element_type=jnp.bfloat16)
        value = adaptive_squared_distance_bf16(a,b,dots,row_tile=2,column_tile=3,
                                               custom_backward=custom)
        return jnp.sum(value*weights,dtype=jnp.bfloat16)
    for a,b in ((x,y),(jnp.zeros_like(x),jnp.zeros_like(y))):
        result = jax.jit(jax.value_and_grad(lambda a,b:loss(a,b,True),(0,1)))(a,b)
        reference = jax.jit(jax.value_and_grad(lambda a,b:loss(a,b,False),(0,1)))(a,b)
        np.testing.assert_array_equal(result[0], reference[0])
        # Many row tiles change BF16 gradient summation order. Check against
        # analytic FP32 derivatives, not the reference's sequential BF16 sum.
        delta = np.asarray(a.astype(jnp.float32))[:, None, :] - np.asarray(b.astype(jnp.float32))[None, :, :]
        weighted = 2 * delta * np.asarray(weights.astype(jnp.float32))[..., None]
        expected_gradients = (weighted.sum(axis=1), -weighted.sum(axis=0))
        for actual,expected in zip(result[1],expected_gradients,strict=True):
            np.testing.assert_allclose(actual.astype(jnp.float32),expected,rtol=.01,atol=.001)


@pytest.mark.parametrize('width', [64, 129])
def test_exact_zero_pairs_preserve_independent_dot_derivatives(width):
    from flaxchat.yat import adaptive_squared_distance_bf16
    x = jnp.zeros((3, width), jnp.bfloat16)
    y = jnp.zeros((5, width), jnp.bfloat16)
    dots = jnp.zeros((3, 5), jnp.bfloat16)
    def fn(x, y, dots):
        return adaptive_squared_distance_bf16(x, y, dots).sum(dtype=jnp.bfloat16)
    value, grads = jax.jit(jax.value_and_grad(fn, (0, 1, 2)))(x, y, dots)
    assert float(value) == 0
    for g in grads:
        np.testing.assert_array_equal(g, jnp.zeros_like(g))


def test_underflowed_norms_are_not_treated_as_exact_zeros():
    from flaxchat.yat import adaptive_squared_distance_bf16
    x = jnp.full((1, 64), 1e-20, jnp.bfloat16)
    y = -x
    def fn(x):
        dots = jnp.matmul(x, y.T, preferred_element_type=jnp.bfloat16)
        return adaptive_squared_distance_bf16(x, y, dots).sum(dtype=jnp.bfloat16)
    actual = jax.grad(fn)(x)
    expected = jax.grad(lambda a: squared_distance_bf16(a, y, block=32).sum(dtype=jnp.bfloat16))(x)
    np.testing.assert_array_equal(actual, expected)
    assert np.any(np.asarray(actual.astype(jnp.float32)) != 0)


@pytest.mark.parametrize('length', [17, 512, 8192])
def test_explicit_softmax_forward_and_gradient_oracle(length):
    logits=(jax.random.normal(jax.random.key(61),(2,length))*.7).astype(jnp.bfloat16)
    logits=logits.at[:,-3:].set(-jnp.inf)
    cotangent=jax.random.normal(jax.random.key(62),logits.shape).astype(jnp.bfloat16)
    actual=softmax_bf16(logits)
    np.testing.assert_array_equal(actual,softmax_bf16(logits,custom_backward=False))
    grad_fn=jax.jit(jax.grad(lambda x:(softmax_bf16(x)*cotangent).sum(dtype=jnp.bfloat16)))
    gradient=grad_fn(logits).astype(jnp.float32)
    oracle=jax.grad(lambda x:(jax.nn.softmax(x,-1)*cotangent.astype(jnp.float32)).sum())(logits.astype(jnp.float32))
    assert np.isfinite(gradient).all()
    assert float(jnp.linalg.norm(gradient-oracle)/jnp.linalg.norm(oracle))<.02
    np.testing.assert_array_equal(gradient[:,-3:],0)
    assert 'f32' not in str(grad_fn.lower(logits).compiler_ir(dialect='stablehlo'))


def test_softmax_automatic_reference_retains_forward_mode():
    logits=jnp.array([[1.,2.,3.]],jnp.bfloat16)
    value,tangent=jax.jvp(lambda x:softmax_bf16(x,custom_backward=False),
                        (logits,),(jnp.ones_like(logits),))
    np.testing.assert_array_equal(value,softmax_bf16(logits))
    assert np.isfinite(tangent.astype(jnp.float32)).all()
