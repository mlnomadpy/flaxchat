from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flaxchat.encoder import EncoderConfig, ModernBert, yat_glu


def direct(x, kernel, epsilon, alpha):
    a, b = jnp.split(kernel, 2, axis=-1)
    # Independent explicit Euclidean differences instead of expanded norms.
    distances = jnp.sum((x[..., :, None] - a)**2, axis=-2)
    return alpha * (x @ a + 1)**2 / (distances + epsilon) * (x @ b)


def test_formula_and_input_weight_gradients_match_direct_distances():
    x = jax.random.normal(jax.random.key(4), (2, 3, 8))
    w = jax.random.normal(jax.random.key(7), (8, 12))
    def fn(x, w):
        return yat_glu(x, w, alpha=1.2)
    def ref(x, w):
        return direct(x, w, .01, 1.2)
    np.testing.assert_allclose(fn(x, w), ref(x, w), atol=2e-5, rtol=2e-5)
    for a, b in zip(jax.grad(lambda x,w: fn(x,w).sum(), (0,1))(x,w),
                    jax.grad(lambda x,w: ref(x,w).sum(), (0,1))(x,w), strict=True):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_zero_coincident_antipodal_orthogonal_are_finite(dtype):
    x = jnp.array([[0.,0.], [1.,0.], [-1.,0.], [0.,1.]], dtype=dtype)
    w = jnp.array([[1.,0.,1.,-1.], [0.,1.,1.,1.]], dtype=dtype)
    y = jax.jit(yat_glu)(x,w)
    assert y.dtype == dtype and np.isfinite(y.astype(jnp.float32)).all()
    assert np.isfinite(jax.grad(lambda z: yat_glu(z,w).astype(jnp.float32).sum())(x).astype(jnp.float32)).all()
    assert np.array_equal(np.asarray(y[0], dtype=np.float32), [0,0])
    assert float(y[1,1]) < 0  # fixed numerator bias retains orthogonal response
    assert float(y[2,0]) == 0  # dot=-1 cancels the fixed numerator bias


def test_same_parameter_layout_and_opt_in_changes_model():
    c = EncoderConfig(vocab_size=16,hidden_size=8,intermediate_size=12,num_hidden_layers=1,num_attention_heads=2,use_remat=False)
    baseline = ModernBert(c, rngs=nnx.Rngs(17))
    candidate = ModernBert(replace(c,ffn_type='yat_glu'), rngs=nnx.Rngs(17))
    a,b = nnx.state(baseline,nnx.Param),nnx.state(candidate,nnx.Param)
    assert sum(v.size for v in jax.tree.leaves(b)) == sum(v.size for v in jax.tree.leaves(a)) + c.num_hidden_layers
    for old, new in zip(baseline.layers, candidate.layers, strict=True):
        np.testing.assert_array_equal(old.wi.kernel[...], new.wi.kernel[...])
        assert float(new.yat_alpha[...]) == 1
    tokens=jnp.array([[1,5,6,0]])
    assert not np.allclose(baseline(tokens),candidate(tokens))


@pytest.mark.parametrize('kwargs', [{'ffn_type':'unknown'},{'yat_epsilon':0},{'yat_alpha':float('nan')}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EncoderConfig(**kwargs)


def test_alpha_is_trainable_with_correct_gradient_and_fixed_constants():
    x = jnp.array([[.4, .7]])
    w = jnp.array([[.2, .3, .5, -.1], [.1, -.4, .8, .7]])
    derivative = jax.grad(lambda a: yat_glu(x, w, alpha=a).sum())(jnp.float32(1))
    np.testing.assert_allclose(derivative, direct(x,w,.01,1).sum(), rtol=1e-5)
    assert abs(float(derivative)) > 0
    for kwargs in ({'yat_bias':0}, {'yat_epsilon':.001}, {'yat_alpha_trainable':False}):
        with pytest.raises(ValueError):
            EncoderConfig(ffn_type='yat_glu', **kwargs)


def test_yat_attention_matches_explicit_distances_and_all_gradients():
    from flaxchat.encoder import bidirectional_attention
    q,k,v=[jax.random.normal(jax.random.key(i),(1,4,2,4))*.3 for i in (3,4,5)]
    segments=jnp.zeros((1,4),dtype=jnp.int32)
    def actual(q,k,v,a):
        return bidirectional_attention(q,k,v,segments,score='yat_softmax',alpha=a)
    def reference(q,k,v,a):
        qh,kh,vh=[z.transpose(0,2,1,3) for z in (q,k,v)]
        dots=jnp.einsum('bhqd,bhkd->bhqk',qh,kh)
        distance=jnp.sum((qh[:,:,:,None,:]-kh[:,:,None,:,:])**2,-1)
        weights=jax.nn.softmax(a*(dots+1)**2/(distance+.01),-1)
        return (weights@vh).transpose(0,2,1,3)
    args=(q,k,v,jnp.float32(.7))
    np.testing.assert_allclose(actual(*args),reference(*args),rtol=2e-5,atol=2e-6)
    ga=jax.grad(lambda *args:jnp.square(actual(*args)).sum(),(0,1,2,3))(*args)
    gr=jax.grad(lambda *args:jnp.square(reference(*args)).sum(),(0,1,2,3))(*args)
    for a,b in zip(ga,gr,strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-5)
    assert abs(float(ga[3]))>1e-5


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.bfloat16])
def test_yat_attention_masks_and_padding_gradients(packed,dtype):
    from flaxchat.encoder import bidirectional_attention
    q=k=jnp.zeros((2,5,1,2),dtype=dtype)
    v=jnp.broadcast_to(jnp.arange(5)[None,:,None,None],q.shape).astype(dtype)
    segments=jnp.array([[0,0,0,1,-1],[-1,-1,-1,-1,-1]])
    def fn(v):
        return bidirectional_attention(q,k,v,segments,score='yat_softmax',radius=1,packed=packed)
    result=jax.jit(fn)(v)
    expected=[.5,1,1.5,3,0] if packed else [.5,1,2,2.5,0]
    np.testing.assert_allclose(result[0,:,0,0].astype(jnp.float32),expected,atol=.01)
    assert np.array_equal(result[1],jnp.zeros_like(result[1]))
    grads=jax.grad(lambda v:fn(v).astype(jnp.float32).sum())(v)
    assert np.isfinite(grads.astype(jnp.float32)).all()
    assert np.array_equal(grads[1],jnp.zeros_like(grads[1]))
    with pytest.raises(ValueError):
        bidirectional_attention(q,k,v,segments,score='yat_softmax',backend='splash')
