import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import pytest
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_tasks import EncoderTokenClassifier, token_classification_statistics


def test_token_loss_and_gradient_match_independent_reference():
    logits=jnp.array([[[2.,-1.,0.],[1.,2.,3.],[3.,1.,2.],[0.,2.,1.]]])
    labels=jnp.array([[0,-100,2,1]])
    mask=jnp.array([[True,True,True,False]])
    def fn(x):
        return token_classification_statistics(x,labels,token_mask=mask)[0]
    actual=jax.jit(jax.value_and_grad(fn))(logits)
    reference=jax.value_and_grad(lambda x:-(jax.nn.log_softmax(x)[0,0,0]+jax.nn.log_softmax(x)[0,2,2])/2)(logits)
    np.testing.assert_allclose(actual[0],reference[0],rtol=1e-6)
    np.testing.assert_allclose(actual[1],reference[1],rtol=1e-6,atol=1e-8)
    _,correct,count=token_classification_statistics(logits,labels,token_mask=mask)
    assert int(correct)==1 and int(count)==2
    np.testing.assert_array_equal(actual[1][0,[1,3]],0)


def test_all_ignored_batch_has_zero_loss_and_gradients():
    x=jnp.ones((2,3,4),jnp.bfloat16)
    labels=jnp.full((2,3),-100)
    def fn(x):
        return token_classification_statistics(x,labels,token_mask=jnp.ones((2,3),bool))[0]
    loss,gradient=jax.jit(jax.value_and_grad(fn))(x)
    assert float(loss)==0
    np.testing.assert_array_equal(gradient,0)


def test_invalid_active_label_is_nonfinite_but_padding_is_ignored():
    x=jnp.ones((1,2,3))
    labels=jnp.array([[3,0]])
    assert not np.isfinite(token_classification_statistics(x,labels,token_mask=jnp.ones((1,2),bool))[0])
    assert np.isfinite(token_classification_statistics(x,labels,token_mask=jnp.array([[False,True]]))[0])
    with pytest.raises(ValueError,match='integers'):
        token_classification_statistics(x,labels.astype(jnp.float32),token_mask=jnp.ones((1,2),bool))
    with pytest.raises(ValueError,match='Expected logits'):
        token_classification_statistics(x,labels[:,:1],token_mask=jnp.ones((1,2),bool))


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
def test_token_adapter_backpropagates_into_encoder_and_masks_padding(dtype):
    config=EncoderConfig(vocab_size=16,hidden_size=8,intermediate_size=12,
                         num_hidden_layers=1,num_attention_heads=2,
                         max_position_embeddings=16,compute_dtype=dtype)
    model=EncoderTokenClassifier(ModernBert(config,rngs=nnx.Rngs(71)),3,rngs=nnx.Rngs(72))
    tokens=jnp.array([[5,6,7,0]])
    labels=jnp.array([[0,1,-100,2]])
    logits=nnx.jit(lambda model,tokens:model(tokens))(model,tokens)
    assert logits.shape==(1,4,3)
    np.testing.assert_array_equal(logits[:,3],0)
    loss,grads=nnx.jit(nnx.value_and_grad(lambda model:model.statistics(tokens,labels)[0]))(model)
    assert np.isfinite(loss)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads['encoder']['embedding']['embedding'][...]))>0
    assert float(jnp.linalg.norm(grads['classifier']['kernel'][...]))>0
    for invalid in (True,1,0):
        with pytest.raises(ValueError):
            EncoderTokenClassifier(model.encoder,invalid,rngs=nnx.Rngs(0))
