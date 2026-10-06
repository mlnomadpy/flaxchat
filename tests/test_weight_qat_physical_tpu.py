"""Run only with FLAXCHAT_PHYSICAL_TPU=1 JAX_PLATFORMS=tpu on a real TPU.

No CPU model fallback. This suite is source coverage until a physical receipt exists.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('FLAXCHAT_PHYSICAL_TPU') != '1',
                                reason='Requires explicit physical TPU admission')


def test_int8_forward_export_parity_and_identity_ste():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flaxchat.weight_qat import fake_quantize_int8
    from scripts.quantize_yat_encoder import quantize_rows
    assert jax.default_backend() == 'tpu'
    rows = np.array([[0, 0, 0, 0], [-1, .25, .75, 1],
                     [.5, 1.5, 2.5, 127], [-.5, -1.5, -2.5, -127]], np.float32)
    integers, scales, _ = quantize_rows(rows)
    expected = integers.astype(np.float32) * scales
    native = jax.device_put(rows.T)
    actual = jax.jit(lambda x: fake_quantize_int8(x, 0))(native)
    np.testing.assert_array_equal(np.asarray(actual).T, expected)
    grad = jax.jit(jax.grad(lambda x: fake_quantize_int8(x, 0).sum()))(native)
    np.testing.assert_array_equal(np.asarray(grad), np.ones_like(rows.T))
    # Duplicate gathered tokens must accumulate gradients into FP32 masters.
    indices = jnp.array([1, 1, 2])
    gather_grad = jax.grad(lambda table: fake_quantize_int8(table[indices], -1).sum())(jax.device_put(rows))
    np.testing.assert_array_equal(np.asarray(gather_grad),
        np.array([[0]*4, [2]*4, [1]*4, [0]*4], np.float32))


def test_qat_pool_backward_preserves_master_alpha_and_rejects_mlm():
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from scripts.quantize_yat_encoder import quantize_rows
    from flaxchat.encoder import EncoderConfig, ModernBert
    assert jax.default_backend() == 'tpu'
    config = EncoderConfig(vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
        local_attention=128, ffn_type='yat_glu', attention_score='yat_softmax',
        compute_dtype='bfloat16', residual_dtype='float32',
        weight_quantization='int8_per_channel_ste', use_remat=True,
        yat_compute_mode='bf16', yat_ffn_compute_mode='bf16_adaptive',
        yat_attention_implementation='centered_fp32_scores',
        yat_softmax_backward='factored', yat_attention_block_size=64,
        yat_global_attention_block_size=64)
    model = ModernBert(config, rngs=nnx.Rngs(29))
    ids = (jnp.arange(2 * 128).reshape(2, 128) % 127 + 1).astype(jnp.int32)
    ids = ids.at[0, 97:].set(0)
    plain_config = replace(config, weight_quantization='none')
    original = ModernBert(plain_config, rngs=nnx.Rngs(31))
    reconstructed = ModernBert(plain_config, rngs=nnx.Rngs(37))
    nnx.update(original, nnx.state(model))
    nnx.update(reconstructed, nnx.state(model))
    pool = nnx.jit(lambda m, tokens: m.pool(tokens))
    # Same original FP32 masters, different constructor seeds: restoration wins.
    original_vectors = np.asarray(pool(original, ids))
    np.testing.assert_array_equal(np.asarray(pool(reconstructed, ids)), original_vectors)

    def offline_reconstruct(variable, *, transpose):
        master = np.asarray(variable[...])
        rows = master.T if transpose else master
        integers, scales, _ = quantize_rows(rows)
        dequantized = integers.astype(np.float32) * scales
        variable[...] = jax.device_put(dequantized.T if transpose else dequantized)

    # Independent production exporter rounds actual host tensor bytes. Model
    # forward/backward still execute only on the physical TPU asserted above.
    offline_reconstruct(reconstructed.embedding.embedding, transpose=False)
    for layer in reconstructed.layers:
        for name in ('qkv', 'attn_out', 'wi', 'wo'):
            offline_reconstruct(getattr(layer, name).kernel, transpose=True)
    quantized_vectors = np.asarray(pool(model, ids))
    np.testing.assert_array_equal(quantized_vectors, np.asarray(pool(reconstructed, ids)))
    assert np.isfinite(original_vectors).all() and np.isfinite(quantized_vectors).all()
    assert not np.array_equal(original_vectors, quantized_vectors)
    # Reconstructing a separate model must never mutate the original masters.
    np.testing.assert_array_equal(np.asarray(model.embedding.embedding[...]),
                                  np.asarray(original.embedding.embedding[...]))
    before = np.asarray(model.layers[0].wi.kernel[...]).copy()
    loss, gradients = nnx.value_and_grad(lambda m: jnp.square(m.pool(ids)).mean())(model)
    assert np.isfinite(np.asarray(loss))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradients))
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    assert np.isfinite(np.asarray(gradients.layers[0].yat_alpha[...])).all()
    assert np.isfinite(np.asarray(gradients.layers[0].yat_attention_alpha[...])).all()
    np.testing.assert_array_equal(np.asarray(model.layers[0].wi.kernel[...]), before)
    assert config.yat_bias == 1 and config.yat_epsilon == .01 and config.yat_alpha_trainable
    with pytest.raises(ValueError, match='not MLM'):
        model(ids)
