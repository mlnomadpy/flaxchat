from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from flaxchat.encoder import EncoderConfig, ModernBert


def configuration():
    return EncoderConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2,
                         global_attn_every_n_layers=2, local_attention=4,
                         compute_dtype='bfloat16', ffn_type='yat_glu',
                         attention_score='yat_softmax', yat_compute_mode='bf16',
                         yat_attention_implementation='centered_fp32_scores',
                         yat_attention_block_size=4, yat_global_attention_block_size=4,
                         yat_ffn_compute_mode='bf16_adaptive', use_remat=False)


def test_adaptive_ffn_keeps_direct_centered_attention(monkeypatch):
    import flaxchat.encoder as encoder
    actual_ffn, actual_attention = encoder.yat_glu, encoder.bidirectional_attention
    observed = {'ffn': [], 'attention': []}

    def ffn(*args, **kwargs):
        observed['ffn'].append(kwargs['compute_mode'])
        return actual_ffn(*args, **kwargs)

    def attention(*args, **kwargs):
        observed['attention'].append((kwargs['yat_compute_mode'], kwargs['yat_attention_implementation']))
        return actual_attention(*args, **kwargs)

    monkeypatch.setattr(encoder, 'yat_glu', ffn)
    monkeypatch.setattr(encoder, 'bidirectional_attention', attention)
    model = ModernBert(configuration(), rngs=nnx.Rngs(11))
    ids = jnp.array([[5, 6, 7, 8], [9, 10, 11, 12]])
    loss, gradients = nnx.jit(nnx.value_and_grad(lambda m: m(ids, ids)))(model)
    assert np.isfinite(loss)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(gradients))
    assert observed['ffn'] and set(observed['ffn']) == {'bf16_adaptive'}
    assert observed['attention'] and set(observed['attention']) == {('bf16', 'centered_fp32_scores')}


def test_compute_mode_default_and_invalid_combinations():
    assert EncoderConfig().yat_ffn_compute_mode is None
    with pytest.raises(ValueError, match='Unknown yat_ffn_compute_mode'):
        replace(configuration(), yat_ffn_compute_mode='invalid')
    with pytest.raises(ValueError, match='requires YAT FFN'):
        replace(configuration(), ffn_type='geglu')
    with pytest.raises(ValueError, match='BF16 FFN requires'):
        replace(configuration(), compute_dtype='float32')
    with pytest.raises(ValueError, match='requires direct BF16'):
        replace(configuration(), yat_ffn_backward_block=32)
