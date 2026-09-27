from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from flaxchat.encoder import EncoderConfig, bidirectional_attention


def centered_config():
    return EncoderConfig(
        compute_dtype='bfloat16', attention_score='yat_softmax', yat_compute_mode='bf16',
        yat_attention_implementation='centered_fp32_scores',
        yat_attention_block_size=4, yat_global_attention_block_size=4,
    )


@pytest.mark.parametrize('changes', [
    {'yat_compute_mode': 'mixed'}, {'yat_compute_mode': 'bf16_adaptive'},
    {'attention_score': 'dot_product'}, {'yat_attention_block_size': 0},
    {'yat_global_attention_block_size': 0}, {'yat_softmax_backward': 'max_centered'},
])
def test_centered_precision_must_be_selected_consistently(changes):
    with pytest.raises(ValueError, match='centered_fp32_scores'):
        replace(centered_config(), **changes)


@pytest.mark.parametrize('packed', [True, False])
def test_centered_encoder_routing_respects_document_isolation(packed):
    shape = (1, 5, 1, 4)
    q = k = jnp.zeros(shape, jnp.bfloat16)
    v = jnp.broadcast_to(jnp.array([1, 1, 9, 9, 100], jnp.bfloat16)[None, :, None, None], shape)
    segments = jnp.array([[0, 0, 1, 1, -1]])
    output = bidirectional_attention(
        q, k, v, segments, score='yat_softmax', yat_compute_mode='bf16',
        yat_attention_implementation='centered_fp32_scores', yat_attention_block_size=4,
        packed=packed,
    )
    expected = np.array([1, 1, 9, 9, 0] if packed else [5, 5, 5, 5, 0])
    np.testing.assert_array_equal(np.asarray(output[0, :, 0, 0], float), expected)


def test_default_encoder_does_not_enable_experimental_precision():
    assert EncoderConfig().yat_attention_implementation == 'standard'
