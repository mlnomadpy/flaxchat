"""Configuration-only checks; no model execution or device arrays."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from flaxchat.encoder import EncoderConfig


@pytest.mark.parametrize('architecture', ['goat', 'goat_input'])
def test_goat_is_explicit_and_fixed_constants_are_enforced(architecture):
    original = EncoderConfig()
    assert original.attention_score == 'dot_product'
    goat = replace(original, attention_score=architecture)
    assert goat.yat_bias == 1 and goat.yat_epsilon == .01 and goat.yat_alpha_trainable
    for changes in ({'yat_bias': 0}, {'yat_epsilon': .1}, {'yat_alpha_trainable': False},
                    {'attention_backend': 'splash'}, {'yat_local_shards': True},
                    {'weight_quantization': 'int8_per_channel_ste'}):
        with pytest.raises(ValueError):
            replace(goat, **changes)


@pytest.mark.parametrize('architecture', ['goat', 'goat_input'])
def test_goat_accepts_parent_precision_policy(architecture):
    EncoderConfig(attention_score=architecture, compute_dtype='bfloat16', yat_compute_mode='bf16',
                  yat_attention_implementation='centered_fp32_scores',
                  yat_attention_block_size=64, yat_global_attention_block_size=64)


@pytest.mark.parametrize('architecture', ['goat', 'goat_input'])
def test_raw_hf_import_requires_explicit_architecture_migration(architecture):
    from flaxchat.encoder import import_hf_weights
    model = SimpleNamespace(config=EncoderConfig(attention_score=architecture))
    with pytest.raises(ValueError, match='explicit QKV-to-V migration'):
        import_hf_weights(model, {})


def test_input_geometry_has_distinct_checkpoint_configuration():
    original = EncoderConfig(attention_score='goat')
    corrected = replace(original, attention_score='goat_input')
    assert original != corrected
    assert original.attention_score == 'goat'
