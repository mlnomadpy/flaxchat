"""Optional YAT gradient strategy must persist through training and resume."""
from dataclasses import replace
import pytest
from flaxchat.encoder import EncoderConfig
from tests.test_encoder_training import test_preparation_training_and_exact_resume as _training_check


@pytest.mark.parametrize('recipe', ['yat_max_centered', 'yat_max_centered_tiled'])
def test_max_centered_training_exact_resume_and_identity(tmp_path, monkeypatch, recipe):
    _training_check(tmp_path, monkeypatch, 'bfloat16', 'bfloat16', 'dense', 'xla', recipe)


def test_backward_strategy_requires_bf16_yat_and_has_legacy_default():
    assert EncoderConfig().yat_softmax_backward == 'factored'
    base = EncoderConfig(attention_score='yat_softmax', yat_compute_mode='bf16_adaptive',
                         compute_dtype='bfloat16', residual_dtype='bfloat16')
    assert replace(base, yat_softmax_backward='max_centered').yat_bias == 1
    for changes in [dict(yat_softmax_backward='other'),
                    dict(yat_softmax_backward='max_centered', attention_score='dot_product'),
                    dict(yat_softmax_backward='max_centered', yat_compute_mode='mixed')]:
        with pytest.raises(ValueError):
            replace(base, **changes)


def test_new_stage_does_not_treat_backward_strategy_as_execution_only(monkeypatch):
    from dataclasses import asdict
    from scripts.train_encoder import initialization_metadata

    config = EncoderConfig(attention_score='yat_softmax', yat_compute_mode='bf16_adaptive',
                           compute_dtype='bfloat16', residual_dtype='bfloat16')
    legacy = asdict(config)
    legacy.pop('yat_softmax_backward')
    parent = dict(model_family='modernbert', resolved_config=dict(encoder=legacy),
                  tokenizer_identity='tokenizer', step=3)
    monkeypatch.setattr('scripts.train_encoder.load_checkpoint_metadata', lambda *a: parent)
    _, receipt = initialization_metadata('parent', 3, config, 'tokenizer')
    assert receipt['step'] == 3
    with pytest.raises(ValueError, match='configuration differs'):
        initialization_metadata('parent', 3, replace(config, yat_softmax_backward='max_centered'), 'tokenizer')
