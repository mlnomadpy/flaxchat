import json
from dataclasses import asdict

import pytest

from flaxchat.encoder import EncoderConfig
from scripts.train_yat_mmbert import parser, validate_recipe


def arguments(tmp_path, extra=(), **changes):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(asdict(EncoderConfig()) | {'model_type': 'modernbert'} | changes))
    return parser().parse_args(['--config', str(config), '--pretrained', str(tmp_path),
                               '--data', 'data', '--output', 'output', *extra])


def test_agreed_architecture_and_parameter_count(tmp_path):
    config, count = validate_recipe(arguments(tmp_path))
    assert 300_000_000 <= count <= 500_000_000
    assert config.yat_bias == 1 and config.yat_epsilon == .01
    assert config.yat_alpha_trainable
    assert config.ffn_type == 'yat_glu' and config.attention_score == 'yat_softmax'
    assert config.compute_dtype == config.residual_dtype == 'bfloat16'


@pytest.mark.parametrize('extra', [('--ffn-type', 'geglu'), ('--attention-score', 'dot_product'),
    ('--yat-compute-mode', 'mixed'), ('--dtype', 'float32'),
    ('--attention-backend', 'splash'), ('--mlm-projection', 'dense')])
def test_recipe_cannot_silently_train_baseline(tmp_path, extra):
    with pytest.raises(ValueError, match='requires'):
        validate_recipe(arguments(tmp_path, extra))


def test_size_and_fixed_constants_are_enforced(tmp_path):
    with pytest.raises(ValueError, match='parameters'):
        validate_recipe(arguments(tmp_path, vocab_size=16))
    with pytest.raises(ValueError, match='fixed bias'):
        validate_recipe(arguments(tmp_path, yat_bias=2))
    args = arguments(tmp_path)
    args.pretrained = None
    with pytest.raises(ValueError, match='pretrained'):
        validate_recipe(args)


def test_cli_rejects_nonfixed_epsilon(tmp_path):
    with pytest.raises(SystemExit) as error:
        arguments(tmp_path, ('--yat-epsilon', '.1'))
    assert error.value.code == 2


def test_recipe_propagates_optional_backward_strategy(tmp_path):
    config, _ = validate_recipe(arguments(tmp_path, ('--yat-softmax-backward', 'max_centered')))
    assert config.yat_softmax_backward == 'max_centered'


def test_attention_alpha_override_is_separate_trainable_and_checkpoint_identified(tmp_path, monkeypatch):
    import jax.numpy as jnp
    from flax import nnx
    from flaxchat.encoder import EncoderBlock
    from scripts.train_encoder import initialization_metadata
    args = arguments(tmp_path, ('--yat-alpha', '2', '--yat-attention-alpha', '.1'))
    config, _ = validate_recipe(args)
    block = EncoderBlock(config, 0, rngs=nnx.Rngs(0))
    assert isinstance(block.yat_attention_alpha, nnx.Param)
    assert float(block.yat_alpha[...]) == 2
    assert float(block.yat_attention_alpha[...]) == pytest.approx(.1)
    block.yat_attention_alpha[...] += jnp.float32(.01)
    assert float(block.yat_attention_alpha[...]) == pytest.approx(.11)
    legacy = asdict(config)
    legacy.pop('yat_attention_alpha')
    parent = dict(model_family='modernbert', resolved_config=dict(encoder=legacy),
                  tokenizer_identity='tok', step=3)
    monkeypatch.setattr('scripts.train_encoder.load_checkpoint_metadata', lambda *a: parent)
    with pytest.raises(ValueError, match='configuration differs'):
        initialization_metadata('parent', 3, config, 'tok')
    inherited = EncoderConfig(**legacy)
    initialization_metadata('parent', 3, inherited, 'tok')
    legacy_block = EncoderBlock(inherited, 0, rngs=nnx.Rngs(0))
    assert float(legacy_block.yat_attention_alpha[...]) == 2


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_attention_alpha_rejects_invalid_initial_values(value):
    with pytest.raises(ValueError, match='attention alpha'):
        EncoderConfig(yat_attention_alpha=value)


def test_local_accumulation_recipe_preserves_model_constants(tmp_path):
    args = arguments(tmp_path, ('--local-gradient-accumulation', '--accumulation-steps', '4'))
    config, count = validate_recipe(args)
    assert args.local_gradient_accumulation and args.accumulation_steps == 4
    assert config.yat_bias == 1 and config.yat_epsilon == .01
    assert config.yat_alpha_trainable and 300_000_000 <= count <= 500_000_000
