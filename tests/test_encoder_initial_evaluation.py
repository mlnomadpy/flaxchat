import pytest
from flaxchat.encoder import EncoderConfig
from scripts.evaluate_encoder import initial_weight_identity
from scripts import train_encoder


def test_initial_baseline_requires_exact_checkpoint_weight_identity(monkeypatch):
    expected = {'model.safetensors': 'abc'}
    seen = []
    def inventory(directory, config):
        seen.append((directory, config))
        return expected.copy()
    monkeypatch.setattr(train_encoder, 'pretrained_inventory', inventory)
    config = EncoderConfig()
    assert initial_weight_identity({'initial_weights_sha256': expected}, config, 'snapshot') == expected
    assert seen == [('snapshot', config)]
    with pytest.raises(ValueError, match='identity mismatch'):
        initial_weight_identity({'initial_weights_sha256': {'model.safetensors':'different'}}, config, 'snapshot')


@pytest.mark.parametrize('metadata', [{}, {'initial_weights_sha256':None}, {'initial_weights_sha256':{}}])
def test_missing_initial_identity_is_not_a_baseline(metadata):
    with pytest.raises(ValueError, match='does not record'):
        initial_weight_identity(metadata, EncoderConfig(), 'unused')


@pytest.mark.parametrize('config', [EncoderConfig(ffn_type='yat_glu'), EncoderConfig(attention_score='yat_softmax')])
def test_baseline_does_not_silently_change_architecture(config):
    with pytest.raises(ValueError, match='released GeGLU'):
        initial_weight_identity({'initial_weights_sha256':{'model.safetensors':'abc'}}, config, 'unused')


def test_initial_evaluation_uses_starting_parameters_and_same_mask_policy(monkeypatch):
    from dataclasses import asdict
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from scripts import evaluate_encoder as evaluator
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
                           num_hidden_layers=1, num_attention_heads=2,
                           max_position_embeddings=16)
    hashes = {'model.safetensors':'pinned'}
    metadata = dict(step=1972, model_family='modernbert', tokenizer_identity='tokenizer',
                    data_manifest_identity='manifest', initial_weights_sha256=hashes,
                    resolved_config=dict(encoder=asdict(config), special_token_ids=[0,1,2,3,4],
                                         mask_probability=1.0))
    monkeypatch.setattr(evaluator, 'load_checkpoint_metadata', lambda *args, **kwargs:metadata)
    monkeypatch.setattr(evaluator, 'file_hash', lambda _:'manifest')
    def rows(path, config):
        token = 5 if str(path) == 'train' else 6
        return np.array([[token,token,0,0]],np.int32),dict(tokenizer_sha256='tokenizer',special_token_ids=[0,1,2,3,4])
    monkeypatch.setattr(evaluator, 'load_prepared_rows', rows)
    monkeypatch.setattr(train_encoder, 'pretrained_inventory', lambda *args:hashes)
    def load(model, path):
        assert path == 'initial'
        nnx.update(model, jax.tree.map(jnp.zeros_like, nnx.state(model)))
        return hashes
    monkeypatch.setattr(train_encoder, 'load_pretrained', load)
    def forbidden(*args, **kwargs):
        raise AssertionError('Baseline must not restore trained parameters')
    monkeypatch.setattr(evaluator, 'restore_model_from_checkpoint', forbidden)
    result = evaluator.evaluate('candidate', 'validation', train_data='train', initial_pretrained='initial')
    assert result['parameter_source'] == 'initial_pretrained'
    assert result['initial_weights_sha256'] == hashes
    assert result['masked_tokens'] == 2 and result['evaluated_rows'] == 1
    assert result['masked_token_loss'] == pytest.approx(np.log(16),abs=1e-5)
    def restore_pinned(model, checkpoint, *, step, expected_identity):
        assert checkpoint == 'candidate' and step == 1972
        assert expected_identity == {'resolved_config':metadata['resolved_config']}
        load(model, 'initial')
    monkeypatch.setattr(evaluator, 'restore_model_from_checkpoint', restore_pinned)
    candidate = evaluator.evaluate('candidate', 'validation', train_data='train', checkpoint_step=1972)
    assert candidate['checkpoint_step'] == 1972
    assert candidate['parameter_source'] == 'checkpoint'
    assert candidate['selected_rows_sha256'] == result['selected_rows_sha256']
    monkeypatch.setattr(train_encoder, 'load_pretrained', lambda *args:{'model.safetensors':'changed'})
    with pytest.raises(ValueError, match='changed while loading'):
        evaluator.evaluate('candidate','validation',train_data='train',initial_pretrained='initial')
