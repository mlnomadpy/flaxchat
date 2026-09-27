"""Weight inventory failures must be rejected before paid model initialization."""
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.train_encoder import pretrained_inventory


def test_single_snapshot_inventory_is_content_addressed(tmp_path):
    path = tmp_path / 'model.safetensors'
    save_file({'weight': np.ones((2, 3), dtype=np.float32)}, str(path))
    original = pretrained_inventory(tmp_path)
    assert set(original) == {'model.safetensors'}
    save_file({'weight': np.zeros((2, 3), dtype=np.float32)}, str(path))
    assert pretrained_inventory(tmp_path) != original


@pytest.mark.parametrize('defect', ['missing', 'truncated', 'wrong-index', 'duplicate',
                                   'traversal', 'empty-index'])
def test_snapshot_rejects_invalid_shards(tmp_path, defect):
    first = tmp_path / 'model.safetensors'
    save_file({'weight': np.ones((2, 3), dtype=np.float32)}, str(first))
    if defect == 'missing':
        first.unlink()
    elif defect == 'truncated':
        first.write_bytes(first.read_bytes()[:-1])
    else:
        mapping = {'weight': first.name}
        if defect == 'wrong-index':
            mapping = {'absent': first.name}
        elif defect == 'empty-index':
            mapping = {}
        elif defect == 'traversal':
            mapping = {'weight': '../outside.safetensors'}
        elif defect == 'duplicate':
            save_file({'weight': np.zeros((2, 3), dtype=np.float32)}, str(tmp_path / 'second.safetensors'))
            mapping['other'] = 'second.safetensors'
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    from safetensors import SafetensorError
    with pytest.raises((ValueError, SafetensorError)):
        pretrained_inventory(tmp_path)


def test_sharded_snapshot_inventory(tmp_path):
    mapping = {}
    for name in ('a', 'b'):
        shard = f'{name}.safetensors'
        save_file({name: np.ones((2,), dtype=np.float32)}, str(tmp_path / shard))
        mapping[name] = shard
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    assert set(pretrained_inventory(tmp_path)) == {'a.safetensors', 'b.safetensors'}


@pytest.mark.parametrize('defect', ['missing-weights', 'worker-weights', 'worker-runtime', None])
def test_input_preflight_checks_snapshot_before_model_allocation(tmp_path, monkeypatch, defect):
    from scripts import train_encoder as trainer
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(model_type='modernbert', vocab_size=8, hidden_size=8,
        intermediate_size=12, num_hidden_layers=1, num_attention_heads=2)))
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    data = tmp_path / 'data'
    data.mkdir()
    np.save(data / 'tokens.npy', np.array([[1, 5, 6, 0]] * 4, dtype=np.int32))
    (data / 'manifest.json').write_text(json.dumps(dict(format='flaxchat-encoder-rows-v1',
        tokenizer_sha256=trainer.file_hash(tokenizer), vocab_size=8, pad_token_id=0,
        mask_token_id=4, special_token_ids=[0, 1, 4], tokens_sha256=trainer.file_hash(data/'tokens.npy'))))
    if defect != 'missing-weights':
        from flaxchat.encoder import EncoderConfig
        shapes = trainer.pretrained_shapes(EncoderConfig.from_hf(json.loads(config.read_text())))
        save_file({key: np.ones(shape, dtype=np.float32) for key, shape in shapes.items()}, str(tmp_path/'model.safetensors'))
    def agreement(value):
        assert value['initial_weights_sha256']
        assert value['resolved_config']['runtime']
        other = json.loads(json.dumps(value))
        if defect == 'worker-weights':
            other['initial_weights_sha256']['model.safetensors'] = 'different'
        if defect == 'worker-runtime':
            other['resolved_config']['runtime']['python'] = 'different'
        return [value, other]
    monkeypatch.setattr(trainer, 'gather_process_metadata', agreement)
    monkeypatch.setattr(trainer, 'ModernBert', lambda *a, **kw: pytest.fail('Preflight must not allocate a model'))
    args = trainer.parser().parse_args(['--config', str(config), '--pretrained', str(tmp_path),
        '--data', str(data), '--output', str(tmp_path/'output'), '--batch-size', '4', '--preflight-only'])
    if defect:
        with pytest.raises(ValueError, match='shard|Workers disagree'):
            trainer.run(args)
    else:
        trainer.run(args)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('defect', ['shape', 'name', 'missing', 'decoder', 'nan', 'untied'])
def test_snapshot_checks_architecture_headers_without_model(tmp_path, defect):
    from flaxchat.encoder import EncoderConfig
    from scripts.train_encoder import pretrained_shapes
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
                           num_hidden_layers=2, num_attention_heads=2)
    tensors = {key: np.zeros(shape, np.float32) for key, shape in pretrained_shapes(config).items()}
    if defect == 'shape':
        tensors['head.dense.weight'] = np.zeros((1, 1), np.float32)
    elif defect == 'name':
        tensors['unexpected'] = np.zeros(1, np.float32)
    elif defect == 'missing':
        del tensors['head.norm.weight']
    elif defect == 'nan':
        tensors['head.norm.weight'][0] = np.nan
    elif defect == 'untied':
        tensors['decoder.weight'] = np.ones((16, 8), np.float32)
    else:
        tensors['decoder.weight'] = np.zeros((1, 1), np.float32)
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    with pytest.raises(ValueError, match='shape|architecture|Nonfinite|tied'):
        pretrained_inventory(tmp_path, config)
