"""New data/schedule stages transfer weights without relaxing exact-run resume."""
from dataclasses import asdict, replace
import json

import jax
from flax import nnx
import numpy as np
import optax
import pytest

from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_data import file_hash
from scripts import train_encoder as trainer


def test_new_stage_preserves_parent_weights_and_resets_optimizer(tmp_path, monkeypatch):
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
                           num_hidden_layers=1, num_attention_heads=2, loss_chunk_size=4,
                           compute_dtype='float32', residual_dtype='float32',
                           ffn_type='yat_glu', attention_score='yat_softmax')
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(asdict(config)))
    def data(name, token):
        path = tmp_path / name
        path.mkdir()
        np.save(path / 'tokens.npy', np.array([[1, token, token, 0]]*4, dtype=np.int32))
        (path / 'manifest.json').write_text(json.dumps(dict(format='flaxchat-encoder-rows-v1',
            vocab_size=16, pad_token_id=0, special_token_ids=[0, 1, 4],
            tokenizer_sha256='synthetic', tokens_sha256=file_hash(path / 'tokens.npy'))))
        return path
    old_data, new_data = data('old', 5), data('new', 7)
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    common = ['--config', str(cfg), '--dtype', 'float32', '--residual-dtype', 'float32',
              '--loss-chunk-size', '4', '--batch-size', '4', '--mask-probability', '1', '--save-every', '1']
    def args(output, dataset, horizon, extra=()):
        return trainer.parser().parse_args(common + ['--output', str(output), '--data', str(dataset),
                                                    '--steps', str(horizon), *extra])
    trainer.run(args(parent, old_data, 2))
    original = ModernBert(config, rngs=nnx.Rngs(99))
    restore_model_from_checkpoint(original, str(parent), step=1)
    parent_weights = jax.tree.map(np.asarray, nnx.state(original))
    restore = trainer.restore_model_from_checkpoint
    observed = []
    def verify_restore(model, path, **kwargs):
        metadata = restore(model, path, **kwargs)
        if path == str(parent):
            assert kwargs['step'] == 1
            for a, b in zip(jax.tree.leaves(parent_weights), jax.tree.leaves(nnx.state(model)), strict=True):
                np.testing.assert_array_equal(a, b)
            observed.append(True)
        return metadata
    monkeypatch.setattr(trainer, 'restore_model_from_checkpoint', verify_restore)
    initialize = ['--initialize-from-checkpoint', str(parent), '--initialize-step', '1']
    _, receipt = trainer.initialization_metadata(str(parent), 1, replace(config,
        yat_local_shards=True, yat_attention_block_size=64, yat_global_attention_block_size=32), 'synthetic')
    assert receipt['execution_overrides']['yat_attention_block_size'] == dict(parent=0, current=64)
    assert receipt['execution_overrides']['yat_global_attention_block_size'] == dict(parent=0, current=32)
    trainer.run(args(child, new_data, 4, initialize + ['--preflight-only']))
    assert not child.exists()
    trainer.run(args(child, new_data, 4, initialize + ['--stop-after', '1']))
    assert observed == [True]
    metadata = load_checkpoint_metadata(str(child))
    assert metadata['initialization']['step'] == 1
    assert 'fresh_optimizer' in metadata['initialization']['policy']
    assert metadata['data_manifest_identity'] != load_checkpoint_metadata(str(parent))['data_manifest_identity']
    restored = ModernBert(config, rngs=nnx.Rngs(10))
    optimizer = nnx.Optimizer(restored, optax.chain(optax.clip_by_global_norm(1.),
                              optax.adamw(2e-5, weight_decay=.01)), wrt=nnx.Param)
    _, state = restore_model_from_checkpoint(restored, str(child), optimizer=optimizer, load_training_state=True)
    assert state['completed_steps'].tolist() == [1]
    assert state['optimizer_updates'].tolist() == [1]
    # Adam's counter and moments must belong to the new stage, not the parent.
    counters = [int(v) for v in jax.tree.leaves(optimizer.opt_state)
                if np.shape(v) == () and np.issubdtype(v.dtype, np.integer)]
    assert counters and set(counters) == {1}
    trainer.run(args(child, new_data, 4, ['--resume', '--stop-after', '2']))
    assert load_checkpoint_metadata(str(child))['initialization'] == metadata['initialization']
    with pytest.raises(ValueError, match='identity mismatch'):
        trainer.run(args(child, new_data, 4, ['--resume', '--yat-global-attention-block-size', '32']))
    with pytest.raises(ValueError, match='identity mismatch'):
        trainer.run(args(child, old_data, 4, ['--resume']))
    with pytest.raises(ValueError, match='identity mismatch'):
        trainer.run(args(child, new_data, 5, ['--resume']))
    with pytest.raises(ValueError, match='Choose checkpoint'):
        trainer.run(args(child, new_data, 4, initialize + ['--resume']))
    with pytest.raises(ValueError, match='different output'):
        trainer.run(args(parent, new_data, 4, initialize))
    with pytest.raises(ValueError, match='encoder configuration differs'):
        trainer.run(args(tmp_path/'wrong-model', new_data, 4, initialize + ['--ffn-type', 'geglu', '--preflight-only']))
    path = new_data / 'manifest.json'
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps(manifest | {'tokenizer_sha256': 'different'}))
    with pytest.raises(ValueError, match='tokenizer differs'):
        trainer.run(args(tmp_path/'wrong-tokenizer', new_data, 4, initialize + ['--preflight-only']))
