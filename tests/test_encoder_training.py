import json
import pytest
from dataclasses import asdict
import numpy as np
from flax import nnx
import optax
import jax

from flaxchat.checkpoint import restore_model_from_checkpoint
from flaxchat.encoder import EncoderConfig, ModernBert
from scripts.prepare_encoder_data import prepare
from scripts.train_encoder import parser, run


@pytest.mark.parametrize('compute,residual,projection,backend', [
    ('float32', 'float32', 'dense', 'xla'), ('bfloat16', 'float32', 'dense', 'xla'),
    ('bfloat16', 'bfloat16', 'dense', 'xla'), ('bfloat16', 'float32', 'masked', 'xla'),
    ('bfloat16', 'float32', 'masked', 'pallas')])
def test_preparation_training_and_exact_resume(tmp_path, monkeypatch, compute, residual, projection, backend):
    if backend == 'pallas' and jax.default_backend() == 'cpu':
        import flaxchat.fused_cross_entropy as fused
        original = fused.fused_cross_entropy
        monkeypatch.setattr(fused, 'fused_cross_entropy',
                            lambda *args, **kwargs: original(*args, **(kwargs | {'interpret': True})))
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
    tokenizer = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2,
                                           '[SEP]': 3, '[MASK]': 4, 'hello': 5, 'world': 6}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in
                                  ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']])
    tokpath = tmp_path / 'tokenizer.json'
    tokenizer.save(str(tokpath))
    source = tmp_path / 'texts.jsonl'
    source.write_text('\n'.join(json.dumps({'text': 'hello world hello world'}) for _ in range(8)))
    data = tmp_path / 'data'
    manifest = prepare(source, tokpath, data, sequence_length=6, max_rows=8, pad_token_id=0)
    assert manifest['rows'] == 8
    config = EncoderConfig(vocab_size=7, hidden_size=128 if backend == 'pallas' else 8,
        intermediate_size=192 if backend == 'pallas' else 12,
        num_hidden_layers=1, num_attention_heads=2, loss_chunk_size=4,
        use_remat=True, compute_dtype=compute, residual_dtype=residual, mlm_projection=projection, mlm_loss_backend=backend, mlm_vocab_tile=128)
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(asdict(config)))
    common = ['--config', str(cfg), '--data', str(data), '--steps', '3', '--batch-size', '4',
              '--mask-probability', '1', '--save-every', '2', '--dtype', compute, '--residual-dtype', residual,
              '--loss-chunk-size', '4', '--mlm-projection', projection, '--mlm-loss-backend', backend, '--mlm-vocab-tile', '128']
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    run(parser().parse_args(common + ['--output', str(full), '--preflight-only']))
    assert not full.exists()
    run(parser().parse_args(common + ['--output', str(full)]))
    run(parser().parse_args(common + ['--output', str(resumed), '--stop-after', '2']))
    run(parser().parse_args(common + ['--output', str(resumed), '--resume']))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(parser().parse_args(common + ['--output', str(full), '--resume',
            '--dtype', 'bfloat16', '--residual-dtype', 'bfloat16' if residual == 'float32' else 'float32']))
    restored = []
    for path in (full, resumed):
        model = ModernBert(config, rngs=nnx.Rngs(99))
        optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.),
            optax.adamw(2e-5, weight_decay=.01)), wrt=nnx.Param)
        _, state = restore_model_from_checkpoint(model, str(path), optimizer=optimizer, load_training_state=True)
        assert int(state['completed_steps'][0]) == 3
        assert int(state['optimizer_updates'][0]) == 3
        for value in jax.tree.leaves(optimizer.opt_state):
            if jax.numpy.issubdtype(value.dtype, jax.numpy.floating):
                assert value.dtype == np.float32
        restored.append((nnx.state(model), optimizer.opt_state))
    for a, b in zip(jax.tree.leaves(restored[0]), jax.tree.leaves(restored[1]), strict=True):
        np.testing.assert_array_equal(a, b)

    from scripts.evaluate_encoder import evaluate
    with pytest.raises(ValueError, match='No held-out rows'):
        evaluate(str(full), str(data), train_data=str(data))
    validation_source = tmp_path / 'validation.jsonl'
    validation_source.write_text(json.dumps({'text': 'world hello world hello world'}) + '\n')
    validation = tmp_path / 'validation'
    prepare(validation_source, tokpath, validation, sequence_length=6, max_rows=4, pad_token_id=0)
    report = evaluate(str(full), str(validation), train_data=str(data), batch_size=2)
    assert report['masked_tokens'] == 5
    assert np.isfinite(report['masked_token_loss'])
    with pytest.raises(ValueError, match='Output already contains'):
        run(parser().parse_args(common + ['--output', str(full)]))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(parser().parse_args(common + ['--output', str(full), '--resume', '--seed', '99']))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(parser().parse_args(common + ['--output', str(full), '--resume', '--mlm-projection',
                                         'masked' if projection == 'dense' else 'dense']))


def test_two_process_encoder_resume(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path
    from scripts.train_encoder import file_hash
    if os.environ.get('FLAXCHAT_RUN_DISTRIBUTED_CPU') != '1':
        pytest.skip('Set FLAXCHAT_RUN_DISTRIBUTED_CPU=1 for localhost multiprocess test')
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=1, num_attention_heads=2, loss_chunk_size=4, compute_dtype='bfloat16')
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(asdict(config)))
    data = tmp_path / 'data'
    data.mkdir()
    np.save(data / 'tokens.npy', np.array([[1, 5, 6, 0], [1, 7, 8, 0], [1, 9, 10, 0], [1, 11, 12, 0]], dtype=np.int32))
    (data / 'manifest.json').write_text(json.dumps(dict(format='flaxchat-encoder-rows-v1',
        vocab_size=16, pad_token_id=0, special_token_ids=[0, 1, 4], tokenizer_sha256='synthetic-only',
        tokens_sha256=file_hash(data / 'tokens.npy'))))
    root = Path(__file__).resolve().parents[1]
    common = ['--config', str(cfg), '--data', str(data), '--output', str(tmp_path / 'checkpoints'),
        '--steps', '2', '--batch-size', '4', '--mask-probability', '1', '--save-every', '1',
        '--dtype', 'bfloat16', '--loss-chunk-size', '4', '--distributed', '--shared-local-checkpoints']
    for phase in (['--stop-after', '1'], ['--resume']):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        jobs = []
        try:
            for rank in range(2):
                env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_COORDINATOR_ADDRESS=f'127.0.0.1:{port}',
                           JAX_PROCESS_COUNT='2', JAX_PROCESS_INDEX=str(rank),
                           XLA_FLAGS='--xla_force_host_platform_device_count=1')
                path = tmp_path / f'worker-{phase[0].strip("-")}-{rank}.log'
                log = path.open('w')
                jobs.append((subprocess.Popen([sys.executable, '-m', 'scripts.train_encoder', *common, *phase],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log, path))
            for process, _, path in jobs:
                assert process.wait(timeout=90) == 0, path.read_text()[-12000:]
        finally:
            for process, log, _ in jobs:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                log.close()
    from flaxchat.checkpoint import load_checkpoint_metadata
    metadata = load_checkpoint_metadata(str(tmp_path / 'checkpoints'))
    assert metadata['model_family'] == 'modernbert'


def test_training_precision_default():
    args = parser().parse_args(['--config', 'config.json', '--data', 'data', '--output', 'output'])
    assert args.dtype == 'bfloat16'
    assert args.residual_dtype == 'float32'


def test_preparer_protects_mask_token_even_when_tokenizer_flag_is_false(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<unk>': 1, 'a': 2,
                                            'b': 3, '<mask>': 4, 'hello': 5}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_tokens([AddedToken('<mask>', special=False)])
    path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(path))
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'text': '<mask> hello'}))
    manifest = prepare(source, path, tmp_path / 'data', sequence_length=4,
                       max_rows=2, pad_token_id=0, mask_token_id=4)
    assert manifest['mask_token_id'] == 4
    assert 4 in manifest['special_token_ids']
    from flaxchat.mlm import mask_tokens
    tokens = np.load(tmp_path / 'data/tokens.npy')
    _, targets = mask_tokens(tokens, seed=0, step=0, example_ids=np.array([0]),
        vocab_size=6, mask_token_id=4, special_token_ids=manifest['special_token_ids'], probability=1)
    assert targets[0, 0] == -1
    assert targets[0, 1] == 5
