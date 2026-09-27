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
from scripts.train_encoder import parser, run, learning_rate_schedule


@pytest.mark.parametrize('compute,residual,projection,backend', [
    ('float32', 'float32', 'dense', 'xla'), ('bfloat16', 'float32', 'dense', 'xla'),
    ('bfloat16', 'bfloat16', 'dense', 'xla'), ('bfloat16', 'float32', 'masked', 'xla'),
    ('bfloat16', 'float32', 'masked', 'pallas'),
    ('bfloat16', 'float32', 'masked', 'xla_full'),
    ('bfloat16', 'float32', 'masked', 'xla_local')])
@pytest.mark.parametrize('recipe', ['original', 'scheduled_accumulation'])
def test_preparation_training_and_exact_resume(tmp_path, monkeypatch, compute, residual, projection, backend, recipe, attention_alpha=None, local_gradient_accumulation=False, ffn_backward_block=None, projection_capacity=None, ffn_compute_mode=None):
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
    fixture_rows = max(8, 2 * jax.device_count()) if local_gradient_accumulation else 8
    source.write_text('\n'.join(json.dumps({'text': 'hello world hello world'}) for _ in range(fixture_rows)))
    data = tmp_path / 'data'
    manifest: dict[str, object] = dict(prepare(
        source, tokpath, data, sequence_length=6, max_rows=fixture_rows, pad_token_id=0))
    assert manifest['rows'] == fixture_rows
    if recipe in ('language', 'mixture'):
        documents = [dict(id='first', language_script='a', first_row=0, end_row=4),
                     dict(id='second', language_script='b', first_row=4, end_row=8)]
        (data / 'documents.jsonl').write_text('\n'.join(map(json.dumps, documents)))
        from flaxchat.encoder_data import file_hash
        prepared = np.load(data / 'tokens.npy')
        manifest.update(split='train', documents=2, documents_sha256=file_hash(data / 'documents.jsonl'),
                        language_counts={name: dict(rows=4, nonpadding_tokens=int((prepared[start:start+4] != 0).sum()))
                                         for name, start in [('a', 0), ('b', 4)]})
        if recipe == 'mixture':
            documents = [dict(id=str(i), language_script='a' if i < 2 else 'b',
                              source_group='hq' if i % 2 == 0 else 'english',
                              first_row=i*2, end_row=i*2+2) for i in range(4)]
            manifest['documents'] = 4
            (data / 'documents.jsonl').write_text('\n'.join(map(json.dumps, documents)))
            manifest.update(documents_sha256=file_hash(data / 'documents.jsonl'),
                            source_token_shares=dict(hq=.6, english=.4))
        (data / 'manifest.json').write_text(json.dumps(manifest))
    config = EncoderConfig(vocab_size=7, yat_attention_alpha=attention_alpha, ffn_type='yat_glu' if recipe.startswith('yat') else 'geglu',
        attention_score='yat_softmax' if recipe in ('yat_both', 'yat_bf16', 'yat_adaptive', 'yat_adaptive_local', 'yat_max_centered', 'yat_max_centered_tiled', 'yat_centered_fp32') else 'dot_product',
        yat_compute_mode={'yat_centered_fp32': 'bf16', 'yat_bf16': 'bf16', 'yat_adaptive': 'bf16_adaptive', 'yat_adaptive_local': 'bf16_adaptive', 'yat_max_centered': 'bf16_adaptive', 'yat_max_centered_tiled': 'bf16_adaptive'}.get(recipe, 'mixed'), hidden_size=128 if backend == 'pallas' else 8,
        intermediate_size=192 if backend == 'pallas' else 12,
        num_hidden_layers=1, num_attention_heads=2, loss_chunk_size=4,
        use_remat=True, compute_dtype=compute, residual_dtype=residual, mlm_projection=projection, mlm_loss_backend=backend, mlm_vocab_tile=128)
    if recipe == 'yat_adaptive_local':
        from dataclasses import replace
        config = replace(config, yat_local_shards=True)
    if recipe.startswith('yat_max_centered'):
        from dataclasses import replace
        config = replace(config, yat_softmax_backward='max_centered',
                         yat_global_attention_block_size=4 if recipe.endswith('_tiled') else 0)
    if recipe == 'yat_centered_fp32':
        from dataclasses import replace
        config = replace(config, yat_attention_implementation='centered_fp32_scores',
                         yat_attention_block_size=4, yat_global_attention_block_size=4,
                         num_hidden_layers=2, global_attn_every_n_layers=2)
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(asdict(config)))
    common = ['--config', str(cfg), '--data', str(data), '--steps', '3', '--batch-size', '4',
              '--mask-probability', '1', '--save-every', '2', '--dtype', compute, '--residual-dtype', residual,
              '--loss-chunk-size', '4', '--mlm-projection', projection, '--mlm-loss-backend', backend, '--mlm-vocab-tile', '128']
    if ffn_compute_mode is not None:
        common += ['--yat-ffn-compute-mode', ffn_compute_mode]
    if projection_capacity is not None:
        common += ['--mlm-projection-capacity', str(projection_capacity)]
    if ffn_backward_block is not None:
        common += ['--yat-ffn-backward-block', str(ffn_backward_block)]
    if recipe == 'yat_adaptive_local':
        common += ['--yat-local-shards', '--compile-diagnostics']
        if config.yat_attention_block_size:
            common += ['--yat-attention-block-size', str(config.yat_attention_block_size)]
        if config.yat_global_attention_block_size:
            common += ['--yat-global-attention-block-size', str(config.yat_global_attention_block_size)]
    if recipe.startswith('yat_max_centered'):
        common += ['--yat-softmax-backward', 'max_centered',
                   '--yat-global-attention-block-size', str(config.yat_global_attention_block_size)]
    if recipe == 'yat_centered_fp32':
        common += ['--yat-attention-implementation', 'centered_fp32_scores',
                   '--yat-attention-block-size', '4', '--yat-global-attention-block-size', '4']
    if local_gradient_accumulation:
        common += ['--local-gradient-accumulation', '--accumulation-steps', '2', '--batch-size', str(2 * jax.device_count())]
    if recipe == 'donation':
        common += ['--donate-state', '--compile-diagnostics']
    if recipe == 'profile':
        common += ['--profile-dir', str(tmp_path / 'traces'), '--profile-skip-steps', '1', '--profile-steps', '1']
    if recipe == 'scheduled_accumulation':
        common += ['--compile-diagnostics', '--accumulation-steps', '2', '--lr-schedule', 'cosine', '--warmup-steps', '1', '--shuffle']
    if recipe.startswith('yat'):
        common += ['--ffn-type', 'yat_glu', '--yat-epsilon', '.01', '--yat-alpha', '1']
    if recipe in ('yat_both', 'yat_bf16', 'yat_adaptive', 'yat_adaptive_local', 'yat_max_centered', 'yat_max_centered_tiled', 'yat_centered_fp32'):
        common += ['--attention-score', 'yat_softmax']
    if recipe in ('language', 'mixture'):
        common += ['--language-exponent', '.5']
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    run(parser().parse_args(common + ['--output', str(full), '--preflight-only']))
    assert not full.exists()
    run(parser().parse_args(common + ['--output', str(full)]))
    run(parser().parse_args(common + ['--output', str(resumed), '--stop-after', '2']))
    with pytest.raises(ValueError, match='precedes restored'):
        run(parser().parse_args(common + ['--output', str(resumed), '--resume', '--stop-after', '1']))
    run(parser().parse_args(common + ['--output', str(resumed), '--resume']))
    run(parser().parse_args(common + ['--output', str(resumed), '--resume']))  # explicit completed no-op
    if recipe.startswith('yat_max_centered'):
        from flaxchat.checkpoint import load_checkpoint_metadata
        metadata = load_checkpoint_metadata(str(resumed))
        assert metadata['resolved_config']['encoder']['yat_softmax_backward'] == 'max_centered'
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume',
                                             '--yat-softmax-backward', 'factored']))
    if recipe == 'yat_centered_fp32':
        from flaxchat.checkpoint import load_checkpoint_metadata
        metadata = load_checkpoint_metadata(str(resumed))
        assert metadata['resolved_config']['encoder']['yat_attention_implementation'] == 'centered_fp32_scores'
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume',
                                             '--yat-attention-implementation', 'standard']))
    if ffn_compute_mode is not None:
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume',
                                             '--yat-ffn-compute-mode', 'bf16']))
    if projection_capacity is not None:
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume',
                                             '--mlm-projection-capacity', str(projection_capacity + 1)]))
    if ffn_backward_block is not None:
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume',
                                             '--yat-ffn-backward-block', '8']))
    if recipe in ('language', 'mixture'):
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume', '--language-exponent', '.3']))
    if recipe == 'mixture':
        changed_manifest = manifest | {'source_token_shares': dict(hq=.5, english=.5)}
        (data / 'manifest.json').write_text(json.dumps(changed_manifest))
        with pytest.raises(ValueError, match='identity mismatch'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume']))
        (data / 'manifest.json').write_text(json.dumps(manifest))
    if recipe.startswith('yat'):
        with pytest.raises(ValueError, match='identity mismatch|Live state paths do not match checkpoint|yat_ffn_backward_block requires direct BF16 YAT FFN|yat_ffn_compute_mode requires YAT FFN'):
            run(parser().parse_args(common + ['--output', str(resumed), '--resume', '--ffn-type', 'geglu']))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(parser().parse_args(common + ['--output', str(full), '--resume',
            '--dtype', 'bfloat16', '--residual-dtype', 'bfloat16' if residual == 'float32' else 'float32']))
    restored = []
    for path in (full, resumed):
        model = ModernBert(config, rngs=nnx.Rngs(99))
        optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.),
            optax.adamw(learning_rate_schedule(parser().parse_args(common + ['--output', str(path)]))
                        if recipe == 'scheduled_accumulation' else 2e-5, weight_decay=.01)), wrt=nnx.Param)
        _, state = restore_model_from_checkpoint(model, str(path), optimizer=optimizer, load_training_state=True)
        assert int(state['completed_steps'][0]) == 3
        assert int(state['optimizer_updates'][0]) == 3
        if recipe.startswith('yat'):
            assert any(float(layer.yat_alpha[...]) != config.yat_alpha for layer in model.layers)
            if config.attention_score == 'yat_softmax':
                initial_alpha = config.yat_attention_alpha if config.yat_attention_alpha is not None else config.yat_alpha
                assert any(float(layer.yat_attention_alpha[...]) != initial_alpha for layer in model.layers)
        for value in jax.tree.leaves(optimizer.opt_state):
            if jax.numpy.issubdtype(value.dtype, jax.numpy.floating):
                assert value.dtype == np.float32
        restored.append((nnx.state(model), nnx.state(optimizer)))
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
    held_out_loss = report['masked_token_loss']
    assert isinstance(held_out_loss, (float, int))
    assert np.isfinite(held_out_loss)
    with pytest.raises(ValueError, match='Output already contains'):
        run(parser().parse_args(common + ['--output', str(full)]))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(parser().parse_args(common + ['--output', str(full), '--resume', '--seed', '99']))
    with pytest.raises(ValueError, match='requires masked projection' if projection_capacity is not None else 'identity mismatch'):
        run(parser().parse_args(common + ['--output', str(full), '--resume', '--mlm-projection',
                                         'masked' if projection == 'dense' else 'dense']))


@pytest.mark.parametrize('projection,backend', [('dense', 'xla'), ('masked', 'xla'), ('masked', 'xla_full'), ('masked', 'xla_local')])
def test_two_process_encoder_resume(tmp_path, projection, backend):
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
    common = ['--config', str(cfg), '--data', str(data),
        '--mlm-projection', projection, '--mlm-loss-backend', backend,
        '--steps', '4', '--batch-size', '4', '--mask-probability', '1', '--save-every', '3',
        '--dtype', 'bfloat16', '--loss-chunk-size', '4', '--distributed', '--shared-local-checkpoints']
    if backend == 'xla_local':
        common += ['--compile-diagnostics', '--accumulation-steps', '2', '--lr-schedule', 'cosine', '--warmup-steps', '1', '--shuffle']
    for output, module, phase, expected_code in (
        ('baseline', 'scripts.train_encoder', [], 0),
        ('checkpoints', 'scripts.validate_encoder_interruption', [], -9),
        ('checkpoints', 'scripts.train_encoder', ['--resume'], 0),
    ):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        jobs = []
        try:
            for rank in range(2):
                env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_COORDINATOR_ADDRESS=f'127.0.0.1:{port}',
                           JAX_PROCESS_COUNT='2', JAX_PROCESS_INDEX=str(rank),
                           XLA_FLAGS='--xla_force_host_platform_device_count=1')
                path = tmp_path / f'worker-{output}-{bool(phase and phase[0] == "--resume")}-{rank}.log'
                log = path.open('w')
                jobs.append((subprocess.Popen([sys.executable, '-m', module, *common, '--output', str(tmp_path / output), *phase],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log, path))
            for process, _, path in jobs:
                assert process.wait(timeout=90) == expected_code, path.read_text()[-12000:]
                if expected_code == -9:
                    assert 'FAULT_INJECTION: encoder SIGKILL after committed step 3' in path.read_text()
        finally:
            for process, log, _ in jobs:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                log.close()
    from flaxchat.checkpoint import load_checkpoint_metadata
    metadata = load_checkpoint_metadata(str(tmp_path / 'checkpoints'))
    assert metadata['model_family'] == 'modernbert'
    baseline, resumed = [json.loads((tmp_path / output / '4/manifest/metadata').read_text())
                         for output in ('baseline', 'checkpoints')]
    for key in ('model_state', 'optimizer_state', 'training_state'):
        assert baseline[key] == resumed[key], key


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
    protected_ids = manifest['special_token_ids']
    assert isinstance(protected_ids, list)
    assert 4 in protected_ids
    from flaxchat.mlm import mask_tokens
    tokens = np.load(tmp_path / 'data/tokens.npy')
    _, targets = mask_tokens(tokens, seed=0, step=0, example_ids=np.array([0]),
        vocab_size=6, mask_token_id=4, special_token_ids=manifest['special_token_ids'], probability=1)
    assert targets[0, 0] == -1
    assert targets[0, 1] == 5


@pytest.mark.parametrize('completed,updates,stop', [(4, 4, 3), (-1, 0, 3), (2, 3, 3), (5, 4, 5)])
def test_resume_cursor_rejects_inconsistent_state(completed, updates, stop):
    from scripts.train_encoder import validate_resume_cursor
    with pytest.raises(ValueError):
        validate_resume_cursor({'completed_steps': np.array([completed]),
                                'optimizer_updates': np.array([updates])}, horizon=4, stop=stop)


@pytest.mark.parametrize('defect', ['vocabulary', 'empty_length', 'manifest_shape', 'special', 'format'])
def test_shared_prepared_validation_rejects_checksum_consistent_bad_inputs(tmp_path, defect):
    from flaxchat.encoder_data import file_hash, load_prepared_rows
    config = EncoderConfig(vocab_size=7, hidden_size=8, intermediate_size=12,
                           num_hidden_layers=1, num_attention_heads=2)
    tokens = np.array([[5, 6, 0]], dtype=np.int32)
    manifest = dict(format='flaxchat-encoder-rows-v1', vocab_size=7, pad_token_id=0,
                    mask_token_id=4, special_token_ids=[0, 4], tokenizer_sha256='test')
    if defect == 'vocabulary':
        tokens[0, 0] = 7
    elif defect == 'empty_length':
        tokens = tokens[:, :0]
    elif defect == 'manifest_shape':
        manifest['sequence_length'] = 999
    elif defect == 'special':
        manifest['special_token_ids'] = [0, 4, 7]
    else:
        manifest['format'] = 'unknown'
    np.save(tmp_path / 'tokens.npy', tokens)
    manifest['tokens_sha256'] = file_hash(tmp_path / 'tokens.npy')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_prepared_rows(tmp_path, config)


def test_epoch_shuffling_replays_across_boundaries():
    from flaxchat.encoder_data import EpochRows
    sampler = EpochRows(7, 42, shuffle=True)
    ids = np.concatenate([sampler.batch(i, 4) for i in range(7)])
    for epoch in range(4):
        assert sorted(ids[epoch * 7:(epoch + 1) * 7]) == list(range(7))
    np.testing.assert_array_equal(EpochRows(7, 42, shuffle=True).batch(3, 4), ids[12:16])
    assert not np.array_equal(ids[:7], ids[7:14])


def test_mlm_accumulation_weights_targets_and_empty_microbatches():
    from flaxchat.training import gradients_for_mlm_microbatches
    import jax.numpy as jnp
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=1, num_attention_heads=2, compute_dtype='float32', loss_chunk_size=4)
    model = ModernBert(config, rngs=nnx.Rngs(0))
    x = jnp.array([[5, 6, 7, 0], [5, 7, 8, 0], [6, 7, 8, 0], [5, 9, 8, 0]])
    y = jnp.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [6, -1, -1, -1], [5, 9, 8, -1]])
    expected = nnx.jit(nnx.value_and_grad(lambda m: m(x, y)))(model)
    actual = nnx.jit(gradients_for_mlm_microbatches)(model, x.reshape(2, 2, 4), y.reshape(2, 2, 4))
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)


def test_document_windows_preserve_tails_and_report_padding(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    tokenizer = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, 'a': 2, 'b': 3, '[MASK]': 4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokpath = tmp_path / 'tokenizer.json'
    tokenizer.save(str(tokpath))
    source = tmp_path / 'text.jsonl'
    source.write_text(json.dumps({'text': 'a b a b a b a'}) + '\n' + json.dumps({'text': 'b'}))
    output = tmp_path / 'windows'
    manifest = prepare(source, tokpath, output, sequence_length=3, max_rows=10, pad_token_id=0,
                       long_document_policy='window')
    np.testing.assert_array_equal(np.load(output / 'tokens.npy'), [[2, 3, 2], [3, 2, 3], [2, 0, 0], [3, 0, 0]])
    assert manifest['nonpadding_tokens'] == 8
    assert manifest['nonpadding_fraction'] == 8 / 12


@pytest.mark.parametrize('compute', ['float32', 'bfloat16'])
def test_yat_training_and_exact_resume(tmp_path, monkeypatch, compute):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, compute,
                                              'float32', 'dense', 'xla', 'yat')


def test_language_sampling_training_and_exact_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'float32',
                                              'float32', 'dense', 'xla', 'language')


def test_yat_attention_and_ffn_training_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16',
                                              'float32', 'dense', 'xla', 'yat_both')


def test_yat_bf16_arithmetic_training_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16',
                                              'float32', 'dense', 'xla', 'yat_bf16')


def test_yat_adaptive_bf16_training_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16',
                                              'float32', 'dense', 'xla', 'yat_adaptive')


def test_yat_mmbert_bf16_masked_training_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16',
                                              'bfloat16', 'masked', 'xla', 'yat_adaptive')


def test_mixture_training_and_exact_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch, 'bfloat16',
                                              'bfloat16', 'masked', 'xla', 'mixture')


def test_training_logs_gradient_health(tmp_path, monkeypatch, capsys):
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'bfloat16', 'dense', 'xla', 'original')
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()
               if line.startswith('{')]
    steps = [row for row in records if row.get('event') == 'train_step']
    assert steps
    for row in steps:
        norm, scale = row['gradient_norm_before_clip'], row['gradient_clip_scale']
        assert np.isfinite(norm) and norm >= 0
        assert 0 < scale <= 1
        np.testing.assert_allclose(scale, min(1., 1. / max(norm, 1.)), rtol=1e-6)


def test_training_logs_structured_compiled_memory(tmp_path, monkeypatch, capsys):
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'float32', 'masked', 'xla_full', 'donation')
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()
               if line.startswith('{')]
    memory = [row for row in records if row.get('event') == 'compiled_memory']
    assert memory
    for row in memory:
        assert isinstance(row['statistics'], str)
        assert 'not measured device peak' in row['scope']
        for name in ('argument_size_in_bytes', 'output_size_in_bytes',
                     'temp_size_in_bytes', 'alias_size_in_bytes'):
            assert type(row['bytes'][name]) is int and row['bytes'][name] >= 0
        assert row['bytes']['argument_size_in_bytes'] > 0


def test_separate_attention_alpha_training_and_exact_resume(tmp_path, monkeypatch):
    from flaxchat.checkpoint import load_checkpoint_metadata
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'bfloat16', 'dense', 'xla', 'yat_adaptive', .1)
    for directory in ('full', 'resumed'):
        metadata = load_checkpoint_metadata(str(tmp_path / directory))
        config = metadata['resolved_config']['encoder']
        assert config['yat_attention_alpha'] == .1 and config['yat_alpha'] == 1.


def test_centered_fp32_encoder_training_and_exact_resume(tmp_path, monkeypatch):
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'float32', 'masked', 'xla',
        'yat_centered_fp32', attention_alpha=.1,
    )


def test_tiled_ffn_encoder_training_and_exact_resume(tmp_path, monkeypatch):
    from flaxchat.checkpoint import load_checkpoint_metadata
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'float32', 'masked', 'xla',
        'yat_centered_fp32', attention_alpha=.1, ffn_backward_block=32)
    config = load_checkpoint_metadata(str(tmp_path/'resumed'))['resolved_config']['encoder']
    assert config['yat_ffn_backward_block'] == 32


def test_explicit_projection_capacity_training_and_resume_identity(tmp_path, monkeypatch):
    from flaxchat.checkpoint import load_checkpoint_metadata
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'float32', 'masked', 'xla_full',
        'original', projection_capacity=2)
    config = load_checkpoint_metadata(str(tmp_path/'resumed'))['resolved_config']['encoder']
    assert config['mlm_projection_capacity'] == 2


def test_adaptive_ffn_direct_attention_exact_resume(tmp_path, monkeypatch):
    from flaxchat.checkpoint import load_checkpoint_metadata
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'float32', 'masked', 'xla_full',
        'yat_centered_fp32', attention_alpha=.1, ffn_compute_mode='bf16_adaptive')
    config = load_checkpoint_metadata(str(tmp_path/'resumed'))['resolved_config']['encoder']
    assert config['yat_ffn_compute_mode'] == 'bf16_adaptive'
    assert config['yat_compute_mode'] == 'bf16'
    assert config['yat_attention_implementation'] == 'centered_fp32_scores'
