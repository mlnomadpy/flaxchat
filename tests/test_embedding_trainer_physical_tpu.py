"""Physical single-host embedding trainer admission, with real Orbax recovery.

FLAXCHAT_PHYSICAL_TPU=1 JAX_PLATFORMS=tpu python -m pytest -q this_file
Uses a tiny YAT encoder, authentic prepared arrays and the real CLI entrypoint.
This qualifies semantics and restart; production throughput needs its own receipt.
"""
import os
import json
import sys
from dataclasses import asdict
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('FLAXCHAT_PHYSICAL_TPU') != '1',
                                reason='Requires explicit physical TPU admission')


@pytest.mark.parametrize('triplets,cached,fault', [
    (False, False, None), (False, True, None), (True, False, None), (True, True, None),
    (True, True, 'best4'), (False, True, 'baseline0'),
    (True, True, 'gate4'), (False, True, 'gatehorizon'),
    (True, False, 'best4'), (False, False, 'baseline0'),
    (True, False, 'gate4'), (False, False, 'gatehorizon'),
    (False, False, 'independent'), (True, False, 'qat'), (False, False, 'migration')])
def test_real_trainer_resume_pair_triplet_schema_and_best(tmp_path, monkeypatch, capsys, triplets, cached, fault):
    import jax
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.encoder_data import file_hash
    from flaxchat.public_encoder import export_state_safetensors
    from flaxchat.checkpoint import load_checkpoint_metadata
    from scripts.prepare_yat_embedding_finetune import _tokenize
    from scripts.train_yat_embedding_finetune import main
    import scripts.train_yat_embedding_finetune as trainer
    import tokenizers
    assert jax.default_backend() == 'tpu'
    assert jax.process_count() == 1, 'This admission is single-host; multi-host suite is separate'
    qat = fault == 'qat'
    if qat:
        fault = None
    migration_case = fault == 'migration'
    if migration_case:
        fault = 'gate4'
    independent = fault == 'independent'
    if independent:
        fault = None  # Independent metrics are real; no controlled quality override.
    parent = tmp_path / 'parent'
    parent.mkdir()
    config = EncoderConfig(vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=32,
        local_attention=16, ffn_type='yat_glu', attention_score='yat_softmax',
        compute_dtype='bfloat16', residual_dtype='float32', use_remat=False)
    model = ModernBert(config, rngs=nnx.Rngs(29))
    export_state_safetensors(nnx.to_pure_dict(nnx.state(model)), parent / 'model.safetensors')
    (parent / 'config.json').write_text(json.dumps(asdict(config)))
    vocab = {'[PAD]': 0, '[UNK]': 1, 'query': 2, 'document': 3, '[MASK]': 4, 'negative': 5}
    vocab.update({str(i): i + 6 for i in range(100)})
    vocab.update({f'unused{i}': i for i in range(106, 128)})
    if independent:
        for token, index in [('en', 125), ('ar', 126), ('positive', 127)]:
            del vocab[f'unused{index}']
            vocab[token] = index
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(parent / 'tokenizer.json'))
    published = parent / 'manifest.json'
    published.write_text(json.dumps({name: file_hash(parent / name) for name in ('config.json', 'tokenizer.json', 'model.safetensors')}))
    folder = tmp_path / 'fixture'
    folder.mkdir()
    for split, span in [('train', range(32)), ('dev', range(80, 88))]:
        rows = [dict(query=f'query {i}', positive=f'document {i}',
                     negative=f'negative {i}' if triplets else None,
                     group=str(i), language='en') for i in span]
        (folder / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    tokens = _tokenize(folder, parent / 'tokenizer.json', 16, 16, 0)
    manifest = dict(format='flaxchat-yat-embedding-triplets-v2', source='fixture',
        tokenizer_sha256=file_hash(parent / 'tokenizer.json'), rows={'train': 32, 'dev': 8},
        raw_files={f'{split}.jsonl': file_hash(folder / f'{split}.jsonl') for split in ('train', 'dev')},
        query_length=16, document_length=16, pad_id=0, vocab_size=128, tokenization=tokens)
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    independent_manifest = None
    if independent:
        import shutil
        from tests.test_development_quarantine_metadata import candidate
        from flaxchat.embedding_development_quarantine import load_candidate_exclusions
        from scripts.prepare_embedding_retrieval_dev import prepare
        candidates = candidate(tmp_path)
        index = load_candidate_exclusions(candidates)
        manifest['source_identity'] = {'repo': 'fixture/clean', 'revision': 'a' * 40}
        manifest['development_quarantine'] = {'identity': index.identity}
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        historical = tmp_path / 'historical'
        shutil.copytree(folder, historical)
        metadata_path = tmp_path / 'historical-metadata.json'
        metadata_path.write_text(json.dumps({'resolved_config': {
            'data_manifests': {'fixture': file_hash(historical / 'manifest.json')}}}))
        identity = file_hash(metadata_path)
        exposure_path = tmp_path / 'exposure.json'
        exposure_path.write_text(json.dumps({
            'format': 'flaxchat-known-contrastive-exposure-input-v1',
            'parent_files_sha256': json.loads(published.read_text()),
            'known_contrastive_inventory_complete': True,
            'expected_stage_identities': [identity],
            'stages': [{'stage_identity_sha256': identity,
                'checkpoint_metadata': str(metadata_path),
                'sources': [{'name': 'fixture', 'directory': str(historical)}]}]}))
        independent_manifest = prepare(candidates,
            list(json.loads((candidates / 'exclusions.json').read_text())['sources']),
            exposure_path, published, parent / 'tokenizer.json', parent / 'config.json',
            tmp_path / 'independent', query_length=16, document_length=16)
    common = ['--parent-public', str(parent), '--parent-manifest', str(published), '--data', f'fixture={folder}',
        '--source-weights', 'fixture=1', '--steps', '4', '--warmup', '1',
        '--batch-size', str(max(16, 2 * jax.device_count())), '--save-every', '1', '--eval-every', '4',
        '--keep-checkpoints', '1', '--dev-max-rows', '8', '--max-dev-regression', '1',
        '--training-scope', 'qualification']
    if qat:
        common += ['--weight-quantization', 'int8_per_channel_ste']
    if cached:
        common += ['--encoder-chunk-size', str(max(4, jax.device_count()))]
    if independent:
        common += ['--retrieval-dev', f'alignment={tmp_path / "independent"}']
    def run(output, extra=()):
        monkeypatch.setattr(sys, 'argv', ['train_yat_embedding_finetune'] + common + ['--output', str(output)] + list(extra))
        main()
    # Force controlled selection improvements for commit-fault tests while the
    # real TPU model, gradient, quality probes and Orbax persistence all execute.
    # This fixture isolates transaction semantics, not production model quality.
    original_gate = trainer.quality_gate
    if fault:
        def fixture_gate(metrics, baseline, **kwargs):
            for result in metrics.values():
                result['mrr'] = (.8 if metrics is baseline else 0.) if fault.startswith('gate') else (.5 if metrics is baseline else .9)
            return original_gate(metrics, baseline, **kwargs)
        monkeypatch.setattr(trainer, 'quality_gate', fixture_gate)
    complete, resumed = tmp_path / 'complete', tmp_path / 'resumed'
    if fault and fault.startswith('gate'):
        common[common.index('--steps') + 1] = '8' if fault == 'gate4' else '4'
        common[common.index('--max-dev-regression') + 1] = '.02'
        with pytest.raises(RuntimeError, match='representation gate failed'):
            run(resumed)
        assert load_checkpoint_metadata(str(resumed))['step'] == 4
        capsys.readouterr()
        for _ in range(2):
            with pytest.raises(ValueError, match='blocked by failed development gate'):
                run(resumed, ['--resume'])
            events = capsys.readouterr().out
            assert 'embedding_step' not in events and 'embedding_already_completed' not in events
        assert load_checkpoint_metadata(str(resumed / 'best'))['step'] == 0
        if migration_case:
            from flaxchat.embedding_recovery import checkpoint_stage_identity
            from flaxchat.embedding_contract import source_identity
            from pathlib import Path
            old = load_checkpoint_metadata(str(resumed), 4, include_receipt=True)
            old_best = load_checkpoint_metadata(str(resumed / 'best'), 0, include_receipt=True)
            specification = {'format': 'flaxchat-quality-policy-migration-v1',
                'checkpoint_prefix': str(resumed),
                'target_source_python_sha256': source_identity(Path(__file__).resolve().parents[1])['sha256'],
                'previous_identity': checkpoint_stage_identity(old),
                'checkpoint': {key: old['committed_receipt'][key] for key in ('step', 'manifest_sha256')},
                'best': {key: old_best['committed_receipt'][key] for key in ('step', 'manifest_sha256')}}
            migration_path = tmp_path / 'migration.json'
            migration_path.write_text(json.dumps(specification))
            common[common.index('--steps') + 1] = '12'
            common += ['--schedule-steps', '8', '--quality-regression-action', 'report-only']
            # Same seed, data and fixed8-step schedule;12updates uninterrupted
            # is the reference for preserved optimizer moments after migration.
            run(complete)
            common += ['--resume-quality-migration', str(migration_path)]
            run(resumed, ['--resume', '--stop-after', '8'])
            run(resumed, ['--resume'])
            migrated = load_checkpoint_metadata(str(resumed), 12, include_receipt=True)
            assert migrated['resolved_config']['schedule_steps'] == 8
            assert migrated['resolved_config']['steps'] == 12
            assert migrated['resolved_config']['resume_policy_migration']['source_checkpoint']['step'] == 4
            assert load_checkpoint_metadata(str(resumed / 'best'), 0, include_receipt=True) == old_best
            import optax
            from flaxchat.checkpoint import create_checkpoint_manager, load_checkpoint
            optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.),
                optax.adamw(optax.warmup_cosine_decay_schedule(0., 3e-5, 1, 8, end_value=3e-6),
                    weight_decay=.01, mask=lambda params: jax.tree.map(lambda value: value.ndim > 1, params))), wrt=nnx.Param)
            restored = []
            for output in (complete, resumed):
                manager = create_checkpoint_manager(str(output), async_checkpointing=False)
                try:
                    restored.append(load_checkpoint(manager, 12, model=model, optimizer=optimizer, load_training_state=True))
                finally:
                    manager.close()
            for element in (0, 1):
                for expected, actual in zip(jax.tree.leaves(restored[0][element]), jax.tree.leaves(restored[1][element]), strict=True):
                    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
            for key in restored[0][3]:
                if key != 'quality_state':
                    np.testing.assert_array_equal(restored[0][3][key], restored[1][3][key])
            state = json.loads(bytes(np.asarray(restored[1][3]['quality_state'], np.uint8)).decode())
            assert state['last_evaluation']['gate']['passed'] is False
            assert state['last_evaluation']['step'] == 12
        return
    run(complete)
    complete_events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert [event['step'] for event in complete_events if event.get('event') == 'embedding_dev'] == [0, 4]
    if independent:
        for event in complete_events:
            if event.get('event') == 'embedding_dev':
                assert 'independent/alignment' in event['source_metrics']
                assert 'independent/alignment/language/en' in event['source_metrics']
                assert 'independent/alignment/language/ar' in event['source_metrics']
    original_save = trainer.save_checkpoint
    fault_fired = False
    best_writes = []
    def fault_save(manager, step, *args, **kwargs):
        nonlocal fault_fired
        saved = original_save(manager, step, *args, **kwargs)
        if str(manager.directory).rstrip('/').endswith('/best'):
            best_writes.append(step)
            target = 4 if fault == 'best4' else 0
            if fault and not fault_fired and step == target:
                manager.wait_until_finished()
                fault_fired = True
                raise RuntimeError('Injected after durable best commit, before recovery')
        return saved
    monkeypatch.setattr(trainer, 'save_checkpoint', fault_save)
    if fault == 'baseline0':
        with pytest.raises(RuntimeError, match='Injected after durable best'):
            run(resumed, ['--stop-after', '2'])
        run(resumed, ['--resume'])
    else:
        run(resumed, ['--stop-after', '2'])
        partial_events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
        assert [event['step'] for event in partial_events if event.get('event') == 'embedding_dev'] == [0]
        if fault == 'best4':
            with pytest.raises(RuntimeError, match='Injected after durable best'):
                run(resumed, ['--resume'])
            # The recent cursor is 3 while the selected durable best is 4.
            assert load_checkpoint_metadata(str(resumed))['step'] == 3
            assert load_checkpoint_metadata(str(resumed / 'best'))['step'] == 4
        run(resumed, ['--resume'])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert all(event['step'] != 2 for event in events if event.get('event') == 'embedding_dev')
    if fault:
        assert fault_fired
    if fault == 'best4':
        assert best_writes == [0, 4]  # Replay must not request an older/duplicate best save.
        assert any(event.get('best_ahead_of_cursor') is True for event in events)
    if fault == 'baseline0':
        assert any(event.get('bootstrap_resume') is True for event in events)
    left = load_checkpoint_metadata(str(complete), 4, include_receipt=True)
    right = load_checkpoint_metadata(str(resumed), 4, include_receipt=True)
    assert left['committed_receipt']['model_state'] == right['committed_receipt']['model_state']
    assert left['resolved_config'] == right['resolved_config']
    if qat:
        assert right['resolved_config']['encoder']['weight_quantization'] == 'int8_per_channel_ste'
        assert right['resolved_config']['quantization_aware_training']['gradient'] == 'identity-ste-including-scale'
    if independent:
        receipt = right['resolved_config']['independent_retrieval_development']['alignment']
        assert receipt['candidate_identity'] == independent_manifest['candidate_identity']
        assert receipt['parent_exposure'] == independent_manifest['parent_exposure']
    # Compare complete optimizer moments/counter and exact recovery bookkeeping,
    # rather than using matching weights as a proxy for exact continuation.
    import optax
    from flaxchat.checkpoint import create_checkpoint_manager, load_checkpoint
    schedule = optax.warmup_cosine_decay_schedule(0., 3e-5, 1, 4, end_value=3e-6)
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.),
        optax.adamw(schedule, weight_decay=.01,
            mask=lambda params: jax.tree.map(lambda value: value.ndim > 1, params))), wrt=nnx.Param)
    restored = []
    for output in (complete, resumed):
        manager = create_checkpoint_manager(str(output), async_checkpointing=False)
        try:
            restored.append(load_checkpoint(manager, 4, model=model, optimizer=optimizer, load_training_state=True))
        finally:
            manager.close()
    for element in (0, 1, 3):
        for expected, actual in zip(jax.tree.leaves(restored[0][element]), jax.tree.leaves(restored[1][element]), strict=True):
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    # A separate best manager must retain its quality-selected checkpoint even
    # after the recent manager removes earlier recovery checkpoints.
    best = load_checkpoint_metadata(str(resumed / 'best'))
    assert best['step'] in (0, 4)
    assert best['quality_selection']['step'] == best['step']
    manager = create_checkpoint_manager(str(resumed), async_checkpointing=False)
    try:
        assert list(manager.all_steps()) == [4]
    finally:
        manager.close()

    if independent:
        # Changing an independent raw artifact must reject exact resume before
        # even constructing/loading the TPU model, preserving the durable state.
        raw = tmp_path / 'independent' / 'raw.jsonl'
        raw.write_text(raw.read_text() + '{}\n')
        def forbidden_model_load(*args, **kwargs):
            raise AssertionError('Malformed independent data reached model loading')
        monkeypatch.setattr(trainer, 'load_public_encoder', forbidden_model_load)
        with pytest.raises(ValueError):
            run(resumed, ['--resume'])
        assert load_checkpoint_metadata(str(resumed), 4, include_receipt=True) == right


def test_physical_rejected_update_preserves_model_and_optimizer():
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    from flaxchat.training import apply_gradients_if_finite
    assert jax.default_backend() == 'tpu'
    model = nnx.Linear(128, 128, rngs=nnx.Rngs(29))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(model))
    before_opt = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(optimizer))
    gradients = jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param))
    @nnx.jit
    def reject(model, optimizer, gradients):
        return apply_gradients_if_finite(model, optimizer, gradients, jnp.array(float('nan')))
    assert not bool(reject(model, optimizer, gradients))
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(model)), strict=True):
        np.testing.assert_array_equal(expected, np.asarray(actual))
    for expected, actual in zip(jax.tree.leaves(before_opt), jax.tree.leaves(nnx.state(optimizer)), strict=True):
        np.testing.assert_array_equal(expected, np.asarray(actual))
