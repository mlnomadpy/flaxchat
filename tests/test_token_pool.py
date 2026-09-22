"""Packed data integrity, boundaries, and exact training continuation."""
import json

import numpy as np
import pytest

from flaxchat.token_pool import TokenPool, write_pool


def pool_at(path, size=257):
    return write_pool(path, {'train': np.arange(size) % 32, 'validation': np.arange(65) % 32},
                      {'tokenizer': 'synthetic-test-only'}, 32)


def test_packing_has_no_padding_and_resume_is_exact(tmp_path):
    pool = TokenPool(pool_at(tmp_path / 'pool'))
    x, y = pool.batch('train', 1, 2, 8)
    np.testing.assert_array_equal(x.ravel(), np.arange(16, 32))
    np.testing.assert_array_equal(y.ravel(), np.arange(17, 33) % 32)
    restored = TokenPool(pool.path)
    np.testing.assert_array_equal(restored.batch('train', 1, 2, 8), (x, y))
    with pytest.raises(ValueError, match='exhausted'):
        pool.batch('train', 16, 2, 8)


def test_corruption_and_path_escape_rejected(tmp_path):
    path = pool_at(tmp_path / 'pool')
    shard = path.parent / 'train.bin'
    with shard.open('r+b') as handle:
        handle.write(b'BAD!')
    with pytest.raises(ValueError, match='Corrupt'):
        TokenPool(path)
    manifest = json.loads(path.read_text())
    manifest['splits']['train']['file'] = '../outside.bin'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inside'):
        TokenPool(path)


@pytest.mark.integration
@pytest.mark.parametrize('cadence', [['--save-every', '2'], ['--checkpoint-interval-seconds', '100000']])
def test_gpt_resume_matches_uninterrupted(tmp_path, cadence):
    import jax
    from scripts.train_gpt2 import main
    from flaxchat.checkpoint import create_checkpoint_manager, load_checkpoint
    # Global batch adapts to actual TPU devices; identical schedule across legs.
    count = jax.device_count()
    manifest = write_pool(tmp_path / 'pool',
                          {'train': np.arange(count * 8 * 4 + 1) % 32,
                           'validation': np.arange(count * 8 + 1) % 32},
                          {'tokenizer': 'synthetic-test-only'}, 32)
    base = ['--token-manifest', str(manifest), '--depth', '1', '--seq-len', '8',
            '--batch-per-device', '1', '--tokens', str(count * 8 * 4),
            '--warmup-steps', '1', '--eval-every', '2'] + cadence
    def run(name, extra):
        return main(base + ['--ckpt-dir', str(tmp_path / name / 'checkpoints'),
                            '--artifact-dir', str(tmp_path / name)] + extra)
    assert run('full', []) == 0
    assert run('resumed', ['--stop-after', '2']) == 0
    assert run('resumed', ['--resume']) == 0
    states = []
    for name in ('full', 'resumed'):
        manager = create_checkpoint_manager(str(tmp_path / name / 'checkpoints'), async_checkpointing=False)
        try:
            state, _ = load_checkpoint(manager)
            states.append(state)
        finally:
            manager.close()
    for a, b in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(a, b)
    manifests = []
    for name in ('full', 'resumed'):
        path = tmp_path / name / 'checkpoints/4/manifest/metadata'
        manifests.append(json.loads(path.read_text()))
    for key in ('model_state', 'optimizer_state', 'training_state'):
        assert manifests[0][key] == manifests[1][key]
    summary = json.loads((tmp_path / 'resumed/training_summary.json').read_text())
    assert summary['start_step'] == 2 and summary['completed_steps'] == 4
    assert np.isfinite(summary['final_validation_loss'])
    assert summary['model_config']['tie_embeddings'] is True
    with pytest.raises(ValueError, match='not empty'):
        run('resumed', [])


def test_invalid_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match='valid token'):
        write_pool(tmp_path / 'bad', {'train': [1.5, 2.5]}, {}, 32)


def test_manifest_changes_identity(tmp_path):
    path = pool_at(tmp_path / 'pool')
    before = TokenPool(path).identity
    manifest = json.loads(path.read_text())
    manifest['identity']['revision'] = 'different'
    path.write_text(json.dumps(manifest))
    assert TokenPool(path).identity != before


@pytest.mark.parametrize('processes', [1, 2, 4, 8, 16])
def test_host_batches_reconstruct_canonical_order(tmp_path, processes):
    pool = TokenPool(pool_at(tmp_path / 'pool', size=1025))
    for step in (0, 1, 3):
        global_pair = pool.batch('train', step, 16, 8)
        local_pairs = [pool.local_batch('train', step, 16, 8, rank, processes)
                       for rank in range(processes)]
        for side in (0, 1):
            np.testing.assert_array_equal(np.concatenate([pair[side] for pair in local_pairs]), global_pair[side])


@pytest.mark.parametrize('rank,count,batch', [(0, 0, 8), (-1, 2, 8), (2, 2, 8), (0, 3, 8)])
def test_invalid_host_partition_rejected(tmp_path, rank, count, batch):
    pool = TokenPool(pool_at(tmp_path / 'pool'))
    with pytest.raises(ValueError):
        pool.local_batch('train', 0, batch, 8, rank, count)


@pytest.mark.integration
def test_fsdp_training_and_checkpoint_bridge(tmp_path):
    import jax
    from scripts.train_gpt2 import main as train
    from scripts.validate_checkpoint_bridge import main as bridge
    devices = jax.device_count()
    if devices < 2:
        pytest.skip('Requires multiple devices; run virtual-device or TPU suite')
    manifest = write_pool(tmp_path / 'pool',
                          {'train': np.arange(devices * 8 * 4 + 1) % 32,
                           'validation': np.arange(devices * 8 + 1) % 32}, {}, 32)
    checkpoints = tmp_path / 'train/checkpoints'
    base = ['--token-manifest', str(manifest), '--depth', '1', '--seq-len', '8',
            '--global-batch-size', str(devices), '--tokens', str(devices * 8 * 4),
            '--warmup-steps', '1', '--eval-every', '2', '--save-every', '2',
            '--ckpt-dir', str(checkpoints), '--artifact-dir', str(tmp_path / 'train'),
            '--fsdp', '2']
    assert train(base + ['--stop-after', '2']) == 0
    assert train(base + ['--resume']) == 0
    full = list(base)
    full[full.index('--ckpt-dir') + 1] = str(tmp_path / 'full/checkpoints')
    full[full.index('--artifact-dir') + 1] = str(tmp_path / 'full')
    assert train(full) == 0
    reference = json.loads((tmp_path / 'full/checkpoints/4/manifest/metadata').read_text())
    resumed = json.loads((checkpoints / '4/manifest/metadata').read_text())
    assert all(reference[key] == resumed[key] for key in ('model_state', 'optimizer_state', 'training_state'))
    output = tmp_path / 'bridge.json'
    assert bridge(['--source', str(checkpoints), '--destination', str(tmp_path / 'bridged'),
                   '--output', str(output)]) == 0
    assert all(json.loads(output.read_text())['equal'].values())


@pytest.mark.parametrize('batch', ['0', '-1'])
def test_trainer_rejects_nonpositive_explicit_global_batch(batch):
    from scripts.train_gpt2 import main
    with pytest.raises(SystemExit) as error:
        main(['--token-manifest', 'unused', '--global-batch-size', batch])
    assert error.value.code == 2


def test_noncontiguous_host_rows_preserve_canonical_global_batch(tmp_path):
    pool = TokenPool(pool_at(tmp_path / 'pool', size=1025))
    # Host zero owns rows 0,2,4,6; host one owns 1,3,5,7.
    expected = pool.batch('train', 2, 8, 8)
    actual = [np.empty_like(expected[0]), np.empty_like(expected[1])]
    for rows in ([0, 2, 4, 6], [1, 3, 5, 7]):
        for row in rows:
            pair = pool.batch_rows('train', 2, 8, 8, row, row + 1)
            for side in (0, 1):
                actual[side][row:row + 1] = pair[side]
    for side in (0, 1):
        np.testing.assert_array_equal(actual[side], expected[side])


@pytest.mark.parametrize('start,stop', [(-1, 1), (0, 0), (1, 0), (0, 9)])
def test_invalid_global_row_interval_rejected(tmp_path, start, stop):
    pool = TokenPool(pool_at(tmp_path / 'pool'))
    with pytest.raises(ValueError, match='row interval'):
        pool.batch_rows('train', 0, 8, 8, start, stop)


def test_muon_accumulated_resume_matches_uninterrupted(tmp_path):
    from scripts.train_gpt2 import main as train
    import jax
    devices = jax.device_count()
    batch = devices * 2
    manifest = write_pool(tmp_path / 'pool',
                          {'train': np.arange(batch * 8 * 4 + 1) % 32,
                           'validation': np.arange(batch * 8 + 1) % 32}, {}, 32)
    base = ['--token-manifest', str(manifest), '--depth', '1', '--seq-len', '8',
            '--global-batch-size', str(batch), '--tokens', str(batch * 8 * 4),
            '--optimizer', 'muon', '--accumulation-steps', '2', '--lr', '.001',
            '--loss-chunk-size', '3', '--remat',
            '--warmup-steps', '1', '--eval-every', '2', '--save-every', '2']
    def run_at(name, extra=()):
        return train(base + ['--ckpt-dir', str(tmp_path / name / 'checkpoints'),
                            '--artifact-dir', str(tmp_path / name)] + list(extra))
    assert run_at('resumed', ['--stop-after', '2']) == 0
    assert run_at('resumed', ['--resume']) == 0
    assert run_at('full') == 0
    def state(name):
        return json.loads((tmp_path / name / 'checkpoints/4/manifest/metadata').read_text())
    assert all(state('resumed')[key] == state('full')[key]
               for key in ('model_state', 'optimizer_state', 'training_state'))
