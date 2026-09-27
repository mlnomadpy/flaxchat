import pytest
from scripts.benchmark_encoder_projection import summarize_steps


def test_summary_excludes_warmup_and_uses_token_weighted_rate():
    records = [dict(event='train_step', step=i+1, seconds=s, tokens=100, updated=True, loss=2,
                    masked_tokens=15, projection_dense_fallback=(s == 3)) for i, s in enumerate((99, 1, 3))]
    records.append(dict(event='checkpoint', step=3, seconds=20))
    r = summarize_steps(records, warmup=1, expected_steps=3, expected_tokens=100)
    assert r['steady_tokens_per_second'] == 50
    assert r['first_step_compile_and_execute_seconds'] == 99
    assert r['checkpoint_seconds'] == 20
    assert r['overflow_steps'] == 1
    records[1]['updated'] = False
    with pytest.raises(ValueError):
        summarize_steps(records, warmup=1, expected_steps=3, expected_tokens=100)


def test_compare_never_matches_different_batch_and_reports_missing():
    from scripts.compare_encoder_projection import compare
    def case(batch, projection, rate):
        return dict(name=f'train-{projection}-xla-b{batch}-s512', passed=True, returncode=0,
            batch=batch, length=512, projection=projection, backend='xla', steady_tokens_per_second=rate,
            first_loss=2, final_loss=1, steady_usd_per_billion_input_tokens=4,
            invocation_usd_per_billion_input_tokens=40, overflow_steps=0)
    report = compare({'results':[case(4,'dense',10), case(4,'masked',20),case(16,'masked',100)]})
    assert [r['speedup'] for r in report['comparisons']] == [1, 2]
    assert not report['complete']
    assert not report['fused_kernel_qualified']
    assert 'train-dense-xla-b16-s512' in report['missing_or_failed_training']


def test_gradient_diagnostics_rank_absolute_error_without_hiding_near_zero():
    import jax.numpy as jnp
    from scripts.validate_encoder_projection import gradient_diagnostics
    reference = {'large': jnp.array([3.,4.]), 'tiny': jnp.array([0.,0.])}
    candidate = {'large': jnp.array([6.,8.]), 'tiny': jnp.array([1e-6,0.])}
    rows = gradient_diagnostics(reference, candidate)
    assert 'large' in rows[0]['parameter']
    assert rows[0]['reference_l2'] == 5
    assert rows[0]['difference_l2'] == 5
    assert rows[0]['relative_l2'] == 1
    assert rows[0]['cosine_similarity'] == pytest.approx(1)
    assert rows[1]['difference_l2'] == pytest.approx(1e-6)
    assert rows[1]['relative_l2'] > 1


def test_preflight_rejects_undersized_benchmark_before_hardware(tmp_path):
    import numpy as np
    from scripts.benchmark_encoder_projection import preflight_cases
    path = tmp_path / 'data-512'
    path.mkdir()
    np.save(path / 'tokens.npy', np.zeros((32,512), dtype=np.int32))
    assert preflight_cases([(32,512)], tmp_path)[0]['available_rows'] == 32
    with pytest.raises(ValueError, match='requires at least 64'):
        preflight_cases([(64,512)], tmp_path)


@pytest.mark.parametrize('error,expected_calls', [(.02, 2), (.07, 1)])
def test_campaign_gates_throughput_on_original_regression(tmp_path, monkeypatch, error, expected_calls):
    import argparse
    import json
    from pathlib import Path
    from scripts import validate_projection_campaign as campaign
    calls = []
    def run(command, *, timeout):
        assert 0 < timeout <= 1500
        calls.append(command)
        if command[2] == 'scripts.diagnose_encoder_projection':
            item = dict(reference_loss=.02, candidate_loss=.02001, gradient_relative_l2=error)
            Path(command[command.index('--output') + 1]).write_text(json.dumps(
                {'stages': {'full_bf16_xla_full': item, 'full_bf16_pallas': item}}))
        return argparse.Namespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    code = campaign.main(argparse.Namespace(output=str(tmp_path/'out'), prefix='gs://test/fresh',
        data_root='fixture', hourly_usd='1', max_seconds=1500))
    assert len(calls) == expected_calls
    assert code == (0 if error < .03 else 1)


@pytest.mark.parametrize('defect', ['truncated', 'reordered', 'duplicate', 'nan-loss', 'empty-masks',
    'excess-masks', 'nan-masks', 'infinite-time', 'zero-time', 'wrong-tokens', 'truthy-update',
    'missing-checkpoint', 'early-checkpoint', 'wrong-checkpoint', 'missing-fallback'])
def test_benchmark_rejects_malformed_evidence(defect):
    records = [dict(event='train_step', step=i, seconds=1., tokens=100, updated=True,
                    loss=2., masked_tokens=15, projection_dense_fallback=False) for i in range(1, 7)]
    records.append(dict(event='checkpoint', step=6, seconds=2.))
    if defect == 'truncated':
        records.pop(3)
    elif defect == 'reordered':
        records[0], records[1] = records[1], records[0]
    elif defect == 'duplicate':
        records[1]['step'] = 1
    elif defect == 'missing-checkpoint':
        records.pop()
    elif defect == 'early-checkpoint':
        records.insert(0, records.pop())
    elif defect == 'wrong-checkpoint':
        records[-1]['step'] = 5
    elif defect == 'missing-fallback':
        records[0].pop('projection_dense_fallback')
    else:
        field, value = {'nan-loss': ('loss', float('nan')), 'empty-masks': ('masked_tokens', 0),
            'excess-masks': ('masked_tokens', 101), 'nan-masks': ('masked_tokens', float('nan')),
            'infinite-time': ('seconds', float('inf')), 'zero-time': ('seconds', 0),
            'wrong-tokens': ('tokens', 99), 'truthy-update': ('updated', 1)}[defect]
        records[0][field] = value
    with pytest.raises(ValueError):
        summarize_steps(records, expected_steps=6, expected_tokens=100)


def test_benchmark_default_requires_all_55_steps():
    records = [dict(event='train_step', step=i, seconds=1., tokens=100, updated=True,
                    loss=2., masked_tokens=15, projection_dense_fallback=False) for i in range(1, 7)]
    with pytest.raises(ValueError, match='horizon'):
        summarize_steps(records)


def test_full_benchmark_horizon_has_fifty_measured_steps():
    records = [dict(event='train_step', step=i, seconds=2. if i > 5 else 100.,
                    tokens=32768, updated=True, loss=2., masked_tokens=4915,
                    projection_dense_fallback=False) for i in range(1, 56)]
    records.append(dict(event='checkpoint', step=55, seconds=40.))
    report = summarize_steps(records, expected_tokens=32768)
    assert report['measured_steps'] == 50
    assert report['steady_tokens_per_second'] == 16384
    assert report['checkpoint_seconds'] == 40


@pytest.mark.parametrize('warmup,horizon', [(-1, 55), (55, 55), (0, 0), (True, 55)])
def test_benchmark_rejects_invalid_measurement_window(warmup, horizon):
    with pytest.raises(ValueError, match='horizon'):
        summarize_steps([], warmup=warmup, expected_steps=horizon)


@pytest.mark.parametrize('defect', [None, 'missing-log', 'changed-log', 'wrong-backend', 'wrong-step', 'summary-rate', 'missing-source'])
def test_comparison_requires_raw_updates_and_checkpoint_identity(tmp_path, defect):
    import hashlib
    import json
    from scripts.compare_encoder_projection import verify_training_evidence
    name = 'train-masked-xla_local-b4-s512'
    config = dict(event='run_config', backend='tpu', devices=4, processes=1,
                  mlm_loss_backend='xla_local', mlm_projection='masked')
    records = [config] + [dict(event='train_step', step=i, seconds=1., tokens=2048,
        updated=True, loss=2., masked_tokens=100, projection_dense_fallback=False) for i in range(1, 56)]
    records.append(dict(event='checkpoint', step=55, seconds=2.))
    case = dict(name=name, batch=4, length=512, backend='xla_local', projection='masked',
                **summarize_steps(records, expected_tokens=2048))
    manifest = dict(step=55, model_state={'hash': 'model'}, optimizer_state={'hash': 'optimizer'},
                    training_state={'hash': 'cursor'}, identity=dict(source_python_sha256='a' * 64,
                    resolved_config=dict(steps=55, batch_size=4,
                        encoder=dict(mlm_loss_backend='xla_local', mlm_projection='masked'))))
    if defect == 'wrong-backend':
        config['mlm_loss_backend'] = 'pallas'
    if defect == 'wrong-step':
        manifest['step'] = 54
    if defect == 'summary-rate':
        case['steady_tokens_per_second'] *= 2
    if defect == 'missing-source':
        del manifest['identity']['source_python_sha256']
    log = tmp_path / f'{name}.log'
    raw = tmp_path / f'{name}-manifest.json'
    log.write_text('\n'.join(json.dumps(r) for r in records))
    raw.write_text(json.dumps(manifest))
    case['training_log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
    case['manifest_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    if defect == 'missing-log':
        log.unlink()
    if defect == 'changed-log':
        log.write_text(log.read_text() + '\nchanged')
    if defect:
        with pytest.raises((ValueError, KeyError, OSError)):
            verify_training_evidence(tmp_path, case)
    else:
        assert verify_training_evidence(tmp_path, case)
