import pytest
from scripts.benchmark_encoder_projection import summarize_steps


def test_summary_excludes_warmup_and_uses_token_weighted_rate():
    records = [dict(event='train_step', seconds=s, tokens=100, updated=True, loss=2,
                    masked_tokens=15, projection_dense_fallback=(s == 3)) for s in (99, 1, 3)]
    records.append(dict(event='checkpoint', seconds=20))
    r = summarize_steps(records, warmup=1)
    assert r['steady_tokens_per_second'] == 50
    assert r['first_step_compile_and_execute_seconds'] == 99
    assert r['checkpoint_seconds'] == 20
    assert r['overflow_steps'] == 1
    records[1]['updated'] = False
    with pytest.raises(ValueError):
        summarize_steps(records, warmup=1)


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
    def run(command):
        calls.append(command)
        if command[2] == 'scripts.diagnose_encoder_projection':
            item = dict(reference_loss=.02, candidate_loss=.02001, gradient_relative_l2=error)
            Path(command[command.index('--output') + 1]).write_text(json.dumps(
                {'stages': {'full_bf16_xla_full': item, 'full_bf16_pallas': item}}))
        return 0
    monkeypatch.setattr(campaign.subprocess, 'call', run)
    code = campaign.main(argparse.Namespace(output=str(tmp_path/'out'), prefix='gs://test/fresh',
        data_root='fixture', hourly_usd='1'))
    assert len(calls) == expected_calls
    assert code == (0 if error < .03 else 1)
