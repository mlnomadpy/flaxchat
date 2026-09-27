import argparse
import copy
import json
from pathlib import Path
import subprocess

import pytest
from scripts import validate_encoder_quality_pilot as worker


def fixture(tmp_path, monkeypatch, *, slow=False, fail=None):
    validation = tmp_path / 'validation'
    validation.mkdir()
    (validation / 'manifest.json').write_text(json.dumps({'rows': 3073}))
    (validation / 'tokens.npy').write_bytes(b'fixture')
    pinned = tmp_path / 'input'
    pinned.write_text('frozen corpus')
    recipe = dict(training_argv=['python', '-m', 'scripts.train_encoder', '--output', 'unused',
                                '--steps', '1972', '--batch-size', '64', '--data', 'train'],
                  steps=1972, batch_size=64, calibration_steps=100,
                  calibration_nonpadding_tokens=[25000] * 100,
                  nonpadding_tokens=1972 * 25000,
                  input_sha256={str(p): worker.file_hash(p) for p in
                                (pinned, validation / 'manifest.json', validation / 'tokens.npy')})
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    args = argparse.Namespace(prefix='gs://test/quality', recipe=path, devices=4,
                              output=tmp_path / 'evidence', validation=validation,
                              hourly_usd=4.8, compute_budget_usd=12)
    identity = dict(resolved_config=dict(steps=1972, batch_size=64), tokenizer='tokenizer',
                    data_manifest='corpus', source_python_sha256='source',
                    initial_weights_sha256={'model': 'weights'})
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'gcloud':
            return subprocess.CompletedProcess(argv, 0)
        assert kwargs['env']['JAX_PLATFORMS'] == 'tpu'
        if '--preflight-only' in argv:
            records = [dict(event='input_preflight', passed=True, input_identity=identity,
                            sequence_length=512)]
        elif argv[2] == 'scripts.train_encoder':
            stage = 'calibration' if '--stop-after' in argv else 'pilot'
            if fail == stage:
                return subprocess.CompletedProcess(argv, 1)
            n = 100 if stage == 'calibration' else 1972
            records = [dict(event='run_config', backend='tpu', devices=4, processes=1,
                            mlm_loss_backend='pallas', input_identity=identity)]
            records += [dict(event='train_step', step=i+1, updated=True,
                             projection_dense_fallback=False, tokens=32768,
                             nonpadding_tokens=25000, masked_tokens=3600, loss=2.,
                             seconds=10. if slow else .2, includes_compilation=i == 0)
                        for i in range(n)]
            records += [dict(event='checkpoint', step=n, seconds=1.)]
        else:
            assert argv[2] == 'scripts.evaluate_encoder'
            assert argv[argv.index('--max-rows') + 1] == '3073'
            Path(argv[argv.index('--output') + 1]).write_text('{"masked_token_loss": 2.0}')
            records = []
        kwargs['stdout'].write('\n'.join(map(json.dumps, records)) + '\n')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(worker.subprocess, 'run', execute)
    monkeypatch.setenv('FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS', '7000')
    monkeypatch.setenv('JAX_PROCESS_COUNT', '1')
    return args, recipe, calls


def test_full_pilot_follows_real_calibration_and_full_evaluation(tmp_path, monkeypatch):
    args, recipe, calls = fixture(tmp_path, monkeypatch)
    worker.run(args)
    report = json.loads((args.output / 'summary.json').read_text())
    assert report['passed'] is True and report['quality_qualified'] is False
    assert [s['name'] for s in report['stages']] == ['preflight', 'calibration', 'pilot', 'evaluation']
    trains = [c for c in calls if len(c) > 2 and c[2] == 'scripts.train_encoder']
    assert all(c[c.index('--steps')+1] == str(recipe['steps']) for c in trains)
    assert '--stop-after' not in trains[-1] and '--resume' not in trains[-1]


def test_insufficient_lease_never_runs_shortened_pilot(tmp_path, monkeypatch):
    args, _, calls = fixture(tmp_path, monkeypatch, slow=True)
    with pytest.raises(ValueError, match='lease'):
        worker.run(args)
    assert not any('gs://test/quality/pilot' in c for c in calls)
    assert json.loads((args.output / 'summary.json').read_text())['passed'] is False


def test_calibration_failure_stops_before_pilot(tmp_path, monkeypatch):
    args, _, calls = fixture(tmp_path, monkeypatch, fail='calibration')
    with pytest.raises(RuntimeError, match='calibration'):
        worker.run(args)
    assert not any('gs://test/quality/pilot' in c for c in calls)


def test_stale_input_rejected_before_subprocess(tmp_path, monkeypatch):
    args, recipe, calls = fixture(tmp_path, monkeypatch)
    Path(next(iter(recipe['input_sha256']))).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        worker.run(args)
    assert calls == []


def test_command_does_not_mutate_frozen_recipe():
    recipe = dict(training_argv=['python', '-m', 'scripts.train_encoder', '--output', 'original',
                                '--steps', '1972'], calibration_steps=100)
    original = copy.deepcopy(recipe)
    result = worker.training_command(recipe, 'gs://target', calibration=True)
    assert result[-2:] == ['--stop-after', '100']
    assert result[result.index('--steps') + 1] == '1972'
    assert recipe == original


def test_worker_import_never_initializes_accelerator_runtime():
    import os
    import sys
    result = subprocess.run(
        [sys.executable, '-c',
         "import sys; import scripts.validate_encoder_quality_pilot; "
         "assert 'jax' not in sys.modules; assert 'flaxchat' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ | {'JAX_PLATFORMS': 'tpu'},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
