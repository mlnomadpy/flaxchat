"""Fail-closed qualification orchestration without provisioning cloud resources."""
import json
from types import SimpleNamespace

import pytest

from scripts import validate_encoder_tpu as qualification


@pytest.mark.parametrize('mode,processes,devices,error', [
    ('multi', 1, 4, 'at least two'), ('single', 2, 8, 'exactly one'),
    ('recovery', 2, 8, 'exactly one'), ('multi', 3, 8, 'Invalid global'),
])
def test_reject_wrong_topology(mode, processes, devices, error):
    with pytest.raises(ValueError, match=error):
        qualification.validate_hardware(dict(backend='tpu', process_count=processes, device_count=devices), mode)


def test_hardware_expectations():
    report = dict(backend='tpu', process_count=2, device_count=8)
    qualification.validate_hardware(report, 'multi', 2, 8)
    with pytest.raises(ValueError, match='process count'):
        qualification.validate_hardware(report, 'multi', 4, 8)
    with pytest.raises(ValueError, match='device count'):
        qualification.validate_hardware(report, 'multi', 2, 16)
    with pytest.raises(ValueError, match='TPU backend'):
        qualification.validate_hardware(report | {'backend': 'cpu'}, 'multi')


@pytest.mark.parametrize('fault', ['committed', 'unmarked-kill', 'timeout', 'heldout-failure'])
def test_recovery_requires_committed_fault_and_preserves_backend(tmp_path, monkeypatch, fault):
    monkeypatch.setattr('sys.argv', ['validate', '--prefix', 'gs://test/run', '--mode', 'recovery',
        '--baseline-prefix', 'gs://test/baseline', '--artifact-root', str(tmp_path),
        '--mlm-projection', 'masked', '--mlm-loss-backend', 'pallas', '--mlm-vocab-tile', '512'])
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'gcloud':
            return SimpleNamespace(returncode=0)
        if argv[1] == '-c':
            from pathlib import Path
            Path(argv[-1]).write_text(json.dumps(dict(backend='tpu', process_count=1, device_count=4)))
        if 'scripts.evaluate_encoder' in argv and fault == 'heldout-failure':
            return SimpleNamespace(returncode=1)
        if 'scripts.validate_encoder_interruption' in argv:
            if fault == 'timeout':
                raise qualification.subprocess.TimeoutExpired(argv, 900)
            if fault == 'committed':
                kwargs['stdout'].write('FAULT_INJECTION: encoder SIGKILL after committed step 3\n')
            return SimpleNamespace(returncode=-9)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(qualification.subprocess, 'run', execute)
    manifests = []
    def manifest(argv, **kwargs):
        manifests.append(argv[-1])
        return json.dumps(dict(model_state={'a': 'hash'}, optimizer_state={'b': 'hash'}, training_state={'c': 'hash'}))
    monkeypatch.setattr(qualification.subprocess, 'check_output', manifest)
    assert qualification.main() == (0 if fault == 'committed' else 1)
    summary = json.loads(next(tmp_path.glob('results-*/summary.json')).read_text())
    assert summary['required_passed'] == (fault == 'committed')
    assert summary['quality_qualified'] is False
    assert summary['mlm_loss_backend'] == 'pallas'
    training = [argv for argv in calls if 'scripts.train_encoder' in argv or 'scripts.validate_encoder_interruption' in argv]
    for argv in training:
        assert argv[argv.index('--mlm-loss-backend') + 1] == 'pallas'
        assert argv[argv.index('--mlm-projection') + 1] == 'masked'
        assert argv[argv.index('--mlm-vocab-tile') + 1] == '512'
    if fault == 'committed':
        assert len(training) == 2
        assert 'checkpoints-single-residual-float32-masked-pallas-tile-512' in manifests[0]
    else:
        assert len(training) == (0 if fault == 'heldout-failure' else 1)
        # Never spend on recovery after invalid baseline evaluation, unexplained kill, or timeout.
        assert not manifests


@pytest.mark.parametrize('defect', [None, 'missing-worker', 'duplicate-rank', 'wrong-topology',
                                   'wrong-config', 'missing-stage', 'recovery-mismatch', 'missing-recovery',
                                   'rejected-update', 'dense-fallback', 'missing-updates', 'wrong-worker',
                                   'out-of-order', 'wrong-tokens', 'nonfinite-masks', 'excess-masks',
                                   'invalid-timing', 'boolean-rank', 'zero-batch', 'missing-fault', 'raw-mismatch', 'missing-controller', 'controller-failure'])
def test_multi_host_summary_requires_all_worker_evidence(tmp_path, defect):
    from scripts.summarize_encoder_validation import summarize, REQUIRED_STAGES
    controller = dict(passed=defect != 'controller-failure', workers=[dict(rank=i, returncode=0) for i in range(2)])
    if defect != 'missing-controller':
        (tmp_path / 'controller.json').write_text(json.dumps(controller))
    for rank in range(2):
        if defect == 'missing-worker' and rank == 1:
            continue
        directory = tmp_path / f'worker-{rank}'
        directory.mkdir()
        report = dict(hardware=dict(backend='tpu', process_count=2, device_count=8, process_index=rank),
                      compute_dtype='bfloat16', residual_dtype='float32', mlm_projection='masked',
                      mlm_loss_backend='pallas', mlm_vocab_tile=1024, batch_size=16, lengths=[512],
                      required_passed=True, failures=[], stages=[dict(name=name, code=-9 if name == 'interrupt' else 0, passed=True) for name in REQUIRED_STAGES])
        if defect == 'boolean-rank':
            report['hardware']['process_index'] = bool(rank)
        if defect == 'zero-batch':
            report['batch_size'] = 0
        recovery = dict(model_state=True, optimizer_state=True, training_state=True)
        if rank == 1:
            if defect == 'duplicate-rank':
                report['hardware']['process_index'] = 0
            if defect == 'wrong-topology':
                report['hardware']['backend'] = 'cpu'
            if defect == 'wrong-config':
                report['mlm_loss_backend'] = 'xla_full'
            if defect == 'missing-stage':
                report['stages'].pop()
            if defect == 'recovery-mismatch':
                recovery['optimizer_state'] = False
        for stage_name, steps in (('train-512', range(1, 7)), ('resume', range(4, 7))):
            events = [dict(event='train_step', step=step, updated=True, masked_tokens=12,
                           loss=1.0, tokens=8192, seconds=0.1, projection_dense_fallback=False)
                      for step in steps] if rank == (1 if defect == 'wrong-worker' else 0) else []
            if rank == 0:
                if defect == 'rejected-update':
                    events[0]['updated'] = False
                if defect == 'dense-fallback':
                    events[0]['projection_dense_fallback'] = True
                if defect == 'missing-updates':
                    events.pop()
                if defect == 'out-of-order':
                    events.reverse()
                if defect == 'wrong-tokens':
                    events[0]['tokens'] = 4096
                if defect == 'nonfinite-masks':
                    events[0]['masked_tokens'] = float('nan')
                if defect == 'excess-masks':
                    events[0]['masked_tokens'] = 8193
                if defect == 'invalid-timing':
                    events[0]['seconds'] = 0
            (directory / f'{stage_name}.log').write_text('\n'.join(json.dumps(event) for event in events))
        (directory / 'hardware.json').write_text(json.dumps(report['hardware']))
        (directory / 'interrupt.log').write_text('' if defect == 'missing-fault' else 'FAULT_INJECTION: encoder SIGKILL after committed step 3')
        raw = dict(step=6, model_state={'a': 'hash'}, optimizer_state={'b': 'hash'}, training_state={'c': 'hash'})
        (directory / 'baseline-manifest.json').write_text(json.dumps(raw))
        if defect == 'raw-mismatch':
            raw['model_state']['a'] = 'different'
        (directory / 'recovery-manifest.json').write_text(json.dumps(raw))
        (directory / 'summary.json').write_text(json.dumps(report))
        if defect != 'missing-recovery' or rank == 0:
            (directory / 'recovery.json').write_text(json.dumps(recovery))
    result = summarize(tmp_path, expected_processes=2, expected_devices=8)
    assert result['passed'] == (defect is None)
    assert result['quality_qualified'] is False
    if defect is None:
        assert all(set(worker['update_log_sha256']) == {'train-512', 'resume'}
                   for worker in result['workers'])


def context_records(length=2048):
    return [dict(event='train_step', step=i, updated=True, masked_tokens=123, loss=1.,
                 tokens=4*length, seconds=1., projection_dense_fallback=False) for i in (1, 2, 3)] + [
        dict(event='checkpoint', step=3, seconds=2.),
        dict(event='device_memory', devices=[dict(device=f'TPU:{i}', statistics=dict(peak_bytes_in_use=100, peak_bytes_reserved=80, bytes_limit=200)) for i in range(4)])]


@pytest.mark.parametrize('defect', [None, 'rejected', 'wrong-context', 'fallback', 'missing-checkpoint', 'missing-memory', 'missing-reserved',
    'nan-masks', 'excess-masks', 'boolean-step', 'duplicate-device', 'empty-device',
    'infinite-limit', 'early-checkpoint', 'early-memory'])
def test_context_acceptance_requires_real_updates_checkpoint_and_memory(tmp_path, defect):
    records = context_records()
    if defect == 'rejected':
        records[0]['updated'] = False
    if defect == 'wrong-context':
        records[0]['tokens'] = 4*512
    if defect == 'fallback':
        records[0]['projection_dense_fallback'] = True
    if defect == 'missing-checkpoint':
        records.pop(-2)
    if defect == 'missing-memory':
        records.pop()
    if defect == 'missing-reserved':
        records[-1]['devices'][0]['statistics'].pop('peak_bytes_reserved')
    if defect == 'nan-masks':
        records[0]['masked_tokens'] = float('nan')
    if defect == 'excess-masks':
        records[0]['masked_tokens'] = 8193
    if defect == 'boolean-step':
        records[0]['step'] = True
    if defect == 'duplicate-device':
        records[-1]['devices'][1]['device'] = 'TPU:0'
    if defect == 'empty-device':
        records[-1]['devices'][0]['device'] = ''
    if defect == 'infinite-limit':
        records[-1]['devices'][0]['statistics']['bytes_limit'] = float('inf')
    if defect == 'early-checkpoint':
        records.insert(0, records.pop(-2))
    if defect == 'early-memory':
        records.insert(0, records.pop())
    log = tmp_path / 'train.log'
    log.write_text('\n'.join(json.dumps(record) for record in records))
    if defect:
        with pytest.raises(ValueError):
            qualification.validate_context_log(log, length=2048, batch=4, devices=4, masked=True)
    else:
        assert qualification.validate_context_log(log, length=2048, batch=4, devices=4, masked=True)['passed']


@pytest.mark.parametrize('failure', [None, 'exit-code', 'rejected-update'])
def test_context_sweep_stops_before_larger_case_on_failed_gate(tmp_path, monkeypatch, failure):
    monkeypatch.setattr('sys.argv', ['validate', '--prefix', 'gs://test/context', '--mode', 'context',
        '--artifact-root', str(tmp_path), '--lengths', '2048', '8192', '--mlm-projection', 'masked',
        '--mlm-loss-backend', 'xla_full'])
    lengths = []
    def execute(argv, **kwargs):
        if argv[0] == 'gcloud':
            return SimpleNamespace(returncode=0)
        if argv[1] == '-c':
            from pathlib import Path
            Path(argv[-1]).write_text(json.dumps(dict(backend='tpu', process_count=1, device_count=4)))
        else:
            assert argv[2] == 'scripts.train_encoder'
            length = int(argv[argv.index('--data')+1].rsplit('-',1)[1])
            lengths.append(length)
            if failure == 'exit-code':
                return SimpleNamespace(returncode=1)
            records = context_records(length)
            if failure == 'rejected-update':
                records[0]['updated'] = False
            kwargs['stdout'].write('\n'.join(json.dumps(record) for record in records))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(qualification.subprocess, 'run', execute)
    assert qualification.main() == int(failure is not None)
    assert lengths == ([2048] if failure else [2048, 8192])
    summary = json.loads(next(tmp_path.glob('results-*/summary.json')).read_text())
    assert summary['scope'] == 'context_execution_only'
    assert summary['quality_qualified'] is False


@pytest.mark.parametrize('failed_stage', ['cpu-suite', 'encoder-tpu-tests', 'released-parity-float32'])
def test_required_failure_stops_before_later_paid_stages(tmp_path, monkeypatch, failed_stage):
    monkeypatch.setattr('sys.argv', ['validate', '--prefix', 'gs://test/run',
        '--artifact-root', str(tmp_path), '--cloud-timeout-seconds', '17'])
    stages = []
    def execute(argv, **kwargs):
        if argv[0] == 'gcloud':
            assert kwargs['timeout'] == 17
            return SimpleNamespace(returncode=0)
        name = qualification.Path(kwargs['stdout'].name).stem
        stages.append(name)
        if name == 'hardware':
            qualification.Path(argv[-1]).write_text(json.dumps(
                dict(backend='tpu', process_count=1, device_count=4)))
        return SimpleNamespace(returncode=int(name == failed_stage))
    monkeypatch.setattr(qualification.subprocess, 'run', execute)
    assert qualification.main() == 1
    assert stages[-1] == failed_stage
    summary = json.loads(next(tmp_path.glob('results-*/summary.json')).read_text())
    assert summary['failures'] == [failed_stage]
    assert summary['required_passed'] is False


@pytest.mark.parametrize('failure', ['stage-upload', 'final-upload', 'manifest'])
def test_cloud_failures_are_bounded_and_fail_local_summary(tmp_path, monkeypatch, failure):
    monkeypatch.setattr('sys.argv', ['validate', '--prefix', 'gs://test/run',
        '--mode', 'recovery', '--baseline-prefix', 'gs://test/baseline',
        '--artifact-root', str(tmp_path), '--cloud-timeout-seconds', '17'])
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'gcloud':
            assert kwargs['timeout'] == 17
            is_final = any(value.endswith('/summary.json') for value in argv)
            if (failure == 'final-upload' and is_final) or (failure == 'stage-upload' and not is_final):
                raise qualification.subprocess.TimeoutExpired(argv, 17)
            return SimpleNamespace(returncode=0)
        if argv[1] == '-c':
            qualification.Path(argv[-1]).write_text(json.dumps(
                dict(backend='tpu', process_count=1, device_count=4)))
        if 'scripts.validate_encoder_interruption' in argv:
            kwargs['stdout'].write('FAULT_INJECTION: encoder SIGKILL after committed step 3\n')
            return SimpleNamespace(returncode=-9)
        return SimpleNamespace(returncode=0)
    def metadata(argv, **kwargs):
        assert kwargs['timeout'] == 17
        if failure == 'manifest':
            raise qualification.subprocess.TimeoutExpired(argv, 17)
        return json.dumps(dict(model_state={'a': 'hash'}, optimizer_state={'b': 'hash'}, training_state={'c': 'hash'}))
    monkeypatch.setattr(qualification.subprocess, 'run', execute)
    monkeypatch.setattr(qualification.subprocess, 'check_output', metadata)
    with pytest.raises(qualification.subprocess.TimeoutExpired):
        qualification.main()
    summary = json.loads(next(tmp_path.glob('results-*/summary.json')).read_text())
    assert summary['required_passed'] is False
    assert summary['all_checks_passed'] is False
    assert any('TimeoutExpired' in value for value in summary['failures'])
    if failure == 'stage-upload':
        assert not any('scripts.train_encoder' in argv for argv in calls)
    if failure == 'final-upload':
        assert any('final-evidence-upload' in value for value in summary['failures'])


@pytest.mark.parametrize('launcher', ['JAX_COORDINATOR_ADDRESS', 'JAX_PROCESS_COUNT',
    'JAX_PROCESS_INDEX', 'TPU_WORKER_HOSTNAMES', 'TPU_WORKER_ID', 'SLURM_NTASKS',
    'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'])
def test_cpu_stages_strip_all_distributed_launchers(launcher):
    original = {launcher: '2', 'PATH': '/usr/bin'}
    isolated = qualification.stage_environment(original, cpu=True)
    assert launcher not in isolated
    assert isolated['JAX_PLATFORMS'] == 'cpu'
    assert isolated['PATH'] == '/usr/bin'
    assert original[launcher] == '2'
    assert qualification.stage_environment(original)[launcher] == '2'
