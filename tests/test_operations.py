import pytest
from flaxchat.operations import RunLedger, run_spot_attempt


def test_cloud_retries_timed_out_reads_but_never_mutations(monkeypatch):
    import subprocess
    from scripts import gcp_spot_supervisor as supervisor
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return subprocess.CompletedProcess(argv, 0, '{"state":"READY"}', '')
    monkeypatch.setattr(supervisor.subprocess, 'run', run)
    assert supervisor.cloud(['compute', 'tpus', 'tpu-vm', 'describe', 'test']) == {'state': 'READY'}
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor.cloud(['compute', 'tpus', 'queued-resources', 'create', 'test'])
    assert len(calls) == 1


def test_cleanup_timeout_never_counts_as_absence(monkeypatch):
    import subprocess
    from scripts import gcp_spot_supervisor as supervisor
    calls = []
    def cloud(argv, **kwargs):
        calls.append(argv)
        if len(calls) <= 2:
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return None
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    assert supervisor.cleanup_resource('test', [], timeout=1)
    assert len(calls) == 5
    assert '--async' in calls[0]
    assert calls[-2][2:4] == ['queued-resources', 'describe']
    assert calls[-1][2:4] == ['tpu-vm', 'describe']


def test_cleanup_unknown_state_exhausts_deadline(monkeypatch):
    import subprocess
    from scripts import gcp_spot_supervisor as supervisor
    ticks = iter(range(100))
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    def cloud(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    assert supervisor.cleanup_resource('test', [], timeout=10) is False


def test_provisioning_read_timeout_does_not_recreate_resource(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from scripts import gcp_spot_supervisor as supervisor
    state = {'exists': False, 'timed_out': False, 'creates': 0}
    def cloud(argv, **kwargs):
        if argv[3] == 'create':
            state['exists'] = True
            state['creates'] += 1
        elif argv[3] == 'delete':
            state['exists'] = False
        elif state['exists'] and not state['timed_out']:
            state['timed_out'] = True
            raise subprocess.TimeoutExpired(argv, 60)
        return {'state': 'READY'} if state['exists'] else None
    def arm(argv):
        Path(argv[argv.index('--receipt') + 1]).write_text(json.dumps({'verified': True}))
        return 0
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'main', arm)
    monkeypatch.setattr(supervisor.gcp_tpu_run, 'main', lambda _: 0)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    assert supervisor.main(['--project', 'tpubuilders', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-32',
        '--runtime-version', 'v2-alpha-tpuv5', '--name', 'flaxchat-validation-read-timeout', '--output', str(tmp_path),
        '--hourly-usd', '1', '--budget-usd', '10', '--setup', '["setup"]', '--workload', '["train"]']) == 0
    assert state == {'exists': False, 'timed_out': True, 'creates': 1}


def test_reservations_survive_restart_and_count_retries(tmp_path):
    path = tmp_path / 'ledger.json'
    ledger = RunLedger(path, 10)
    ledger.reserve('a', 'resource-a', 6, 3600)
    restarted = RunLedger(path, 10)
    with pytest.raises(ValueError, match='exhausted'):
        restarted.reserve('b', 'resource-b', 6, 3600)
    with pytest.raises(ValueError, match='already'):
        restarted.reserve('a', 'resource-a', 1, 1)
    assert restarted.summary()['posted_usage_usd'] is None
    restarted.reconcile('a', usage_usd=2, promotional_credits_usd=2, billing_source='billing-row-id')
    summary = restarted.summary()
    assert summary['posted_usage_usd'] == 2
    assert summary['reserved_usd'] == 6


def test_provision_failure_still_deletes_and_missing_guard_never_allocates(tmp_path):
    ledger = RunLedger(tmp_path / 'ledger.json', 10)
    called = []
    def provision(resource):
        called.append('provision')
        raise RuntimeError('partial creation')
    def cleanup(resource):
        called.append('cleanup')
        return True
    with pytest.raises(RuntimeError, match='partial creation'):
        run_spot_attempt(ledger, 'a', 'resource', hourly_usd=1, max_seconds=100,
                         arm_and_verify=lambda *a: {'verified': True}, provision=provision,
                         run=lambda r: 0, cleanup_and_verify=cleanup)
    assert called == ['provision', 'cleanup']
    called.clear()
    with pytest.raises(RuntimeError, match='lease'):
        run_spot_attempt(ledger, 'b', 'resource-b', hourly_usd=1, max_seconds=100,
                         arm_and_verify=lambda *a: None, provision=provision,
                         run=lambda r: 0, cleanup_and_verify=cleanup)
    assert called == []


def test_failed_cleanup_overrides_success(tmp_path):
    ledger = RunLedger(tmp_path / 'ledger.json', 10)
    with pytest.raises(RuntimeError, match='absence'):
        run_spot_attempt(ledger, 'a', 'resource', hourly_usd=1, max_seconds=100,
                         arm_and_verify=lambda *a: {'verified': True}, provision=lambda r: None,
                         run=lambda r: 0, cleanup_and_verify=lambda r: False)
    assert ledger.summary()['attempts']['a']['status'] == 'cleanup_failed'


def test_supervisor_refuses_existing_resource_before_arming(tmp_path, monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor, 'cloud', lambda *a, **k: {'state': 'READY'})
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'main', lambda *a: pytest.fail('must not arm deletion for existing resource'))
    with pytest.raises(ValueError, match='existing'):
        supervisor.main(['--project', 'tpubuilders', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-16',
            '--runtime-version', 'v2-alpha-tpuv5', '--name', 'flaxchat-validation-existing', '--output', str(tmp_path),
            '--hourly-usd', '33.6', '--budget-usd', '40', '--setup', '["true"]', '--workload', '["true"]'])


def test_supervisor_checks_budget_before_guard_and_allocation(tmp_path, monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor, 'cloud', lambda *a, **k: None)
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'main', lambda *a: pytest.fail('must not arm an overbudget attempt'))
    with pytest.raises(ValueError, match='exhausted'):
        supervisor.main(['--project', 'tpubuilders', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-16',
            '--runtime-version', 'v2-alpha-tpuv5', '--name', 'flaxchat-validation-budget', '--output', str(tmp_path),
            '--hourly-usd', '33.6', '--budget-usd', '1', '--setup', '["true"]', '--workload', '["true"]'])


def test_supervisor_retries_only_after_verified_cleanup_with_resume(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from scripts import gcp_spot_supervisor as supervisor
    states, events = {}, []
    def cloud(argv, **kwargs):
        operation, name = argv[3:5]
        if operation == 'create':
            assert not states, 'previous attempt must be absent before retry'
            states[name] = 'READY'
            events.append(('create', name))
            return {}
        if operation == 'delete':
            states.pop(name, None)
            events.append(('delete', name))
            return {}
        return {'state': states[name]} if name in states else None
    def arm(argv):
        Path(argv[argv.index('--receipt') + 1]).write_text(json.dumps({'verified': True}))
        return 0
    def execute(argv):
        name = argv[argv.index('--node') + 1]
        command = argv[argv.index('--') + 1:]
        events.append(('execute', command))
        if command == ['train']:
            states[name] = 'PREEMPTED'
            return 1
        return 0
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'main', arm)
    monkeypatch.setattr(supervisor.gcp_tpu_run, 'main', execute)
    assert supervisor.main(['--project', 'tpubuilders', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-16',
        '--runtime-version', 'v2-alpha-tpuv5', '--name', 'flaxchat-validation-retry', '--output', str(tmp_path),
        '--hourly-usd', '1', '--budget-usd', '10', '--max-attempts', '2', '--setup', '["setup"]',
        '--checks', '["checks"]', '--workload', '["train"]', '--resume-workload', '["resume"]']) == 0
    assert not states
    assert [value for event, value in events if event == 'execute'] == [
        ['setup'], ['checks'], ['train'], ['setup'], ['checks'], ['resume']]
    ledger = RunLedger(tmp_path / 'ledger.json', 10).summary()
    assert ledger['reserved_usd'] == 6
    assert len(ledger['attempts']) == 2
