import pytest
from flaxchat.operations import RunLedger, run_spot_attempt


@pytest.fixture(autouse=True)
def cleanup_scope_provider_fixture(monkeypatch):
    # These lifecycle tests fake all provider I/O. Exact scope semantics are
    # exercised separately in test_gcp_cleanup_guard.py.
    monkeypatch.setattr('scripts.gcp_cleanup_guard.preflight_cleanup_scope',
                        lambda **kwargs: {'server_permission_probes_required': True})


def test_free_compute_still_reserves_ancillary_exposure_and_shared_cap(tmp_path):
    ledger = RunLedger(tmp_path / 'ledger.json', 2)
    assert ledger.reserve('free-scan', 'cloud-shell/owned-job', 0, 900, 1) == 1
    assert ledger.summary()['reserved_usd'] == 1
    assert ledger.summary()['posted_usage_usd'] is None
    with pytest.raises(ValueError, match='positive reservation'):
        ledger.reserve('unaccounted', 'cloud-shell/unaccounted', 0, 900)
    with pytest.raises(ValueError, match='budget exhausted'):
        ledger.reserve('over-cap', 'cloud-shell/over-cap', 0, 900, 2)


def test_attempt_names_cannot_reuse_prior_expiry_guard_identity():
    from scripts.gcp_spot_supervisor import attempt_resource_name
    prefix = 'flaxchat-validation-yat-mmbert-copied-launch'
    first = attempt_resource_name(prefix, 0, 'a'*16)
    second = attempt_resource_name(prefix, 0, 'b'*16)
    retry = attempt_resource_name(prefix, 1, 'a'*16)
    assert len({first, second, retry}) == 3
    assert all(name.startswith('flaxchat-validation-') and len(name) <= 63
               for name in (first, second, retry))
    assert first.endswith('-0-' + 'a'*16)
    long_name = attempt_resource_name(prefix*5, 9, 'f'*16)
    assert len(long_name) <= 63 and long_name.endswith('-9-'+'f'*16)
    for invalid in ('other-prefix', 'flaxchat-validation-INVALID', prefix+'/'+'x'*100):
        with pytest.raises(ValueError):
            attempt_resource_name(invalid, 0, 'a'*16)


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


@pytest.mark.parametrize('action', ['describe', 'delete', 'create'])
def test_not_found_is_not_successful_creation(monkeypatch, action):
    import subprocess
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda argv, **kwargs:
        subprocess.CompletedProcess(argv, 1, '', 'NOT_FOUND: requested accelerator was not found'))
    argv = ['compute', 'tpus', 'queued-resources', action, 'test']
    if action == 'create':
        with pytest.raises(RuntimeError, match='NOT_FOUND'):
            supervisor.cloud(argv)
    else:
        assert supervisor.cloud(argv) is None


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


@pytest.mark.parametrize('duration', [1800, 7200])
@pytest.mark.parametrize('mode', ['spot', 'flex-start'])
def test_provisioning_read_timeout_does_not_recreate_resource(tmp_path, monkeypatch, duration, mode):
    import json
    import subprocess
    from pathlib import Path
    from scripts import gcp_spot_supervisor as supervisor
    state = {'exists': False, 'timed_out': False, 'creates': 0}
    provider_labels = {}
    def cloud(argv, **kwargs):
        if argv[3] == 'create':
            assert ('--spot' in argv) == (mode == 'spot')
            if mode == 'flex-start':
                assert '--provisioning-model=flex-start' in argv
                assert 0 < int(argv[argv.index('--max-run-duration') + 1][:-1]) <= duration
            state['exists'] = True
            state['creates'] += 1
            provider_labels.update(dict(item.split('=', 1) for item in argv[argv.index('--labels') + 1].split(',')))
            return {'state': 'READY'}
        elif argv[3] == 'delete':
            state['exists'] = False
        elif state['exists'] and not state['timed_out']:
            state['timed_out'] = True
            raise subprocess.TimeoutExpired(argv, 60)
        return {'state': 'READY', 'labels': provider_labels} if state['exists'] else None
    def arm(argv):
        Path(argv[argv.index('--receipt') + 1]).write_text(json.dumps({'verified': True}))
        return 0
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'main', arm)
    commands = []
    monkeypatch.setattr(supervisor.gcp_tpu_run, 'main', lambda argv: commands.append(argv) or 0)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    assert supervisor.main(['--project', 'tpubuilders', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-32',
        '--runtime-version', 'v2-alpha-tpuv5', '--name', 'flaxchat-validation-read-timeout', '--output', str(tmp_path),
        '--hourly-usd', '1', '--budget-usd', '10', '--attempt-seconds', str(duration),
        '--provisioning-model', mode,
        '--setup', '["setup"]', '--workload', '["train"]']) == 0
    assert state == {'exists': False, 'timed_out': True, 'creates': 1}
    assert int(commands[0][commands[0].index('--timeout') + 1]) <= 300
    assert duration - 30 < int(commands[1][commands[1].index('--timeout') + 1]) < duration
    attempt = json.loads((tmp_path / 'ledger.json').read_text())['attempts']['0']
    identity = json.loads((tmp_path / 'attempt-0' / 'resource-identity.json').read_text())
    assert attempt['resource'].endswith('/' + identity['name'])
    assert identity['name'].endswith('-0-' + identity['run_nonce'])
    assert commands[0][commands[0].index('--node') + 1] == identity['name']
    assert attempt['max_seconds'] == duration + 1800
    assert attempt['status'] == 'resource_absent'
    assert json.loads((tmp_path / 'attempt-0' / 'create-response.json').read_text()) == {'state': 'READY'}


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
        attempt = ledger.summary()['attempts']['a']
        assert attempt['status'] == 'attempt_failed'
        assert attempt['events'][-1]['error'] == 'RuntimeError: partial creation'
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
    failed = ledger.summary()['attempts']['b']
    assert failed['status'] == 'guard_failed_before_provisioning'
    assert 'Missing verified cloud lease' in failed['events'][-1]['error']


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
    with pytest.raises(SystemExit) as error:
        supervisor.main(['--project', 'tpubuilders', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-16',
            '--runtime-version', 'v2-alpha-tpuv5', '--name', 'flaxchat-validation-budget', '--output', str(tmp_path),
            '--hourly-usd', '33.6', '--budget-usd', '1', '--setup', '["true"]', '--workload', '["true"]'])
    assert error.value.code == 2


def test_supervisor_retries_only_after_verified_cleanup_with_resume(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from scripts import gcp_spot_supervisor as supervisor
    states, events = {}, []
    provider_labels = {}
    def cloud(argv, **kwargs):
        operation, name = argv[3:5]
        if operation == 'create':
            assert '--async' in argv, 'submission must not wait for capacity inside gcloud'
            assert kwargs['timeout'] <= 60
            assert not states, 'previous attempt must be absent before retry'
            states[name] = 'READY'
            provider_labels[name] = dict(item.split('=', 1) for item in argv[argv.index('--labels') + 1].split(','))
            events.append(('create', name))
            return {}
        if operation == 'delete':
            states.pop(name, None)
            provider_labels.pop(name, None)
            events.append(('delete', name))
            return {}
        return {'state': states[name], 'labels': provider_labels[name]} if name in states else None
    def arm(argv):
        Path(argv[argv.index('--receipt') + 1]).write_text(json.dumps({'verified': True}))
        return 0
    def execute(argv):
        name = argv[argv.index('--node') + 1]
        command = argv[argv.index('--') + 1:]
        if '--setup' in argv:
            assert float(argv[argv.index('--timeout') + 1]) <= 300
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


def test_supervisor_deadline_survives_host_sleep_and_clock_changes(monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(supervisor.time, 'time', lambda: 1000.)
    assert supervisor.remaining_seconds(300., 1100.) == 100
    with pytest.raises(TimeoutError, match='lease is expiring'):
        supervisor.remaining_seconds(300., 1020.)
    with pytest.raises(TimeoutError, match='lease is expiring'):
        supervisor.remaining_seconds(120., 3000.)


@pytest.mark.parametrize('queue', [None, {'state': {'state': 'FAILED'}},
                                  {'state': {'state': 'SUSPENDING'}},
                                  {'state': {'state': 'SUSPENDED'}},
                                  {'state': {'state': 'DELETING'}}])
def test_capacity_terminal_request_does_not_wait_or_recreate(monkeypatch, queue):
    from scripts import gcp_spot_supervisor as supervisor
    calls = []
    def cloud(argv, **kwargs):
        calls.append(argv)
        return queue if argv[2] == 'queued-resources' else None
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: pytest.fail('terminal queue must fail promptly'))
    with pytest.raises(RuntimeError, match='Spot request unavailable'):
        supervisor.wait_for_ready('test', [], lambda: 300, 60)
    assert all(argv[3] == 'describe' for argv in calls)


def test_capacity_wait_counts_host_sleep(monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: 100.)
    wall = iter([1000., 1100.])
    monkeypatch.setattr(supervisor.time, 'time', lambda: next(wall))
    monkeypatch.setattr(supervisor, 'cloud', lambda *a, **k: pytest.fail('wait already expired'))
    with pytest.raises(TimeoutError, match='capacity wait budget'):
        supervisor.wait_for_ready('test', [], lambda: 300, 60)


@pytest.mark.parametrize('outcome', ['failure', 'timeout'])
def test_supervisor_local_preflight_precedes_cloud_allocation(tmp_path, monkeypatch, outcome):
    import subprocess
    from scripts import gcp_spot_supervisor as supervisor
    calls = []
    def preflight(argv, **kwargs):
        calls.append(argv)
        assert kwargs['timeout'] == 120
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(argv, 120)
        raise subprocess.CalledProcessError(2, argv)
    monkeypatch.setattr(supervisor.subprocess, 'run', preflight)
    def unexpected_cloud(*args, **kwargs):
        pytest.fail('Cloud access occurred before local preflight passed')
    monkeypatch.setattr(supervisor, 'cloud', unexpected_cloud)
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'lease', unexpected_cloud)
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        supervisor.main(['--project', 'test', '--zone', 'us-east5-a', '--accelerator-type', 'v5p-16',
            '--runtime-version', 'test', '--name', 'flaxchat-validation-test', '--output', str(tmp_path),
            '--hourly-usd', '1', '--budget-usd', '10', '--local-preflight', '["python", "check.py"]',
            '--setup', '["true"]', '--workload', '["true"]'])
    assert calls == [['python', 'check.py']]
    assert not (tmp_path / 'ledger.json').exists()


def test_unverified_cleanup_permissions_never_allocate(tmp_path, monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    def cloud(argv, **kwargs):
        assert argv[3] == 'describe', 'Unverified guard must not create resources'
        return None
    def reject(argv):
        raise ValueError('Cleanup permissions could not be verified; allocation blocked')
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    monkeypatch.setattr(supervisor.gcp_cleanup_guard, 'main', reject)
    with pytest.raises(ValueError, match='allocation blocked'):
        supervisor.main(['--project', 'tpubuilders', '--zone', 'us-west4-a',
            '--accelerator-type', 'v5litepod-16', '--runtime-version', 'v2-alpha-tpuv5-lite',
            '--name', 'flaxchat-validation-guard-test', '--output', str(tmp_path),
            '--hourly-usd', '1', '--budget-usd', '10', '--setup', '["true"]', '--workload', '["true"]'])


def test_supervisor_rejects_underfunded_attempt_before_side_effects(tmp_path, monkeypatch, capsys):
    from scripts import gcp_spot_supervisor as supervisor
    output = tmp_path / 'untouched'
    monkeypatch.setattr(supervisor, 'cloud', lambda *a, **k: pytest.fail('No cloud action permitted'))
    with pytest.raises(SystemExit) as error:
        supervisor.main(['--project', 'project', '--zone', 'zone', '--accelerator-type', 'v5litepod-8',
            '--runtime-version', 'version', '--name', 'flaxchat-validation-test', '--output', str(output),
            '--hourly-usd', '4.8', '--budget-usd', '7', '--attempt-seconds', '2100',
            '--setup', '["true"]', '--workload', '["true"]'])
    assert error.value.code == 2
    assert 'requires $7.20' in capsys.readouterr().err
    assert not output.exists()


def readiness_clock(monkeypatch, supervisor):
    clock = {'elapsed': 0.}
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: clock['elapsed'])
    monkeypatch.setattr(supervisor.time, 'time', lambda: 1000 + clock['elapsed'])
    monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: clock.update(elapsed=clock['elapsed'] + seconds))
    return clock


def test_provisioning_switches_to_startup_budget_and_preserves_state_transitions(monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    clock = readiness_clock(monkeypatch, supervisor)
    calls, observations, phases = [], [], []
    def cloud(argv, **kwargs):
        calls.append(argv)
        if argv[2] == 'queued-resources':
            return {'state': {'state': 'ACCEPTED' if clock['elapsed'] < 10 else 'PROVISIONING'}}
        return None if clock['elapsed'] < 10 else {'state': 'READY' if clock['elapsed'] >= 40 else 'CREATING'}
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    supervisor.wait_for_ready('owned', [], lambda: 300, 15, 60,
                              observe=observations.append, phase=phases.append)
    assert clock['elapsed'] == 40  # Longer than capacity budget, inside startup budget.
    assert phases == ['startup_wait']
    assert observations[0]['queue_state'] == 'ACCEPTED'
    assert any(item['queue_state'] == 'PROVISIONING' and item['wait_phase'] == 'startup_wait' for item in observations)
    assert observations[-1]['node_state'] == 'READY'
    assert all(call[3] == 'describe' for call in calls)


def test_allocated_but_slow_startup_has_distinct_failure(monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    readiness_clock(monkeypatch, supervisor)
    monkeypatch.setattr(supervisor, 'cloud', lambda argv, **kwargs:
        {'state': {'state': 'PROVISIONING'}} if argv[2] == 'queued-resources' else {'state': 'CREATING'})
    observations = []
    with pytest.raises(TimeoutError, match='startup wait budget exhausted after allocation'):
        supervisor.wait_for_ready('owned', [], lambda: 300, 5, 15, observe=observations.append)
    assert observations[-1]['wait_phase'] == 'startup_wait'
    assert observations[-1]['node_state'] == 'CREATING'


@pytest.mark.parametrize('kind', ['timeout', 'transport'])
def test_unknown_provider_read_reattaches_same_resource_and_never_creates(monkeypatch, kind):
    import subprocess
    from scripts import gcp_spot_supervisor as supervisor
    readiness_clock(monkeypatch, supervisor)
    calls, observations = [], []
    def cloud(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            if kind == 'timeout':
                raise subprocess.TimeoutExpired(argv, 10)
            raise RuntimeError('Connection reset by peer')
        return {'state': {'state': 'PROVISIONING'}} if argv[2] == 'queued-resources' else {'state': 'READY'}
    monkeypatch.setattr(supervisor, 'cloud', cloud)
    supervisor.wait_for_ready('same-owned', [], lambda: 300, 60, 60, observe=observations.append)
    assert observations[0]['observation_error'] is not None
    assert all(call[3] == 'describe' and call[4] == 'same-owned' for call in calls)


def test_provider_permission_failure_never_becomes_transport_wait(monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor, 'cloud', lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError('403 Permission denied: service unavailable')))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: pytest.fail('No retry for denied permission'))
    with pytest.raises(RuntimeError, match='Permission denied'):
        supervisor.wait_for_ready('owned', [], lambda: 300, 60, 60)


def test_readiness_does_not_override_expired_global_lease(monkeypatch):
    from scripts import gcp_spot_supervisor as supervisor
    monkeypatch.setattr(supervisor, 'cloud', lambda argv, **kw:
        {'state': {'state': 'ACTIVE'}} if argv[2] == 'queued-resources' else {'state': 'READY'})
    calls = {'remaining': 0}
    def remaining():
        calls['remaining'] += 1
        if calls['remaining'] >= 3:
            raise TimeoutError('Attempt lease is expiring')
        return 300
    with pytest.raises(TimeoutError, match='lease is expiring'):
        supervisor.wait_for_ready('owned', [], remaining, 60, 60)
