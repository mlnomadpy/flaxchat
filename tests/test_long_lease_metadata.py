"""Long-lease admission and clock/cost boundaries; fake provider I/O only."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import gcp_cleanup_guard as guard
from scripts import gcp_spot_supervisor as supervisor
from scripts import gcp_tpu_run as worker
from scripts import representation_run as representation


def campaign_args(root, seconds, opt_in=False, budget=200):
    return ['--project', 'project', '--zone', 'us-west4-a', '--accelerator-type', 'v5litepod-8',
            '--runtime-version', 'runtime', '--name', 'flaxchat-validation-test', '--output', str(root),
            '--hourly-usd', '9.6', '--budget-usd', str(budget), '--attempt-seconds', str(seconds),
            '--setup', '["setup"]', '--workload', '["workload"]',
            *(['--allow-long-lease'] if opt_in else [])]


@pytest.mark.parametrize('seconds,opt_in,accepted', [
    (7200, False, True), (7201, False, False), (7201, True, True),
    (43200, True, True), (43201, True, False), (179, True, False),
])
def test_campaign_explicit_opt_in_and_hard_max(tmp_path, seconds, opt_in, accepted):
    with patch.object(supervisor, 'campaign', return_value=0) as launch:
        if accepted:
            assert supervisor.main(campaign_args(tmp_path, seconds, opt_in)) == 0
            assert launch.call_args.args[0].attempt_seconds == seconds
        else:
            with pytest.raises(SystemExit):
                supervisor.main(campaign_args(tmp_path, seconds, opt_in))
            launch.assert_not_called()


def test_twelve_hour_reservation_keeps_cleanup_and_ancillary_margin(tmp_path):
    # 12h * $9.60 + 30m cleanup * $9.60 + $2 ancillary = $122.
    with patch.object(supervisor, 'campaign', return_value=0) as launch:
        with pytest.raises(SystemExit):
            supervisor.main(campaign_args(tmp_path, 43200, True, 121.99))
        launch.assert_not_called()
        assert supervisor.main(campaign_args(tmp_path, 43200, True, 122)) == 0


@pytest.mark.parametrize('seconds,opt_in,accepted', [
    (7200, False, True), (7201, False, False), (43200, True, True),
    (43201, True, False), (True, True, False), (3600, 'true', False),
])
def test_guard_payload_opt_in(seconds, opt_in, accepted):
    args = ('project', 'us-west4-a', 'flaxchat-validation-test', seconds)
    if not accepted:
        with pytest.raises(ValueError):
            guard.lease(*args, now=1000, allow_long_lease=opt_in)
        return
    lease = guard.lease(*args, now=1000, allow_long_lease=opt_in)
    assert lease['deadline'] == 1000 + seconds
    assert lease.get('allow_long_lease', False) is opt_in


def test_guard_long_lease_still_requires_server_identity_and_original_deadline(monkeypatch):
    lease = guard.lease('project', 'us-west4-a', 'flaxchat-validation-test', 43200,
                        now=1000, allow_long_lease=True)
    receipt = dict(lease=lease, execution='execution-id', location='us-central1')
    execution = dict(state='ACTIVE', argument=json.dumps(lease),
                     status={'currentSteps': [{'step': 'wait_for_expiry'}]})
    monkeypatch.setattr(guard.subprocess, 'check_output', lambda *a, **kw: json.dumps(execution))
    with patch.object(guard, 'verify_cleanup_permissions') as probe:
        assert guard.verify(receipt, project='project', zone='us-west4-a',
                            queue='flaxchat-validation-test', now=1100) == 43100
        probe.assert_called_once()
    # Receipt opt-in cannot extend an existing server lease.
    execution['argument'] = json.dumps({**lease, 'deadline': lease['deadline'] - 1})
    with pytest.raises(ValueError, match='exact lease'):
        guard.verify(receipt, project='project', zone='us-west4-a',
                     queue='flaxchat-validation-test', now=1100)
    with pytest.raises(ValueError, match='expired'):
        guard.verify(receipt, project='project', zone='us-west4-a',
                     queue='flaxchat-validation-test', now=lease['deadline'] - 59)


@pytest.mark.parametrize('seconds,opt_in,accepted', [(7201, False, False), (43200, True, True), (43201, True, False)])
def test_worker_timeout_boundary_before_cloud_io(tmp_path, seconds, opt_in, accepted):
    args = ['--project', 'project', '--zone', 'us-west4-a', '--node', 'node',
            '--output', str(tmp_path / 'worker'), '--timeout', str(seconds),
            *(['--allow-long-lease'] if opt_in else []), '--', 'workload']
    with patch.object(worker.subprocess, 'check_output', side_effect=RuntimeError('read reached')) as read:
        if accepted:
            with pytest.raises(RuntimeError, match='read reached'):
                worker.main(args)
            read.assert_called_once()
        else:
            with pytest.raises(SystemExit):
                worker.main(args)
            read.assert_not_called()


def test_long_attempt_passes_opt_in_and_does_not_reset_workload_deadline(tmp_path):
    clock = {'elapsed': 0., 'created': False}
    guards, commands = [], []
    def cloud(args, **kw):
        if args[3] == 'create':
            clock['created'] = True
            return {'name': 'operation'}
        if not clock['created']:
            return None
        return {'state': 'READY', 'labels': {'flaxchat-run': 'flaxchat-validation-test', 'flaxchat-stage': 'qualification'}}
    def arm(args):
        guards.append(args)
        Path(args[args.index('--receipt') + 1]).write_text('{"verified": true}')
    def run(args):
        commands.append(args)
        clock['elapsed'] += 600
        return 0
    with patch.object(supervisor, 'cloud', side_effect=cloud), \
         patch.object(guard, 'preflight_cleanup_scope', return_value={}), \
         patch.object(guard, 'main', side_effect=arm), \
         patch.object(supervisor, 'wait_for_ready', side_effect=lambda *a, **kw: clock.update(elapsed=1200)), \
         patch.object(supervisor, 'cleanup_resource', return_value=True), \
         patch.object(worker, 'main', side_effect=run), \
         patch.object(supervisor.time, 'monotonic', side_effect=lambda: clock['elapsed']), \
         patch.object(supervisor.time, 'time', side_effect=lambda: 1000 + clock['elapsed']):
        assert supervisor.main(campaign_args(tmp_path / 'campaign', 43200, True)) == 0
    assert '--allow-long-lease' in guards[0]
    assert guards[0][guards[0].index('--seconds') + 1] == '43200'
    assert len(commands) == 2
    assert '--allow-long-lease' in commands[1]
    assert int(commands[1][commands[1].index('--timeout') + 1]) == 43200 - 1800 - 15
    report = json.loads((tmp_path / 'campaign/campaign-receipt.json').read_text())
    assert report['reserved_usd'] == 122
    assert report['allow_long_lease'] is True


def test_manifest_adapter_carries_explicit_flag_only():
    manifest = dict(run_id='campaign', stage_id='stage', source={'uri': 'gs://bucket/source', 'sha256': 'a'*64},
                    deployment={'attempt_seconds': 43200, 'allow_long_lease': True})
    args = representation.supervisor_argv(manifest, 'gs://bucket/run.json', 'b'*64, '/tmp/output')
    assert '--allow-long-lease' in args
    del manifest['deployment']['allow_long_lease']
    args = representation.supervisor_argv(manifest, 'gs://bucket/run.json', 'b'*64, '/tmp/output')
    assert '--allow-long-lease' not in args


@pytest.mark.parametrize('seconds,accepted', [(30, True), (1800, True), (1801, False), (29, False)])
def test_outer_setup_cap_is_bounded(tmp_path, seconds, accepted):
    args = campaign_args(tmp_path, 43200, True) + ['--setup-timeout-seconds', str(seconds)]
    with patch.object(supervisor, 'campaign', return_value=0) as launch:
        if accepted:
            assert supervisor.main(args) == 0
        else:
            with pytest.raises(SystemExit):
                supervisor.main(args)
            launch.assert_not_called()
