"""Run bounded Spot attempts with a durable expiry guard and retry ledger.

Setup and workload are JSON argv arrays executed on every TPU worker. Retries
require a confirmed non-READY node and use the explicitly supplied resume argv.
No retry is allowed while prior resource absence remains unverified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from flaxchat.operations import RunLedger, run_spot_attempt
from scripts import gcp_cleanup_guard, gcp_tpu_run


def attempt_resource_name(prefix, index, nonce):
    """Never reuse an identity targeted by an older server-side expiry guard."""
    if not re.fullmatch(r'flaxchat-validation-[a-z0-9-]+', prefix):
        raise ValueError('Only dedicated validation prefixes are permitted')
    if not isinstance(index, int) or index < 0 or not re.fullmatch(r'[0-9a-f]{16}', nonce):
        raise ValueError('Invalid attempt index or run nonce')
    suffix = f'-{index}-{nonce}'
    return prefix[:63-len(suffix)].rstrip('-') + suffix


def billing_label(value):
    """Normalize provider labels without losing identity through truncation."""
    normalized = re.sub('[^a-z0-9_-]', '-', value.lower()).strip('-_') or 'run'
    if normalized != value or len(normalized) > 63:
        normalized = normalized[:50].rstrip('-_') + '-' + hashlib.sha256(value.encode()).hexdigest()[:12]
    return normalized


def verify_node_labels(node, expected):
    actual = node.get('labels', {}) if isinstance(node, dict) else {}
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError('Provider node labels differ from campaign identities')
    return actual


def cloud(argv, *, timeout: float = 60):
    # Retry only reads, within the original call deadline. A timed-out create
    # may have succeeded and must never be replayed as a fresh allocation.
    read_only = len(argv) > 3 and argv[3] in ('describe', 'list')
    deadline = time.monotonic() + timeout
    for attempt in range(3 if read_only else 1):
        try:
            release = ['alpha'] if '--provisioning-model=flex-start' in argv else []
            result = subprocess.run(['gcloud', *release, *argv, '--format=json', '--quiet'],
                                    capture_output=True, text=True,
                                    timeout=max(.1, min(20 if read_only else timeout, deadline - time.monotonic())))
            break
        except subprocess.TimeoutExpired:
            if not read_only or attempt == 2 or time.monotonic() >= deadline:
                raise
    if result.returncode:
        if (read_only or (len(argv) > 3 and argv[3] == 'delete')) and (
            'NOT_FOUND' in result.stderr or 'was not found' in result.stderr
        ):
            return None
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def cleanup_resource(name, flags, *, timeout=1800, observe=None):
    """Request deletion asynchronously, then verify queue AND worker absence."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        def bounded(argv):
            return cloud(argv, timeout=max(.1, min(30, deadline - time.monotonic())))
        try:
            bounded(['compute', 'tpus', 'queued-resources', 'delete', name, *flags, '--force', '--async'])
        except subprocess.TimeoutExpired:
            # The request may already be accepted. Absence checks are the
            # authority, and this idempotent deletion can be requested again.
            pass
        try:
            queue = bounded(['compute', 'tpus', 'queued-resources', 'describe', name, *flags])
            node = bounded(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags])
            if observe:
                observe(dict(queue_absent=queue is None, node_absent=node is None,
                    queue_state=(queue or {}).get('state'), node_state=(node or {}).get('state'), observed_unix=time.time()))
            if queue is None and node is None:
                return True
        except subprocess.TimeoutExpired:
            pass  # Unknown is not absent; keep the cloud guard armed.
        time.sleep(max(0, min(10, deadline - time.monotonic())))
    return False


def command(value):
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(x, str) and x for x in parsed):
        raise argparse.ArgumentTypeError('Expected nonempty JSON argv array')
    return parsed


def remaining_seconds(monotonic_deadline, wall_deadline):
    # macOS monotonic time may exclude system sleep; the cloud lease does not.
    seconds = int(min(monotonic_deadline - time.monotonic(), wall_deadline - time.time()))
    if seconds < 30:
        raise TimeoutError('Attempt lease is expiring')
    return seconds


def wait_for_ready(name, flags, remaining, capacity_wait_seconds=None,
                   startup_wait_seconds=None, *, observe=None, phase=None):
    """Separate queue capacity from allocated-resource startup under one lease.

    PROVISIONING means selected from the queue and resources being allocated,
    not another capacity wait. Transport uncertainty never creates a new request.
    """
    started, wall_started = time.monotonic(), time.time()
    startup_started = startup_wall_started = None
    queue_state = node_state = None
    last_observation = None
    def publish(error=None):
        nonlocal last_observation
        observation = dict(queue_state=queue_state, node_state=node_state,
            wait_phase='startup_wait' if startup_started is not None else 'capacity_wait',
            observation_error=error)
        if observation != last_observation:
            if observe:
                observe({**observation, 'observed_unix': time.time()})
            last_observation = observation
    def begin_startup():
        nonlocal startup_started, startup_wall_started
        if startup_started is None:
            startup_started, startup_wall_started = time.monotonic(), time.time()
            if phase:
                phase('startup_wait')
    def time_left():
        seconds = remaining()
        if startup_started is not None:
            budget, mono_start, wall_start = startup_wait_seconds, startup_started, startup_wall_started
            label = 'TPU startup wait budget exhausted after allocation was observed'
        else:
            budget, mono_start, wall_start = capacity_wait_seconds, started, wall_started
            label = 'Spot capacity wait budget exhausted; allocation has not been observed'
        if budget is not None:
            elapsed = max(time.monotonic() - mono_start, time.time() - wall_start)
            seconds = min(seconds, budget - elapsed)
            if seconds <= 0:
                raise TimeoutError(f'{label}; queue={queue_state or "unknown"}; node={node_state or "unknown"}')
        return seconds
    while True:
        try:
            queue = cloud(['compute', 'tpus', 'queued-resources', 'describe', name, *flags],
                          timeout=min(60, time_left()))
            state = (queue or {}).get('state', {})
            queue_state = state.get('state') if isinstance(state, dict) else state
            if queue_state in ('PROVISIONING', 'ACTIVE'):
                begin_startup()
            publish()
            if queue is None or queue_state in ('FAILED', 'SUSPENDING', 'SUSPENDED', 'DELETING'):
                detail = json.dumps(queue) if queue else (
                    'inspect the saved create-response.json operation and TPU audit logs; '
                    'absence alone does not establish capacity exhaustion'
                )
                raise RuntimeError(f'Spot request unavailable: {queue_state or "absent"}; {detail}')
            node = cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags],
                         timeout=min(60, time_left()))
            node_state = (node or {}).get('state')
            if node_state in ('CREATING', 'STARTING', 'READY'):
                begin_startup()
            publish()
            if node_state == 'READY':
                remaining()  # Never admit ready hardware after the global lease expires.
                return
        except subprocess.TimeoutExpired:
            publish('read_timeout; current provider state unknown')
            remaining()
        except RuntimeError as error:
            text = str(error).lower()
            denied = any(token in text for token in ('permission', 'forbidden', '403', '401',
                                                     'unauthenticated', 'credentials'))
            transient = not denied and any(token in text for token in
                ('connection reset', 'connection aborted', 'temporarily unavailable',
                 'service unavailable', '503', 'timed out', 'unexpected eof', 'remote disconnected'))
            if not transient:
                raise
            publish('transient_transport_failure; current provider state unknown')
            remaining()
        time.sleep(max(0, min(10, time_left())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--accelerator-type', required=True)
    parser.add_argument('--runtime-version', required=True)
    parser.add_argument('--provisioning-model', choices=['spot', 'flex-start'], default='spot')
    parser.add_argument('--name', required=True, help='flaxchat-validation-* prefix; actual names include a fresh per-run suffix')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', help='Billing run identity; defaults to requested resource prefix')
    parser.add_argument('--stage-id', default='qualification', help='Billing stage identity')
    parser.add_argument('--campaign-ledger', type=Path, help='Shared durable campaign budget ledger; use the same path and cap for every run')
    parser.add_argument('--hourly-usd', type=float, required=True, help='Conservative whole-slice rate')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--attempt-seconds', type=int, default=1800)
    parser.add_argument('--allow-long-lease', action='store_true',
                        help='Explicitly allow one attempt up to 12 hours; default maximum is 2 hours')
    parser.add_argument('--capacity-wait-seconds', type=int,
                        help='Optional shorter wait before allocation is observed; expires through verified cleanup')
    parser.add_argument('--startup-wait-seconds', type=int,
                        help='Separate bounded startup wait after PROVISIONING; global lease still applies')
    parser.add_argument('--setup-timeout-seconds', type=int, default=300,
                        help='Bound setup and SSH connection delays separately from training')
    parser.add_argument('--tunnel-through-iap', action='store_true')
    parser.add_argument('--iap-fallback', action='store_true',
                        help='Reattach failed public SSH over IAP without launching another remote command')
    parser.add_argument('--max-attempts', type=int, default=1)
    parser.add_argument('--local-preflight', type=command,
                        help='Local input/archive validation argv; must pass before any cloud allocation')
    parser.add_argument('--setup', type=command, required=True)
    parser.add_argument('--workload', type=command, required=True)
    parser.add_argument('--resume-workload', type=command)
    parser.add_argument('--checks', type=command, help='Optional single-worker qualification argv before training')
    parser.add_argument('--acceptance-prefix', help='Fresh gs:// prefix for the nine-stage multi-host recovery campaign')
    parser.add_argument('--directory', default='/tmp/flaxchat-validation')
    parser.add_argument('--ancillary-reserve-usd', type=float, default=2.)
    args = parser.parse_args(argv)
    maximum = 43200 if args.allow_long_lease else 7200
    if not 180 <= args.attempt_seconds <= maximum or not 1 <= args.max_attempts <= 10:
        parser.error(f'Attempts must be 180–{maximum} seconds; count must be 1–10; longer leases require --allow-long-lease')
    if args.capacity_wait_seconds is not None and not 1 <= args.capacity_wait_seconds <= args.attempt_seconds:
        parser.error('Capacity wait must be positive and no longer than the attempt lease')
    if args.startup_wait_seconds is not None and not 1 <= args.startup_wait_seconds <= args.attempt_seconds:
        parser.error('Startup wait must be positive and no longer than the attempt lease')
    if not 30 <= args.setup_timeout_seconds <= 1800:
        parser.error('Setup timeout must be 30–1800 seconds')
    if args.max_attempts > 1 and not args.resume_workload:
        parser.error('Retries require explicit durable-checkpoint resume argv')
    if args.acceptance_prefix and not args.acceptance_prefix.startswith('gs://'):
        parser.error('Acceptance requires a fresh gs:// checkpoint prefix')
    if (not all(math.isfinite(v) for v in
                (args.hourly_usd, args.budget_usd, args.ancillary_reserve_usd))
            or min(args.hourly_usd, args.budget_usd) <= 0 or args.ancillary_reserve_usd < 0):
        parser.error('Rates and budget must be finite and positive; ancillary reserve nonnegative')
    required_reserve = args.hourly_usd * (args.attempt_seconds + 1800) / 3600 + args.ancillary_reserve_usd
    if required_reserve > args.budget_usd:
        parser.error(f'One attempt requires ${required_reserve:.2f}, including the 1800-second cleanup allowance and ancillary reserve; budget is ${args.budget_usd:.2f}')
    return campaign(args)


def campaign(args):
    """Persist original failure and cleanup evidence even before hardware readiness."""
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / 'campaign-receipt.json'
    if receipt_path.exists():
        raise ValueError('Campaign receipt already exists; preserve evidence and use fresh output')
    report = dict(schema_version=1, passed=False, status='running', project=args.project, zone=args.zone,
        run_id=args.run_id or args.name, stage_id=args.stage_id, phase='admission', started_unix=time.time(),
        whole_slice_hourly_usd=args.hourly_usd, attempt_seconds=args.attempt_seconds,
        capacity_wait_seconds=args.capacity_wait_seconds, startup_wait_seconds=args.startup_wait_seconds, max_attempts=args.max_attempts,
        allow_long_lease=getattr(args, 'allow_long_lease', False),
        phases=[], attempts=[], model_execution_state='not_started', posted_usage_usd=None)
    started = time.monotonic()
    phase_started = started
    def persist():
        temporary = receipt_path.with_suffix('.partial')
        with temporary.open('w') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(receipt_path)
    def phase(label):
        nonlocal phase_started
        now = time.monotonic()
        report['phases'].append(dict(phase=report['phase'], elapsed_seconds=now-phase_started))
        report['phase'] = label
        phase_started = now
        persist()
        print(f'Campaign phase: {label}', flush=True)
    persist()
    try:
        code = _run_campaign(args, report, phase, persist)
        report.update(returncode=code, passed=code == 0, status='passed' if code == 0 else 'failed')
        return code
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        discovery = getattr(error, 'cleanup_scope_discovery', None)
        if discovery is not None:
            report['cleanup_scope_discovery'] = discovery
        raise
    finally:
        report['phases'].append(dict(phase=report['phase'], elapsed_seconds=time.monotonic()-phase_started))
        report['finished_unix'] = time.time()
        report['elapsed_seconds'] = time.monotonic()-started
        remote = any(attempt['remote_stages_started'] for attempt in report['attempts'])
        report['no_remote_execution'] = not remote
        report['model_execution_state'] = 'not_inferred' if remote else 'not_started'
        created = [attempt for attempt in report['attempts'] if attempt.get('create_requested')]
        report['cleanup_required'] = bool(created)
        report['cleanup_verified'] = bool(created) and all(attempt['cleanup']['state'] == 'resource_absent' for attempt in created)
        ledger_path = args.campaign_ledger or args.output / 'ledger.json'
        report['ledger_path'] = str(ledger_path)
        if ledger_path.exists():
            try:
                ledger = json.loads(ledger_path.read_text())
                report['reserved_usd'] = sum(item['reservation_usd'] for item in ledger['attempts'].values())
                report['budget_usd'] = ledger['budget_usd']
            except (OSError, ValueError, KeyError) as error:
                report['ledger_snapshot_error'] = f'{type(error).__name__}: {error}'
        persist()


def _run_campaign(args, report, phase, persist):
    if args.local_preflight:
        phase('local_preflight')
        with (args.output / 'local-preflight.log').open('w') as log:
            subprocess.run(args.local_preflight, stdout=log, stderr=subprocess.STDOUT,
                           timeout=120, check=True)
    ledger = RunLedger(args.campaign_ledger or args.output / 'ledger.json', args.budget_usd)
    labels = {'flaxchat-run': billing_label(args.run_id or args.name),
              'flaxchat-stage': billing_label(args.stage_id)}
    flags = ['--project', args.project, '--zone', args.zone]
    run_nonce = uuid.uuid4().hex[:16]
    for index in range(args.max_attempts):
        name = attempt_resource_name(args.name, index, run_nonce)
        attempt = dict(index=index, resource_name=name, attempt_id=(run_nonce + ':' + str(index)) if args.campaign_ledger else str(index),
            cleanup={'state': 'not_required'}, remote_stages_started=[])
        report['attempts'].append(attempt)
        phase('identity_admission')
        gcp_cleanup_guard.lease(args.project, args.zone, name, args.attempt_seconds,
                                allow_long_lease=getattr(args, 'allow_long_lease', False))
        phase('cleanup_scope_admission')
        attempt['cleanup_scope'] = gcp_cleanup_guard.preflight_cleanup_scope(
            project=args.project, zone=args.zone, queue=name)
        persist()
        if cloud(['compute', 'tpus', 'queued-resources', 'describe', name, *flags]) is not None or cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags]) is not None:
            raise ValueError('Refusing an existing queue or node')
        output = args.output / f'attempt-{index}'
        output.mkdir(exist_ok=False)
        (output / 'resource-identity.json').write_text(json.dumps(dict(
            project=args.project, zone=args.zone, name=name, attempt=index,
            requested_prefix=args.name, run_nonce=run_nonce, run_id=args.run_id or args.name,
            stage_id=args.stage_id, requested_labels=labels), indent=2) + '\n')
        attempt['guard_receipt_path'] = str(output / 'guard.json')
        deadline = time.monotonic() + args.attempt_seconds
        wall_deadline = time.time() + args.attempt_seconds
        resource = f'projects/{args.project}/locations/{args.zone}/queuedResources/{name}'
        def remaining(deadline=deadline, wall_deadline=wall_deadline):
            return remaining_seconds(deadline, wall_deadline)
        def arm(resource, seconds, output=output, name=name):
            phase('guard_verification')
            receipt = output / 'guard.json'
            gcp_cleanup_guard.main(['--project', args.project, '--zone', args.zone, '--queue', name,
                                   '--seconds', str(args.attempt_seconds), '--receipt', str(receipt),
                                   *(['--allow-long-lease'] if getattr(args, 'allow_long_lease', False) else [])])
            return json.loads(receipt.read_text())
        def provision(resource, name=name, output=output, attempt=attempt):
            phase('provisioning')
            attempt['create_requested'] = True
            mode = ['--spot'] if args.provisioning_model == 'spot' else [
                '--provisioning-model=flex-start', '--max-run-duration', f'{remaining()}s']
            receipt = cloud(['compute', 'tpus', 'queued-resources', 'create', name, *flags,
                   '--node-id', name, '--accelerator-type', args.accelerator_type,
                   '--runtime-version', args.runtime_version, '--labels',
                   ','.join(key + '=' + value for key, value in labels.items()), *mode, '--async',
                   '--valid-until-duration', f'{args.attempt_seconds}s'], timeout=min(60, remaining()))
            (output / 'create-response.json').write_text(json.dumps(receipt, indent=2) + '\n')
            phase('capacity_wait')
            attempt['resource_state_observations'] = []
            def observe_state(observation):
                attempt['resource_state_observations'].append(observation)
                persist()
            wait_for_ready(name, flags, remaining, args.capacity_wait_seconds,
                           args.startup_wait_seconds, observe=observe_state, phase=phase)
            phase('label_verification')
            node = cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags], timeout=min(60, remaining()))
            observed = verify_node_labels(node, labels)
            (output / 'billing-labels.json').write_text(json.dumps(dict(
                run_id=args.run_id or args.name, stage_id=args.stage_id, requested=labels,
                observed=observed, node_name=node.get('name'), observed_unix=time.time()), indent=2) + '\n')
        def execute(argv, label, setup=False, single_worker=False, name=name, output=output, attempt=attempt):
            phase(label)
            attempt['remote_stages_started'].append(label)
            persist()
            return gcp_tpu_run.main(['--project', args.project, '--zone', args.zone, '--node', name,
                 '--cancel-peers-on-failure',
                 '--output', str(output / label), '--directory', '/tmp' if setup else args.directory,
                 '--timeout', str(min(args.setup_timeout_seconds if setup else (43200 if getattr(args, 'allow_long_lease', False) else 7200), remaining() - 15)),
                 *(['--allow-long-lease'] if getattr(args, 'allow_long_lease', False) else []), *(['--setup'] if setup else []),
                 *(['--tunnel-through-iap'] if args.tunnel_through_iap else []),
                 *(['--iap-fallback'] if args.iap_fallback else []),
                 *(['--single-worker'] if single_worker else []), '--', *argv])
        def run(resource, index=index, name=name, output=output, attempt=attempt):
            if execute(args.setup, 'setup', True):
                raise RuntimeError('Worker setup failed; no automatic retry')
            if args.checks and execute(args.checks, 'checks', single_worker=True):
                raise RuntimeError('Qualification checks failed; no automatic retry')
            if args.acceptance_prefix:
                phase('acceptance')
                attempt['remote_stages_started'].append('acceptance')
                persist()
                from scripts.validate_gcp_multihost import main as acceptance
                if acceptance(['--cleanup-receipt', str(output / 'guard.json'),
                    '--project', args.project, '--zone', args.zone, '--node', name,
                    '--checkpoint-prefix', f'{args.acceptance_prefix.rstrip("/")}/attempt-{index}',
                    '--output', str(output / 'acceptance'), '--timeout', str(min(900, remaining() - 180)),
                    *(['--tunnel-through-iap'] if args.tunnel_through_iap else []),
                    *(['--iap-fallback'] if args.iap_fallback else [])]):
                    raise RuntimeError('Multi-host acceptance failed; no automatic retry')
            result = execute(args.workload if index == 0 else args.resume_workload, 'workload')
            if result:
                node = cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags])
                if node is None or node['state'] in ('PREEMPTED', 'STOPPED', 'STOPPING'):
                    return 75
                raise RuntimeError('Workload failed on live hardware; no automatic retry')
            return 0
        def cleanup(resource, name=name, attempt=attempt):
            phase('cleanup')
            attempt['cleanup'] = {'state': 'pending', 'started_unix': time.time()}
            persist()
            def observe(state):
                attempt['cleanup'].update(state)
                persist()
            try:
                absent = cleanup_resource(name, flags, observe=observe)
                attempt['cleanup'].update(state='resource_absent' if absent else 'unverified', finished_unix=time.time())
                return absent
            except BaseException as error:
                attempt['cleanup'].update(state='unverified', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
                raise
            finally:
                persist()
        def track(callback, attempt=attempt):
            def wrapped(*values):
                try:
                    return callback(*values)
                except BaseException as error:
                    attempt.setdefault('failure', dict(phase=report['phase'], error=f'{type(error).__name__}: {error}', observed_unix=time.time()))
                    persist()
                    raise
            return wrapped
        code = run_spot_attempt(ledger, (run_nonce + ':' + str(index)) if args.campaign_ledger else str(index), resource, hourly_usd=args.hourly_usd,
                    max_seconds=args.attempt_seconds + 1800, ancillary_reserve_usd=args.ancillary_reserve_usd,
                    arm_and_verify=track(arm), provision=track(provision), run=track(run), cleanup_and_verify=cleanup)
        attempt['returncode'] = code
        persist()
        if code == 0:
            return 0
        ledger.event((run_nonce + ':' + str(index)) if args.campaign_ledger else str(index), 'preempted_retry_eligible')
    return 75


if __name__ == '__main__':
    sys.exit(main())
