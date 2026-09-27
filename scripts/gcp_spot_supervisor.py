"""Run bounded Spot attempts with a durable expiry guard and retry ledger.

Setup and workload are JSON argv arrays executed on every TPU worker. Retries
require a confirmed non-READY node and use the explicitly supplied resume argv.
No retry is allowed while prior resource absence remains unverified.
"""
from __future__ import annotations
import argparse
import json
import math
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


def cleanup_resource(name, flags, *, timeout=1800):
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


def wait_for_ready(name, flags, remaining, capacity_wait_seconds=None):
    """Bound capacity waiting separately from compute; never recreate a request."""
    started, wall_started = time.monotonic(), time.time()
    def time_left():
        seconds = remaining()
        if capacity_wait_seconds is not None:
            elapsed = max(time.monotonic() - started, time.time() - wall_started)
            seconds = min(seconds, capacity_wait_seconds - elapsed)
            if seconds <= 0:
                raise TimeoutError('Spot capacity wait budget exhausted')
        return seconds
    while True:
        try:
            node = cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags],
                         timeout=min(60, time_left()))
            if node and node['state'] == 'READY':
                return
            queue = cloud(['compute', 'tpus', 'queued-resources', 'describe', name, *flags],
                          timeout=min(60, time_left()))
            state = (queue or {}).get('state', {})
            state = state.get('state') if isinstance(state, dict) else state
            if queue is None or state in ('FAILED', 'SUSPENDING', 'SUSPENDED', 'DELETING'):
                detail = json.dumps(queue) if queue else (
                    'inspect the saved create-response.json operation and TPU audit logs; '
                    'absence alone does not establish capacity exhaustion'
                )
                raise RuntimeError(f'Spot request unavailable: {state or "absent"}; {detail}')
        except subprocess.TimeoutExpired:
            # A read timeout is not permission to submit a second allocation.
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
    parser.add_argument('--hourly-usd', type=float, required=True, help='Conservative whole-slice rate')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--attempt-seconds', type=int, default=1800)
    parser.add_argument('--capacity-wait-seconds', type=int,
                        help='Optional shorter capacity wait; expires through verified cleanup')
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
    if not 180 <= args.attempt_seconds <= 7200 or not 1 <= args.max_attempts <= 10:
        parser.error('Attempts must be 180–7200 seconds; count must be 1–10')
    if args.capacity_wait_seconds is not None and not 1 <= args.capacity_wait_seconds <= args.attempt_seconds:
        parser.error('Capacity wait must be positive and no longer than the attempt lease')
    if not 30 <= args.setup_timeout_seconds <= 900:
        parser.error('Setup timeout must be 30–900 seconds')
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
    args.output.mkdir(parents=True, exist_ok=True)
    if args.local_preflight:
        with (args.output / 'local-preflight.log').open('w') as log:
            subprocess.run(args.local_preflight, stdout=log, stderr=subprocess.STDOUT,
                           timeout=120, check=True)
    ledger = RunLedger(args.output / 'ledger.json', args.budget_usd)
    flags = ['--project', args.project, '--zone', args.zone]
    run_nonce = uuid.uuid4().hex[:16]
    for index in range(args.max_attempts):
        name = attempt_resource_name(args.name, index, run_nonce)
        gcp_cleanup_guard.lease(args.project, args.zone, name, args.attempt_seconds)
        if cloud(['compute', 'tpus', 'queued-resources', 'describe', name, *flags]) is not None or cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags]) is not None:
            raise ValueError('Refusing an existing queue or node')
        output = args.output / f'attempt-{index}'
        output.mkdir(exist_ok=False)
        (output / 'resource-identity.json').write_text(json.dumps(dict(
            project=args.project, zone=args.zone, name=name, attempt=index,
            requested_prefix=args.name, run_nonce=run_nonce), indent=2) + '\n')
        deadline = time.monotonic() + args.attempt_seconds
        wall_deadline = time.time() + args.attempt_seconds
        resource = f'projects/{args.project}/locations/{args.zone}/queuedResources/{name}'
        def remaining(deadline=deadline, wall_deadline=wall_deadline):
            return remaining_seconds(deadline, wall_deadline)
        def arm(resource, seconds, output=output, name=name):
            receipt = output / 'guard.json'
            gcp_cleanup_guard.main(['--project', args.project, '--zone', args.zone, '--queue', name,
                                   '--seconds', str(args.attempt_seconds), '--receipt', str(receipt)])
            return json.loads(receipt.read_text())
        def provision(resource, name=name, output=output):
            mode = ['--spot'] if args.provisioning_model == 'spot' else [
                '--provisioning-model=flex-start', '--max-run-duration', f'{remaining()}s']
            receipt = cloud(['compute', 'tpus', 'queued-resources', 'create', name, *flags,
                   '--node-id', name, '--accelerator-type', args.accelerator_type,
                   '--runtime-version', args.runtime_version, *mode, '--async',
                   '--valid-until-duration', f'{args.attempt_seconds}s'], timeout=min(60, remaining()))
            (output / 'create-response.json').write_text(json.dumps(receipt, indent=2) + '\n')
            wait_for_ready(name, flags, remaining, args.capacity_wait_seconds)
        def execute(argv, label, setup=False, single_worker=False, name=name, output=output):
            return gcp_tpu_run.main(['--project', args.project, '--zone', args.zone, '--node', name,
                 '--cancel-peers-on-failure',
                 '--output', str(output / label), '--directory', '/tmp' if setup else args.directory,
                 '--timeout', str(min(args.setup_timeout_seconds if setup else 7200, remaining() - 15)), *(['--setup'] if setup else []),
                 *(['--tunnel-through-iap'] if args.tunnel_through_iap else []),
                 *(['--iap-fallback'] if args.iap_fallback else []),
                 *(['--single-worker'] if single_worker else []), '--', *argv])
        def run(resource, index=index, name=name, output=output):
            if execute(args.setup, 'setup', True):
                raise RuntimeError('Worker setup failed; no automatic retry')
            if args.checks and execute(args.checks, 'checks', single_worker=True):
                raise RuntimeError('Qualification checks failed; no automatic retry')
            if args.acceptance_prefix:
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
        def cleanup(resource, name=name):
            return cleanup_resource(name, flags)
        code = run_spot_attempt(ledger, str(index), resource, hourly_usd=args.hourly_usd,
                    max_seconds=args.attempt_seconds + 1800, ancillary_reserve_usd=args.ancillary_reserve_usd,
                    arm_and_verify=arm, provision=provision, run=run, cleanup_and_verify=cleanup)
        if code == 0:
            return 0
        ledger.event(str(index), 'preempted_retry_eligible')
    return 75


if __name__ == '__main__':
    sys.exit(main())
