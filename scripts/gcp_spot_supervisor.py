"""Run bounded Spot attempts with a durable expiry guard and retry ledger.

Setup and workload are JSON argv arrays executed on every TPU worker. Retries
require a confirmed non-READY node and use the explicitly supplied resume argv.
No retry is allowed while prior resource absence remains unverified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from flaxchat.operations import RunLedger, run_spot_attempt
from scripts import gcp_cleanup_guard, gcp_tpu_run


def cloud(argv, *, timeout: float = 60):
    # Retry only reads, within the original call deadline. A timed-out create
    # may have succeeded and must never be replayed as a fresh allocation.
    read_only = len(argv) > 3 and argv[3] in ('describe', 'list')
    deadline = time.monotonic() + timeout
    for attempt in range(3 if read_only else 1):
        try:
            result = subprocess.run(['gcloud', *argv, '--format=json', '--quiet'],
                                    capture_output=True, text=True,
                                    timeout=max(.1, min(20 if read_only else timeout, deadline - time.monotonic())))
            break
        except subprocess.TimeoutExpired:
            if not read_only or attempt == 2 or time.monotonic() >= deadline:
                raise
    if result.returncode:
        if 'NOT_FOUND' in result.stderr or 'was not found' in result.stderr:
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--accelerator-type', required=True)
    parser.add_argument('--runtime-version', required=True)
    parser.add_argument('--name', required=True, help='Unique flaxchat-validation-* prefix covered by cleanup IAM')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hourly-usd', type=float, required=True, help='Conservative whole-slice rate')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--attempt-seconds', type=int, default=1800)
    parser.add_argument('--max-attempts', type=int, default=1)
    parser.add_argument('--setup', type=command, required=True)
    parser.add_argument('--workload', type=command, required=True)
    parser.add_argument('--resume-workload', type=command)
    parser.add_argument('--checks', type=command, help='Optional single-worker qualification argv before training')
    parser.add_argument('--acceptance-prefix', help='Fresh gs:// prefix for the nine-stage multi-host recovery campaign')
    parser.add_argument('--directory', default='/tmp/flaxchat-validation')
    parser.add_argument('--ancillary-reserve-usd', type=float, default=2.)
    args = parser.parse_args(argv)
    if not 180 <= args.attempt_seconds <= 2400 or not 1 <= args.max_attempts <= 10:
        parser.error('Attempts must be 180–2400 seconds; count must be 1–10')
    if args.max_attempts > 1 and not args.resume_workload:
        parser.error('Retries require explicit durable-checkpoint resume argv')
    if args.acceptance_prefix and not args.acceptance_prefix.startswith('gs://'):
        parser.error('Acceptance requires a fresh gs:// checkpoint prefix')
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(args.output / 'ledger.json', args.budget_usd)
    flags = ['--project', args.project, '--zone', args.zone]
    for index in range(args.max_attempts):
        name = f'{args.name}-{index}'
        gcp_cleanup_guard.lease(args.project, args.zone, name, args.attempt_seconds)
        if cloud(['compute', 'tpus', 'queued-resources', 'describe', name, *flags]) is not None or cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags]) is not None:
            raise ValueError('Refusing an existing queue or node')
        output = args.output / f'attempt-{index}'
        output.mkdir(exist_ok=False)
        deadline = time.monotonic() + args.attempt_seconds
        resource = f'projects/{args.project}/locations/{args.zone}/queuedResources/{name}'
        def remaining(deadline=deadline):
            seconds = int(deadline - time.monotonic())
            if seconds < 30:
                raise TimeoutError('Attempt lease is expiring')
            return seconds
        def arm(resource, seconds, output=output, name=name):
            receipt = output / 'guard.json'
            gcp_cleanup_guard.main(['--project', args.project, '--zone', args.zone, '--queue', name,
                                   '--seconds', str(args.attempt_seconds), '--receipt', str(receipt)])
            return json.loads(receipt.read_text())
        def provision(resource, name=name):
            cloud(['compute', 'tpus', 'queued-resources', 'create', name, *flags,
                   '--node-id', name, '--accelerator-type', args.accelerator_type,
                   '--runtime-version', args.runtime_version, '--spot',
                   '--valid-until-duration', f'{args.attempt_seconds}s'], timeout=remaining())
            while True:
                try:
                    node = cloud(['compute', 'tpus', 'tpu-vm', 'describe', name, *flags],
                                 timeout=min(60, remaining()))
                except subprocess.TimeoutExpired:
                    # Provisioning reads can be temporarily unavailable. Keep
                    # the existing guarded request, never submit another one.
                    remaining()
                    time.sleep(10)
                    continue
                if node and node['state'] == 'READY':
                    break
                remaining()
                time.sleep(10)
        def execute(argv, label, setup=False, single_worker=False, name=name, output=output):
            return gcp_tpu_run.main(['--project', args.project, '--zone', args.zone, '--node', name,
                 '--output', str(output / label), '--directory', '/tmp' if setup else args.directory,
                 '--timeout', str(min(2400, remaining() - 15)), *(['--setup'] if setup else []),
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
                    '--output', str(output / 'acceptance'), '--timeout', str(min(900, remaining() - 180))]):
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
