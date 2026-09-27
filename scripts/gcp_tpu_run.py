"""Run one bounded command on every worker of an existing Cloud TPU slice.

Provisioning and deletion belong to the operator/watchdog. Each worker gets an
explicit rank, a unique log, and a remote timeout that survives SSH failure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid


def worker_command(argv, *, rank, count, coordinator, directory, timeout, distributed, execution_id=None):
    if not argv or timeout < 1 or not 0 <= rank < count:
        raise ValueError('Invalid worker command')
    environment = {"FLAXCHAT_COMPILATION_CACHE_DIR": directory.rstrip("/") + "/.jax-cache",
                   # Nested artifact entrypoints and their Python children must
                   # import the frozen checkout, not an inherited installation.
                   "PYTHONPATH": directory,
                   "FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS": str(timeout)}
    if distributed and count > 1:
        environment.update({'JAX_COORDINATOR_ADDRESS': coordinator,
                       'JAX_PROCESS_COUNT': str(count), 'JAX_PROCESS_INDEX': str(rank)})
    command = ['timeout', '--signal=KILL', str(timeout), 'env']
    command.extend(f'{key}={value}' for key, value in environment.items())
    command.extend(argv)
    execution_id = execution_id or uuid.uuid4().hex
    if not re.fullmatch('[0-9a-f]{32}', execution_id):
        raise ValueError('Invalid execution identity')
    marker = f'FLAXCHAT_REMOTE_EXIT_{execution_id}_{rank}='
    lock = f'/tmp/flaxchat-command-{execution_id}-{rank}'
    sdk_python = shlex.quote(directory.rstrip('/') + '/.venv/bin/python')
    # Use the supported, setup-created runtime for workload gcloud children.
    sdk_environment = f'if [ -x {sdk_python} ]; then export CLOUDSDK_PYTHON={sdk_python}; fi; '
    # The worker owns execution; SSH is only an attachment. Replayed transports
    # wait for the same durable result rather than aborting or launching twice.
    runner = (f"(cd {shlex.quote(directory)} && {shlex.join(command)}); command_status=$?; "
              f"printf '%s\\n' \"$command_status\" > {lock}/status.tmp; "
              f"mv {lock}/status.tmp {lock}/status")
    detached = shlex.join(['nohup', 'bash', '-c', runner])
    return (sdk_environment
            + f"if mkdir {lock} 2>/dev/null; then "
            f"date +%s > {lock}/started; "
            f"{detached} > {lock}/output.log 2>&1 < /dev/null & "
            f"elif [ ! -d {lock} ]; then printf '\\n{marker}125\\n'; exit 0; fi; "
            f"attach_deadline=$(( $(date +%s) + {timeout + 10} )); "
            f"while [ ! -f {lock}/status ]; do "
            f"if [ $(date +%s) -ge \"$attach_deadline\" ]; then "
            f"printf '\\n{marker}124\\n'; exit 0; fi; sleep 1; done; "
            f"cat {lock}/output.log; command_status=$(cat {lock}/status); "
            f"printf '\\n{marker}%s\\n' \"$command_status\"; exit 0")


def remote_returncode(log, execution_id, rank, transport_code):
    if transport_code:
        return transport_code
    codes = re.findall(rf'^FLAXCHAT_REMOTE_EXIT_{execution_id}_{rank}=(\d+)$', log, re.MULTILINE)
    return int(codes[0]) if codes and len(set(codes)) == 1 and 0 <= int(codes[0]) <= 255 else 125


def run_transport(argv, log, timeout, cancelled):
    """Bound SSH by wall time too, including laptop sleep; reap its process group."""
    monotonic_deadline, wall_deadline = time.monotonic() + timeout, time.time() + timeout
    with subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            while True:
                seconds = min(monotonic_deadline - time.monotonic(), wall_deadline - time.time())
                if cancelled.is_set() or seconds <= 0:
                    return 124
                try:
                    return process.wait(timeout=min(5, seconds))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                try:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()


def ssh_command(node, project, zone, rank, remote, *, tunnel_through_iap=False):
    return ['gcloud', *(['alpha'] if tunnel_through_iap else []),
            'compute', 'tpus', 'tpu-vm', 'ssh', node,
            '--project', project, '--zone', zone, '--worker', str(rank),
            *(['--tunnel-through-iap'] if tunnel_through_iap else []),
            '--ssh-flag=-o ConnectTimeout=20', '--quiet', '--command', remote]


def attach_worker(node, project, zone, rank, remote, log, timeout, cancelled, *,
                  tunnel_through_iap=False, iap_fallback=False):
    """Reattach the same durable command over IAP within one total deadline."""
    mono_end, wall_end = time.monotonic() + timeout, time.time() + timeout
    attempts = []
    routes = [tunnel_through_iap]
    if iap_fallback and not tunnel_through_iap:
        routes.append(True)
    code = 124
    for index, iap in enumerate(routes):
        remaining = min(mono_end - time.monotonic(), wall_end - time.time())
        if cancelled.is_set() or remaining <= 0:
            break
        # Reserve time for the alternate route. The detached remote execution
        # survives a lost attachment; never generate a new execution ID here.
        allowance = min(90, remaining) if index == 0 and len(routes) > 1 else remaining
        code = run_transport(ssh_command(node, project, zone, rank, remote,
                             tunnel_through_iap=iap), log, allowance, cancelled)
        attempts.append({'route': 'iap' if iap else 'public',
                         'transport_returncode': code, 'timeout_seconds': allowance})
        if code == 0:
            break  # A remote command failure is not a transport retry signal.
    return code, attempts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--tunnel-through-iap', action='store_true',
                        help='Use the supported alpha IAP SSH transport; does not change IAM/firewalls')
    parser.add_argument('--iap-fallback', action='store_true',
                        help='After a failed public attachment, reattach the same execution over IAP within the existing timeout; does not change IAM/firewalls')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--directory', default='/tmp/flaxchat-validation')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--single-worker', action='store_true')
    parser.add_argument('--cancel-peers-on-failure', action='store_true',
                        help='For a supervising owner that tears down the slice on failure; stop peer SSH waits promptly')
    parser.add_argument('--setup', action='store_true', help='Run on every worker without distributed JAX environment')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 7200:
        parser.error('Timeout must be in [1,7200] seconds')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Command required after --')
    args.output.mkdir(parents=True, exist_ok=False)
    resource = json.loads(subprocess.check_output(
        ['gcloud', 'compute', 'tpus', 'tpu-vm', 'describe', args.node,
         '--project', args.project, '--zone', args.zone, '--format=json'], text=True, timeout=60))
    if resource['state'] != 'READY':
        raise RuntimeError(f"TPU is not ready: {resource['state']}")
    endpoints = resource['networkEndpoints']
    count = len(endpoints)
    coordinator = endpoints[0]['ipAddress'] + ':12387'
    ranks = [0] if args.single_worker else list(range(count))
    summary = {'passed': False, 'project': args.project, 'zone': args.zone, 'node': args.node,
               'runtime_version': resource.get('runtimeVersion'),
               'provisioning_model': 'spot' if resource.get('schedulingConfig', {}).get('spot') else 'unknown',
               'accelerator_type': resource.get('acceleratorType'), 'worker_count': count,
               'selected_workers': ranks, 'command': command, 'timeout_seconds': args.timeout,
               'started_epoch': time.time(), 'workers': []}
    execution_id = uuid.uuid4().hex
    summary['execution_id'] = execution_id
    report = args.output / 'summary.json'
    report.write_text(json.dumps(summary, indent=2))
    cancelled = threading.Event()
    def run(rank):
        remote = worker_command(command, rank=rank, count=count, coordinator=coordinator,
                                directory=args.directory, timeout=args.timeout,
                                distributed=not args.single_worker and not args.setup, execution_id=execution_id)
        tick = time.monotonic()
        wall_started = time.time()
        log_path = args.output / f'worker-{rank}.log'
        with log_path.open('w') as log:
            code, transport_attempts = attach_worker(args.node, args.project, args.zone, rank,
                    remote, log, args.timeout + 90, cancelled,
                    tunnel_through_iap=args.tunnel_through_iap, iap_fallback=args.iap_fallback)
        transport_code = code
        code = remote_returncode(log_path.read_text(), execution_id, rank, transport_code)
        elapsed = time.monotonic() - tick
        wall_elapsed = time.time() - wall_started
        return {'rank': rank, 'returncode': code, 'transport_returncode': transport_code, 'seconds': elapsed,
                'transport_attempts': transport_attempts,
                'wall_seconds': wall_elapsed,
                # This detects a timing discrepancy, not its cause: host sleep
                # and wall-clock adjustments can both produce such a gap.
                'wall_minus_monotonic_seconds': wall_elapsed - elapsed,
                'log_path': str(log_path),
                'remote_command': remote}
    with ThreadPoolExecutor(max_workers=len(ranks)) as executor:
        try:
            futures = [executor.submit(run, rank) for rank in ranks]
            for future in as_completed(futures):
                row = future.result()
                summary['workers'].append(row)
                summary['workers'].sort(key=lambda item: item['rank'])
                report.write_text(json.dumps(summary, indent=2))
                if row['returncode'] and args.cancel_peers_on_failure:
                    # A failed worker invalidates the distributed workload.
                    # Reap peer transports promptly so the owning supervisor
                    # can tear down the slice, rather than wait out its lease.
                    cancelled.set()
        except BaseException:
            cancelled.set()
            raise
    summary['passed'] = all(row['returncode'] == 0 for row in summary['workers'])
    summary['finished_epoch'] = time.time()
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
