"""Run one bounded command on every worker of an existing Cloud TPU slice.

Provisioning and deletion belong to the operator/watchdog. Each worker gets an
explicit rank, a unique log, and a remote timeout that survives SSH failure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
    environment = {"FLAXCHAT_COMPILATION_CACHE_DIR": directory.rstrip("/") + "/.jax-cache"}
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
    # gcloud may replay SSH commands after status 255. Always deliver the
    # workload status as data over a successful shell exit, and claim an
    # atomic one-use identity before execution in case the transport drops.
    return (f"if ! mkdir {shlex.quote(lock)}; then printf '\\n{marker}125\\n'; exit 0; fi; "
            f"(cd {shlex.quote(directory)} && {shlex.join(command)}); command_status=$?; "
            f"printf '\\n{marker}%s\\n' \"$command_status\"; exit 0")


def remote_returncode(log, execution_id, rank, transport_code):
    if transport_code:
        return transport_code
    codes = re.findall(rf'^FLAXCHAT_REMOTE_EXIT_{execution_id}_{rank}=(\d+)$', log, re.MULTILINE)
    return int(codes[0]) if len(codes) == 1 and 0 <= int(codes[0]) <= 255 else 125


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--directory', default='/tmp/flaxchat-validation')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--single-worker', action='store_true')
    parser.add_argument('--setup', action='store_true', help='Run on every worker without distributed JAX environment')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 2400:
        parser.error('Timeout must be in [1,2400] seconds')
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
               'provisioning_model': 'spot' if resource.get('schedulingConfig', {}).get('spot') else 'on-demand',
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
        with (args.output / f'worker-{rank}.log').open('w') as log:
            code = run_transport(['gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', args.node,
                    '--project', args.project, '--zone', args.zone, '--worker', str(rank),
                    '--quiet', '--command', remote], log, args.timeout + 90, cancelled)
        transport_code = code
        code = remote_returncode((args.output / f'worker-{rank}.log').read_text(), execution_id, rank, transport_code)
        return {'rank': rank, 'returncode': code, 'transport_returncode': transport_code, 'seconds': time.monotonic() - tick,
                'remote_command': remote}
    with ThreadPoolExecutor(max_workers=len(ranks)) as executor:
        try:
            for row in executor.map(run, ranks):
                summary['workers'].append(row)
                report.write_text(json.dumps(summary, indent=2))
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
