"""Provider-neutral Linux data jobs with immutable admission and detached observation.

This controls already provisioned hosts, not cloud resources. Execution receipts do
not qualify model outputs. Receipt publication to GCS is a separate bounded task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

SCHEMA = 1
TOKEN = 'FLAXCHAT_DATA_JOB_OWNER'


def digest(path, deadline=None):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Source authentication deadline exhausted')
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def write_once(path, value):
    """Commit without overwriting a prior receipt, including a concurrent writer."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write(canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate(spec):
    if type(spec.get('schema_version')) is not int or spec['schema_version'] != SCHEMA:
        raise ValueError('Unsupported job schema')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', spec.get('job_id', '')):
        raise ValueError('Unique job_id required')
    root = Path(spec['root'])
    if not root.is_absolute() or root.parent.resolve() != root.parent or root.name in {'', '.', '..'}:
        raise ValueError('Absolute owned root without symlink ancestors required')
    if not root.name.startswith('flaxchat-data-'):
        raise ValueError('Owned root must have flaxchat-data- prefix')
    for key, ceiling in [('timeout_seconds', 86400), ('kill_after_seconds', 60)]:
        value = spec.get(key)
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError('Finite integer ' + key + ' required')
    command = spec.get('command')
    if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v or '\0' in v for v in command):
        raise ValueError('Nonempty literal argv required')
    inputs = spec.get('inputs')
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 64:
        raise ValueError('Immutable source/input inventory required')
    names = set()
    for item in inputs:
        name = item['name']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'} or name in names:
            raise ValueError('Unique flat input names required')
        names.add(name)
        if not re.fullmatch('[0-9a-f]{64}', item.get('sha256', '')) or type(item.get('bytes')) is not int or not 0 <= item['bytes'] <= 4 * 1024**3:
            raise ValueError('Input SHA256 and byte count required')
        source = Path(item['path'])
        if not source.is_absolute() or source.resolve() != source or not source.is_file():
            raise ValueError('Regular absolute source without symlinks required')
    expected = digest(Path(__file__))
    if spec.get('controller_sha256') != expected:
        raise ValueError('Controller source identity differs')
    if not any('{input:' + name + '}' in arg for name in names for arg in command):
        raise ValueError('Workload must consume a frozen source/input')
    return root


def process_identity(pid, proc=Path('/proc')):
    """Linux boot identity plus start ticks prevents accidental PID reuse."""
    try:
        stat = (proc / str(pid) / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': (proc / 'sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, IndexError):
        return None


def owned_processes(owner, scratch, proc=Path('/proc')):
    result = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            env = (entry / 'environ').read_bytes().split(b'\0')
            if (TOKEN + '=' + owner).encode() in env and ('TMPDIR=' + str(scratch)).encode() in env:
                identity = process_identity(int(entry.name), proc)
                if identity:
                    result.append(identity)
        except OSError:
            continue
    return result


def launch(spec):
    root = validate(spec)
    if not Path('/proc/self/stat').exists():
        raise ValueError('Physical Linux host with /proc required; no inferred liveness')
    timeout = shutil.which('timeout')
    if timeout is None:
        raise ValueError('GNU timeout is required for independent outer deadline')
    subprocess.run([timeout, '--version'], check=True, capture_output=True, timeout=5)
    # mkdir is the once-only admission lock. An existing/missing handle is NEVER
    # a reason to run the same job again, even after a failed launch.
    root.mkdir(mode=0o700)
    identity = hashlib.sha256(canonical(spec)).hexdigest()
    write_once(root / 'spec.json', spec)
    owner = uuid.uuid4().hex
    sources = root / 'inputs'
    sources.mkdir(mode=0o700)
    try:
        freeze_deadline = time.monotonic() + 120
        for item in spec['inputs']:
            target = sources / item['name']
            with Path(item['path']).open('rb') as source, target.open('xb') as destination:
                while block := source.read(1024 * 1024):
                    if time.monotonic() >= freeze_deadline:
                        raise TimeoutError('Input freeze deadline exhausted')
                    destination.write(block)
            if target.stat().st_size != item['bytes'] or digest(target, freeze_deadline) != item['sha256']:
                raise ValueError('Frozen source/input bytes differ: ' + item['name'])
        controller = root / 'controller.py'
        shutil.copyfile(__file__, controller)
        if digest(controller) != spec['controller_sha256']:
            raise ValueError('Controller changed while freezing')
        (root / 'work').mkdir(mode=0o700)
        admission = {'schema_version': SCHEMA, 'job_id': spec['job_id'], 'owner': owner,
                     'spec_sha256': identity, 'controller_sha256': spec['controller_sha256'],
                     'admitted_unix': time.time(), 'model_execution_qualified': False}
        write_once(root / 'admission.json', admission)
        with (root / 'supervisor.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(controller), 'supervise', '--root', str(root)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        # The supervisor writes authoritative ownership after starting. Returning
        # from this attachment is not evidence of worker success or even startup.
        return {**admission, 'attachment_returned': True, 'supervisor_pid_hint': process.pid,
                'status': 'admitted; observe ownership and terminal receipts'}
    except BaseException as error:
        write_once(root / 'admission-failure.json', {'spec_sha256': identity, 'error_type': type(error).__name__})
        raise


def supervise(root):
    root = Path(root)
    spec = json.loads((root / 'spec.json').read_text())
    admission = json.loads((root / 'admission.json').read_text())
    identity = hashlib.sha256(canonical(spec)).hexdigest()
    if identity != admission['spec_sha256'] or spec['controller_sha256'] != digest(__file__):
        raise ValueError('Immutable supervisor admission differs')
    # No duplicate supervise invocation can start a second workload.
    write_once(root / 'supervisor-claim.json', {'pid': os.getpid(), 'spec_sha256': identity})
    scratch = root / 'work'
    receipt = {'schema_version': SCHEMA, 'job_id': spec['job_id'], 'owner': admission['owner'],
               'spec_sha256': identity, 'started_unix': time.time(), 'model_execution_qualified': False}
    process = None
    interrupted = None
    started = time.monotonic()
    deadline = started + spec['timeout_seconds']

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = signum
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        targets = {}
        for item in spec['inputs']:
            target = root / 'inputs' / item['name']
            if target.stat().st_size != item['bytes'] or digest(target, min(deadline, started + 120)) != item['sha256']:
                raise ValueError('Frozen input changed')
            targets[item['name']] = target
        command = []
        for argument in spec['command']:
            argument = argument.replace('{scratch}', str(scratch))
            for name, target in targets.items():
                argument = argument.replace('{input:' + name + '}', str(target))
            if re.search(r'\{(?:input:|scratch)', argument):
                raise ValueError('Unknown command placeholder')
            command.append(argument)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Global workload deadline exhausted during authentication')
        env = {**os.environ, TOKEN: admission['owner'], 'FLAXCHAT_DATA_JOB_ID': spec['job_id'],
               'FLAXCHAT_DATA_JOB_SPEC_SHA256': identity, 'TMPDIR': str(scratch)}
        with (root / 'worker.log').open('xb') as log:
            process = subprocess.Popen(['timeout', '--signal=TERM',
                                        '--kill-after=' + str(spec['kill_after_seconds']) + 's',
                                        str(remaining) + 's', *command],
                                       env=env, cwd=scratch, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            ownership = {**receipt, 'supervisor': process_identity(os.getpid()),
                         'timeout_process': process_identity(process.pid),
                         'deadline_unix': receipt['started_unix'] + spec['timeout_seconds'] + spec['kill_after_seconds']}
            if not ownership['supervisor'] or not ownership['timeout_process']:
                raise RuntimeError('Cannot authenticate Linux process ownership')
            write_once(root / 'ownership.json', ownership)
            receipt['worker_returncode'] = process.wait(timeout=remaining + spec['kill_after_seconds'] + 5)
            receipt['status'] = 'execution_succeeded' if receipt['worker_returncode'] == 0 else 'execution_failed'
    except BaseException as error:
        receipt.update(status='execution_failed', error_type=type(error).__name__, termination_signal=interrupted)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        receipt.update(cleanup_verified=False, scratch_removed=False)
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            killed = []
            for entry in owned_processes(admission['owner'], scratch):
                if process_identity(entry['pid']) == entry:
                    try:
                        os.kill(entry['pid'], signal.SIGKILL)
                        killed.append(entry)
                    except ProcessLookupError:
                        pass
            # Verify termination before deleting scratch. Bound the cleanup wait.
            until = time.monotonic() + 5
            residual = owned_processes(admission['owner'], scratch)
            while residual and time.monotonic() < until:
                time.sleep(.05)
                residual = owned_processes(admission['owner'], scratch)
            receipt['residual_processes'] = residual
            receipt['owned_processes_killed'] = killed
            receipt['scratch_removed'] = False
            if not residual and scratch.is_dir() and not scratch.is_symlink():
                shutil.rmtree(scratch)
                receipt['scratch_removed'] = not scratch.exists()
            receipt['cleanup_verified'] = not residual and receipt['scratch_removed']
        except BaseException as error:
            receipt['cleanup_error_type'] = type(error).__name__
        receipt['finished_unix'] = time.time()
        write_once(root / 'terminal.json', receipt)
    return 0 if receipt['status'] == 'execution_succeeded' and receipt['cleanup_verified'] else 1


def observe(root, expected_spec_sha256):
    """Read-only: unknown, missing or expired handles never imply a retry."""
    if (not isinstance(expected_spec_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_spec_sha256) is None):
        raise ValueError('Independently retained expected specification SHA256 required')
    root = Path(root)
    spec = json.loads((root / 'spec.json').read_text())
    admission = json.loads((root / 'admission.json').read_text())
    identity = hashlib.sha256(canonical(spec)).hexdigest()
    if admission['spec_sha256'] != identity or identity != expected_spec_sha256:
        raise ValueError('Admission identity differs')
    terminal = root / 'terminal.json'
    if terminal.exists():
        result = json.loads(terminal.read_text())
        if any(result.get(key) != admission[key] for key in ('spec_sha256', 'owner', 'job_id')):
            raise ValueError('Foreign terminal receipt')
        return {'state': 'terminal', 'worker': result, 'retry_permitted': False}
    ownership = root / 'ownership.json'
    if not ownership.exists():
        return {'state': 'startup_unobserved', 'retry_permitted': False}
    value = json.loads(ownership.read_text())
    if any(value.get(key) != admission[key] for key in ('spec_sha256', 'owner', 'job_id')):
        raise ValueError('Foreign ownership receipt')
    live = {key: process_identity(value[key]['pid']) == value[key] for key in ('supervisor', 'timeout_process')}
    return {'state': 'live' if all(live.values()) else 'terminal_receipt_missing',
            'processes_verified_live': live, 'retry_permitted': False, 'ownership': value}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'observe', 'supervise'])
    parser.add_argument('--spec')
    parser.add_argument('--root')
    parser.add_argument('--expected-spec-sha256')
    args = parser.parse_args(argv)
    if args.action == 'launch':
        print(json.dumps(launch(json.loads(Path(args.spec).read_text())), sort_keys=True))
    elif args.action == 'observe':
        if not args.expected_spec_sha256:
            parser.error('observe requires --expected-spec-sha256 retained independently of the remote root')
        print(json.dumps(observe(args.root, args.expected_spec_sha256), sort_keys=True))
    else:
        return supervise(args.root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
