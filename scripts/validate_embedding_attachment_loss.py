"""Deliberate local SSH-attachment loss; no allocation or remote process killing.

Reattach byte-identical existing runner commands under the same execution_id.
This does not qualify network outages or whole-controller SIGKILL/lease cleanup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

from scripts import gcp_tpu_run


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_plan(value):
    if value.get('schema_version') != 1 or value.get('fault') != 'local-attachment-sigkill':
        raise ValueError('Expected local-attachment-sigkill schema1')
    for key in ('project', 'zone', 'node', 'guard_receipt', 'directory', 'remote_python'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError('Missing ' + key)
    for key in ('source_sha256', 'runtime_sha256', 'parent_identity'):
        if not re.fullmatch('[0-9a-f]{64}', value.get(key, '')):
            raise ValueError('Missing SHA256 ' + key)
    if type(value.get('processes')) is not int or value['processes'] < 2:
        raise ValueError('Requires declared genuine multi-host execution')
    if not Path(value['directory']).is_absolute() or not Path(value['remote_python']).is_absolute():
        raise ValueError('Frozen absolute source/interpreter paths required')
    if type(value.get('timeout_seconds')) is not int or not 180 <= value['timeout_seconds'] <= 3600:
        raise ValueError('Shared deadline must be180–3600seconds')
    stage = value.get('worker_timeout_seconds', 600)
    if type(stage) is not int or not 60 <= stage <= value['timeout_seconds'] - 90:
        raise ValueError('Finite worker deadline plus attachment overhead must fit campaign')
    argv = value.get('training_argv', [])
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError('Explicit training argv required')
    for item in argv:
        option = item.split('=', 1)[0]
        if item == '--' or any(option.startswith('--') and forbidden.startswith(option) or option == forbidden for forbidden in ('--output', '--resume', '--stop-after')):
            raise ValueError('Recipe must not override campaign output/resume/stop ownership')
        if '=' in item and option in {'--steps', '--save-every'}:
            raise ValueError('Use one separate canonical value for steps/save cadence')
    if '--distributed' not in argv or any(item.startswith(('--output', '--resume', '--stop-after')) for item in argv):
        raise ValueError('Provide distributed new-stage recipe without output/resume/stop')
    if type(value.get('final_step')) is not int or value['final_step'] < 1 or '--steps' not in argv:
        raise ValueError('Explicit positive final horizon required')
    i = argv.index('--steps')
    if argv.count('--steps') != 1 or i + 1 >= len(argv) or int(argv[i+1]) != value['final_step']:
        raise ValueError('Recipe differs from final horizon')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+/[^*?#]+', value.get('checkpoint_output', '')) or '..' in value['checkpoint_output'].split('/'):
        raise ValueError('Fresh explicit GCS checkpoint namespace required')
    return value


def stamp_path(execution_id, rank):
    if not re.fullmatch('[0-9a-f]{32}', execution_id) or type(rank) is not int or rank < 0:
        raise ValueError('Invalid durable execution/rank')
    return '/tmp/flaxchat-attachment-start-' + execution_id + '-' + str(rank) + '.json'


def verify_stamps(stamps, plan, execution_id):
    count = plan['processes']
    if len(stamps) != count or {s.get('launcher_rank') for s in stamps} != set(range(count)) or {s.get('runtime_rank') for s in stamps} != set(range(count)):
        raise ValueError('Incomplete/duplicate launcher or runtime ownership')
    for stamp in stamps:
        if (stamp.get('execution_id') != execution_id or stamp.get('application_starts') != 1
                or stamp.get('backend') != 'tpu' or stamp.get('processes') != count
                or stamp.get('source_sha256') != plan['source_sha256'] or stamp.get('runtime_sha256') != plan['runtime_sha256']
                or type(stamp.get('pid')) is not int or stamp['pid'] <= 0):
            raise ValueError('Application-start stamp differs from accepted execution')
    if len({s.get('devices') for s in stamps}) != 1 or any(type(s.get('devices')) is not int or s['devices'] < count for s in stamps):
        raise ValueError('Physical device ownership incomplete')


def worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-id', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    args, recipe = parser.parse_known_args(argv)
    recipe = recipe[1:] if recipe[:1] == ['--'] else recipe
    from scripts import train_yat_embedding_finetune as trainer
    from flaxchat.embedding_contract import source_identity
    from flaxchat.runtime import runtime_identity
    import jax
    if not jax.distributed.is_initialized():
        jax.distributed.initialize(initialization_timeout=120)
    if jax.default_backend() != 'tpu' or jax.process_count() < 2:
        raise ValueError('Physical multi-host worker required')
    source = source_identity(Path(__file__).resolve().parents[1])['sha256']
    runtime = canonical_hash(runtime_identity())
    if source != args.source_sha256 or runtime != args.runtime_sha256:
        raise ValueError('Source/runtime differs from accepted identity')
    rank = int(os.environ['JAX_PROCESS_INDEX'])
    stamp = dict(event='embedding_application_start', execution_id=args.execution_id,
        application_starts=1, launcher_rank=rank, runtime_rank=jax.process_index(), pid=os.getpid(),
        backend='tpu', processes=jax.process_count(), devices=jax.device_count(), source_sha256=source, runtime_sha256=runtime)
    # O_EXCL makes duplicate application execution a hard failure, not another stamp.
    path = Path(stamp_path(args.execution_id, rank))
    temporary = path.with_suffix('.partial')
    # Publish a complete JSON inode atomically, refusing an existing final stamp.
    with temporary.open('x') as handle:
        try:
            json.dump(stamp, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(stamp, sort_keys=True), flush=True)
    stage_parser = argparse.ArgumentParser(allow_abbrev=False)
    trainer.add_stage_arguments(stage_parser)
    return trainer.run(stage_parser.parse_args(recipe))


def kill_local_attachment(process):
    if process.poll() is not None:
        raise ValueError('Attachment already completed; no fault was injected')
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    if process.returncode != -signal.SIGKILL:
        raise ValueError('Local attachment did not terminate by SIGKILL')


def verify_terminal(log, stamp, execution_id, rank, transport_code):
    if transport_code != 0:
        raise ValueError('Reattachment transport failed; remote outcome unknown')
    if gcp_tpu_run.remote_returncode(log, execution_id, rank, transport_code) != 0:
        raise ValueError('Remote workload failed or terminal marker absent')
    events = []
    for line in log.splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if row.get('event') == 'embedding_application_start':
                events.append(row)
    if events != [stamp]:
        raise ValueError('Exactly one matching physical application start required')


def execute(plan, output):
    from scripts import gcp_cleanup_guard
    from flaxchat.checkpoint_metadata import read_committed_metadata
    value = validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, passed=False, status='incomplete', scope='local-ssh-attachment-loss-v1',
        controller_sigkill_qualified=False, network_outage_qualified=False, lease_cleanup_qualified=False,
        plan=value, plan_sha256=canonical_hash(value), workers=[])
    path = output / 'receipt.json'
    def persist():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    persist()
    began, wall_began = time.monotonic(), time.time()
    def remaining():
        return min(value['timeout_seconds']-(time.monotonic()-began), value['timeout_seconds']-(time.time()-wall_began))
    processes, files = [], []
    try:
        guard = json.loads(Path(value['guard_receipt']).read_text())
        available = gcp_cleanup_guard.verify(guard, project=value['project'], zone=value['zone'], queue=value['node'])
        if available < value['timeout_seconds'] + 60:
            raise ValueError('Existing independent guard insufficient for campaign')
        resource = json.loads(subprocess.check_output(['gcloud', 'compute', 'tpus', 'tpu-vm', 'describe', value['node'],
            '--project', value['project'], '--zone', value['zone'], '--format=json'], timeout=min(60, remaining())))
        if resource.get('state') != 'READY' or len(resource.get('networkEndpoints', [])) != value['processes']:
            raise ValueError('Existing node differs from declared READY multi-host slice')
        probe = subprocess.run(['gcloud', 'storage', 'ls', value['checkpoint_output'].rstrip('/')+'/**', '--json'], capture_output=True, text=True, timeout=min(30, remaining()))
        if probe.returncode == 0 and json.loads(probe.stdout or '[]'):
            raise ValueError('Checkpoint output exists')
        if probe.returncode != 0 and 'matched no objects' not in probe.stderr.lower():
            raise ValueError('Cannot prove fresh checkpoint namespace')
        execution_id = uuid.uuid4().hex
        report['execution_id'] = execution_id
        command = [value['remote_python'], '-m', 'scripts.validate_embedding_attachment_loss', 'worker',
            '--execution-id', execution_id, '--source-sha256', value['source_sha256'], '--runtime-sha256', value['runtime_sha256'], '--',
            *value['training_argv'], '--output', value['checkpoint_output']]
        coordinator = resource['networkEndpoints'][0]['ipAddress'] + ':12387'
        remotes = [gcp_tpu_run.worker_command(command, rank=rank, count=value['processes'], coordinator=coordinator,
            directory=value['directory'], timeout=value.get('worker_timeout_seconds', 600), distributed=True, execution_id=execution_id)
            for rank in range(value['processes'])]
        report['remote_commands_sha256'] = [hashlib.sha256(remote.encode()).hexdigest() for remote in remotes]
        persist()
        for rank, remote in enumerate(remotes):
            log = (output / f'initial-{rank}.log').open('w')
            files.append(log)
            argv = gcp_tpu_run.ssh_command(value['node'], value['project'], value['zone'], rank, remote,
                tunnel_through_iap=value.get('tunnel_through_iap', False))
            processes.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        stamps = []
        for rank in range(value['processes']):
            while remaining() > 30:
                if any(process.poll() is not None for process in processes):
                    raise ValueError('Original attachment finished before fault injection')
                argv = gcp_tpu_run.ssh_command(value['node'], value['project'], value['zone'], rank,
                    'cat ' + stamp_path(execution_id, rank), tunnel_through_iap=value.get('tunnel_through_iap', False))
                probe = subprocess.run(argv, capture_output=True, text=True, timeout=min(30, remaining()))
                if probe.returncode == 0:
                    stamps.append(json.loads(probe.stdout))
                    break
                time.sleep(min(1, max(0, remaining())))
            else:
                raise TimeoutError('Physical application ownership stamps did not arrive')
        verify_stamps(stamps, value, execution_id)
        report['physical_application_stamps'] = stamps
        report['local_attachment_processes'] = []
        for rank, process in enumerate(processes):
            kill_local_attachment(process)
            report['local_attachment_processes'].append(dict(rank=rank, local_pid=process.pid, signal='SIGKILL', returncode=process.returncode))
        report['local_attachment_sigkill'] = True
        persist()
        for rank, remote in enumerate(remotes):
            with (output / f'reattached-{rank}.log').open('w') as log:
                code = gcp_tpu_run.run_transport(gcp_tpu_run.ssh_command(value['node'], value['project'], value['zone'], rank,
                    remote, tunnel_through_iap=value.get('tunnel_through_iap', False)), log,
                    max(1, remaining()-15), threading.Event())
            content = (output / f'reattached-{rank}.log').read_text()
            verify_terminal(content, stamps[rank], execution_id, rank, code)
            report['workers'].append(dict(rank=rank, same_execution_id=execution_id, application_starts=1,
                transport_returncode=code, workload_returncode=0))
            persist()
        committed = read_committed_metadata(value['checkpoint_output'], value['final_step'], include_receipt=True)
        recipe = committed.get('resolved_config', {})
        if committed.get('source_python_sha256') != value['source_sha256'] or canonical_hash(recipe.get('runtime')) != value['runtime_sha256'] or canonical_hash(recipe.get('parent')) != value['parent_identity']:
            raise ValueError('Final committed source/runtime/parent identity differs')
        raw = subprocess.check_output(['gcloud', 'storage', 'cat', value['checkpoint_output'].rstrip('/')+'/'+str(value['final_step'])+'/manifest/metadata'], timeout=min(30, remaining()))
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('Final state evidence exceeds bounded size')
        manifest = json.loads(raw)
        if canonical_hash(manifest) != committed['committed_receipt']['manifest_sha256']:
            raise ValueError('Final state manifest changed after committed admission')
        if any(not isinstance(manifest.get(key), dict) or not manifest[key] for key in ('model_state', 'optimizer_state', 'training_state')):
            raise ValueError('Final complete model/optimizer/training state is missing')
        report['final_state_inventory_sha256'] = {key: canonical_hash(manifest[key]) for key in ('model_state', 'optimizer_state', 'training_state')}
        report['final_committed_receipt'] = committed['committed_receipt']
        report['guard_seconds_remaining'] = gcp_cleanup_guard.verify(guard, project=value['project'], zone=value['zone'], queue=value['node'])
        if remaining() <= 0:
            raise TimeoutError('Shared deadline exhausted')
        report.update(passed=True, status='passed')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        # Only reap local SSH processes. Remote commands remain independently bounded.
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        for handle in files:
            handle.close()
        report['elapsed_seconds'] = time.monotonic()-began
        persist()
    return 0 if report['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'execute', 'worker'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    args, rest = parser.parse_known_args(argv)
    if args.mode == 'worker':
        return worker(rest)
    if rest or args.plan is None:
        parser.error('Explicit plan required; unknown arguments forbidden')
    value = validate_plan(json.loads(args.plan.read_text()))
    if args.mode == 'check':
        print(json.dumps({'valid': True, 'physically_qualified': False}, indent=2))
        return 0
    if args.output is None:
        parser.error('Fresh local output required')
    previous_handler = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise TimeoutError('Shared attachment campaign deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, value['timeout_seconds'])
    started = time.monotonic()
    try:
        return execute(value, args.output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0] - (time.monotonic()-started)), previous_timer[1])


if __name__ == '__main__':
    raise SystemExit(main())
