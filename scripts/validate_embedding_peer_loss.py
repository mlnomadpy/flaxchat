"""Abrupt peer-loss acceptance on an existing leased TPU; never allocates hardware.

Run plan/check locally. Execute only under the campaign owner on a qualified slice.
Controller/transport loss are explicitly outside this campaign's acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time


def validate_plan(value):
    if value.get('schema_version') != 1 or value.get('fault') != 'peer-sigkill':
        raise ValueError('Expected peer-sigkill schema1 campaign')
    for key in ('project', 'zone', 'node', 'source_sha256', 'runtime_sha256', 'parent_identity'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError('Missing identity ' + key)
    for key in ('source_sha256', 'runtime_sha256', 'parent_identity'):
        if not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('Expected SHA256 identity for ' + key)
    if type(value.get('stage_timeout_seconds', 600)) is not int or not 60 <= value.get('stage_timeout_seconds', 600) <= 900:
        raise ValueError('Stage timeout must be60–900seconds')
    hosts, target = value.get('processes'), value.get('target_rank')
    if type(hosts) is not int or hosts < 2 or type(target) is not int or not 1 <= target < hosts:
        raise ValueError('Kill a declared non-coordinator peer in genuine multi-host topology')
    step, final = value.get('fault_step'), value.get('final_step')
    if type(step) is not int or type(final) is not int or not 0 < step < final:
        raise ValueError('Fault must follow a positive committed step before final step')
    if type(value.get('timeout_seconds')) is not int or not 180 <= value['timeout_seconds'] <= 3600:
        raise ValueError('Shared deadline must be180–3600seconds')
    for key in ('baseline_output', 'recovery_output'):
        if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+/[^*?#]+', value.get(key, '')) or '..' in value[key].split('/'):
            raise ValueError('Fresh explicit GCS output required')
    left, right = (value[key].rstrip('/') for key in ('baseline_output', 'recovery_output'))
    if left == right or left.startswith(right + '/') or right.startswith(left + '/'):
        raise ValueError('Baseline and recovery namespaces must not overlap')
    argv = value.get('training_argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError('Training argv required')
    for item in argv:
        option = item.split('=', 1)[0]
        if item == '--' or any(option.startswith('--') and forbidden.startswith(option) or option == forbidden for forbidden in ('--output', '--resume', '--stop-after')):
            raise ValueError('Recipe must not override campaign output/resume/stop ownership')
        if '=' in item and option in {'--steps', '--save-every'}:
            raise ValueError('Use one separate canonical value for steps/save cadence')
    if '--distributed' not in argv or '--resume' in argv or '--stop-after' in argv or any(x.startswith('--output') for x in argv):
        raise ValueError('Supply distributed recipe without output/resume/stop arguments')
    for option in ('--steps', '--save-every'):
        if option in argv and (argv.count(option) != 1 or argv.index(option) + 1 >= len(argv)):
            raise ValueError('Require one value for ' + option)
    if '--steps' not in argv or int(argv[argv.index('--steps') + 1]) != final:
        raise ValueError('Recipe final horizon differs')
    if '--save-every' not in argv or int(argv[argv.index('--save-every') + 1]) < 1 or step % int(argv[argv.index('--save-every') + 1]):
        raise ValueError('Fault step must be on declared save cadence')
    if not value.get('remote_python') or not value.get('directory') or not value.get('guard_receipt'):
        raise ValueError('Frozen remote interpreter/directory and existing guard receipt required')
    if not Path(value['remote_python']).is_absolute() or not Path(value['directory']).is_absolute():
        raise ValueError('Select absolute frozen runtime and checkout paths')
    return value


def worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault-step', type=int, required=True)
    parser.add_argument('--target-rank', type=int, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--token', required=True)
    args, recipe = parser.parse_known_args(argv)
    recipe = recipe[1:] if recipe[:1] == ['--'] else recipe
    from scripts import train_yat_embedding_finetune as trainer
    from flaxchat.embedding_contract import source_identity, canonical_hash
    from flaxchat.runtime import runtime_identity
    import jax
    from jax.experimental import multihost_utils
    if not jax.distributed.is_initialized():
        jax.distributed.initialize(initialization_timeout=120)
    if jax.default_backend() != 'tpu' or jax.process_count() < 2:
        raise ValueError('Fault worker requires genuine physical multi-host TPU')
    source = source_identity(Path(__file__).resolve().parents[1])['sha256']
    runtime = canonical_hash(runtime_identity())
    if source != args.source_sha256 or runtime != args.runtime_sha256:
        raise ValueError('Fault worker source/runtime differs from accepted identity')
    stage_parser = argparse.ArgumentParser(allow_abbrev=False)
    trainer.add_stage_arguments(stage_parser)
    stage_args = stage_parser.parse_args(recipe)
    original = trainer.save_checkpoint
    def interrupted_save(manager, step, *positional, **kwargs):
        saved = original(manager, step, *positional, **kwargs)
        if step == args.fault_step:
            # Do not inject in the independent best-checkpoint namespace.
            directory = str(manager.directory).rstrip('/')
            if directory != str(stage_args.output).rstrip('/'):
                return saved
            manager.wait_until_finished()
            from flaxchat.checkpoint_metadata import read_committed_metadata
            committed = read_committed_metadata(directory, step, include_receipt=True)['committed_receipt']
            multihost_utils.sync_global_devices('embedding-peer-committed-' + args.token)
            event = dict(event='embedding_peer_fault_ready', token=args.token, rank=jax.process_index(),
                launcher_rank=int(os.environ['JAX_PROCESS_INDEX']), pid=os.getpid(), backend='tpu',
                processes=jax.process_count(), devices=jax.device_count(), committed_step=step,
                committed_manifest_sha256=committed['manifest_sha256'], source_sha256=source, runtime_sha256=runtime)
            print(json.dumps(event, sort_keys=True), flush=True)
            multihost_utils.sync_global_devices('embedding-peer-ready-recorded-' + args.token)
            if event['launcher_rank'] == args.target_rank:
                print(json.dumps({**event, 'event': 'embedding_peer_sigkill'}), flush=True)
                os.kill(os.getpid(), signal.SIGKILL)
            multihost_utils.sync_global_devices('embedding-peer-lost-' + args.token)
        return saved
    trainer.save_checkpoint = interrupted_save
    try:
        trainer.run(stage_args)
    finally:
        trainer.save_checkpoint = original
    raise ValueError('Fault injection never executed; campaign is incomplete')


def verify_drain(summary, processes):
    rows = summary.get('workers', [])
    if len(rows) != processes or any(type(row.get('rank')) is not int for row in rows) or {row['rank'] for row in rows} != set(range(processes)):
        raise ValueError('Missing terminal worker ownership; resume prohibited')
    for row in rows:
        if row.get('transport_returncode') != 0 or type(row.get('returncode')) is not int or not 0 <= row['returncode'] <= 255 or row['returncode'] == 125:
            raise ValueError('Transport failure is not proof of remote termination; resume prohibited')
    if not summary.get('finished_epoch'):
        raise ValueError('Original distributed execution has not drained')


def remote_status_command(execution_id, rank):
    if not re.fullmatch('[0-9a-f]{32}', execution_id) or type(rank) is not int or rank < 0:
        raise ValueError('Invalid execution status identity')
    path = f'/tmp/flaxchat-command-{execution_id}-{rank}/status'
    return 'test -f ' + path + ' && cat ' + path


def verify_durable_status(payload, expected):
    if not re.fullmatch(r'[0-9]{1,3}\s*', payload) or not 0 <= int(payload) <= 255 or int(payload) != expected:
        raise ValueError('Durable remote terminal status missing/different; resume prohibited')
    return int(payload)


def verify_fault(events, value, token):
    ready = [row for row in events if row.get('event') == 'embedding_peer_fault_ready']
    killed = [row for row in events if row.get('event') == 'embedding_peer_sigkill']
    if len(ready) != value['processes'] or {row['launcher_rank'] for row in ready} != set(range(value['processes'])) or len(killed) != 1:
        raise ValueError('Incomplete rank ownership/fault event evidence')
    for row in ready + killed:
        if (row.get('token') != token or row.get('backend') != 'tpu' or row.get('committed_step') != value['fault_step']
                or row.get('processes') != value['processes'] or row.get('source_sha256') != value['source_sha256']
                or row.get('runtime_sha256') != value['runtime_sha256']):
            raise ValueError('Fault event identity mismatch')
    if killed[0]['launcher_rank'] != value['target_rank'] or {r['rank'] for r in ready} != set(range(value['processes'])):
        raise ValueError('Wrong peer or duplicate runtime process ownership')
    target = next(row for row in ready if row['launcher_rank'] == value['target_rank'])
    if killed[0] != {**target, 'event': 'embedding_peer_sigkill'} or any(type(row.get('pid')) is not int or row['pid'] <= 0 for row in ready):
        raise ValueError('SIGKILL event does not identify the verified peer process')
    if any(not re.fullmatch('[0-9a-f]{64}', str(row.get('committed_manifest_sha256', ''))) for row in ready):
        raise ValueError('Invalid committed checkpoint identity')
    devices = {row.get('devices') for row in ready}
    if len(devices) != 1 or any(type(n) is not int or n < value['processes'] for n in devices):
        raise ValueError('Incomplete or inconsistent physical device ownership')
    if len({r['committed_manifest_sha256'] for r in ready}) != 1:
        raise ValueError('Hosts disagree about committed recovery state')
    return killed[0]


def compare_final(left, right):
    result = {key: bool(left.get(key)) and left.get(key) == right.get(key)
              for key in ('model_state', 'optimizer_state', 'training_state')}
    if not all(result.values()):
        raise ValueError('Final exact state differs: ' + str(result))
    return result


def execute(plan, output):
    from scripts import gcp_cleanup_guard
    from flaxchat.checkpoint_metadata import read_committed_metadata
    value = validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, passed=False, status='incomplete', scope='physical-embedding-peer-loss-v1',
        controller_loss_qualified=False, transport_loss_qualified=False, plan=value, stages=[])
    receipt_path = output / 'receipt.json'
    def persist():
        receipt_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    persist()
    start, wall_start = time.monotonic(), time.time()
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
    def remaining():
        return min(value['timeout_seconds'] - (time.monotonic()-start), value['timeout_seconds'] - (time.time()-wall_start))
    try:
        guard = json.loads(Path(value['guard_receipt']).read_text())
        lease_left = gcp_cleanup_guard.verify(guard, project=value['project'], zone=value['zone'], queue=value['node'])
        if lease_left < value['timeout_seconds'] + 60:
            raise ValueError('Existing independent lease does not cover campaign plus cleanup')
        for prefix in (value['baseline_output'], value['recovery_output']):
            probe = subprocess.run(['gcloud', 'storage', 'ls', prefix.rstrip('/')+'/**', '--json'], capture_output=True, text=True, timeout=30)
            if probe.returncode == 0 and json.loads(probe.stdout or '[]'):
                raise ValueError('Output exists; preserve old evidence')
            if probe.returncode != 0 and 'matched no objects' not in probe.stderr.lower():
                raise ValueError('Cannot verify fresh output namespace')
        common = value['training_argv']
        for name, prefix, fault, resume in [('baseline', value['baseline_output'], False, False),
                                           ('interruption', value['recovery_output'], True, False),
                                           ('resume', value['recovery_output'], False, True)]:
            allowance = int(min(value.get('stage_timeout_seconds', 600), remaining()-90))
            if allowance < 30:
                raise TimeoutError('Shared campaign deadline exhausted before ' + name)
            command = [value['remote_python'], '-m', 'scripts.train_yat_embedding_finetune']
            if fault:
                command = [value['remote_python'], '-m', 'scripts.validate_embedding_peer_loss', 'worker',
                    '--fault-step', str(value['fault_step']), '--target-rank', str(value['target_rank']),
                    '--source-sha256', value['source_sha256'], '--runtime-sha256', value['runtime_sha256'], '--token', token, '--']
            command += common + ['--output', prefix] + (['--resume'] if resume else [])
            stage = output / name
            argv = [sys.executable, '-m', 'scripts.gcp_tpu_run', '--project', value['project'], '--zone', value['zone'],
                    '--node', value['node'], '--directory', value['directory'], '--output', str(stage), '--timeout', str(allowance)]
            if value.get('tunnel_through_iap'):
                argv += ['--tunnel-through-iap']
            # Do not cancel peer transports: every original remote must drain before resume.
            completed = subprocess.run(argv + ['--'] + command, timeout=min(remaining(), allowance+90))
            summary = json.loads((stage / 'summary.json').read_text())
            verify_drain(summary, value['processes'])
            from scripts.gcp_tpu_run import ssh_command
            for row in summary['workers']:
                command = ssh_command(value['node'], value['project'], value['zone'], row['rank'],
                    remote_status_command(summary['execution_id'], row['rank']),
                    tunnel_through_iap=value.get('tunnel_through_iap', False))
                status = subprocess.check_output(command, text=True, timeout=min(30, remaining()))
                row['durable_returncode'] = verify_durable_status(status, row['returncode'])
            if fault:
                events = []
                for log in sorted(stage.glob('worker-*.log')):
                    for line in log.read_text().splitlines():
                        if line.startswith('{'):
                            row = json.loads(line)
                            if row.get('event', '').startswith('embedding_peer_'):
                                events.append(row)
                killed = verify_fault(events, value, token)
                if completed.returncode == 0 or not any(row['rank'] == value['target_rank'] and row['returncode'] == 137 for row in summary['workers']):
                    raise ValueError('Declared peer did not terminate by SIGKILL')
                committed = read_committed_metadata(prefix, value['fault_step'], include_receipt=True)['committed_receipt']
                if committed['manifest_sha256'] != killed['committed_manifest_sha256']:
                    raise ValueError('Recovery checkpoint changed after fault')
                report['fault_event'] = killed
            elif completed.returncode or summary.get('passed') is not True:
                raise ValueError(name + ' failed')
            report['stages'].append(dict(name=name, remote_drained=True, execution_id=summary['execution_id'], durable_workers=summary['workers']))
            persist()
        manifests = []
        report['final_committed_receipts'] = []
        for prefix in (value['baseline_output'], value['recovery_output']):
            admission = read_committed_metadata(prefix, value['final_step'], include_receipt=True)
            if admission.get('source_python_sha256') != value['source_sha256']:
                raise ValueError('Final checkpoint source differs from campaign')
            parent = admission.get('resolved_config', {}).get('parent')
            parent_hash = hashlib.sha256(json.dumps(parent, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if parent_hash != value['parent_identity']:
                raise ValueError('Final checkpoint parent differs from campaign')
            runtime = admission.get('resolved_config', {}).get('runtime')
            runtime_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if runtime_hash != value['runtime_sha256']:
                raise ValueError('Final checkpoint runtime differs from campaign')
            raw = subprocess.check_output(['gcloud', 'storage', 'cat', prefix.rstrip('/')+'/'+str(value['final_step'])+'/manifest/metadata'], timeout=30)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError('Final manifest exceeds bounded evidence size')
            manifest = json.loads(raw)
            actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if actual != admission['committed_receipt']['manifest_sha256']:
                raise ValueError('Final state manifest differs from authenticated commit')
            manifests.append(manifest)
            report['final_committed_receipts'].append(dict(prefix=prefix, step=value['final_step'],
                manifest_sha256=actual, state_inventory_sha256={key: hashlib.sha256(json.dumps(manifest[key], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                for key in ('model_state', 'optimizer_state', 'training_state')}))
        report['state_equality'] = compare_final(*manifests)
        report['guard_seconds_remaining'] = gcp_cleanup_guard.verify(guard, project=value['project'], zone=value['zone'], queue=value['node'])
        if remaining() <= 0:
            raise TimeoutError('Shared campaign deadline exhausted')
        report.update(passed=True, status='passed', final_step=value['final_step'])
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
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
        parser.error('Plan required; unknown arguments forbidden')
    value = validate_plan(json.loads(args.plan.read_text()))
    if args.mode == 'check':
        print(json.dumps({'valid': True, 'physical_qualified': False, 'plan': value}, indent=2))
        return 0
    if args.output is None:
        parser.error('Fresh local output required for execute')
    previous_handler = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise TimeoutError('Shared peer-loss campaign deadline exceeded')
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
