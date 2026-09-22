"""Bounded single-host TPU acceptance with complete logs, JUnit, and provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def source_digest(root):
    digest = hashlib.sha256()
    files = sorted(p for folder in ('flaxchat', 'scripts', 'tests', 'benchmarks', 'tasks', 'infra', 'accelerators')
                   for p in (root / folder).rglob('*.py'))
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def junit_inventory(path):
    root = ET.parse(path).getroot()
    result = []
    for case in root.iter('testcase'):
        status = 'passed'
        message = ''
        for name in ('failure', 'error', 'skipped'):
            item = case.find(name)
            if item is not None:
                status = name
                message = item.get('message', '')
                break
        result.append({'test': case.get('classname', '') + '::' + case.get('name', ''),
                       'status': status, 'seconds': float(case.get('time', '0')), 'message': message})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--token-manifest', required=True)
    parser.add_argument('--minimum-devices', type=int, default=8)
    parser.add_argument('--timeout-seconds', type=int, default=4500)
    parser.add_argument('--checks', nargs='+', help='Explicit subset; recorded as partial acceptance')
    args = parser.parse_args(argv)
    if not 60 <= args.timeout_seconds <= 5400:
        parser.error('Acceptance must be bounded to at most 90 minutes')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    base = [py, '-m', 'scripts.train_gpt2', '--token-manifest', str(Path(args.token_manifest).resolve()),
            '--batch-per-device', '1', '--seq-len', '128', '--warmup-steps', '2',
            '--eval-every', '5', '--save-every', '5']
    checks = [
        ('devices', [py, '-c',
         "import json,jax; from flaxchat.common import compute_init; compute_init(); "
         "d=jax.devices(); print(json.dumps({'backend':jax.default_backend(),'devices':[str(x) for x in d],'processes':jax.process_count()})); "
         f"assert jax.default_backend()=='tpu' and len(d)>={args.minimum_devices} and jax.process_count()==1"]),
        ('fineweb-stop', base + ['--depth', '2', '--tokens', '51200', '--stop-after', '5',
         '--ckpt-dir', str(output / 'fineweb/checkpoints'), '--artifact-dir', str(output / 'fineweb')]),
        ('fineweb-resume', base + ['--depth', '2', '--tokens', '51200', '--resume',
         '--ckpt-dir', str(output / 'fineweb/checkpoints'), '--artifact-dir', str(output / 'fineweb')]),
        ('gpt2-size-smoke', base + ['--depth', '12', '--tokens', '10240',
         '--ckpt-dir', str(output / 'gpt2/checkpoints'), '--artifact-dir', str(output / 'gpt2')]),
        ('checkpoint-generation', [py, '-m', 'scripts.validate_gpt_checkpoint',
         '--checkpoint-dir', str(output / 'gpt2/checkpoints'), '--token-manifest', args.token_manifest,
         '--output', str(output / 'checkpoint-generation.json')]),
        ('pretrain-smoke', [py, '-m', 'scripts.pretrain', '--cpu-smoke']),
        ('pytest-all', [py, '-m', 'pytest', 'tests', '-v', '--tb=short', '--durations=30',
         f'--junitxml={output / "pytest.xml"}']),
        ('tinystories-e2e', [py, '-m', 'scripts.run_tinystories', '--smoke',
         '--output-dir', str(output / 'tinystories'), '--sequence-length', '32',
         '--embedding-dim', '24', '--batch-size', '8', '--max-new-tokens', '1']),
    ]
    all_names = [name for name, _ in checks]
    if args.checks:
        if set(args.checks) - set(all_names):
            parser.error('Unknown acceptance check')
        checks = [check for check in checks if check[0] in args.checks or check[0] == 'devices']
    started = time.monotonic()
    summary = {'scope': 'full' if args.checks is None else 'selected-checks',
               'selected_checks': [name for name, _ in checks],
               'source_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
               'source_python_sha256': source_digest(root), 'checks': [], 'passed': False,
               'environment': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint')},
               'limitations': ['Single physical host; does not close multi-host issue #12',
                               'GCP does not reproduce Kaggle runtime; issue #31 needs Kaggle acceptance',
                               'Short smoke establishes execution, not model quality'],
               'started_epoch': time.time()}
    (output / 'pip-freeze.txt').write_text(subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    def persist():
        temporary = output / 'summary.tmp'
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(output / 'summary.json')
    persist()
    dependencies = {'fineweb-resume': 'fineweb-stop', 'checkpoint-generation': 'gpt2-size-smoke'}
    for name, command in checks:
        prerequisite = dependencies.get(name)
        if prerequisite and not any(x['name'] == prerequisite and x['status'] == 'passed' for x in summary['checks']):
            summary['checks'].append({'name': name, 'status': 'blocked', 'reason': f'{prerequisite} failed'})
            persist()
            continue
        remaining = args.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            summary['checks'].append({'name': name, 'status': 'not_run', 'reason': 'campaign deadline'})
            persist()
            continue
        tick = time.monotonic()
        print(f'Running {name}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=min(remaining, 2400 if name == 'pytest-all' else 900))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 124
        summary['checks'].append({'name': name, 'argv': command, 'returncode': returncode,
                                  'status': 'passed' if returncode == 0 else 'failed',
                                  'seconds': time.monotonic() - tick})
        if name == 'pytest-all' and (output / 'pytest.xml').exists():
            summary['tests'] = junit_inventory(output / 'pytest.xml')
        persist()
        if name == 'devices' and returncode:
            break
    summary['passed'] = len(summary['checks']) == len(checks) and all(x['status'] == 'passed' for x in summary['checks'])
    summary['elapsed_seconds'] = time.monotonic() - started
    persist()
    print(json.dumps({k: v for k, v in summary.items() if k != 'tests'}, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
