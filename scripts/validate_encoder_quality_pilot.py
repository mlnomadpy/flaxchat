"""Run a fixed real-corpus MLM pilot only after physical calibration fits its lease.

This is a development pilot, not production model-quality acceptance. No TPU is
provisioned by this worker entrypoint; an external supervisor owns its lease.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts.plan_encoder_quality import plan
from scripts.workload_deadline import WorkloadDeadline
from scripts.validate_encoder_scale import validate_updates


def file_hash(path):
    # Importing even flaxchat.encoder_data executes flaxchat.__init__, which
    # initializes JAX and can take the TPU away from the child training process.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_command(recipe, prefix, *, calibration):
    argv = list(recipe['training_argv'])
    if argv[1:3] != ['-m', 'scripts.train_encoder']:
        raise ValueError('Expected fixed encoder training entrypoint')
    argv[0] = sys.executable
    argv[argv.index('--output') + 1] = prefix + ('/calibration' if calibration else '/pilot')
    if '--resume' in argv or '--stop-after' in argv:
        raise ValueError('Require a fresh fixed recipe')
    if calibration:
        argv += ['--stop-after', str(recipe['calibration_steps'])]
    return argv


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def run(args):
    if (not args.prefix.startswith('gs://') or args.devices != 4
            or int(os.environ.get('JAX_PROCESS_COUNT', '1')) != 1):
        raise ValueError('This pilot requires a single-host four-device TPU and GCS prefix')
    recipe = json.loads(args.recipe.read_text())
    for name in ('manifest.json', 'tokens.npy'):
        if str(Path(args.validation) / name) not in recipe['input_sha256']:
            raise ValueError('Evaluation data must be pinned by the frozen recipe')
    for name, digest in recipe['input_sha256'].items():
        if file_hash(name) != digest:
            raise ValueError(f'Frozen pilot input changed: {name}')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    deadline = WorkloadDeadline(min(6500, int(os.environ['FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS']) - 90))
    remaining_start = deadline.remaining()
    env = os.environ | {'JAX_PLATFORMS': 'tpu', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}
    summary: dict[str, Any] = dict(passed=False, quality_qualified=False, stages=[], training_steps=recipe['steps'])

    def persist():
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        subprocess.run(['gcloud', 'storage', 'cp', *map(str, output.iterdir()),
                        args.prefix + '/evidence/'], check=True, timeout=deadline.remaining(60))

    def stage(name, argv):
        path = output / (name + '.log')
        remaining = deadline.remaining() - 65
        if remaining <= 0:
            raise TimeoutError('Insufficient time for stage and evidence upload')
        with path.open('w') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, timeout=remaining)
        summary['stages'].append(dict(name=name, returncode=result.returncode))
        persist()
        if result.returncode:
            raise RuntimeError(f'Quality pilot stage failed: {name}')
        return path

    try:
        calibration = training_command(recipe, args.prefix, calibration=True)
        preflight = records(stage('preflight', calibration + ['--preflight-only']))
        checked = [r for r in preflight if r.get('event') == 'input_preflight' and r.get('passed') is True]
        if len(checked) != 1:
            raise ValueError('Missing unique independent input preflight')
        expected_identity = checked[0]['input_identity']
        log = stage('calibration', calibration)
        elapsed = remaining_start - deadline.remaining()
        estimate = plan(records(log), expected_identity=expected_identity,
                        expected_nonpadding=recipe['calibration_nonpadding_tokens'],
                        devices=args.devices, processes=1, steps=recipe['steps'],
                        batch_size=recipe['batch_size'], sequence_length=checked[0]['sequence_length'],
                        remaining_seconds=deadline.remaining() - 65,
                        hourly_usd=args.hourly_usd,
                        budget_usd=args.compute_budget_usd - elapsed * args.hourly_usd / 3600)
        (output / 'cost-plan.json').write_text(json.dumps(estimate, indent=2) + '\n')
        persist()
        pilot = stage('pilot', training_command(recipe, args.prefix, calibration=False))
        validate_updates(pilot, first=1, last=recipe['steps'], batch=recipe['batch_size'],
                         length=checked[0]['sequence_length'])
        updates = [r for r in records(pilot) if r.get('event') == 'train_step']
        if sum(r['nonpadding_tokens'] for r in updates) != recipe['nonpadding_tokens']:
            raise ValueError('Full pilot token schedule mismatch')
        # Evaluate the complete frozen development validation pool; no test-set
        # selection or downsampled score is presented as production acceptance.
        train = recipe['training_argv'][recipe['training_argv'].index('--data') + 1]
        validation = Path(args.validation)
        manifest = json.loads((validation / 'manifest.json').read_text())
        stage('evaluation', [sys.executable, '-m', 'scripts.evaluate_encoder',
                            '--checkpoint', args.prefix + '/pilot', '--data', str(validation),
                            '--train-data', train, '--batch-size', '8', '--max-rows', str(manifest['rows']),
                            '--output', str(output / 'evaluation.json')])
        summary['passed'] = True
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/quality-pilot-evidence'))
    parser.add_argument('--validation', required=True)
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--compute-budget-usd', type=float, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
