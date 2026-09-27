"""Bounded encoder qualification workload; launch under the Spot supervisor.

This orchestrator intentionally does not import JAX: each stage owns its runtime.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def stage_environment(environment, *, cpu=False, matmul_precision='highest'):
    env = dict(environment, JAX_DEFAULT_MATMUL_PRECISION=matmul_precision)
    if cpu:
        env['JAX_PLATFORMS'] = 'cpu'
        for key in ('JAX_COORDINATOR_ADDRESS', 'JAX_PROCESS_COUNT', 'JAX_PROCESS_INDEX',
                    'TPU_WORKER_HOSTNAMES', 'TPU_WORKER_ID', 'SLURM_NTASKS',
                    'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'):
            env.pop(key, None)
    return env


def validate_hardware(report, mode, expected_processes=None, expected_devices=None):
    """Reject a valid TPU job running on the wrong qualification topology."""
    if report['backend'] != 'tpu':
        raise ValueError('TPU backend required')
    processes, devices = report['process_count'], report['device_count']
    if processes < 1 or devices < processes or devices % processes:
        raise ValueError('Invalid global process/device topology')
    if mode == 'multi' and processes < 2:
        raise ValueError('Multi-host qualification requires at least two processes')
    if mode in ('single', 'recovery', 'context') and processes != 1:
        raise ValueError('Single-host qualification requires exactly one process')
    if expected_processes is not None and processes != expected_processes:
        raise ValueError('Unexpected process count')
    if expected_devices is not None and devices != expected_devices:
        raise ValueError('Unexpected global device count')


def projection_arguments(args):
    return ['--mlm-projection', args.mlm_projection, '--mlm-loss-backend', args.mlm_loss_backend,
            '--mlm-vocab-tile', str(args.mlm_vocab_tile)]


def qualification_label(mode, residual_dtype, projection, backend, vocab_tile):
    # Preserve existing default paths for recorded baseline recovery commands.
    label = f'{mode}-residual-{residual_dtype}'
    if (projection, backend, vocab_tile) != ('dense', 'xla', 1024):
        label += f'-{projection}-{backend}-tile-{vocab_tile}'
    return label


def validate_context_log(path, *, length, batch, devices, masked):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith('{')]
    steps = [record for record in records if record.get('event') == 'train_step']
    if ([record['step'] for record in steps] != [1, 2, 3]
            or any(type(record['step']) is not int for record in steps)):
        raise ValueError('Require three ordered update records')
    for record in steps:
        if (record['updated'] is not True or not math.isfinite(record['masked_tokens']) or
            not 0 < record['masked_tokens'] <= batch * length or
            type(record['tokens']) is not int or
            not math.isfinite(record['loss']) or record['tokens'] != batch * length or
            not math.isfinite(record['seconds']) or record['seconds'] <= 0):
            raise ValueError('Invalid, empty, or rejected update')
        if masked and record['projection_dense_fallback'] is not False:
            raise ValueError('Masked projection fell back to dense')
    checkpoints = [record for record in records if record.get('event') == 'checkpoint']
    if len(checkpoints) != 1 or checkpoints[0]['step'] != 3 or not math.isfinite(checkpoints[0]['seconds']) or checkpoints[0]['seconds'] <= 0:
        raise ValueError('Require the final committed checkpoint')
    memory = [record for record in records if record.get('event') == 'device_memory']
    if len(memory) != 1 or len(memory[0]['devices']) != devices:
        raise ValueError('Require memory evidence from every device')
    if not records.index(steps[-1]) < records.index(checkpoints[0]) < records.index(memory[0]):
        raise ValueError('Checkpoint and memory evidence must follow completed updates')
    identities = [device['device'] for device in memory[0]['devices']]
    if (any(not isinstance(identity, str) or not identity.strip() for identity in identities)
            or len(set(identities)) != devices):
        raise ValueError('Require distinct device identities for memory evidence')
    for device in memory[0]['devices']:
        stats = device['statistics']
        if any(not math.isfinite(stats.get(key, float('nan'))) for key in ('peak_bytes_in_use', 'peak_bytes_reserved', 'bytes_limit')):
            raise ValueError('Require finite device memory counters')
        if (not 0 < stats['peak_bytes_in_use'] <= stats['bytes_limit'] or
            not 0 <= stats.get('peak_bytes_reserved', -1) <= stats['bytes_limit']):
            raise ValueError('Invalid device memory evidence')
    return dict(passed=True, scope='context_execution_only', quality_qualified=False,
                sequence_length=length, batch_size=batch, steps=steps,
                checkpoint=checkpoints[0], memory=memory[0]['devices'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', required=True)
    p.add_argument('--mode', choices=['single', 'multi', 'recovery', 'context'], default='single')
    p.add_argument('--baseline-prefix', help='Existing successful single-host run prefix for recovery-only verification')
    p.add_argument('--cpu-tests', choices=['full', 'focused'], default='full',
                   help='Focused rerun after a recorded full-suite pass')
    p.add_argument('--artifact-root', default='artifacts/encoder-validation-0922')
    p.add_argument('--lengths', type=int, nargs='+', default=[512],
                   help='Explicit context sweep; start with a 512-token pilot')
    p.add_argument('--residual-dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--mlm-projection', choices=['dense', 'masked'], default='dense')
    p.add_argument('--mlm-loss-backend', choices=['xla', 'xla_full', 'xla_local', 'pallas'], default='xla')
    p.add_argument('--mlm-vocab-tile', type=int, default=1024)
    p.add_argument('--expected-processes', type=int)
    p.add_argument('--expected-devices', type=int)
    p.add_argument('--batch-size', type=int, help='Global batch, divisible by the actual global device count')
    p.add_argument('--stage-timeout-seconds', type=int, default=900)
    p.add_argument('--cloud-timeout-seconds', type=int, default=120,
                   help='Bound each evidence upload and checkpoint metadata read')
    a = p.parse_args()
    if a.cloud_timeout_seconds <= 0 or a.stage_timeout_seconds <= 0 or a.mlm_vocab_tile <= 0 or a.mlm_vocab_tile % 128:
        p.error('Require positive stage/cloud timeouts and vocabulary tile multiple of 128')
    if any(value is not None and value <= 0 for value in (a.expected_processes, a.expected_devices, a.batch_size)):
        p.error('Expected topology and batch size must be positive')
    if not a.prefix.startswith('gs://') or any(n not in (128, 512, 2048, 8192) for n in a.lengths):
        p.error('Require gs:// output and lengths from 128, 512, 2048, 8192')
    if len(set(a.lengths)) != len(a.lengths):
        p.error('Repeated lengths would overwrite qualification checkpoints')
    if a.mode == 'context' and (a.lengths != sorted(a.lengths) or any(n not in (2048, 8192) for n in a.lengths)):
        p.error('Context mode requires an ascending sweep of 2048 and/or 8192')
    if a.mode == 'multi' and a.lengths != [512]:
        p.error('Run the single-host context sweep before multi-host 512-token qualification')
    if a.mode == 'recovery' and (not a.baseline_prefix or not a.baseline_prefix.startswith('gs://') or a.lengths != [512]):
        p.error('Recovery requires a gs:// baseline-prefix and length 512')
    if a.mode != 'recovery' and a.baseline_prefix:
        p.error('baseline-prefix is only valid for recovery mode')
    root = Path(a.artifact_root)
    rank = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
    run_label = qualification_label(a.mode, a.residual_dtype, a.mlm_projection, a.mlm_loss_backend, a.mlm_vocab_tile)
    result_dir = root / f'results-{run_label}-{rank}'
    result_dir.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    experimental_failures = []
    hardware = None
    def stage(name, argv, *, expected_code=0, required=True, cpu=False, matmul_precision='highest', expected_marker=None):
        path = result_dir / f'{name}.log'
        started = time.monotonic()
        env = stage_environment(os.environ, cpu=cpu, matmul_precision=matmul_precision)
        with path.open('w') as log:
            try:
                code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                      timeout=a.stage_timeout_seconds).returncode
            except subprocess.TimeoutExpired:
                code = 124
        passed = code == expected_code and (expected_marker is None or expected_marker in path.read_text())
        record = dict(name=name, code=code, expected_code=expected_code, required=required, passed=passed, seconds=time.monotonic()-started)
        records.append(record)
        print(json.dumps(record), flush=True)
        if not passed:
            (failures if required else experimental_failures).append(name)
        subprocess.run(['gcloud','storage','cp',str(path),f'{a.prefix}/results-{run_label}-{rank}/'],
                       check=True, timeout=a.cloud_timeout_seconds)
        return passed
    python = sys.executable
    try:
        probe = ("import json, jax, flax, sys; import flaxchat.common; from pathlib import Path; "
                 "report=dict(backend=jax.default_backend(), devices=[str(d) for d in jax.devices()], "
                 "device_count=jax.device_count(), process_count=jax.process_count(), "
                 "process_index=jax.process_index(), jax=jax.__version__, flax=flax.__version__, python=sys.version); "
                 "Path(sys.argv[1]).write_text(json.dumps(report,indent=2)); print(json.dumps(report)); "
                 "assert report['backend'] == 'tpu', 'TPU backend required'")
        if not stage('hardware', [python, '-c', probe, str(result_dir/'hardware.json')]):
            return 1
        hardware = json.loads((result_dir/'hardware.json').read_text())
        validate_hardware(hardware, a.mode, a.expected_processes, a.expected_devices)
        batch = a.batch_size or (16 if a.mode == 'multi' else 4)
        if batch % hardware['device_count']:
            raise ValueError('Global batch must be divisible by actual global device count')
        if a.mode == 'single':
            if a.cpu_tests == 'full':
                stage('cpu-suite', [python,'-m','pytest','tests/','-q','--tb=short','-m','not accelerator',
                    '--cov=flaxchat','--cov-report=term','--cov-report=json:'+str(result_dir/'coverage.json')], cpu=True)
            else:
                stage('cpu-focused', [python,'-m','pytest','tests/test_encoder.py',
                    'tests/test_encoder_training.py','tests/test_gcp_tpu_run.py',
                    '-q','--tb=short','-m','not accelerator'], cpu=True)
            if failures:
                return 1
            stage('encoder-tpu-tests', [python,'-m','pytest','tests/test_encoder.py',
                'tests/test_encoder_training.py','-q','--tb=short','-m','not accelerator'])
            if failures:
                return 1
            # Keep optional Splash diagnostics isolated so an abort cannot erase XLA evidence.
            stage('encoder-splash', [python,'-m','pytest','tests/test_encoder.py',
                '-q','--tb=short','-m','accelerator'], required=False)
            stage('gpt-splash', [python,'-m','pytest','tests/test_attention_accelerator.py',
                '-q','--tb=short'], required=False, matmul_precision='default')
            if failures:
                return 1
            stage('released-parity-float32', [python,'-m','scripts.validate_encoder_checkpoint','compare',
                '--snapshot','artifacts/mmbert-base','--reference',str(root/'reference.npz'),
                '--output',str(result_dir/'parity-float32.json'),'--dtype','float32'])
            if failures:
                return 1
            # Numerical drift against FP32 is evidence, never a BF16 qualification pass.
            stage('bf16-drift-diagnostic', [python,'-m','scripts.validate_encoder_checkpoint','compare',
                '--snapshot','artifacts/mmbert-base','--reference',str(root/'reference.npz'),
                '--output',str(result_dir/'bf16-diagnostic.json'),'--dtype','bfloat16',
                '--residual-dtype',a.residual_dtype,'--diagnostic'])
            if failures:
                return 1
        lengths = a.lengths
        for length in lengths:
            steps = 6 if length <= 512 else 3
            base = [python,'-m','scripts.train_encoder','--config','artifacts/mmbert-base/config.json',
                '--pretrained','artifacts/mmbert-base','--data',str(root/f'data-{length}'),
                '--steps',str(steps),'--batch-size',str(batch),'--save-every',str(steps),
                '--loss-chunk-size','128','--dtype','bfloat16','--residual-dtype',a.residual_dtype, *projection_arguments(a)]
            ckpt = f'{a.prefix}/checkpoints-{run_label}/{length}'
            if a.mode == 'recovery':
                baseline_label = qualification_label('single', a.residual_dtype, a.mlm_projection,
                                                     a.mlm_loss_backend, a.mlm_vocab_tile)
                ckpt = f'{a.baseline_prefix}/checkpoints-{baseline_label}/{length}'
            elif not stage(f'train-{length}',base+['--output',ckpt]):
                break
            if a.mode == 'context':
                try:
                    evidence = validate_context_log(result_dir / f'train-{length}.log', length=length,
                                                    batch=batch, devices=hardware['device_count'],
                                                    masked=a.mlm_projection == 'masked')
                    (result_dir / f'context-{length}.json').write_text(json.dumps(evidence, indent=2))
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append(f'context-{length}: {exc}')
                    break
            if length == 512:
                if a.mode in ('single', 'recovery'):
                    if not stage('heldout', [python,'-m','scripts.evaluate_encoder','--checkpoint',ckpt,
                        '--train-data',str(root/'data-512'),'--data',str(root/'validation-512'),
                        '--max-rows','16','--output',str(result_dir/'heldout.json')]):
                        break
                recovery = f'{a.prefix}/checkpoints-{run_label}/{length}-resume'
                if not stage('interrupt', [python,'-m','scripts.validate_encoder_interruption',*base[3:],
                       '--output',recovery,'--save-every','3'], expected_code=-9,
                       expected_marker='FAULT_INJECTION: encoder SIGKILL after committed step 3'):
                    break
                if not stage('resume',base+['--output',recovery,'--resume']):
                    break
                def manifest(path, steps=steps):
                    return json.loads(subprocess.check_output(['gcloud','storage','cat',f'{path}/{steps}/manifest/metadata'], text=True,
                                                              timeout=a.cloud_timeout_seconds))
                baseline, resumed = manifest(ckpt), manifest(recovery)
                (result_dir / 'baseline-manifest.json').write_text(json.dumps(baseline, indent=2))
                (result_dir / 'recovery-manifest.json').write_text(json.dumps(resumed, indent=2))
                equal = {k:baseline[k] == resumed[k] for k in ('model_state','optimizer_state','training_state')}
                (result_dir/'recovery.json').write_text(json.dumps(equal,indent=2))
                if not all(equal.values()):
                    failures.append('exact-recovery')
    except BaseException as exc:
        failures.append(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        summary = dict(scope='context_execution_only' if a.mode == 'context' else 'smoke_and_recovery_only', cpu_tests=a.cpu_tests if a.mode == 'single' else None, baseline_prefix=a.baseline_prefix, quality_qualified=False,
            compute_dtype='bfloat16', residual_dtype=a.residual_dtype, lengths=a.lengths,
            mlm_projection=a.mlm_projection, mlm_loss_backend=a.mlm_loss_backend, mlm_vocab_tile=a.mlm_vocab_tile,
            expected_processes=a.expected_processes, expected_devices=a.expected_devices, hardware=hardware,
            batch_size=a.batch_size or (16 if a.mode == 'multi' else 4), stage_timeout_seconds=a.stage_timeout_seconds,
            cloud_timeout_seconds=a.cloud_timeout_seconds, required_passed=not failures,all_checks_passed=not failures and not experimental_failures,failures=failures,experimental_failures=experimental_failures,stages=records)
        summary_path = result_dir/'summary.json'
        summary_path.write_text(json.dumps(summary, indent=2))
        try:
            subprocess.run(['gcloud','storage','cp',*map(str,result_dir.glob('*')),
                            f'{a.prefix}/results-{run_label}-{rank}/'],
                           check=True, timeout=a.cloud_timeout_seconds)
        except (subprocess.SubprocessError, OSError) as exc:
            failures.append('final-evidence-upload: ' + type(exc).__name__ + ': ' + str(exc))
            summary.update(required_passed=False, all_checks_passed=False)
            summary_path.write_text(json.dumps(summary, indent=2))
            raise
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
