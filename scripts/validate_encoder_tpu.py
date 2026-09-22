"""Bounded encoder qualification workload; launch under the Spot supervisor.

This orchestrator intentionally does not import JAX: each stage owns its runtime.
"""
import argparse
import json
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
                    'TPU_WORKER_HOSTNAMES', 'TPU_WORKER_ID'):
            env.pop(key, None)
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', required=True)
    p.add_argument('--mode', choices=['single', 'multi', 'recovery'], default='single')
    p.add_argument('--baseline-prefix', help='Existing successful single-host run prefix for recovery-only verification')
    p.add_argument('--cpu-tests', choices=['full', 'focused'], default='full',
                   help='Focused rerun after a recorded full-suite pass')
    p.add_argument('--artifact-root', default='artifacts/encoder-validation-0922')
    p.add_argument('--lengths', type=int, nargs='+', default=[512],
                   help='Explicit context sweep; start with a 512-token pilot')
    p.add_argument('--residual-dtype', choices=['float32', 'bfloat16'], default='float32')
    a = p.parse_args()
    if not a.prefix.startswith('gs://') or any(n not in (128, 512, 2048, 8192) for n in a.lengths):
        p.error('Require gs:// output and lengths from 128, 512, 2048, 8192')
    if len(set(a.lengths)) != len(a.lengths):
        p.error('Repeated lengths would overwrite qualification checkpoints')
    if a.mode == 'multi' and a.lengths != [512]:
        p.error('Run the single-host context sweep before multi-host 512-token qualification')
    if a.mode == 'recovery' and (not a.baseline_prefix or not a.baseline_prefix.startswith('gs://') or a.lengths != [512]):
        p.error('Recovery requires a gs:// baseline-prefix and length 512')
    if a.mode != 'recovery' and a.baseline_prefix:
        p.error('baseline-prefix is only valid for recovery mode')
    root = Path(a.artifact_root)
    rank = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
    run_label = f'{a.mode}-residual-{a.residual_dtype}'
    result_dir = root / f'results-{run_label}-{rank}'
    result_dir.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    experimental_failures = []
    def stage(name, argv, *, expected_code=0, required=True, cpu=False, matmul_precision='highest'):
        path = result_dir / f'{name}.log'
        started = time.monotonic()
        env = stage_environment(os.environ, cpu=cpu, matmul_precision=matmul_precision)
        with path.open('w') as log:
            code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        record = dict(name=name, code=code, expected_code=expected_code, required=required, seconds=time.monotonic()-started)
        records.append(record)
        print(json.dumps(record), flush=True)
        subprocess.run(['gcloud','storage','cp',str(path),f'{a.prefix}/results-{run_label}-{rank}/'], check=True)
        if code != expected_code:
            (failures if required else experimental_failures).append(name)
        return code == expected_code
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
        if a.mode == 'single':
            if a.cpu_tests == 'full':
                stage('cpu-suite', [python,'-m','pytest','tests/','-q','--tb=short','-m','not accelerator',
                    '--cov=flaxchat','--cov-report=term','--cov-report=json:'+str(result_dir/'coverage.json')], cpu=True)
            else:
                stage('cpu-focused', [python,'-m','pytest','tests/test_encoder.py',
                    'tests/test_encoder_training.py','tests/test_gcp_tpu_run.py',
                    '-q','--tb=short','-m','not accelerator'], cpu=True)
            stage('encoder-tpu-tests', [python,'-m','pytest','tests/test_encoder.py',
                'tests/test_encoder_training.py','-q','--tb=short','-m','not accelerator'])
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
            # Numerical drift against FP32 is evidence, never a BF16 qualification pass.
            stage('bf16-drift-diagnostic', [python,'-m','scripts.validate_encoder_checkpoint','compare',
                '--snapshot','artifacts/mmbert-base','--reference',str(root/'reference.npz'),
                '--output',str(result_dir/'bf16-diagnostic.json'),'--dtype','bfloat16',
                '--residual-dtype',a.residual_dtype,'--diagnostic'])
            if failures:
                return 1
        batch = 16 if a.mode == 'multi' else 4
        lengths = a.lengths
        for length in lengths:
            steps = 6 if length <= 512 else 3
            base = [python,'-m','scripts.train_encoder','--config','artifacts/mmbert-base/config.json',
                '--pretrained','artifacts/mmbert-base','--data',str(root/f'data-{length}'),
                '--steps',str(steps),'--batch-size',str(batch),'--save-every',str(steps),
                '--loss-chunk-size','128','--dtype','bfloat16','--residual-dtype',a.residual_dtype]
            ckpt = f'{a.prefix}/checkpoints-{run_label}/{length}'
            if a.mode == 'recovery':
                ckpt = f'{a.baseline_prefix}/checkpoints-single-residual-{a.residual_dtype}/{length}'
            elif not stage(f'train-{length}',base+['--output',ckpt]):
                break
            if length == 512:
                if a.mode in ('single', 'recovery'):
                    stage('heldout', [python,'-m','scripts.evaluate_encoder','--checkpoint',ckpt,
                        '--train-data',str(root/'data-512'),'--data',str(root/'validation-512'),
                        '--max-rows','16','--output',str(result_dir/'heldout.json')])
                recovery = f'{a.prefix}/checkpoints-{run_label}/{length}-resume'
                if not stage('interrupt', [python,'-m','scripts.validate_encoder_interruption',*base[3:],
                       '--output',recovery,'--save-every','3'], expected_code=-9):
                    break
                if not stage('resume',base+['--output',recovery,'--resume']):
                    break
                def manifest(path, steps=steps):
                    return json.loads(subprocess.check_output(['gcloud','storage','cat',f'{path}/{steps}/manifest/metadata'], text=True))
                baseline, resumed = manifest(ckpt), manifest(recovery)
                equal = {k:baseline[k] == resumed[k] for k in ('model_state','optimizer_state','training_state')}
                (result_dir/'recovery.json').write_text(json.dumps(equal,indent=2))
                if not all(equal.values()):
                    failures.append('exact-recovery')
    except BaseException as exc:
        failures.append(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        (result_dir/'summary.json').write_text(json.dumps(dict(scope='smoke_and_recovery_only', cpu_tests=a.cpu_tests if a.mode == 'single' else None, baseline_prefix=a.baseline_prefix, quality_qualified=False,
            compute_dtype='bfloat16', residual_dtype=a.residual_dtype, lengths=a.lengths, required_passed=not failures,all_checks_passed=not failures and not experimental_failures,failures=failures,experimental_failures=experimental_failures,stages=records),indent=2))
        subprocess.run(['gcloud','storage','cp',*map(str,result_dir.glob('*')),f'{a.prefix}/results-{run_label}-{rank}/'],check=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
