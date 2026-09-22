"""Bounded physical GCP multi-host acceptance on an already prepared TPU slice.

Run setup and arm the deletion watchdog first. This never provisions hardware.
Use a fresh GCS prefix. CPU bridge evidence is explicitly distinguished from TPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleanup-receipt', type=Path, required=True, help='Active cloud cleanup execution receipt')
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--checkpoint-prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--global-batch', type=int, default=16)
    parser.add_argument('--fsdp', type=int, default=4)
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--loss-chunk-size', type=int, default=0)
    parser.add_argument('--remat', action='store_true')
    parser.add_argument('--timeout', type=int, default=2400)
    args = parser.parse_args(argv)
    if not args.checkpoint_prefix.startswith('gs://') or args.timeout > 3000 or args.timeout < 60:
        parser.error('A fresh gs:// prefix and a bounded 60–3000 second campaign are required')
    from scripts.gcp_cleanup_guard import verify
    remaining = verify(json.loads(args.cleanup_receipt.read_text()), project=args.project, zone=args.zone, queue=args.node)
    if remaining < args.timeout + 120:
        parser.error("Cloud cleanup deadline must cover the campaign plus 120 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = args.checkpoint_prefix.rstrip('/')
    started = time.monotonic()
    report = {'passed': False, 'started_epoch': time.time(), 'stages': [],
              'checkpoint_prefix': prefix, 'limitations': ['Single-host bridge uses CPU with bfloat16 constants.',
              'Bounded correctness smoke; not convergence or pod-scale performance evidence.']}
    def persist():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    persist()
    remote_python = '.venv/bin/python'
    base = ['--token-manifest', 'artifacts/gcp-validation/fineweb-pool/manifest.json',
            '--global-batch-size', str(args.global_batch), '--depth', '2', '--seq-len', '32',
            '--tokens', str(args.global_batch * 32 * 8), '--warmup-steps', '2',
            '--eval-every', '4', '--save-every', '4', '--optimizer', args.optimizer,
            '--accumulation-steps', str(args.accumulation_steps),
            '--loss-chunk-size', str(args.loss_chunk_size)]
    if args.remat:
        base.append('--remat')
    report['training_recipe'] = {'optimizer': args.optimizer, 'accumulation_steps': args.accumulation_steps,
                                 'loss_chunk_size': args.loss_chunk_size, 'remat': args.remat}
    def trainer(name, extra=(), module='scripts.train_gpt2'):
        return [remote_python, '-m', module, *base, '--ckpt-dir', f'{prefix}/{name}',
                '--artifact-dir', f'artifacts/multihost/{name}', *extra]
    stages = [
        ('probe', [remote_python, '-m', 'scripts.multihost_acceptance', 'run',
                   '--project', args.project, '--zone', args.zone, '--slice', args.node,
                   '--environment-file', '/tmp/flaxchat-env.txt',
                   '--output-dir', 'artifacts/multihost/probe'], [], False),
        ('dp-baseline', trainer('dp-baseline'), [], False),
        ('dp-interruption', trainer('dp-recovery', module='scripts.validate_multihost_interruption'), [], True),
        ('dp-resume', trainer('dp-recovery', ['--resume']), [], False),
        ('fsdp-baseline', trainer('fsdp-baseline', ['--fsdp', str(args.fsdp)]), [], False),
        ('fsdp-stop', trainer('fsdp-recovery', ['--fsdp', str(args.fsdp), '--stop-after', '4']), [], False),
        ('fsdp-resume', trainer('fsdp-recovery', ['--fsdp', str(args.fsdp), '--resume']), [], False),
        ('single-host-bridge', ['env', 'JAX_PLATFORMS=cpu', 'FLAXCHAT_DTYPE=bfloat16', remote_python,
                               '-m', 'scripts.validate_checkpoint_bridge', '--source', f'{prefix}/dp-baseline',
                               '--step', '4', '--destination', f'{prefix}/bridge',
                               '--output', 'artifacts/multihost/bridge.json'], ['--single-worker'], False),
        ('multi-host-restore', trainer('bridge', ['--resume']), [], False),
    ]
    for name, command, extra, expect_kill in stages:
        remaining = args.timeout - (time.monotonic() - started)
        if remaining < 30:
            report['stages'].append({'name': name, 'status': 'not_run', 'reason': 'campaign deadline'})
            persist()
            return 1
        print(f'Running {name}', flush=True)
        stage_dir = args.output / name
        result = subprocess.run([sys.executable, '-m', 'scripts.gcp_tpu_run', '--project', args.project,
            '--zone', args.zone, '--node', args.node, '--output', str(stage_dir),
            '--timeout', str(min(480, int(remaining))), *extra, '--', *command])
        passed = result.returncode == 0
        if name == 'probe' and passed:
            from scripts.multihost_acceptance import validate_records
            records = []
            for log in sorted(stage_dir.glob('worker-*.log')):
                for line in log.read_text().splitlines():
                    if line.startswith('{'):
                        record = json.loads(line)
                        if 'topology' in record and 'training' in record:
                            records.append(record)
            report['physical_probe'] = validate_records(records, cost_usd=None)
        if expect_kill:
            # TPU runtime process ordering can differ from gcloud SSH worker
            # ordering. Locate the actual JAX rank-zero event across all logs.
            fault_logs = [path.name for path in stage_dir.glob('worker-*.log')
                          if 'FAULT_INJECTION: SIGKILL coordinator_rank=0 committed_step=4' in path.read_text()]
            passed = result.returncode != 0 and len(fault_logs) == 1
            report['coordinator_fault_worker_log'] = fault_logs[0] if len(fault_logs) == 1 else None
        report['stages'].append({'name': name, 'passed': passed, 'returncode': result.returncode,
                                'expected_rank_zero_kill': expect_kill})
        persist()
        if not passed:
            return 1
    def manifest(name):
        return json.loads(subprocess.check_output(['gcloud', 'storage', 'cat',
                          f'{prefix}/{name}/8/manifest/metadata'], text=True, timeout=60))
    comparisons = {}
    for baseline, restored in [('dp-baseline', 'dp-recovery'), ('dp-baseline', 'bridge'),
                               ('fsdp-baseline', 'fsdp-recovery')]:
        left, right = manifest(baseline), manifest(restored)
        comparisons[f'{baseline}=>{restored}'] = {
            key: left[key] == right[key] for key in ('model_state', 'optimizer_state', 'training_state')}
    report['exact_state_comparisons'] = comparisons
    report['passed'] = all(all(result.values()) for result in comparisons.values())
    report['finished_epoch'] = time.time()
    persist()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
