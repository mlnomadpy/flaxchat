"""Kill a training process after a committed checkpoint and verify exact recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-manifest', required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Completed FineWeb acceptance output')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((args.baseline / 'training_summary.json').read_text())
    resolved = baseline['resolved_config']
    config = resolved['model']
    batch_per_device = resolved['global_batch_size'] // baseline['devices']
    steps = resolved['steps']
    checkpoint_step = 5
    if steps <= checkpoint_step:
        parser.error('Baseline must contain more than five updates')
    command = [sys.executable, '-m', 'scripts.train_gpt2', '--token-manifest', args.token_manifest,
               '--depth', str(config['n_layer']), '--seq-len', str(config['sequence_len']),
               '--batch-per-device', str(batch_per_device), '--tokens', str(baseline['tokens']),
               '--lr', str(resolved['lr']), '--warmup-steps', str(resolved['warmup_steps']),
               '--eval-every', '5', '--save-every', '5',
               '--ckpt-dir', str(args.output / 'checkpoints'), '--artifact-dir', str(args.output)]
    if not config['tie_embeddings']:
        command.append('--untie-embeddings')
    summary = {'passed': False, 'command': command, 'baseline': str(args.baseline),
               'method': 'SIGKILL after committed checkpoint; recreate process; compare final state digests',
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = args.output / 'recovery-summary.json'
    report.write_text(json.dumps(summary, indent=2))
    with (args.output / 'interrupted.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        try:
            while not (args.output / 'checkpoints' / str(checkpoint_step) / 'manifest' / 'metadata').exists():
                if process.poll() is not None:
                    raise RuntimeError('Trainer exited before interruption checkpoint')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Checkpoint not committed before deadline')
                time.sleep(.05)
            process.kill()
            summary['interrupted_returncode'] = process.wait(timeout=30)
            if summary['interrupted_returncode'] != -9:
                raise RuntimeError('Trainer was not killed by SIGKILL')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
    report.write_text(json.dumps(summary, indent=2))
    with (args.output / 'resume.log').open('w') as log:
        resumed = subprocess.run(command + ['--resume'], stdout=log, stderr=subprocess.STDOUT, timeout=600)
    summary['resume_returncode'] = resumed.returncode
    report.write_text(json.dumps(summary, indent=2))
    if resumed.returncode:
        raise RuntimeError('Interrupted training did not resume')
    relative = Path('checkpoints') / str(steps) / 'manifest/metadata'
    expected = json.loads((args.baseline / relative).read_text())
    actual = json.loads((args.output / relative).read_text())
    summary['equal'] = {key: actual[key] == expected[key] for key in ('model_state', 'optimizer_state', 'training_state')}
    summary['completed_steps'] = steps
    summary['passed'] = all(summary['equal'].values())
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
