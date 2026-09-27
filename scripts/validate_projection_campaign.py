"""Gate a projection benchmark on the original released-weight regression fixture."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.workload_deadline import WorkloadDeadline


def main(args):
    deadline = WorkloadDeadline(args.max_seconds)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    # Keep diagnosis separate from throughput, and save it even if qualification fails.
    command = [sys.executable, '-m', 'scripts.diagnose_encoder_projection',
               '--output', str(root / 'diagnosis.json'), '--prefix', args.prefix]
    code = subprocess.run(command, timeout=deadline.remaining()).returncode
    if code:
        return code
    diagnostic = json.loads((root / 'diagnosis.json').read_text())
    passed = all(abs(diagnostic['stages'][name]['reference_loss'] - diagnostic['stages'][name]['candidate_loss']) < 1e-3
                 and diagnostic['stages'][name]['gradient_relative_l2'] < .03
                 for name in ('full_bf16_xla_full', 'full_bf16_pallas'))
    if not passed:
        print('Original released-weight regression failed; skipping throughput campaign', flush=True)
        return 1
    return subprocess.run([sys.executable, '-m', 'scripts.benchmark_encoder_projection',
        '--output', str(root / 'benchmark'), '--prefix', args.prefix.rstrip('/') + '/benchmark',
        '--data-root', args.data_root, '--cases', '64:512', '--kernel-rows', '128', '512', '1024',
        '--tiles', '1024', '--hourly-usd', args.hourly_usd, '--max-seconds', str(max(1, int(deadline.remaining())))], timeout=deadline.remaining()).returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=int, default=1500)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--hourly-usd', required=True)
    raise SystemExit(main(parser.parse_args()))
