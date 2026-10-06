"""Bounded data-only candidate export, full-history collision filter and recheck.

Run near GCS under the independent data-job supervisor. No model execution or
resource allocation. Original authenticated history must be staged first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--spec')
    inputs.add_argument('--candidate')
    parser.add_argument('--relocation-receipt')
    parser.add_argument('--stage-metadata', action='append', required=True)
    parser.add_argument('--prepared-directory', action='append', required=True)
    parser.add_argument('--object-receipt', required=True)
    parser.add_argument('--producer-policies', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--total-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 600 <= args.total_seconds <= 3600:
        raise ValueError('Finite 600..3600 second data-only budget required')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.total_seconds
    object_receipt = args.object_receipt
    if args.relocation_receipt:
        original = json.loads(Path(args.object_receipt).read_text())
        relocated = json.loads(Path(args.relocation_receipt).read_text())
        if (relocated.get('status') != 'passed' or
                relocated.get('all_original_members_verified') is not True or
                len(relocated.get('objects', [])) != 3):
            raise ValueError('Complete original history relocation required')
        original['objects'].extend(relocated['objects'])
        object_receipt = str(root / 'combined-objects.json')
        Path(object_receipt).write_text(json.dumps(original, indent=2) + '\n')
    receipt = {'format': 'flaxchat-next-stage-development-job-v1',
               'status': 'running', 'model_execution': False, 'commands': []}

    def run(command, limit):
        remaining = min(limit, deadline - time.monotonic() - 10)
        if remaining <= 0:
            raise TimeoutError('Data preparation budget exhausted')
        item = {'argv': command, 'timeout_seconds': remaining}
        receipt['commands'].append(item)
        with (root / f'command-{len(receipt["commands"])}.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                item['returncode'] = process.wait(timeout=remaining)
                if item['returncode']:
                    raise RuntimeError('Data preparation command failed; inspect retained log')
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def scan(candidate, name, details=False):
        command = [sys.executable, '-m', 'scripts.scan_gcs_parent_exposure',
                   '--object-receipt', object_receipt,
                   '--producer-policies', args.producer_policies,
                   '--development-exclusions', str(candidate),
                   '--output', str(root / name), '--timeout-seconds', '450',
                   '--max-total-bytes', str(4 * 1024**3)]
        for path in args.stage_metadata:
            command += ['--stage-metadata', path]
        for path in args.prepared_directory:
            command += ['--prepared-directory', path]
        if details:
            command += ['--overlap-details-output', str(root / 'collisions.json'),
                        '--max-overlap-text-identities', '100000',
                        '--materialize-directory', str(root / 'historical')]
        run(command, 480)

    try:
        candidate = Path(args.candidate).resolve() if args.candidate else root / 'candidate'
        filtered = root / 'filtered'
        if not args.candidate:
            run([sys.executable, '-m', 'scripts.prepare_representation_development',
                 '--spec', args.spec, '--output', str(candidate),
                 '--timeout-seconds', '600'], 630)
        scan(candidate, 'scan.json', details=True)
        def digest(name):
            return hashlib.sha256((root / name).read_bytes()).hexdigest()
        run([sys.executable, '-m', 'scripts.filter_representation_development',
             '--candidate', str(candidate), '--scan', str(root / 'scan.json'),
             '--scan-sha256', digest('scan.json'),
             '--overlap-details', str(root / 'collisions.json'),
             '--overlap-details-sha256', digest('collisions.json'),
             '--output', str(filtered), '--min-rows-per-config', '64',
             '--timeout-seconds', '120'], 150)
        scan(filtered, 'filtered-scan.json')
        result = json.loads((root / 'filtered-scan.json').read_text())
        if (result.get('coverage_complete') is not True or
                result.get('candidate_exposure_checked') is not True or
                result.get('exact_aligned_overlap_rows') != 0):
            raise ValueError('Filtered candidate full supplied-history recheck failed')
        receipt.update(status='passed', filtered_scan_sha256=digest('filtered-scan.json'),
                       candidate_identity=result['candidate_identity'],
                       scope='supplied known-history exact overlap only; tokenization and TPU admission pending')
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root / 'job.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
