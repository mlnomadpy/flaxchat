"""Current-source direct-path qualification before unattended continuation."""
import argparse
import json
import os
from pathlib import Path
from scripts.representation_v2_cloud_campaign import run_logged
import sys
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.root
    output = root / 'qualification'
    if output.exists():
        raise ValueError('Qualification output must be fresh')
    parent_receipt = root / 'parent-qualification.json'
    os.environ.update(JAX_PLATFORMS='tpu', FLAXCHAT_PHYSICAL_TPU='1')
    manifest = json.loads((root / 'run.json').read_text())
    parent_log = root / 'parent-qualification.log'
    try:
        run_logged([sys.executable, '-m', 'scripts.validate_production_parent_tpu',
                    '--parent', str(root / 'parent'), '--output', str(parent_receipt)],
                   timeout=240, log_path=parent_log)
    finally:
        parent_error = sys.exception()
        upload_errors = []
        for evidence in (parent_receipt, parent_log):
            if evidence.exists():
                try:
                    run_logged(['gcloud', 'storage', 'cp', str(evidence),
                                manifest['output_prefix'] + '/qualification/' + evidence.name],
                               timeout=30, log_path=root / ('upload-' + evidence.name + '.log'))
                except (OSError, subprocess.SubprocessError) as error:
                    upload_errors.append(str(error))
        if upload_errors:
            (root / 'parent-qualification-upload-errors.json').write_text(json.dumps(upload_errors))
            if parent_error is None:
                raise RuntimeError('Parent qualification evidence upload failed: ' + '; '.join(upload_errors))
    parent = json.loads(parent_receipt.read_text())
    if parent.get('status') != 'passed' or parent.get('weight_import_qualified') is not True:
        raise ValueError('Actual parent restoration did not pass')
    source = Path(__file__).resolve().parents[1]
    nodes = source / 'infra/tpu/representation-v2-direct-nodes.json'
    run_logged([sys.executable, '-m', 'scripts.validate_test_suite',
                    '--output', str(output), '--expected-devices', '8',
                    '--prefix', manifest['output_prefix'] + '/qualification',
                    '--timeout-seconds', '1050', '--module-timeout-seconds', '800',
                    '--selected-nodes', str(nodes), '--required-nodes', str(nodes),
                    '--test-files', *json.loads(nodes.read_text())], timeout=1080, log_path=root / 'qualification-suite.log')


if __name__ == '__main__':
    main()
