"""Preserve historical evaluation/release invocations through frozen manifests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts import representation_run as runner


def validate_workflow(manifest, kind, index=None):
    contract = manifest.get('workflow_contract', {})
    if contract.get('kind') != kind:
        raise ValueError('Manifest workflow_contract must select this workflow')
    argv = manifest['workload']
    module = {'miracl': 'scripts.evaluate_yat_mteb_subsets',
              'publish-flax': 'scripts.publish_yat_embedding_from_gcp',
              'publish-torch': 'scripts.publish_yat_torch_from_gcp'}[kind]
    selected_module = len(argv) > 2 and argv[1:3] == ['-m', module]
    selected_script = len(argv) > 1 and argv[1].endswith('/' + module.rsplit('.', 1)[1] + '.py')
    if not (selected_module or selected_script):
        raise ValueError('Manifest workload must execute the selected workflow')
    for option in ('--task', '--subsets-file', '--repo', '--parity-evidence', '--host-ram-budget-bytes', '--evidence-budget-bytes'):
        if option in argv and (argv.count(option) != 1 or argv.index(option) + 1 >= len(argv)):
            raise ValueError('Workflow requires a single value for ' + option)
    targets = {item['target'] for item in manifest.get('artifacts', [])}
    if kind == 'miracl':
        if index not in range(2, 8) or contract.get('shard_index') != index:
            raise ValueError('Manifest must bind the requested MIRACL shard 2–7')
        if '--task' not in argv or argv[argv.index('--task') + 1] != 'MIRACLRetrievalHardNegatives':
            raise ValueError('MIRACL workflow must preserve its benchmark task')
        target = contract.get('subsets_target')
        if target not in targets or '--subsets-file' not in argv or argv[argv.index('--subsets-file') + 1] != '{root}/' + target:
            raise ValueError('MIRACL shard input must be a hashed manifest artifact')
    else:
        repo = contract.get('repo')
        if not repo or '--repo' not in argv or argv[argv.index('--repo') + 1] != repo:
            raise ValueError('Publication repo must match the explicit workflow contract')
        target = contract.get('release_target')
        if target not in targets or '{root}/' + target not in argv:
            raise ValueError('Publication release must be a hashed manifest artifact')
        if kind == 'publish-torch':
            evidence = contract.get('parity_evidence_target')
            if (not evidence or evidence not in targets or evidence == target
                    or '--parity-evidence' not in argv
                    or argv[argv.index('--parity-evidence') + 1] != '{root}/' + evidence):
                raise ValueError('Torch publication must bind separate hashed durable parity evidence')
            for option in ('--host-ram-budget-bytes', '--evidence-budget-bytes'):
                if option not in argv or not argv[argv.index(option) + 1].isdigit() or int(argv[argv.index(option) + 1]) < 1:
                    raise ValueError('Torch publication requires explicit positive replay capacity: ' + option)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['miracl', 'publish-flax', 'publish-torch'], required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--manifest', type=Path, default=os.environ.get('FLAXCHAT_RUN_MANIFEST'))
    parser.add_argument('--root', type=Path, default=os.environ.get('FLAXCHAT_RUN_ROOT'))
    parser.add_argument('--run-only', action='store_true')
    args = parser.parse_args(argv)
    if args.manifest is None or args.root is None:
        parser.error('Set FLAXCHAT_RUN_MANIFEST and FLAXCHAT_RUN_ROOT, or supply --manifest and --root')
    manifest = runner.load_manifest(args.manifest)
    validate_workflow(manifest, args.kind, args.index)
    if not args.run_only:
        runner.setup(args.manifest, args.root)
    return runner.run_worker(args.manifest, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
