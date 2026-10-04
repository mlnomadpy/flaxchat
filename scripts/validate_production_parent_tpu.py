"""Authenticate and import an actual portable production parent on physical TPU.

Model-free preflight does not import JAX. Physical mode checks every restored
leaf against its committed hash; it does not evaluate quality or optimizer resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.release_contract import artifact_hashes, canonical_hash, validate_export

FILES = ('model.safetensors', 'config.json', 'tokenizer.json',
         'checkpoint-metadata.json', 'checkpoint-manifest.json', 'export.json')
DEFAULT_METADATA = 'a318eb6d7ae45a87f66dbb74f5022d86f35c988597d97407fa69f5dc09b0cfd1'
DEFAULT_MANIFEST = 'ebde175cdfffaa01d1f8bc095342a7e8a4552305abe6622f9f5679d58a9acfe0'
MAX_JSON_BYTES = 8 * 1024 * 1024
# The pinned multilingual tokenizer is 17,525,329 bytes. Match the bounded
# public-tokenizer download policy without expanding metadata/manifest limits.
MAX_TOKENIZER_JSON_BYTES = 32 * 1024 * 1024


def read_json(path):
    path = Path(path)
    maximum = MAX_TOKENIZER_JSON_BYTES if path.name == 'tokenizer.json' else MAX_JSON_BYTES
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Bounded regular parent JSON artifact required')
    # Bound the actual read too, including a file that grows after stat().
    with path.open('rb') as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError('Bounded regular parent JSON artifact required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate parent JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite parent JSON value')
    return json.loads(payload.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def preflight(root, *, expected_step=14000, expected_metadata=DEFAULT_METADATA,
              expected_manifest=DEFAULT_MANIFEST, expected_leaves=181):
    """Hash actual artifacts and admit declared identity without numerical imports."""
    root = Path(root)
    if (type(expected_step) is not int or expected_step < 1
            or type(expected_leaves) is not int or not 1 <= expected_leaves <= 10000
            or not isinstance(expected_metadata, str)
            or re.fullmatch('[0-9a-f]{64}', expected_metadata) is None
            or not isinstance(expected_manifest, str)
            or re.fullmatch('[0-9a-f]{64}', expected_manifest) is None):
        raise ValueError('Explicit positive step/leaf count and metadata SHA-256 required')
    if any((root / name).is_symlink() or not (root / name).is_file() for name in FILES):
        raise ValueError('Complete regular parent export inventory required')
    before = artifact_hashes(root, FILES)
    for name in FILES:
        if name.endswith('.json'):
            read_json(root / name)
    export = read_json(root / 'export.json')
    validate_export(root, export)
    manifest = read_json(root / 'checkpoint-manifest.json')
    config = read_json(root / 'config.json')
    if (export['source_model_family'] != 'yat_embedding_finetune'
            or export['source_checkpoint_step'] != expected_step
            or manifest['metadata_sha256'] != expected_metadata
            or canonical_hash(manifest) != expected_manifest
            or len(manifest.get('model_state', {})) != expected_leaves
            or config.get('yat_bias') != 1 or config.get('yat_epsilon') != .01
            or config.get('yat_alpha_trainable') is not True):
        raise ValueError('Actual parent identity or fixed YAT architecture differs')
    for name, record in manifest['model_state'].items():
        if (not isinstance(name, str) or not isinstance(record, dict)
                or not isinstance(record.get('shape'), list)
                or any(type(size) is not int or size < 0 for size in record['shape'])
                or record.get('dtype') not in {'float32', 'bfloat16', 'float16', 'int32', 'int64', 'bool'}
                or not isinstance(record.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', record['sha256']) is None):
            raise ValueError('Invalid committed model leaf schema')
    if artifact_hashes(root, FILES) != before:
        raise ValueError('Parent artifacts changed during admission')
    return dict(artifacts_sha256=before, expected_step=expected_step,
                expected_metadata_sha256=expected_metadata,
                expected_manifest_sha256=expected_manifest, expected_leaves=expected_leaves,
                model_state=manifest['model_state'], model_execution=False)


def record_leaf(name, observed, expected, receipt, persist):
    """Persist observed mismatches before raising, including partial inventory."""
    passed = all(observed[key] == expected[key] for key in ('shape', 'dtype', 'sha256'))
    receipt['leaves'][name] = {'observed': observed, 'expected': expected, 'passed': passed}
    persist()
    if not passed:
        raise ValueError(f'Restored production model leaf differs: {name}')


def run_physical(root, receipt, persist):
    import jax
    # Refuse CPU before importing the model implementation or constructing it.
    if jax.default_backend() != 'tpu' or jax.process_count() != 1:
        raise RuntimeError('Actual parent import requires physical single-host TPU')
    import numpy as np
    from flax import nnx
    from flaxchat.checkpoint import _canonical_manifest_paths
    from flaxchat.public_encoder import load_public_encoder, _named_leaves
    from scripts.evaluation_contract import provenance
    repository = Path(__file__).resolve().parents[1]
    source_paths = sorted((repository / 'flaxchat').glob('**/*.py')) + [
        Path(__file__).resolve(), repository / 'scripts/release_contract.py',
        repository / 'scripts/evaluation_contract.py']
    receipt['runtime'] = provenance(source_paths, devices=[{
        'id': device.id, 'process_index': device.process_index,
        'device_kind': device.device_kind, 'platform': device.platform}
        for device in jax.devices()], batch_size=0)
    receipt['model_execution_started'] = True
    persist()
    model = load_public_encoder(root)
    leaves = _named_leaves(nnx.to_pure_dict(nnx.state(model)))
    expected = _canonical_manifest_paths(receipt['admission']['model_state'])
    if set(leaves) != set(expected):
        raise ValueError('Restored production model leaf inventory differs')
    for name, value in leaves.items():
        array = np.asarray(jax.device_get(value)).copy(order='C')
        observed = {'shape': list(array.shape), 'dtype': str(array.dtype),
                    'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        record_leaf(name, observed, expected[name], receipt, persist)
    if artifact_hashes(Path(root), FILES) != receipt['admission']['artifacts_sha256']:
        raise ValueError('Parent artifacts changed during physical import')
    receipt['weight_import_qualified'] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, default=14000)
    parser.add_argument('--expected-metadata-sha256', default=DEFAULT_METADATA)
    parser.add_argument('--expected-manifest-sha256', default=DEFAULT_MANIFEST)
    parser.add_argument('--expected-leaves', type=int, default=181)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.parent.resolve()):
        raise ValueError('Use a fresh evidence output outside the parent artifact directory')
    from scripts.evaluation_contract import write_atomic
    receipt = dict(format='actual-production-parent-import-v1', status='started',
        model_execution_started=False, weight_import_qualified=False,
        quality_qualified=False, optimizer_resume_qualified=False, leaves={}, started_unix=time.time())
    def persist():
        write_atomic(args.output, receipt)
    persist()
    try:
        receipt['admission'] = preflight(args.parent, expected_step=args.expected_step,
            expected_metadata=args.expected_metadata_sha256,
            expected_manifest=args.expected_manifest_sha256, expected_leaves=args.expected_leaves)
        persist()
        if not args.preflight_only:
            run_physical(args.parent, receipt, persist)
        receipt['status'] = 'preflight_passed' if args.preflight_only else 'passed'
        return 0
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['finished_unix'] = time.time()
        persist()


if __name__ == '__main__':
    raise SystemExit(main())
