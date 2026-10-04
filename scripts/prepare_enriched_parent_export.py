"""Enrich a materialized historical public export without loading a model.

Run near retained data. This tool does not download, upload, or overwrite a Hub
release. Independent identity receipt and full committed files are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time

from scripts.release_contract import artifact_hashes, canonical_hash, digest
from scripts.validate_production_parent_tpu import preflight, read_json

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_INPUT_BYTES = 4 * 1024 ** 3


def stable_file_identity(stat):
    """Content/namespace stability excludes access time changed by normal reads."""
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def authenticate_safetensors(path, manifest, check_deadline=lambda: None):
    """Bounded header and streaming leaf-byte authentication, without tensors."""
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate Safetensors header key')
            result[key] = value
        return result
    def canonical_paths(records):
        result = {}
        for name, record in records.items():
            key = re.sub(r'\[(\d+)\]', r"['\1']", name)
            key = re.sub(r'\.([A-Za-z_]\w*)', r"['\1']", key)
            if key in result:
                raise ValueError('Duplicate canonical tensor path')
            result[key] = record
        return result
    expected = canonical_paths(manifest['model_state'])
    widths = {'F32': ('float32', 4), 'BF16': ('bfloat16', 2), 'F16': ('float16', 2),
              'I32': ('int32', 4), 'I64': ('int64', 8), 'BOOL': ('bool', 1)}
    with Path(path).open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('Missing Safetensors header length')
        length = int.from_bytes(length_bytes, 'little')
        if not 2 <= length <= 1024 * 1024 or length + 8 > Path(path).stat().st_size:
            raise ValueError('Bounded Safetensors header required')
        header = json.loads(stream.read(length), object_pairs_hook=unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite header')))
        header.pop('__metadata__', None)
        tensors = canonical_paths(header)
        if set(tensors) != set(expected) or not 1 <= len(tensors) <= 10000:
            raise ValueError('Safetensors complete leaf inventory differs')
        size = Path(path).stat().st_size - 8 - length
        records = []
        for name, record in tensors.items():
            shape, offsets, dtype = record.get('shape'), record.get('data_offsets'), record.get('dtype')
            if (not isinstance(shape, list) or len(shape) > 16
                    or any(type(v) is not int or not 0 <= v <= MAX_INPUT_BYTES for v in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int for v in offsets) or not 0 <= offsets[0] <= offsets[1] <= size
                    or dtype not in widths):
                raise ValueError('Invalid bounded Safetensors leaf schema')
            if (offsets[1] - offsets[0] != math.prod(shape) * widths[dtype][1]
                    or shape != expected[name]['shape'] or widths[dtype][0] != expected[name]['dtype']):
                raise ValueError('Safetensors leaf shape/dtype differs: ' + name)
            records.append((offsets[0], offsets[1], name))
        cursor = 0
        for first, last, name in sorted(records):
            check_deadline()
            if first != cursor:
                raise ValueError('Safetensors payload gap or overlapping leaves')
            stream.seek(8 + length + first)
            h = hashlib.sha256()
            remaining = last - first
            while remaining:
                check_deadline()
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError('Truncated Safetensors leaf')
                h.update(chunk)
                remaining -= len(chunk)
            if h.hexdigest() != expected[name]['sha256']:
                raise ValueError('Safetensors committed leaf bytes differ: ' + name)
            cursor = last
        if cursor != size:
            raise ValueError('Safetensors trailing unaccounted bytes')
    return {'checked_leaves': len(records), 'payload_bytes': size, 'header_bytes': length,
            'policy': 'bounded-header-and-streamed-committed-leaf-sha256-v1', 'tensor_math': False}


def prepare(public, metadata, manifest, identity, *, identity_sha256, output,
            max_input_bytes=2 * 1024 ** 3, timeout_seconds=600):
    public, metadata, manifest, identity, output = map(Path, (public, metadata, manifest, identity, output))
    if (type(max_input_bytes) is not int or not 1 <= max_input_bytes <= MAX_INPUT_BYTES
            or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800
            or not isinstance(identity_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', identity_sha256) is None):
        raise ValueError('Finite byte/deadline bounds and independent identity SHA required')
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh enriched output required; historical artifacts are immutable')
    if public.resolve() in output.resolve().parents or output.resolve() == public.resolve():
        raise ValueError('Enriched output must be outside the historical public directory')
    sources = {name: public / name for name in ('model.safetensors', 'config.json', 'tokenizer.json',
                                               'export.json', 'files.sha256.json')}
    sources.update({'checkpoint-metadata.json': metadata, 'checkpoint-manifest.json': manifest,
                    'public-identity-receipt.json': identity})
    if any(path.is_symlink() or not path.is_file() for path in sources.values()):
        raise ValueError('Complete regular input artifacts required')
    if any(output.resolve() == path.resolve() or output.resolve() in path.resolve().parents
           or path.resolve().parent == output.resolve() for path in sources.values()):
        raise ValueError('Enriched output must be separate from original input artifacts')
    total = sum(path.stat().st_size for path in sources.values())
    if total > max_input_bytes or any(path.stat().st_size > MAX_JSON_BYTES
                                   for name, path in sources.items()
                                   if name.endswith('.json') and name != 'tokenizer.json'):
        raise ValueError('Input artifact byte budget exceeded')
    if digest(identity) != identity_sha256:
        raise ValueError('Independent public identity receipt changed')
    pin = read_json(identity)
    if (not isinstance(pin.get('model_id'), str)
            or re.fullmatch('[0-9a-f]{40}', pin.get('immutable_revision', '')) is None):
        raise ValueError('Immutable public model revision required')
    weights = pin['weights']
    expected = dict(pin['small_metadata_actual_bytes_sha256'])
    expected.update({'model.safetensors': weights['sha256'], 'tokenizer.json': pin['tokenizer_sha256']})
    expected['public-identity-receipt.json'] = identity_sha256
    if any(re.fullmatch('[0-9a-f]{64}', expected.get(name, '')) is None
           for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'export.json', 'files.sha256.json')):
        raise ValueError('Complete externally pinned public artifact SHA inventory required')
    deadline = time.monotonic() + timeout_seconds
    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('Enriched parent preparation deadline exceeded')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + 32 * 1024 * 1024:
        raise ValueError('Insufficient scratch bytes for separate enriched export')
    staging = Path(tempfile.mkdtemp(prefix='.parent-export-', dir=output.parent))
    observed = {}
    snapshots = {}
    try:
        for name, path in sources.items():
            check_deadline()
            before = path.stat()
            target_name = {'export.json': 'historical-export.json',
                           'files.sha256.json': 'historical-files.sha256.json'}.get(name, name)
            h = hashlib.sha256()
            copied = 0
            with path.open('rb') as source, (staging / target_name).open('xb') as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    check_deadline()
                    copied += len(chunk)
                    if copied > before.st_size:
                        raise ValueError('Input artifact grew during preparation')
                    h.update(chunk)
                    target.write(chunk)
            if (path.is_symlink() or stable_file_identity(path.stat()) != stable_file_identity(before)
                    or copied != before.st_size):
                raise ValueError('Input artifact changed during preparation: ' + name
                                 + '; expected stable identity=' + str(stable_file_identity(before))
                                 + '; observed=' + str(stable_file_identity(path.stat())))
            snapshots[name] = before
            observed[name] = h.hexdigest()
            if name in expected and observed[name] != expected[name]:
                raise ValueError('Externally pinned public artifact SHA mismatch: ' + name)
        if weights['bytes'] != (staging / 'model.safetensors').stat().st_size:
            raise ValueError('Pinned model byte count differs')
        original = read_json(staging / 'historical-export.json')
        public_hashes = read_json(staging / 'historical-files.sha256.json')
        if any(public_hashes.get(name) != observed[name]
               for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'export.json')):
            raise ValueError('Historical public checksum inventory differs')
        checkpoint = pin['checkpoint']
        if (original['source_checkpoint_step'] != checkpoint['step']
                or original['source_checkpoint_metadata_sha256'] != checkpoint['metadata_canonical_sha256']
                or original['source_checkpoint_manifest_identity_sha256'] != checkpoint['export_checkpoint_manifest_identity_sha256']):
            raise ValueError('Pinned export checkpoint identity differs')
        enriched = dict(original)
        enriched['artifacts_sha256'] = artifact_hashes(staging, ('model.safetensors', 'config.json', 'tokenizer.json',
                                                               'checkpoint-metadata.json', 'checkpoint-manifest.json'))
        (staging / 'export.json').write_text(json.dumps(enriched, indent=2) + '\n')
        check_deadline()
        admitted = preflight(staging, expected_step=checkpoint['step'],
                             expected_metadata=checkpoint['metadata_canonical_sha256'],
                             expected_manifest=checkpoint['retained_committed_manifest_canonical_sha256'],
                             expected_leaves=checkpoint['model_leaves'])
        inventory = authenticate_safetensors(staging / 'model.safetensors',
                                            read_json(staging / 'checkpoint-manifest.json'), check_deadline)
        (staging / 'files.sha256.json').write_text(json.dumps(admitted['artifacts_sha256'], indent=2) + '\n')
        check_deadline()
        if any(path.is_symlink() or stable_file_identity(path.stat()) != stable_file_identity(snapshots[name])
               for name, path in sources.items()):
            raise ValueError('Original inputs changed before enriched export commit')
        receipt = {'format': 'flaxchat-enriched-parent-export-v1', 'immutable_public_revision': pin['immutable_revision'],
                   'model_id': pin['model_id'], 'public_identity_receipt_sha256': identity_sha256,
                   'original_input_sha256': observed, 'enriched_artifacts_sha256': admitted['artifacts_sha256'],
                   'checkpoint_manifest_canonical_sha256': canonical_hash(read_json(staging / 'checkpoint-manifest.json')),
                   'input_bytes': total, 'max_input_bytes': max_input_bytes, 'timeout_seconds': timeout_seconds,
                   'safetensors_inventory': inventory,
                   'model_execution': False, 'weight_import_qualified': False, 'quality_qualified': False,
                   'historical_public_release_changed': False}
        (staging / 'preparation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if output.exists() or output.is_symlink():
            raise ValueError('Fresh enriched output was created concurrently')
        staging.rename(output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public', 'metadata', 'manifest', 'identity', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--identity-sha256', required=True)
    parser.add_argument('--max-input-bytes', type=int, default=2 * 1024 ** 3)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    args = vars(parser.parse_args(argv))
    prepare(**args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
