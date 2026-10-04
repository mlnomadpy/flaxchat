"""Literal artifact bytes and lineage receipts only; no model execution."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_enriched_parent_export as preparation
from scripts.release_contract import canonical_hash, digest
from tests.test_production_parent_tpu_metadata import export_fixture


def fixture(tmp_path):
    public = tmp_path / 'historical-public'
    metadata_sha = export_fixture(public)
    export = json.loads((public / 'export.json').read_text())
    export.pop('artifacts_sha256')
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (public / 'model.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + b'leaf')
    export.update(sha256=digest(public / 'model.safetensors'), bytes=(public / 'model.safetensors').stat().st_size)
    (public / 'export.json').write_text(json.dumps(export))
    hashes = {name: digest(public / name) for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'export.json')}
    (public / 'files.sha256.json').write_text(json.dumps(hashes))
    manifest = json.loads((public / 'checkpoint-manifest.json').read_text())
    pin = {'model_id': 'fixture/parent', 'immutable_revision': 'a' * 40,
           'weights': {'sha256': hashes['model.safetensors'], 'bytes': (public / 'model.safetensors').stat().st_size},
           'tokenizer_sha256': hashes['tokenizer.json'],
           'small_metadata_actual_bytes_sha256': {name: digest(public / name) for name in ('config.json', 'export.json', 'files.sha256.json')},
           'checkpoint': {'step': 14000, 'model_leaves': 1, 'metadata_canonical_sha256': metadata_sha,
                          'export_checkpoint_manifest_identity_sha256': manifest['identity_sha256'],
                          'retained_committed_manifest_canonical_sha256': canonical_hash(manifest)}}
    identity = tmp_path / 'pin.json'
    identity.write_text(json.dumps(pin))
    return dict(public=public, metadata=public / 'checkpoint-metadata.json', manifest=public / 'checkpoint-manifest.json',
                identity=identity, identity_sha256=digest(identity), output=tmp_path / 'enriched')


def test_import_does_not_initialize_numerical_backend():
    subprocess.run([sys.executable, '-c', "import sys; import scripts.prepare_enriched_parent_export; assert 'jax' not in sys.modules; assert 'flax' not in sys.modules"], check=True)


def test_enriches_separate_artifact_and_keeps_historical_release_unchanged(tmp_path):
    args = fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in args['public'].iterdir()}
    receipt = preparation.prepare(**args)
    assert {path.name: path.read_bytes() for path in args['public'].iterdir()} == before
    assert receipt['model_execution'] is False and receipt['weight_import_qualified'] is False
    assert (args['output'] / 'historical-export.json').read_bytes() == before['export.json']
    assert 'artifacts_sha256' in json.loads((args['output'] / 'export.json').read_text())
    assert receipt['safetensors_inventory']['checked_leaves'] == 1
    checksums = json.loads((args['output'] / 'files.sha256.json').read_text())
    assert checksums['export.json'] == digest(args['output'] / 'export.json')
    assert not list(tmp_path.glob('.parent-export-*'))
    with pytest.raises(ValueError, match='Fresh enriched'):
        preparation.prepare(**args)


def test_rehashed_committed_leaf_cannot_replace_independently_pinned_manifest(tmp_path):
    args = fixture(tmp_path)
    manifest = json.loads(args['manifest'].read_text())
    manifest['model_state']['weight']['sha256'] = 'b' * 64
    args['manifest'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Actual parent identity'):
        preparation.prepare(**args)
    assert not args['output'].exists() and not list(tmp_path.glob('.parent-export-*'))


def test_self_consistent_public_checksums_do_not_replace_pinned_weights(tmp_path):
    args = fixture(tmp_path)
    (args['public'] / 'model.safetensors').write_bytes(b'substituted')
    checksums = json.loads((args['public'] / 'files.sha256.json').read_text())
    checksums['model.safetensors'] = digest(args['public'] / 'model.safetensors')
    (args['public'] / 'files.sha256.json').write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match='Externally pinned'):
        preparation.prepare(**args)
    assert not args['output'].exists()


def test_declared_bounds_and_symlink_inputs_rejected_before_output(tmp_path):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match='byte budget'):
        preparation.prepare(**args, max_input_bytes=1)
    with pytest.raises(ValueError, match='Finite'):
        preparation.prepare(**args, timeout_seconds=True)
    model = args['public'] / 'model.safetensors'
    actual = tmp_path / 'actual-model'
    model.rename(actual)
    model.symlink_to(actual)
    with pytest.raises(ValueError, match='regular'):
        preparation.prepare(**args)
    assert not args['output'].exists()


def test_nested_output_and_insufficient_scratch_cannot_modify_public_directory(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        preparation.prepare(**(args | {'output': args['public'] / 'nested'}))
    monkeypatch.setattr(preparation.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 0})())
    with pytest.raises(ValueError, match='scratch'):
        preparation.prepare(**args)
    assert not args['output'].exists() and not list(tmp_path.glob('.parent-export-*'))


def test_deadline_failure_removes_staging_and_never_commits_output(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    ticks = iter([0, 1, 601])
    monkeypatch.setattr(preparation.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='deadline'):
        preparation.prepare(**args)
    assert not args['output'].exists() and not list(tmp_path.glob('.parent-export-*'))


def test_safetensors_header_cannot_claim_unbounded_shape_or_hide_extra_payload(tmp_path):
    args = fixture(tmp_path)
    manifest = json.loads(args['manifest'].read_text())
    path = tmp_path / 'bad.safetensors'
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [2 ** 60], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, 'little') + header + b'leaf')
    with pytest.raises(ValueError, match='bounded Safetensors leaf schema'):
        preparation.authenticate_safetensors(path, manifest)
    valid = (args['public'] / 'model.safetensors').read_bytes()
    path.write_bytes(valid + b'hidden')
    with pytest.raises(ValueError, match='trailing'):
        preparation.authenticate_safetensors(path, manifest)


def test_safetensors_raw_leaf_hash_must_match_committed_state(tmp_path):
    args = fixture(tmp_path)
    manifest = json.loads(args['manifest'].read_text())
    path = args['public'] / 'model.safetensors'
    path.write_bytes(path.read_bytes()[:-4] + b'evil')
    with pytest.raises(ValueError, match='committed leaf bytes'):
        preparation.authenticate_safetensors(path, manifest)


@pytest.mark.parametrize('changed_field', ['st_atime_ns', 'st_mtime_ns', 'st_ino'])
def test_read_access_time_is_allowed_but_modified_or_replaced_input_is_rejected(tmp_path, monkeypatch, changed_field):
    args = fixture(tmp_path)
    model = args['public'] / 'model.safetensors'
    original_stat, original_open = Path.stat, Path.open
    reading = False
    class Observation:
        def __init__(self, original):
            self.original = original
        def __getattr__(self, name):
            value = getattr(self.original, name)
            return value + 1 if name == changed_field else value
    def read_hook(path, *a, **kw):
        nonlocal reading
        if path == model and a and a[0] == 'rb':
            reading = True
        return original_open(path, *a, **kw)
    def observe(path, *a, **kw):
        value = original_stat(path, *a, **kw)
        return Observation(value) if reading and path == model else value
    monkeypatch.setattr(Path, 'open', read_hook)
    monkeypatch.setattr(Path, 'stat', observe)
    if changed_field == 'st_atime_ns':
        receipt = preparation.prepare(**args)
        assert receipt['safetensors_inventory']['checked_leaves'] == 1
        assert args['output'].is_dir()
    else:
        with pytest.raises(ValueError, match='changed during preparation'):
            preparation.prepare(**args)
        assert not args['output'].exists()
        assert not list(tmp_path.glob('.parent-export-*'))
