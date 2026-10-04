"""Actual-parent admission receipts without model execution or JAX import."""
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from scripts import validate_production_parent_tpu as validator
from scripts.release_contract import artifact_hashes, canonical_hash, checkpoint_identity, digest


def export_fixture(root):
    root.mkdir()
    (root / 'model.safetensors').write_bytes(b'not-loaded-metadata-fixture')
    (root / 'tokenizer.json').write_text('{}')
    config = {'yat_bias': 1, 'yat_epsilon': .01, 'yat_alpha_trainable': True}
    (root / 'config.json').write_text(json.dumps(config))
    metadata = {'model_family': 'yat_embedding_finetune', 'resolved_config': {'encoder': config},
                'tokenizer_identity': digest(root / 'tokenizer.json')}
    manifest = {'step': 14000, 'metadata_sha256': canonical_hash(metadata),
        'identity': checkpoint_identity(metadata), 'model_state': {'weight': {
            'shape': [1], 'dtype': 'float32', 'sha256': hashlib.sha256(b'leaf').hexdigest()}}}
    manifest['identity_sha256'] = canonical_hash(manifest['identity'])
    (root / 'checkpoint-metadata.json').write_text(json.dumps(metadata))
    (root / 'checkpoint-manifest.json').write_text(json.dumps(manifest))
    export = {'source_model_family': 'yat_embedding_finetune', 'source_checkpoint_step': 14000,
        'source_checkpoint_metadata_sha256': manifest['metadata_sha256'],
        'source_checkpoint_manifest_identity_sha256': manifest['identity_sha256'],
        'tokenizer_identity': metadata['tokenizer_identity'], 'tensors': 1,
        'sha256': digest(root / 'model.safetensors'), 'bytes': (root / 'model.safetensors').stat().st_size,
        'artifacts_sha256': artifact_hashes(root, validator.FILES[:-1])}
    (root / 'export.json').write_text(json.dumps(export))
    return manifest['metadata_sha256']


def test_import_has_no_numerical_backend_side_effect():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import scripts.validate_production_parent_tpu; "
        "assert 'jax' not in sys.modules; assert 'flax' not in sys.modules"], check=True)
    assert result.returncode == 0


def test_preflight_authenticates_actual_inventory_without_loading_model(tmp_path):
    root = tmp_path / 'parent'
    metadata = export_fixture(root)
    receipt = validator.preflight(root, expected_metadata=metadata, expected_manifest=canonical_hash(validator.read_json(root / 'checkpoint-manifest.json')), expected_leaves=1)
    assert receipt['expected_step'] == 14000
    assert receipt['model_execution'] is False
    assert len(receipt['artifacts_sha256']) == 6
    (root / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        validator.preflight(root, expected_metadata=metadata, expected_manifest=canonical_hash(validator.read_json(root / 'checkpoint-manifest.json')), expected_leaves=1)


@pytest.mark.parametrize('changes', [{'expected_step': 14001}, {'expected_leaves': 181},
    {'expected_metadata': 'a' * 64}, {'expected_step': True}])
def test_preflight_expected_parent_identity_cannot_drift(tmp_path, changes):
    root = tmp_path / 'parent'
    metadata = export_fixture(root)
    with pytest.raises(ValueError):
        validator.preflight(root, **({'expected_metadata': metadata, 'expected_manifest': canonical_hash(validator.read_json(root / 'checkpoint-manifest.json')), 'expected_leaves': 1} | changes))


def test_preflight_only_persists_explicit_unqualified_receipt(tmp_path, monkeypatch):
    root = tmp_path / 'parent'
    metadata = export_fixture(root)
    monkeypatch.setattr(validator, 'run_physical', lambda *a: pytest.fail('No model permitted'))
    output = tmp_path / 'evidence.json'
    assert validator.main(['--parent', str(root), '--output', str(output), '--expected-leaves', '1',
        '--expected-metadata-sha256', metadata, '--expected-manifest-sha256', canonical_hash(validator.read_json(root / 'checkpoint-manifest.json')), '--preflight-only']) == 0
    receipt = json.loads(output.read_text())
    assert receipt['status'] == 'preflight_passed'
    assert not receipt['weight_import_qualified']
    assert not receipt['quality_qualified'] and not receipt['optimizer_resume_qualified']
    with pytest.raises(ValueError, match='fresh evidence'):
        validator.main(['--parent', str(root), '--output', str(output)])


def test_cpu_backend_refused_before_model_import_or_load(monkeypatch):
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(default_backend=lambda: 'cpu'))
    receipt = {'model_execution_started': False}
    with pytest.raises(RuntimeError, match='physical single-host TPU'):
        validator.run_physical('/unused', receipt, lambda: pytest.fail('No model permitted'))
    assert not receipt['model_execution_started']


def test_failed_admission_retains_failure_without_model_execution(tmp_path):
    output = tmp_path / 'failed.json'
    with pytest.raises(ValueError):
        validator.main(['--parent', str(tmp_path / 'missing'), '--output', str(output)])
    receipt = json.loads(output.read_text())
    assert receipt['status'] == 'failed' and 'error' in receipt
    assert not receipt['model_execution_started']
    assert receipt['leaves'] == {}


def test_partial_leaf_mismatch_is_durable_before_terminal_failure(tmp_path, monkeypatch):
    root = tmp_path / 'parent'
    metadata = export_fixture(root)
    output = tmp_path / 'partial.json'
    def synthetic_leaf_check(parent, receipt, persist):
        expected = {'shape': [1], 'dtype': 'float32', 'sha256': 'a' * 64}
        validator.record_leaf('first', expected, expected, receipt, persist)
        validator.record_leaf('second', expected | {'sha256': 'b' * 64}, expected, receipt, persist)
    monkeypatch.setattr(validator, 'run_physical', synthetic_leaf_check)
    with pytest.raises(ValueError, match='second'):
        validator.main(['--parent', str(root), '--output', str(output), '--expected-leaves', '1',
            '--expected-metadata-sha256', metadata, '--expected-manifest-sha256', canonical_hash(validator.read_json(root / 'checkpoint-manifest.json'))])
    receipt = json.loads(output.read_text())
    assert receipt['status'] == 'failed'
    assert receipt['leaves']['first']['passed']
    assert not receipt['leaves']['second']['passed']
    assert receipt['leaves']['second']['observed']['sha256'] == 'b' * 64
    assert not receipt['weight_import_qualified']


def test_rehashed_model_manifest_cannot_replace_authenticated_parent_leaf_hashes(tmp_path):
    root = tmp_path / 'parent'
    metadata = export_fixture(root)
    original_manifest = canonical_hash(validator.read_json(root / 'checkpoint-manifest.json'))
    manifest = validator.read_json(root / 'checkpoint-manifest.json')
    manifest['model_state']['weight']['sha256'] = 'b' * 64
    (root / 'checkpoint-manifest.json').write_text(json.dumps(manifest))
    export = validator.read_json(root / 'export.json')
    export['artifacts_sha256'] = artifact_hashes(root, validator.FILES[:-1])
    (root / 'export.json').write_text(json.dumps(export))
    # Self-consistent export hashes and unchanged metadata are insufficient.
    with pytest.raises(ValueError, match='Actual parent identity'):
        validator.preflight(root, expected_metadata=metadata, expected_manifest=original_manifest,
                            expected_leaves=1)


def test_actual_size_multilingual_tokenizer_passes_full_metadata_preflight(tmp_path):
    root = tmp_path / 'parent'
    export_fixture(root)
    # Real released tokenizer byte count, valid JSON padding; no tokenizer/model load.
    (root / 'tokenizer.json').write_bytes(b'{}' + b' ' * (17_525_329 - 2))
    metadata = validator.read_json(root / 'checkpoint-metadata.json')
    metadata['tokenizer_identity'] = digest(root / 'tokenizer.json')
    (root / 'checkpoint-metadata.json').write_text(json.dumps(metadata))
    manifest = validator.read_json(root / 'checkpoint-manifest.json')
    manifest.update(metadata_sha256=canonical_hash(metadata), identity=checkpoint_identity(metadata))
    manifest['identity_sha256'] = canonical_hash(manifest['identity'])
    (root / 'checkpoint-manifest.json').write_text(json.dumps(manifest))
    export = validator.read_json(root / 'export.json')
    export.update(source_checkpoint_metadata_sha256=manifest['metadata_sha256'],
                  source_checkpoint_manifest_identity_sha256=manifest['identity_sha256'],
                  tokenizer_identity=metadata['tokenizer_identity'],
                  artifacts_sha256=artifact_hashes(root, validator.FILES[:-1]))
    (root / 'export.json').write_text(json.dumps(export))
    admitted = validator.preflight(root, expected_metadata=canonical_hash(metadata),
                                   expected_manifest=canonical_hash(manifest), expected_leaves=1)
    assert admitted['model_execution'] is False
    assert admitted['artifacts_sha256']['tokenizer.json'] == metadata['tokenizer_identity']


def test_tokenizer_new_bound_is_finite_and_inclusive(tmp_path):
    path = tmp_path / 'tokenizer.json'
    with path.open('wb') as stream:
        stream.write(b'{}')
        stream.truncate(validator.MAX_TOKENIZER_JSON_BYTES + 1)
    with pytest.raises(ValueError, match='Bounded regular'):
        validator.read_json(path)
    path.write_bytes(b'{}' + b' ' * (validator.MAX_TOKENIZER_JSON_BYTES - 2))
    assert validator.read_json(path) == {}


@pytest.mark.parametrize('name', ['config.json', 'checkpoint-manifest.json',
                                  'checkpoint-metadata.json', 'export.json', 'identity.json'])
def test_small_artifact_limits_remain_unchanged(tmp_path, name):
    path = tmp_path / name
    with path.open('wb') as stream:
        stream.write(b'{}')
        stream.truncate(validator.MAX_JSON_BYTES + 1)
    with pytest.raises(ValueError, match='Bounded regular'):
        validator.read_json(path)


@pytest.mark.parametrize('payload', [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_large_tokenizer_retains_strict_json_checks(tmp_path, payload):
    path = tmp_path / 'tokenizer.json'
    path.write_bytes(payload + b' ' * validator.MAX_JSON_BYTES)
    with pytest.raises(ValueError, match='Duplicate|Nonfinite'):
        validator.read_json(path)


def test_large_tokenizer_symlink_is_rejected(tmp_path):
    target = tmp_path / 'target.json'
    target.write_bytes(b'{}' + b' ' * validator.MAX_JSON_BYTES)
    path = tmp_path / 'tokenizer.json'
    path.symlink_to(target)
    with pytest.raises(ValueError, match='Bounded regular'):
        validator.read_json(path)
