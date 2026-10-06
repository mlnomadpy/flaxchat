"""Objective-state exclusion contracts only; no numerical model execution."""
import copy
from types import SimpleNamespace

import pytest

from flaxchat.embedding_stage import contrastive_objective_identity
from scripts.release_contract import encoder_export_manifest


def fixture():
    metadata = {'resolved_config': {'contrastive_objective': contrastive_objective_identity(
        SimpleNamespace(contrastive_similarity='yat', yat_infonce_alpha_init=.01))}}
    manifest = {'model_state': {
        "['weight']": {'shape': [2, 2], 'dtype': 'float32', 'sha256': 'a' * 64},
        "['contrastive_raw_alpha']": {'shape': [], 'dtype': 'float32', 'sha256': 'b' * 64},
    }}
    return metadata, manifest


def test_serving_excludes_only_declared_alpha_preserving_full_manifest():
    metadata, manifest = fixture()
    original = copy.deepcopy(manifest)
    serving, excluded = encoder_export_manifest(metadata, manifest)
    assert set(serving) == {"['weight']"}
    assert excluded[0]['record'] == manifest['model_state']["['contrastive_raw_alpha']"]
    assert manifest == original
    manifest['model_state']["['unknown_extra']"] = {'shape': []}
    serving, _ = encoder_export_manifest(metadata, manifest)
    assert "['unknown_extra']" in serving


def test_cosine_exports_remain_unchanged_but_cannot_hide_alpha():
    _, manifest = fixture()
    with pytest.raises(ValueError, match='Undeclared'):
        encoder_export_manifest({}, manifest)
    del manifest['model_state']["['contrastive_raw_alpha']"]
    assert encoder_export_manifest({}, manifest) == (manifest['model_state'], [])


@pytest.mark.parametrize('mutation', ['extra_objective_field', 'wrong_bias', 'missing_alpha', 'wrong_shape', 'wrong_dtype', 'wrong_hash', 'duplicate_alpha'])
def test_fail_closed_on_bad_objective_or_alpha(mutation):
    metadata, manifest = fixture()
    objective = metadata['resolved_config']['contrastive_objective']
    record = manifest['model_state']["['contrastive_raw_alpha']"]
    if mutation == 'extra_objective_field':
        objective['extra'] = True
    elif mutation == 'wrong_bias':
        objective['bias'] = 2
    elif mutation == 'missing_alpha':
        del manifest['model_state']["['contrastive_raw_alpha']"]
    elif mutation == 'wrong_shape':
        record['shape'] = [1]
    elif mutation == 'wrong_dtype':
        record['dtype'] = 'bfloat16'
    elif mutation == 'wrong_hash':
        record['sha256'] = 'bad'
    elif mutation == 'duplicate_alpha':
        manifest['model_state']['.contrastive_raw_alpha'] = dict(record)
    with pytest.raises(ValueError):
        encoder_export_manifest(metadata, manifest)


def test_authenticated_yat_export_and_parent_admission(tmp_path):
    import json
    from scripts.release_contract import artifact_hashes, canonical_hash, checkpoint_identity, validate_export
    from scripts.validate_production_parent_tpu import preflight
    from tests.test_production_parent_tpu_metadata import export_fixture

    tmp_path = tmp_path / "export"
    export_fixture(tmp_path)
    metadata_path = tmp_path / 'checkpoint-metadata.json'
    manifest_path = tmp_path / 'checkpoint-manifest.json'
    export_path = tmp_path / 'export.json'
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    export = json.loads(export_path.read_text())
    objective, extra = fixture()
    metadata['resolved_config']['contrastive_objective'] = objective['resolved_config']['contrastive_objective']
    manifest['model_state']["['contrastive_raw_alpha']"] = extra['model_state']["['contrastive_raw_alpha']"]
    manifest['metadata_sha256'] = canonical_hash(metadata)
    manifest['identity'] = checkpoint_identity(metadata)
    manifest['identity_sha256'] = canonical_hash(manifest['identity'])
    metadata_path.write_text(json.dumps(metadata))
    manifest_path.write_text(json.dumps(manifest))
    _, exclusions = encoder_export_manifest(metadata, manifest)
    export.update(source_checkpoint_metadata_sha256=manifest['metadata_sha256'],
                  source_checkpoint_manifest_identity_sha256=manifest['identity_sha256'],
                  excluded_training_only_tensors=exclusions)
    export['artifacts_sha256'] = artifact_hashes(tmp_path, (
        'model.safetensors', 'config.json', 'tokenizer.json', 'checkpoint-metadata.json', 'checkpoint-manifest.json'))
    export_path.write_text(json.dumps(export))
    validate_export(tmp_path, export)
    admitted = preflight(tmp_path, expected_step=14000, expected_leaves=1,
                         expected_metadata=manifest['metadata_sha256'], expected_manifest=canonical_hash(manifest))
    assert len(admitted['model_state']) == 1
    assert admitted['excluded_training_only_tensors'] == exclusions
    export.pop('excluded_training_only_tensors')
    with pytest.raises(ValueError, match='exclusions'):
        validate_export(tmp_path, export)
