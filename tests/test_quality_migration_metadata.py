import copy
import json

import pytest

from flaxchat.embedding_recovery import (admit_quality_migration, migration_restore_identity,
    validate_resume_evaluation)
from flaxchat.embedding_quality import quality_gate


def fixture(tmp_path):
    old = {'resolved_config': {'steps': 20000, 'warmup': 100, 'learning_rate': 1e-5,
        'development_policy': {'eval_every': 100}, 'source_receipt': {'sha256': 'a' * 64},
        'data_manifests': {'source': 'c' * 64}}, 'tokenizer': 'd' * 64,
        'data_manifest': 'e' * 64, 'source_python_sha256': 'a' * 64}
    expected = copy.deepcopy(old)
    expected['source_python_sha256'] = 'b' * 64
    expected['resolved_config'].update(steps=60000, schedule_steps=20000, source_receipt={'sha256': 'b' * 64})
    expected['resolved_config']['development_policy']['regression_action'] = 'report-only'
    spec = {'format': 'flaxchat-quality-policy-migration-v1', 'checkpoint_prefix': 'gs://bucket/stage',
        'target_source_python_sha256': 'b' * 64, 'previous_identity': old,
        'checkpoint': {'step': 500, 'manifest_sha256': 'f' * 64},
        'best': {'step': 0, 'manifest_sha256': '1' * 64}}
    path = tmp_path / 'migration.json'
    path.write_text(json.dumps(spec))
    return spec, expected, path


def test_only_explicit_policy_and_preserved_schedule_migrate(tmp_path):
    spec, expected, path = fixture(tmp_path)
    admitted, lineage = admit_quality_migration(path, expected, spec['checkpoint_prefix'])
    assert admitted == spec and lineage['preserved_schedule_steps'] == 20000
    assert lineage['source_checkpoint']['step'] == 500
    old = spec['previous_identity']
    metadata = {'step': 500, 'committed_receipt': spec['checkpoint'],
        'resolved_config': old['resolved_config'], 'tokenizer_identity': old['tokenizer'],
        'data_manifest_identity': old['data_manifest'], 'source_python_sha256': old['source_python_sha256']}
    assert migration_restore_identity(metadata, expected, spec) == old
    for field, value in [('step', 499), ('committed_receipt', spec['best'])]:
        with pytest.raises(ValueError, match='pinned'):
            migration_restore_identity({**metadata, field: value}, expected, spec)
    with pytest.raises(ValueError, match='differs'):
        migration_restore_identity(metadata, expected, None)


@pytest.mark.parametrize('field,value', [('learning_rate', 2e-5), ('warmup', 101),
    ('schedule_steps', 60000), ('steps', 10000), ('data_manifests', {'source': 'changed'})])
def test_migration_refuses_optimizer_schedule_and_data_changes(tmp_path, field, value):
    spec, expected, path = fixture(tmp_path)
    expected['resolved_config'][field] = value
    with pytest.raises(ValueError):
        admit_quality_migration(path, expected, spec['checkpoint_prefix'])


def test_report_only_preserves_failure_and_authentication():
    baseline = {'source': {'mrr': .5, 'recall_at_1': .4, 'recall_at_10': .8}}
    metrics = {'source': {'mrr': .1, 'recall_at_1': .1, 'recall_at_10': .2}}
    record = {'step': 500, 'metrics': metrics, 'gate': quality_gate(metrics, baseline, max_regression=.02)}
    quality = {'baseline': baseline, 'last_evaluation': record}
    assert validate_resume_evaluation(quality, cursor=500, horizon=60000, every=100,
        max_regression=.02, regression_action='report-only')['gate']['passed'] is False
    with pytest.raises(ValueError, match='blocked'):
        validate_resume_evaluation(quality, cursor=500, horizon=60000, every=100, max_regression=.02)
    record['gate']['passed'] = True
    with pytest.raises(ValueError, match='disagrees'):
        validate_resume_evaluation(quality, cursor=500, horizon=60000, every=100,
            max_regression=.02, regression_action='report-only')
