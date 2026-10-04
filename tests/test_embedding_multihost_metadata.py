"""Worker admission metadata only; no JAX backend or model execution."""
import copy
import pytest
from scripts.validate_embedding_multihost import validate_ownership, compare_manifests


def records():
    return [{'rank': rank, 'backend': 'tpu', 'processes': 2, 'global_devices': 4,
             'local_devices': list(range(rank * 2, (rank + 1) * 2)),
             'rows': list(range(rank * 4, (rank + 1) * 4)), 'device_kind': 'TPU v5 lite',
             'source_sha256': 'source', 'worker_sha256': 'worker', 'runtime_sha256': 'runtime',
             'campaign_output': 'gs://bucket/fresh'}
            for rank in range(2)]


def test_actual_rank_inventory_disjoint_complete_and_consistent():
    report = validate_ownership(records(), devices=4, processes=2, batch=8)
    assert report['passed'] and report['physical_processes'] == 2


@pytest.mark.parametrize('field,value', [
    ('rank', 0), ('backend', 'cpu'), ('processes', 1), ('global_devices', 8),
    ('local_devices', [0, 1]), ('rows', [0, 1, 2, 3]),
    ('source_sha256', 'other'), ('worker_sha256', 'other'), ('runtime_sha256', 'other'),
    ('campaign_output', 'gs://bucket/conflicting')])
def test_invalid_ownership_and_source_consensus_refused(field, value):
    inventory = copy.deepcopy(records())
    inventory[1][field] = value
    with pytest.raises(ValueError):
        validate_ownership(inventory, devices=4, processes=2, batch=8)


def test_single_host_and_missing_rank_never_qualify_multihost():
    with pytest.raises(ValueError):
        validate_ownership(records()[:1], devices=4, processes=2, batch=8)
    with pytest.raises(ValueError):
        validate_ownership(records(), devices=4, processes=1, batch=8)


def test_exact_checkpoint_comparison_requires_all_three_persistent_trees():
    reference = {key: {'leaf': {'sha256': key}} for key in ('model_state', 'optimizer_state', 'training_state')}
    assert all(compare_manifests(reference, copy.deepcopy(reference)).values())
    for key in reference:
        changed = copy.deepcopy(reference)
        changed[key]['leaf']['sha256'] = 'different'
        with pytest.raises(ValueError):
            compare_manifests(reference, changed)
    with pytest.raises(ValueError):
        compare_manifests({}, {})
