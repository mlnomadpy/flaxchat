import json
import pytest
from scripts.gcp_cleanup_guard import lease, verify


def test_lease_is_bounded_and_resource_specific():
    assert lease('tpubuilders', 'us-east5-a', 'flaxchat-validation-test', 60, now=100)['deadline'] == 160


@pytest.mark.parametrize('changes', [dict(seconds=0), dict(seconds=7201), dict(queue='production'),
                                    dict(queue='flaxchat-validation-../other'), dict(project='../other'),
                                    dict(zone='us-east5-a/other')])
def test_rejects_unbounded_or_unowned_resources(changes):
    args = dict(project='tpubuilders', zone='us-east5-a', queue='flaxchat-validation-test', seconds=300)
    with pytest.raises(ValueError):
        lease(**(args | changes))


def test_receipt_requires_live_cloud_execution(monkeypatch):
    payload = lease('tpubuilders', 'us-east5-a', 'flaxchat-validation-test', 300, now=100)
    receipt = dict(lease=payload, execution='execution-id', location='us-central1')
    state = {'state': 'ACTIVE', 'argument': json.dumps(payload)}
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', lambda *a, **kw: json.dumps(state))
    assert verify(receipt, project='tpubuilders', zone='us-east5-a', queue='flaxchat-validation-test', now=110) == 290
    state['state'] = 'FAILED'
    with pytest.raises(ValueError, match='not active'):
        verify(receipt, project='tpubuilders', zone='us-east5-a', queue='flaxchat-validation-test', now=110)
    with pytest.raises(ValueError, match='identity'):
        verify(receipt, project='tpubuilders', zone='us-east5-a', queue='flaxchat-validation-other', now=110)


def test_guard_yaml_has_no_arbitrary_delete_url():
    import yaml
    from pathlib import Path
    workflow = yaml.safe_load(Path('infra/tpu/cleanup_workflow.yaml').read_text())
    text = json.dumps(workflow)
    assert 'GOOGLE_CLOUD_PROJECT_ID' in text
    assert 'flaxchat-validation-' in text
    assert 'queuedResources/' in text
    assert 'http.get' in text  # verify disappearance, not just HTTP acceptance
