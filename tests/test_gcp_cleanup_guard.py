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
    state = {'state': 'ACTIVE', 'argument': json.dumps(payload), 'status': {'currentSteps': [{'step':'wait_for_expiry'}]}}
    monkeypatch.setattr('scripts.gcp_cleanup_guard.verify_cleanup_permissions', lambda *a, **kw: None)
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', lambda *a, **kw: json.dumps(state))
    assert verify(receipt, project='tpubuilders', zone='us-east5-a', queue='flaxchat-validation-test', now=110) == 290
    state['state'] = 'FAILED'
    state['error'] = {'context': 'HTTP 403', 'payload': json.dumps({
        'body': {'error': {'message': "Permission 'tpu.nodes.get' denied"}}})}
    with pytest.raises(ValueError, match='not active'):
        verify(receipt, project='tpubuilders', zone='us-east5-a', queue='flaxchat-validation-test', now=110)
    with pytest.raises(ValueError, match="tpu.nodes.get.*denied"):
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



def test_permission_proof_requires_reviewed_workflow_and_probe_completion(monkeypatch):
    from pathlib import Path
    from scripts.gcp_cleanup_guard import verify_cleanup_permissions
    execution = dict(name='projects/tpubuilders/locations/us-central1/workflows/cleanup/executions/e1',
                     workflowRevisionId='r1', status={'currentSteps':[{'step':'wait_for_expiry'}]})
    workflow = dict(revisionId='r1', sourceContents=Path('infra/tpu/cleanup_workflow.yaml').read_text())
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', lambda *a, **kw: json.dumps(workflow))
    def check():
        verify_cleanup_permissions(execution, project='tpubuilders', location='us-central1',
                                   zone='us-west4-a', queue='flaxchat-validation-test', deadline=300)
    check()
    execution['status']['currentSteps'] = [{'step':'probe_delete'}]
    with pytest.raises(ValueError, match='allocation blocked'):
        check()
    execution['status']['currentSteps'] = [{'step':'wait_for_expiry'}]
    workflow['sourceContents'] = 'main: {}'
    with pytest.raises(ValueError, match='source does not match'):
        check()
    workflow['revisionId'] = 'r2'
    with pytest.raises(ValueError, match='revision changed'):
        check()


def test_workflow_permission_probe_cannot_delete_existing_resource():
    from pathlib import Path
    import yaml
    steps = yaml.safe_load(Path('infra/tpu/cleanup_workflow.yaml').read_text())['main']['steps']
    names = [next(iter(step)) for step in steps]
    assert names.index('probe_read') + 1 == names.index('refuse_existing')
    assert names.index('probe_delete') + 1 == names.index('refuse_race')
    read = next(step['probe_read'] for step in steps if 'probe_read' in step)
    check = read['except']['steps'][0]['read_absent']['switch'][0]
    assert '== 404' in check['condition'] and check['next'] == 'probe_delete'
    delete = next(step['probe_delete'] for step in steps if 'probe_delete' in step)
    check = delete['except']['steps'][0]['delete_absent']['switch'][0]
    assert '== 404' in check['condition'] and check['next'] == 'wait_for_expiry'


def test_canonical_project_number_is_resolved_not_blindly_trusted(monkeypatch):
    from pathlib import Path
    from scripts.gcp_cleanup_guard import verify_cleanup_permissions
    execution = dict(name='projects/540509768325/locations/us-central1/workflows/cleanup/executions/e1',
                     workflowRevisionId='r1', status={'currentSteps':[{'step':'wait_for_expiry'}]})
    def output(cmd, **kw):
        if cmd[1] == 'projects':
            return '540509768325\n'
        return json.dumps(dict(revisionId='r1', sourceContents=Path('infra/tpu/cleanup_workflow.yaml').read_text()))
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', output)
    kwargs = dict(project='tpubuilders', location='us-central1', zone='us-west4-a', queue='flaxchat-validation-test', deadline=300)
    verify_cleanup_permissions(execution, **kwargs)
    execution['name'] = execution['name'].replace('540509768325', '123')
    with pytest.raises(ValueError, match='identity'):
        verify_cleanup_permissions(execution, **kwargs)
