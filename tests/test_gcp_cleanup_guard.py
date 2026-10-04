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


SA = 'cleanup@azettaai.iam.gserviceaccount.com'
ROLE = 'projects/azettaai/roles/flaxchatValidationCleanup'
PREFIX = 'projects/azettaai/locations/us-west4-a/'


def scope_inputs():
    condition = " || ".join(f"resource.name.startsWith('{PREFIX}{kind}/flaxchat-validation-yat-embed-torch-')"
                           for kind in ('queuedResources', 'nodes'))
    policy = {'bindings': [{'role': ROLE, 'members': ['serviceAccount:' + SA],
                           'condition': {'expression': condition}}]}
    # Actual deployed custom-role permission inventory. TPU queued resources
    # use nodes permissions, while IAM CEL checks distinct resource names.
    roles = {ROLE: {'includedPermissions': ['tpu.nodes.get', 'tpu.nodes.delete']}}
    return policy, roles


def check_scope(policy, roles, **changes):
    from scripts.gcp_cleanup_guard import validate_cleanup_scope
    return validate_cleanup_scope(policy, roles, **(dict(service_account=SA, project='azettaai',
        zone='us-west4-a', queue='flaxchat-validation-yat-embed-torch-test') | changes))


def test_scope_accepts_exact_supported_prefix_but_is_not_effective_iam():
    policy, roles = scope_inputs()
    receipt = check_scope(policy, roles)
    assert receipt['supported_project_allow_coverage']
    assert not receipt['effective_iam_verified']
    assert receipt['server_permission_probes_required']
    assert len(receipt['policy_sha256']) == 64
    assert len(receipt['roles_sha256']) == 64


@pytest.mark.parametrize('changes', [dict(zone='us-west1-c'),
    dict(queue='flaxchat-validation-yat-cache-west1-test'), dict(project='other-project'),
    dict(service_account='different@azettaai.iam.gserviceaccount.com')])
def test_scope_rejects_actual_failed_region_and_prefix_before_execution(changes):
    with pytest.raises(ValueError, match='exact zone/name'):
        check_scope(*scope_inputs(), **changes)


@pytest.mark.parametrize('expression', ["true", "resource.name.contains('yat')",
    "resource.name.startsWith('projects/') && true", "request.time < timestamp('2027-01-01')",
    "resource.name.startsWith('projects/') || true"])
def test_scope_unknown_cel_never_becomes_allow(expression):
    policy, roles = scope_inputs()
    policy['bindings'][0]['condition']['expression'] = expression
    with pytest.raises(ValueError, match='unsupported conditional'):
        check_scope(policy, roles)


def test_scope_requires_node_and_queue_read_and_delete_permissions():
    policy, roles = scope_inputs()
    roles[ROLE]['includedPermissions'].remove('tpu.nodes.delete')
    with pytest.raises(ValueError, match='tpu.nodes.delete'):
        check_scope(policy, roles)
    policy, roles = scope_inputs()
    roles[ROLE]['deleted'] = True
    with pytest.raises(ValueError):
        check_scope(policy, roles)


def test_queue_condition_requires_distinct_resource_coverage_with_nodes_permissions():
    policy, roles = scope_inputs()
    policy['bindings'][0]['condition']['expression'] = (
        f"resource.name.startsWith('{PREFIX}nodes/flaxchat-validation-yat-embed-torch-')")
    with pytest.raises(ValueError, match='queuedResources:tpu.nodes'):
        check_scope(policy, roles)
    policy['bindings'][0]['condition']['expression'] = (
        f"resource.name.startsWith('{PREFIX}queuedResources/flaxchat-validation-yat-embed-torch-')")
    with pytest.raises(ValueError, match='nodes:tpu.nodes'):
        check_scope(policy, roles)


def test_scope_readonly_fetch_binds_policy_roles_and_reviewed_workflow(monkeypatch):
    from pathlib import Path
    from scripts.gcp_cleanup_guard import preflight_cleanup_scope
    policy, roles = scope_inputs()
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        if command[1] == 'workflows':
            return json.dumps({'serviceAccount': SA, 'revisionId': 'r1',
                'sourceContents': Path('infra/tpu/cleanup_workflow.yaml').read_text()})
        if command[1] == 'projects':
            return json.dumps(policy)
        return json.dumps(roles[ROLE])
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', output)
    receipt = preflight_cleanup_scope(project='azettaai', zone='us-west4-a',
        queue='flaxchat-validation-yat-embed-torch-test')
    assert receipt['workflow_revision'] == 'r1'
    assert len(calls) == 3
    assert all('execute' not in call and 'set-iam-policy' not in call for call in calls)


@pytest.mark.parametrize('failure', ['timeout', 'reset'])
def test_scope_read_retries_only_bounded_transient_transport(monkeypatch, failure):
    import subprocess
    from pathlib import Path
    from scripts.gcp_cleanup_guard import preflight_cleanup_scope
    policy, roles = scope_inputs()
    calls = []
    def output(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            raise subprocess.CalledProcessError(1, command, stderr='Connection reset by peer')
        if command[1] == 'workflows':
            return json.dumps({'serviceAccount': SA, 'revisionId': 'r1',
                'sourceContents': Path('infra/tpu/cleanup_workflow.yaml').read_text()})
        return json.dumps(policy if command[1] == 'projects' else roles[ROLE])
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', output)
    receipt = preflight_cleanup_scope(project='azettaai', zone='us-west4-a',
        queue='flaxchat-validation-yat-embed-torch-test')
    assert len(calls) == 4
    assert all(0 < kwargs['timeout'] <= 20 for _, kwargs in calls)
    assert receipt['discovery']['reads'][0]['status'] == 'transient_failure'
    assert receipt['discovery']['reads'][1]['status'] == 'succeeded'


@pytest.mark.parametrize('message', ["Permission denied", "HTTP 403 service unavailable",
    "Unauthenticated connection reset", "Invalid argument", "Credentials timed out"])
def test_scope_denied_or_unknown_failure_never_retries(monkeypatch, message):
    import subprocess
    from scripts.gcp_cleanup_guard import preflight_cleanup_scope
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr=message)
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', output)
    with pytest.raises(subprocess.CalledProcessError) as captured:
        preflight_cleanup_scope(project='azettaai', zone='us-west4-a',
            queue='flaxchat-validation-yat-embed-torch-test')
    assert len(calls) == 1
    assert captured.value.cleanup_scope_discovery['reads'][0]['status'] == 'terminal_failure'


def test_scope_timeout_retry_exhaustion_preserves_both_observations(monkeypatch):
    import subprocess
    from scripts.gcp_cleanup_guard import preflight_cleanup_scope
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', output)
    with pytest.raises(subprocess.TimeoutExpired) as captured:
        preflight_cleanup_scope(project='azettaai', zone='us-west4-a',
            queue='flaxchat-validation-yat-embed-torch-test')
    assert len(calls) == 2
    assert len(captured.value.cleanup_scope_discovery['reads']) == 2


def test_scope_total_deadline_prevents_further_commands(monkeypatch):
    from unittest.mock import Mock
    from scripts.gcp_cleanup_guard import preflight_cleanup_scope
    clock = iter([0, 121])
    monkeypatch.setattr('scripts.gcp_cleanup_guard.time.monotonic', lambda: next(clock))
    command = Mock()
    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', command)
    with pytest.raises(TimeoutError, match='total deadline') as captured:
        preflight_cleanup_scope(project='azettaai', zone='us-west4-a',
            queue='flaxchat-validation-yat-embed-torch-test')
    command.assert_not_called()
    assert captured.value.cleanup_scope_discovery['reads'] == []
