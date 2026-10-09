"""Ledger-only fixtures: no cloud operations or numerical/model execution."""
import copy
import json

import pytest

from flaxchat.operations import RunLedger


def evidence(tmp_path):
    ledger = RunLedger(tmp_path / 'ledger.json', 900)
    resource = 'projects/p/locations/z/queuedResources/tpu'
    ledger.reserve('a:0', resource, 9.6, 45000, 10)
    state = ledger._read()
    state['attempts']['a:0'].update(status='resource_absent', events=[
        {'event': 'provisioning', 'time_unix': 1010},
        {'event': 'resource_absent', 'time_unix': 1602}])
    ledger._write(state)
    campaign = dict(status='failed', passed=False, cleanup_verified=True, cleanup_required=True,
        started_unix=1000, finished_unix=1603, project='p', zone='z', whole_slice_hourly_usd=9.6,
        attempts=[dict(attempt_id='a:0', resource_name='tpu', cleanup=dict(
            state='resource_absent', queue_absent=True, node_absent=True,
            queue_state=None, node_state=None, observed_unix=1601, finished_unix=1602))])
    observation = dict(schema='flaxchat-independent-tpu-absence-v1', project='p', zone='z',
        attempt_id='a:0', resource_name='tpu', queue_absent=True, node_absent=True,
        observed_unix=1700, source='independent gcloud queue/node inventory',
        queue_read_sha256='a'*64, node_read_sha256='b'*64)
    return ledger, campaign, observation


@pytest.mark.parametrize('status,passed', [('failed', False), ('passed', True)])
def test_terminal_bound_preserves_events_billing_cap_and_original(tmp_path, status, passed):
    ledger, campaign, observation = evidence(tmp_path)
    campaign.update(status=status, passed=passed)
    before = ledger._read()
    kwargs = dict(campaign=campaign, observation=observation)
    dry = ledger.reconcile_completed_reservation('a:0', **kwargs)
    assert ledger._read() == before
    assert dry['retained_reservation_usd'] == 11.61
    result = ledger.reconcile_completed_reservation('a:0', **kwargs, apply=True)
    after = ledger._read()
    item = after['attempts']['a:0']
    assert after['budget_usd'] == 900
    assert item['original_reservation_usd'] == 130
    assert item['reservation_usd'] == 11.61
    assert item['events'] == before['attempts']['a:0']['events']
    assert item['posted_usage_usd'] is item['promotional_credits_usd'] is None
    assert ledger.reconcile_completed_reservation('a:0', **kwargs, apply=True) == result
    assert ledger._read() == after
    assert ledger.summary()['posted_usage_usd'] is None
    assert ledger.summary()['billing_complete'] is False


@pytest.mark.parametrize('change', [
    lambda c, o: c.update(status='running'),
    lambda c, o: c.update(cleanup_verified=False),
    lambda c, o: c.pop('finished_unix'),
    lambda c, o: c.update(finished_unix=float('nan')),
    lambda c, o: c.update(whole_slice_hourly_usd=1),
    lambda c, o: c['attempts'][0]['cleanup'].update(node_absent=False),
    lambda c, o: c['attempts'][0]['cleanup'].update(queue_state='ACTIVE'),
    lambda c, o: c['attempts'][0].update(attempt_id='wrong'),
    lambda c, o: o.update(observed_unix=1500),
    lambda c, o: o.update(resource_name='other'),
    lambda c, o: o.update(node_absent=1),
    lambda c, o: o.pop('queue_read_sha256'),
])
def test_incomplete_or_contradictory_evidence_cannot_release(tmp_path, change):
    ledger, campaign, observation = evidence(tmp_path)
    before = ledger._read()
    change(campaign, observation)
    with pytest.raises(ValueError):
        ledger.reconcile_completed_reservation('a:0', campaign=campaign, observation=observation, apply=True)
    assert ledger._read() == before


def test_active_second_attempt_blocks_release(tmp_path):
    ledger, campaign, observation = evidence(tmp_path)
    active = copy.deepcopy(campaign['attempts'][0])
    active.update(attempt_id='a:1', resource_name='active')
    active['cleanup']['state'] = 'unknown'
    campaign['attempts'].append(active)
    with pytest.raises(ValueError, match='Active'):
        ledger.reconcile_completed_reservation('a:0', campaign=campaign, observation=observation, apply=True)


def test_long_elapsed_time_never_releases_more_than_original(tmp_path):
    ledger, campaign, observation = evidence(tmp_path)
    campaign['started_unix'] = 1
    campaign['finished_unix'] = 100000
    observation['observed_unix'] = 100001
    result = ledger.reconcile_completed_reservation('a:0', campaign=campaign, observation=observation)
    assert result['retained_reservation_usd'] == 130
    assert result['released_reservation_usd'] == 0


def test_evidence_file_pins_are_required(tmp_path):
    from scripts.reconcile_completed_run import pinned_json
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps({'identity': 'pinned'}))
    with pytest.raises(ValueError, match='SHA256'):
        pinned_json(path, '0'*64)


def interrupted_evidence(tmp_path):
    ledger, campaign, observation = evidence(tmp_path)
    campaign['returncode'] = 75
    campaign['attempts'][0]['returncode'] = 75
    state = ledger._read()
    state['attempts']['a:0'].update(status='preempted_retry_eligible', events=[
        {'event': 'work_finished', 'returncode': 75, 'time_unix': 1600},
        {'event': 'resource_absent', 'time_unix': 1602},
        {'event': 'preempted_retry_eligible', 'time_unix': 1603}])
    ledger._write(state)
    return ledger, campaign, observation


def test_final_interrupted_campaign_reconciles_without_rewriting_history(tmp_path):
    ledger, campaign, observation = interrupted_evidence(tmp_path)
    before = ledger._read()
    result = ledger.reconcile_completed_reservation('a:0', campaign=campaign,
        observation=observation, apply=True)
    item = ledger._read()['attempts']['a:0']
    assert result['retained_reservation_usd'] == 11.61
    assert item['events'] == before['attempts']['a:0']['events']
    assert item['status'] == 'preempted_retry_eligible'
    assert ledger.reconcile_completed_reservation('a:0', campaign=campaign,
        observation=observation, apply=True) == result


@pytest.mark.parametrize('mutation', ['campaign_code', 'attempt_code', 'work_code',
    'missing_absence', 'extra_event', 'late_event', 'early_absence', 'unordered', 'nonterminal'])
def test_interrupted_reconciliation_rejects_contradictory_lifecycle(tmp_path, mutation):
    ledger, campaign, observation = interrupted_evidence(tmp_path)
    state = ledger._read()
    events = state['attempts']['a:0']['events']
    if mutation == 'campaign_code':
        campaign['returncode'] = 0
    elif mutation == 'attempt_code':
        campaign['attempts'][0]['returncode'] = 0
    elif mutation == 'work_code':
        events[0]['returncode'] = 0
    elif mutation == 'missing_absence':
        events.pop(1)
    elif mutation == 'extra_event':
        events.append({'event': 'running', 'time_unix': 1603})
    elif mutation == 'late_event':
        events[-1]['time_unix'] = 1604
    elif mutation == 'early_absence':
        events[1]['time_unix'] = 1601
    elif mutation == 'unordered':
        events[0]['time_unix'] = 1603
    else:
        campaign['status'] = 'running'
    ledger._write(state)
    before = ledger._read()
    with pytest.raises(ValueError):
        ledger.reconcile_completed_reservation('a:0', campaign=campaign,
            observation=observation, apply=True)
    assert ledger._read() == before


@pytest.mark.parametrize('status,passed', [
    ('failed', True), ('passed', False), ('passed', 1), ('failed', 0),
    ('passed', None), ('running', True), ('running', False),
    ('stopped', True), ('completed', True), ('cleanup', False),
])
def test_inconsistent_or_nonterminal_campaign_never_releases(tmp_path, status, passed):
    ledger, campaign, observation = evidence(tmp_path)
    campaign.update(status=status, passed=passed)
    before = ledger._read()
    with pytest.raises(ValueError, match='consistent terminal'):
        ledger.reconcile_completed_reservation('a:0', campaign=campaign,
            observation=observation, apply=True)
    assert ledger._read() == before
