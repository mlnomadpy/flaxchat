"""Durable budget reservations and a bounded Spot attempt lifecycle.

Cloud adapters must supply a server-verified expiry lease and verify actual
resource absence. Estimates and posted billing are deliberately separate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from filelock import FileLock


class RunLedger:
    def __init__(self, path, budget_usd):
        if not math.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError('Budget must be finite and positive')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = budget_usd
        self.lock = FileLock(str(self.path) + '.lock')

    def _read(self):
        if not self.path.exists():
            return {'budget_usd': self.budget, 'attempts': {}}
        state = json.loads(self.path.read_text())
        if state['budget_usd'] != self.budget:
            raise ValueError('Cannot silently change a run budget')
        return state

    def _write(self, state):
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        with temporary.open('w') as handle:
            json.dump(state, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def reserve(self, attempt_id, resource, hourly_usd, max_seconds, ancillary_reserve_usd=0.):
        if not resource or not attempt_id or not all(math.isfinite(x) and x >= 0 for x in
            (hourly_usd, max_seconds, ancillary_reserve_usd)) or max_seconds <= 0:
            raise ValueError('Invalid attempt reservation')
        estimate = hourly_usd * max_seconds / 3600 + ancillary_reserve_usd
        if estimate <= 0:
            raise ValueError('A positive reservation is required, including ancillary exposure for free compute')
        with self.lock:
            state = self._read()
            if attempt_id in state['attempts']:
                raise ValueError('Attempt identity already used')
            spent = sum(item['reservation_usd'] for item in state['attempts'].values())
            if spent + estimate > self.budget:
                raise ValueError('Total retry budget exhausted')
            state['attempts'][attempt_id] = {'resource': resource, 'reservation_usd': estimate,
                'whole_slice_hourly_usd': hourly_usd, 'max_seconds': max_seconds,
                'ancillary_reserve_usd': ancillary_reserve_usd, 'status': 'reserved',
                'events': [], 'posted_usage_usd': None, 'promotional_credits_usd': None}
            self._write(state)
        return estimate

    def event(self, attempt_id, event, **details):
        with self.lock:
            state = self._read()
            item = state['attempts'][attempt_id]
            item['events'].append({'event': event, 'time_unix': time.time(), **details})
            item['status'] = event
            self._write(state)

    def reconcile(self, attempt_id, *, usage_usd, promotional_credits_usd, billing_source):
        if not billing_source or not all(math.isfinite(x) and x >= 0 for x in
                                         (usage_usd, promotional_credits_usd)):
            raise ValueError('Posted amounts and billing evidence required')
        with self.lock:
            state = self._read()
            item = state['attempts'][attempt_id]
            item.update(posted_usage_usd=usage_usd, promotional_credits_usd=promotional_credits_usd,
                        billing_source=billing_source)
            # Reconciliation can increase obligations but never release a safety reserve.
            item['reservation_usd'] = max(item['reservation_usd'], usage_usd)
            self._write(state)

    def reconcile_completed_reservation(self, attempt_id, *, campaign, observation, apply=False):
        """Reduce unused reservation after independently verified terminal cleanup.

        This is an elapsed-time upper-bound estimate, never provider billing.
        Original events, posted amounts and the budget cap are not modified.
        Only terminal passed/failed TPU campaigns with complete cleanup evidence are eligible.
        """
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()

        def timestamp(value):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 < value <= time.time() + 5):
                raise ValueError('Valid completed timestamps required')
            return value

        terminal = ((campaign.get('status') == 'failed' and campaign.get('passed') is False)
                    or (campaign.get('status') == 'passed' and campaign.get('passed') is True))
        if (not terminal or campaign.get('cleanup_verified') is not True
                or campaign.get('cleanup_required') is not True):
            raise ValueError('Only consistent terminal campaigns with verified cleanup can be reconciled')
        started = timestamp(campaign.get('started_unix'))
        finished = timestamp(campaign.get('finished_unix'))
        if finished < started:
            raise ValueError('Campaign finished before it started')
        attempts = campaign.get('attempts')
        if not isinstance(attempts, list) or not attempts:
            raise ValueError('Completed campaign attempt evidence required')
        matched = []
        ids = set()
        for attempt in attempts:
            key = attempt.get('attempt_id')
            if not isinstance(key, str) or not key or key in ids:
                raise ValueError('Ambiguous campaign attempt identity')
            ids.add(key)
            cleanup = attempt.get('cleanup', {})
            if (cleanup.get('state') != 'resource_absent'
                    or cleanup.get('queue_absent') is not True
                    or cleanup.get('node_absent') is not True
                    or cleanup.get('queue_state') is not None
                    or cleanup.get('node_state') is not None):
                raise ValueError('Active or unverified campaign resource')
            observed = timestamp(cleanup.get('observed_unix'))
            cleaned = timestamp(cleanup.get('finished_unix'))
            if not started <= observed <= cleaned <= finished:
                raise ValueError('Contradictory cleanup timestamps')
            if key == attempt_id:
                matched.append(attempt)
        if len(matched) != 1:
            raise ValueError('Requested attempt is not uniquely identified in campaign')
        attempt = matched[0]
        project, zone, name = (campaign.get('project'), campaign.get('zone'), attempt.get('resource_name'))
        if any(not isinstance(x, str) or not x or '/' in x for x in (project, zone, name)):
            raise ValueError('Complete project, zone and resource identity required')
        resource = f'projects/{project}/locations/{zone}/queuedResources/{name}'
        scope = attempt.get('cleanup_scope', {}).get('resources')
        if scope is not None and scope != {
                'queuedResources': resource,
                'nodes': f'projects/{project}/locations/{zone}/nodes/{name}'}:
            raise ValueError('Cleanup scope contradicts campaign resource identity')
        expected_observation = dict(schema='flaxchat-independent-tpu-absence-v1',
            project=project, zone=zone, attempt_id=attempt_id, resource_name=name,
            queue_absent=True, node_absent=True)
        if any(observation.get(key) != value for key, value in expected_observation.items()):
            raise ValueError('Independent absence observation identity differs')
        if observation.get('queue_absent') is not True or observation.get('node_absent') is not True:
            raise ValueError('Explicit independent resource absence required')
        independent_time = timestamp(observation.get('observed_unix'))
        if independent_time < finished or not isinstance(observation.get('source'), str) or not observation['source']:
            raise ValueError('Independent observation must follow final campaign receipt')
        for key in ('queue_read_sha256', 'node_read_sha256'):
            value = observation.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Independently retained raw read evidence hashes required')
        receipt_hash, observation_hash = digest(campaign), digest(observation)
        with self.lock:
            state = self._read()
            item = state['attempts'][attempt_id]
            retry_terminal = item['status'] == 'preempted_retry_eligible'
            if item['resource'] != resource or item['status'] not in ('resource_absent', 'preempted_retry_eligible'):
                raise ValueError('Ledger resource identity or terminal state differs')
            events = item.get('events', [])
            if retry_terminal:
                tail = events[-3:]
                codes = (campaign.get('returncode'), attempt.get('returncode'),
                         tail[0].get('returncode') if tail else None)
                if (campaign.get('status') != 'failed' or campaign.get('passed') is not False
                        or any(type(code) is not int or code != 75 for code in codes)
                        or [event.get('event') for event in tail] !=
                        ['work_finished', 'resource_absent', 'preempted_retry_eligible']):
                    raise ValueError('Retry eligibility lacks finalized code75 cleanup evidence')
                times = [timestamp(event.get('time_unix')) for event in tail]
                if times != sorted(times) or times[1] < attempt['cleanup']['finished_unix']:
                    raise ValueError('Retry eligibility contradicts verified cleanup timing')
            if item.get('posted_usage_usd') is not None or item.get('promotional_credits_usd') is not None:
                raise ValueError('Use posted billing reconciliation for an already billed attempt')
            previous = item.get('completed_reservation_reconciliation')
            if previous is not None:
                if (previous['campaign_sha256'] != receipt_hash
                        or previous['independent_observation_sha256'] != observation_hash):
                    raise ValueError('Completed reservation already reconciled with different evidence')
                return previous
            rate, ancillary = item['whole_slice_hourly_usd'], item['ancillary_reserve_usd']
            original = item['reservation_usd']
            if item.get('original_reservation_usd', original) != original:
                raise ValueError('Original reservation differs without a completed reconciliation')
            if (not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                        for x in (rate, ancillary, original))
                    or original <= 0 or ancillary > original
                    or campaign.get('whole_slice_hourly_usd') != rate):
                raise ValueError('Original reservation and campaign pricing differ')
            if (not events or (not retry_terminal and events[-1].get('event') != 'resource_absent')
                    or any(not started <= timestamp(event.get('time_unix')) <= finished
                           for event in events)):
                raise ValueError('Ledger events contradict terminal campaign timing')
            # Include all campaign overhead through final verified cleanup. For
            # multi-attempt campaigns this deliberately overcounts earlier time.
            bound = rate * (finished - started) / 3600 + ancillary
            retained = min(original, math.ceil(bound * 100) / 100)
            record = dict(schema='flaxchat-terminal-reservation-bound-v1',
                attempt_id=attempt_id, resource=resource,
                original_reservation_usd=original, retained_reservation_usd=retained,
                released_reservation_usd=original - retained,
                whole_slice_hourly_usd=rate, ancillary_reserve_usd=ancillary,
                campaign_started_unix=started, cleanup_verified_unix=finished,
                independently_observed_unix=independent_time,
                campaign_sha256=receipt_hash, independent_observation_sha256=observation_hash,
                recorded_unix=time.time(), posted_usage_usd=None,
                accounting_basis='conservative_elapsed_time_bound_not_posted_charges')
            if apply:
                item['original_reservation_usd'] = original
                item['reservation_usd'] = retained
                item['completed_reservation_reconciliation'] = record
                self._write(state)
            return record

    def summary(self):
        with self.lock:
            state = self._read()
        attempts = list(state['attempts'].values())
        complete = bool(attempts) and all(a['posted_usage_usd'] is not None for a in attempts)
        return {**state, 'reserved_usd': sum(a['reservation_usd'] for a in attempts),
                'billing_complete': complete,
                'posted_usage_usd': sum(a['posted_usage_usd'] for a in attempts) if complete else None,
                'promotional_credits_usd': sum(a['promotional_credits_usd'] for a in attempts) if complete else None}


def run_spot_attempt(ledger, attempt_id, resource, *, hourly_usd, max_seconds,
                     arm_and_verify, provision, run, cleanup_and_verify,
                     ancillary_reserve_usd=0.):
    """One attempt; reserve persists even if the controller disappears.

    Each callback must bound its own I/O. arm_and_verify must return a receipt
    proving a live cloud lease covering this exact resource and time window.
    run returns 0 for completion or 75 for a confirmed retryable preemption;
    other errors must not be retried as capacity failures. Callers start a new
    uniquely named attempt with the same ledger and a durable resume cursor.
    """
    ledger.reserve(attempt_id, resource, hourly_usd, max_seconds, ancillary_reserve_usd)
    try:
        receipt = arm_and_verify(resource, max_seconds)
        if not receipt:
            raise RuntimeError('Missing verified cloud lease')
    except BaseException as exc:
        # Provision has not been called. Retain the reservation and diagnostic,
        # without deleting a resource whose guard was never verified.
        ledger.event(attempt_id, 'guard_failed_before_provisioning',
                     error=f'{type(exc).__name__}: {exc}')
        raise
    ledger.event(attempt_id, 'guard_verified', receipt=receipt)
    try:
        ledger.event(attempt_id, 'provisioning')
        provision(resource)
        ledger.event(attempt_id, 'running')
        code = run(resource)
        ledger.event(attempt_id, 'work_finished', returncode=code)
    except BaseException as exc:
        # Preserve the cause before potentially slow cloud deletion. The finally
        # block must still run if writing this diagnostic itself fails.
        ledger.event(attempt_id, 'attempt_failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if not cleanup_and_verify(resource):
                raise RuntimeError('Resource absence was not verified')
            ledger.event(attempt_id, 'resource_absent')
        except BaseException:
            ledger.event(attempt_id, 'cleanup_failed')
            raise
    return code
