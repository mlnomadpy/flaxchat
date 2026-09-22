"""Durable budget reservations and a bounded Spot attempt lifecycle.

Cloud adapters must supply a server-verified expiry lease and verify actual
resource absence. Estimates and posted billing are deliberately separate.
"""
from __future__ import annotations

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
            (hourly_usd, max_seconds, ancillary_reserve_usd)) or min(hourly_usd, max_seconds) <= 0:
            raise ValueError('Invalid attempt reservation')
        estimate = hourly_usd * max_seconds / 3600 + ancillary_reserve_usd
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
    receipt = arm_and_verify(resource, max_seconds)
    if not receipt:
        raise RuntimeError('Missing verified cloud lease')
    ledger.event(attempt_id, 'guard_verified', receipt=receipt)
    try:
        ledger.event(attempt_id, 'provisioning')
        provision(resource)
        ledger.event(attempt_id, 'running')
        code = run(resource)
        ledger.event(attempt_id, 'work_finished', returncode=code)
    finally:
        try:
            if not cleanup_and_verify(resource):
                raise RuntimeError('Resource absence was not verified')
            ledger.event(attempt_id, 'resource_absent')
        except BaseException:
            ledger.event(attempt_id, 'cleanup_failed')
            raise
    return code
