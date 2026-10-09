"""Reconcile only unused reservations after completed, independently observed TPU cleanup.

Defaults to dry-run. No cloud calls, budget amendments or provider-charge claims.
The independent observation must identify this exact attempt and retain hashes
of separate queue/node read evidence; it must postdate the final campaign receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flaxchat.operations import RunLedger


def pinned_json(path, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Bounded regular JSON evidence file required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Evidence file does not match independently retained SHA256')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Evidence JSON object required')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, required=True,
                        help='Existing ledger cap; this command cannot change it')
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--campaign-receipt', type=Path, required=True)
    parser.add_argument('--campaign-sha256', required=True, help='SHA256 of raw receipt file')
    parser.add_argument('--independent-observation', type=Path, required=True)
    parser.add_argument('--observation-sha256', required=True, help='SHA256 of raw observation file')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if not args.ledger.is_file() or args.ledger.is_symlink():
        raise ValueError('Existing regular ledger required')
    receipt = pinned_json(args.campaign_receipt, args.campaign_sha256)
    observation = pinned_json(args.independent_observation, args.observation_sha256)
    result = RunLedger(args.ledger, args.budget_usd).reconcile_completed_reservation(
        args.attempt_id, campaign=receipt, observation=observation, apply=args.apply)
    print(json.dumps({'applied': args.apply, 'reconciliation': result}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
