"""Arm/verify a deployed Cloud Workflows expiry guard before TPU allocation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def lease(project, zone, queue, seconds, *, now=None):
    if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
        raise ValueError('Invalid project')
    if not re.fullmatch(r'[a-z]+-[a-z0-9]+-[a-z]', zone):
        raise ValueError('Invalid zone')
    if not re.fullmatch(r'flaxchat-validation-[a-z0-9-]+', queue):
        raise ValueError('Only dedicated validation queues are permitted')
    if not 60 <= seconds <= 7200:
        raise ValueError('Guard duration must be 60–7200 seconds')
    return dict(project=project, zone=zone, queue=queue,
                deadline=(time.time() if now is None else now) + seconds)


def verify(receipt, *, project, zone, queue, now=None):
    """Read server-side execution state; a receipt alone is not authorization."""
    expected = receipt['lease']
    if any(expected[key] != value for key, value in
           dict(project=project, zone=zone, queue=queue).items()):
        raise ValueError('Guard resource identity mismatch')
    remaining = expected['deadline'] - (time.time() if now is None else now)
    if not 60 <= remaining <= 7200:
        raise ValueError('Guard is expired or too close to expiry')
    execution = json.loads(subprocess.check_output(
        ['gcloud', 'workflows', 'executions', 'describe', receipt['execution'],
         '--project', project, '--location', receipt['location'], '--format=json'],
        text=True, timeout=60))
    if execution['state'] != 'ACTIVE' or json.loads(execution['argument']) != expected:
        raise ValueError('Cloud cleanup execution is not active for this exact lease')
    return remaining


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--location', default='us-central1')
    parser.add_argument('--workflow', default='flaxchat-validation-cleanup')
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    if args.verify:
        receipt = json.loads(args.receipt.read_text())
    else:
        payload = lease(args.project, args.zone, args.queue, args.seconds)
        execution = json.loads(subprocess.check_output(
            ['gcloud', 'workflows', 'execute', args.workflow, '--project', args.project,
             '--location', args.location, '--data', json.dumps(payload), '--format=json'],
            text=True, timeout=60))
        receipt = {'lease': payload, 'execution': execution['name'], 'location': args.location}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    remaining = verify(receipt, project=args.project, zone=args.zone, queue=args.queue)
    print(json.dumps({'verified': True, 'seconds_remaining': remaining, **receipt}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
