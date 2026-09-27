"""Arm/verify a deployed Cloud Workflows expiry guard before TPU allocation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import yaml


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


def verify_cleanup_permissions(execution, *, project, location, zone, queue, deadline):
    """Fail closed on unknown effective IAM; never grant roles or enable APIs."""
    name = execution.get('name', '')
    resource_project = project
    if not name.startswith(f'projects/{project}/'):
        resource_project = subprocess.check_output(
            ['gcloud', 'projects', 'describe', project, '--format=value(projectNumber)'],
            text=True, timeout=60).strip()
        if not resource_project.isdigit():
            raise ValueError('Cannot resolve cleanup project identity')
    match = re.fullmatch(
        rf'projects/{re.escape(resource_project)}/locations/{re.escape(location)}'
        r'/workflows/([a-zA-Z0-9_-]+)/executions/[^/]+', name)
    if match is None:
        raise ValueError('Cleanup execution has an invalid resource identity')
    workflow = json.loads(subprocess.check_output(
        ['gcloud', 'workflows', 'describe', match[1], '--project', project,
         '--location', location, '--format=json'], text=True, timeout=60))
    revision = execution.get('workflowRevisionId')
    if not revision or workflow.get('revisionId') != revision:
        raise ValueError('Cleanup workflow revision changed; re-arm the guard')
    # Bind the observed checkpoint to our reviewed workflow semantics, not an
    # arbitrary workflow with a similarly named step. The TPU API itself checks
    # read/delete access; Policy Troubleshooter cannot resolve queued resources.
    source = Path(__file__).parents[1] / 'infra/tpu/cleanup_workflow.yaml'
    if yaml.safe_load(workflow.get('sourceContents', '')) != yaml.safe_load(source.read_text()):
        raise ValueError('Cleanup workflow source does not match reviewed permission probes')
    if not any(step.get('step') == 'wait_for_expiry' for step in execution.get('status', {}).get('currentSteps', [])):
        raise ValueError('Cleanup permission probes have not succeeded; allocation blocked')


def verify(receipt, *, project, zone, queue, now=None):
    """Read server-side execution state; a receipt alone is not authorization."""
    expected = receipt['lease']
    if any(expected[key] != value for key, value in
           dict(project=project, zone=zone, queue=queue).items()):
        raise ValueError('Guard resource identity mismatch')
    remaining = expected['deadline'] - (time.time() if now is None else now)
    if not 60 <= remaining <= 7200:
        raise ValueError('Guard is expired or too close to expiry')
    polling_deadline = time.monotonic() + 45
    while True:
        execution = json.loads(subprocess.check_output(
            ['gcloud', 'workflows', 'executions', 'describe', receipt['execution'],
             '--project', project, '--location', receipt['location'], '--format=json'],
            text=True, timeout=60))
        if json.loads(execution['argument']) != expected:
            raise ValueError('Cloud cleanup execution is not active for this exact lease')
        if execution['state'] != 'ACTIVE':
            detail = execution.get('error', {}).get('context', '')
            try:
                payload = json.loads(execution.get('error', {}).get('payload', '{}'))
                detail = payload.get('body', {}).get('error', {}).get('message', detail)
            except (ValueError, AttributeError):
                pass
            raise ValueError(
                f'Cloud cleanup execution is not active: {execution["state"]}; '
                f'allocation blocked before provisioning; {str(detail)[:2000]}'
            )
        if any(step.get('step') == 'wait_for_expiry' for step in execution.get('status', {}).get('currentSteps', [])):
            break
        if time.monotonic() >= polling_deadline:
            raise ValueError('Cleanup permission probes timed out; allocation blocked')
        time.sleep(1)
    verify_cleanup_permissions(execution, project=project, location=receipt['location'],
                               zone=zone, queue=queue, deadline=expected['deadline'])
    remaining = expected['deadline'] - (time.time() if now is None else now)
    if remaining < 60:
        raise ValueError('Guard is too close to expiry after permission verification')
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
