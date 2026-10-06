"""Arm/verify a deployed Cloud Workflows expiry guard before TPU allocation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import yaml


def _prefix_condition(expression, resource):
    """Evaluate only the reviewed prefix-disjunction subset, never arbitrary CEL."""
    if not isinstance(expression, str) or len(expression) > 16384:
        raise ValueError('Unsupported cleanup IAM condition')
    terms = expression.split('||')
    prefixes = []
    for term in terms:
        match = re.fullmatch(r"\s*resource\.name\.startsWith\('([A-Za-z0-9/_-]+)'\)\s*", term)
        if match is None:
            raise ValueError('Unsupported cleanup IAM condition; live scope cannot be inferred')
        prefixes.append(match[1])
    return any(resource.startswith(prefix) for prefix in prefixes)


def validate_cleanup_scope(policy, roles, *, service_account, project, zone, queue):
    """Require explicit project allow bindings for both resources; not effective IAM.

    Inherited grants, deny policies and unsupported conditions are not evaluated.
    Live workflow probes remain mandatory even after this model-free admission.
    """
    lease(project, zone, queue, 60)
    if not isinstance(service_account, str) or not re.fullmatch(
            r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.iam\.gserviceaccount\.com', service_account):
        raise ValueError('Cleanup workflow has no explicit service-account identity')
    member = 'serviceAccount:' + service_account
    resources = {
        'queuedResources': f'projects/{project}/locations/{zone}/queuedResources/{queue}',
        'nodes': f'projects/{project}/locations/{zone}/nodes/{queue}',
    }
    coverage = {kind: set() for kind in resources}
    # TPU v2 queuedResources GET/DELETE use tpu.nodes.get/delete too:
    # https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/get
    # https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/delete
    # CEL still evaluates the distinct queue and node resource names.
    needed = {'tpu.nodes.get', 'tpu.nodes.delete'}
    unsupported = []
    for binding in policy.get('bindings', []):
        if member not in binding.get('members', []):
            continue
        role = roles.get(binding.get('role'))
        if not isinstance(role, dict) or role.get('deleted') or role.get('stage') == 'DISABLED':
            continue
        permissions = set(role.get('includedPermissions', []))
        for kind, resource in resources.items():
            if not permissions & needed:
                continue
            try:
                condition = binding.get('condition')
                allowed = condition is None or _prefix_condition(condition.get('expression'), resource)
            except (ValueError, AttributeError):
                unsupported.append(binding.get('role'))
                continue
            if allowed:
                coverage[kind].update(permissions & needed)
    missing = sorted(f'{kind}:{permission}' for kind in resources
                     for permission in needed - coverage[kind])
    if missing:
        raise ValueError('Cleanup IAM scope does not explicitly cover exact zone/name before reservation: '
                         + ', '.join(missing) + ('; unsupported conditional bindings' if unsupported else ''))
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    return dict(schema='cleanup-project-allow-scope-v1', service_account=service_account,
                resources=resources, policy_sha256=digest(policy), roles_sha256=digest(roles),
                supported_project_allow_coverage=True, effective_iam_verified=False,
                server_permission_probes_required=True)


def preflight_cleanup_scope(*, project, zone, queue, location='us-central1',
                            workflow='flaxchat-validation-cleanup'):
    """Read-only scope admission before reservation; does not execute a workflow."""
    lease(project, zone, queue, 60)
    discovery = {'schema': 'cleanup-scope-discovery-v1', 'reads': [],
                 'total_timeout_seconds': 120, 'command_timeout_seconds': 20,
                 'maximum_attempts_per_read': 2}
    deadline = time.monotonic() + 120
    def read(arguments):
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = TimeoutError('Cleanup scope discovery total deadline exhausted before reservation')
                error.cleanup_scope_discovery = discovery
                raise error
            observation = {'operation': arguments[:3], 'attempt': attempt + 1,
                           'timeout_seconds': min(20, remaining)}
            discovery['reads'].append(observation)
            started = time.monotonic()
            try:
                output = subprocess.check_output(['gcloud', *arguments, '--format=json'],
                    text=True, stderr=subprocess.PIPE, timeout=observation['timeout_seconds'])
                result = json.loads(output)
                observation.update(status='succeeded',
                    output_sha256=hashlib.sha256(output.encode()).hexdigest())
                return result
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                transient = isinstance(error, subprocess.TimeoutExpired)
                if isinstance(error, subprocess.CalledProcessError):
                    detail = str(error.stderr or '') + str(error.output or '')
                    detail = detail.lower()
                    denied = any(marker in detail for marker in
                        ('permission', 'forbidden', 'unauthenticated', '403', '401',
                         'access denied', 'credentials'))
                    transient = not denied and any(marker in detail for marker in
                        ('connection reset', 'connection aborted', 'connection refused',
                         'temporarily unavailable', 'service unavailable', '503',
                         'timed out', 'timeout', 'unexpected eof', 'remote disconnected'))
                observation.update(status='transient_failure' if transient else 'terminal_failure',
                                   error_type=type(error).__name__)
                if not transient or attempt == 1:
                    error.cleanup_scope_discovery = discovery
                    raise
            except (ValueError, OSError) as error:
                observation.update(status='terminal_failure', error_type=type(error).__name__)
                error.cleanup_scope_discovery = discovery
                raise
            finally:
                observation['elapsed_seconds'] = time.monotonic() - started
    deployed = read(['workflows', 'describe', workflow, '--project', project, '--location', location])
    source = Path(__file__).parents[1] / 'infra/tpu/cleanup_workflow.yaml'
    if yaml.safe_load(deployed.get('sourceContents', '')) != yaml.safe_load(source.read_text()):
        raise ValueError('Cleanup workflow source does not match reviewed permission probes')
    service_account = deployed.get('serviceAccount', '').split('/')[-1]
    policy = read(['projects', 'get-iam-policy', project])
    roles = {}
    for binding in policy.get('bindings', []):
        name = binding.get('role', '')
        if 'serviceAccount:' + service_account not in binding.get('members', []) or name in roles:
            continue
        if name.startswith('roles/'):
            arguments = ['iam', 'roles', 'describe', name]
        else:
            match = re.fullmatch(r'(projects|organizations)/([A-Za-z0-9_-]+)/roles/([A-Za-z0-9_.]+)', name)
            if match is None:
                raise ValueError('Unsupported cleanup IAM role identity')
            arguments = ['iam', 'roles', 'describe', match[3],
                         '--project' if match[1] == 'projects' else '--organization', match[2]]
        roles[name] = read(arguments)
    receipt = validate_cleanup_scope(policy, roles, service_account=service_account,
                                     project=project, zone=zone, queue=queue)
    receipt.update(workflow_revision=deployed.get('revisionId'),
                   workflow_sha256=hashlib.sha256(json.dumps(deployed, sort_keys=True).encode()).hexdigest(),
                   observed_unix=time.time(), discovery=discovery)
    return receipt


def lease(project, zone, queue, seconds, *, now=None, allow_long_lease=False):
    if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
        raise ValueError('Invalid project')
    if not re.fullmatch(r'[a-z]+-[a-z0-9]+-[a-z]', zone):
        raise ValueError('Invalid zone')
    if not re.fullmatch(r'flaxchat-validation-[a-z0-9-]+', queue):
        raise ValueError('Only dedicated validation queues are permitted')
    if type(allow_long_lease) is not bool:
        raise ValueError('Long lease opt-in must be boolean')
    maximum = 43200 if allow_long_lease else 7200
    if type(seconds) is not int or not 60 <= seconds <= maximum:
        raise ValueError(f'Guard duration must be 60–{maximum} seconds; longer leases require explicit opt-in')
    payload = dict(project=project, zone=zone, queue=queue,
                   deadline=(time.time() if now is None else now) + seconds)
    if allow_long_lease:
        payload['allow_long_lease'] = True
    return payload


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
    opt_in = expected.get('allow_long_lease', False)
    if type(opt_in) is not bool:
        raise ValueError('Long lease opt-in must be boolean')
    if not 60 <= remaining <= (43200 if opt_in else 7200):
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
    parser.add_argument('--allow-long-lease', action='store_true',
                        help='Explicitly allow a guard up to 12 hours; default maximum is 2 hours')
    args = parser.parse_args(argv)
    if args.verify:
        receipt = json.loads(args.receipt.read_text())
    else:
        payload = lease(args.project, args.zone, args.queue, args.seconds, allow_long_lease=args.allow_long_lease)
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
