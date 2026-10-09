"""Immutable, manifest-driven representation worker and guarded launch adapter.

No model imports. A manifest is an immutable deployment artifact; setup verifies
source/overlay/input hashes, installs only exact pins, and writes runtime evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import tarfile
import time

from flaxchat.cost_accounting import validated_rate


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def checked_hash(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected) or digest(path) != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')


def training_output(manifest):
    argv = manifest['workload']
    inferred = any(re.search(r'(?:^|[^A-Za-z0-9_])scripts\.train_[A-Za-z0-9_]+', item)
                   or Path(item).stem.startswith('train_') for item in argv)
    if manifest.get('workload_kind') != 'training' and not inferred:
        return None
    expected = manifest.get('checkpoint_output', manifest['output_prefix'].rstrip('/') + '/checkpoints')
    prefix = manifest['output_prefix'].rstrip('/')
    if not expected.startswith(prefix + '/') or '..' in expected.split('/'):
        raise ValueError('Training checkpoint output must be contained in the declared stage namespace')
    values = []
    for index, item in enumerate(argv):
        if item == '--output':
            if index + 1 >= len(argv) or argv[index + 1].startswith('--'):
                raise ValueError('Training --output requires a value')
            values.append(argv[index + 1])
        elif item.startswith('--output='):
            values.append(item.split('=', 1)[1])
    if len(values) != 1 or values[0].replace('{checkpoint_output}', expected).replace('{output_prefix}', prefix) != expected:
        raise ValueError('Actual training --output must exactly match the declared checkpoint output')
    return expected


def meaningful_setup(argv):
    executable = Path(argv[0]).name
    if executable in {'true', 'false', ':', 'echo', 'printf', 'sleep'} or any(item in {'--help', '-h', '--version'} for item in argv):
        raise ValueError('Setup checks must execute substantive validation, not a trivial/help probe')
    if '-c' in argv:
        code = argv[argv.index('-c') + 1] if argv.index('-c') + 1 < len(argv) else ''
        if re.fullmatch(r'\s*(?:pass|True|False|print\([^;]*\))\s*', code):
            raise ValueError('Setup checks cannot use an empty/print-only Python probe')


def package_inventory(value):
    rows = json.loads(value) if isinstance(value, str) else value
    if not isinstance(rows, list):
        raise ValueError('Package inventory must be a list')
    result = {}
    for item in rows:
        name = item['name'].lower().replace('_', '-').replace('.', '-')
        if name in result:
            raise ValueError('Duplicate runtime package identity')
        result[name] = item['version']
    return result


def source_tree_digest(tree):
    return hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_qualification(manifest, root, identity, source_tree):
    contract = manifest.get('qualification')
    if not contract:
        return None
    path = Path(root) / contract['receipt']
    receipt = json.loads(path.read_text())
    if contract.get('format', 'bound') == 'test_suite':
        from scripts.validate_tpu import source_digest
        source_dir = Path(root) / manifest['source'].get('target', 'source')
        hardware_path = path.with_name('hardware.json')
        hardware = json.loads(hardware_path.read_text())
        if receipt.get('passed') is not True or receipt.get('hardware_passed') is not True or receipt.get('not_run') or receipt.get('source_python_sha256') != source_digest(source_dir):
            raise ValueError('Test-suite qualification incomplete or source identity mismatch')
        if hardware.get('backend') != 'tpu' or hardware.get('devices') != manifest['expected_device_count'] or hardware.get('processes') != manifest['expected_process_count']:
            raise ValueError('Test-suite physical hardware topology mismatch')
        pins = validate_lock(source_dir / manifest['runtime_lock']['path'])
        runtime = hardware.get('runtime', {})
        observed_packages = runtime.get('packages', {})
        required_runtime = {'jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint', 'libtpu'}
        if not required_runtime.issubset(observed_packages) or any(pins.get(name) != version for name, version in observed_packages.items()) or not runtime.get('python'):
            raise ValueError('Test-suite runtime does not match frozen package lock')
        if runtime['python'] != subprocess.check_output([str(Path(root) / 'venv/bin/python'), '--version'], text=True, timeout=30).strip().removeprefix('Python '):
            raise ValueError('Test-suite interpreter identity mismatch')
        results = receipt.get('results')
        if not isinstance(results, list) or not results or any(result.get('passed') is not True or result.get('returncode') != 0 for result in results):
            raise ValueError('Test-suite module acceptance failed')
        tests = [{'id': item.get('test'), 'status': item.get('status')} for result in results for item in result.get('tests', [])]
        extra_identity = {'hardware_sha256': digest(hardware_path), 'manifest_sha256': identity,
            'source_tree_sha256': source_tree_digest(source_tree), 'runtime_lock_sha256': manifest['runtime_lock']['sha256']}
    else:
        expected = {'schema_version': 1, 'passed': True, 'backend': 'tpu', 'manifest_sha256': identity,
            'runtime_lock_sha256': manifest['runtime_lock']['sha256'], 'source_tree_sha256': source_tree_digest(source_tree),
            'devices': manifest['expected_device_count'], 'processes': manifest['expected_process_count']}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError('Physical qualification receipt identity/topology/status mismatch')
        tests = receipt.get('tests')
        extra_identity = {}
    if not isinstance(tests, list) or not tests or any(item.get('status') != 'passed' for item in tests):
        raise ValueError('Physical qualification requires passing tests; skips/failures cannot qualify')
    names = [item.get('id') for item in tests]
    if len(names) != len(set(names)) or not set(contract['required_tests']).issubset(names):
        raise ValueError('Physical qualification test inventory incomplete/duplicated')
    return {'receipt': contract['receipt'], 'sha256': digest(path), 'required_tests': contract['required_tests'], **extra_identity}


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get('schema_version') != 1:
        raise ValueError('Expected representation run schema_version=1')
    for key in ('run_id', 'stage_id'):
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,63}', value.get(key, '')):
            raise ValueError(f'Invalid {key}')
    if value['run_id'] == value['stage_id']:
        raise ValueError('Run and stage identities must be distinct')
    for key in ('parent_identity', 'cleanup_owner', 'expected_topology'):
        if not value.get(key):
            raise ValueError(f'Missing {key}')
    lock = value['runtime_lock']
    lock_path = Path(lock['path'])
    if lock_path.is_absolute() or '..' in lock_path.parts or not re.fullmatch('[0-9a-f]{64}', lock['sha256']):
        raise ValueError('Runtime lock requires contained path and SHA256')
    wheelhouse = lock.get('wheelhouse_target')
    if not isinstance(wheelhouse, str) or Path(wheelhouse).is_absolute() or '..' in Path(wheelhouse).parts:
        raise ValueError('Runtime requires a verified wheelhouse_target')
    if not any(item.get('target') == wheelhouse and item.get('archive') for item in value.get('artifacts', [])):
        raise ValueError('Wheelhouse must be a hashed archive in artifacts')
    if 'deployment' in value:
        deployment = value['deployment']
        opt_in = deployment.get('allow_long_lease', False)
        if type(opt_in) is not bool:
            raise ValueError('Long lease opt-in must be boolean')
        duration = deployment.get('attempt_seconds', 1800)
        maximum = 43200 if opt_in else 7200
        if type(duration) is not int or not 180 <= duration <= maximum:
            raise ValueError(f'Attempt lease must be 180–{maximum} seconds; longer leases require explicit opt-in')
        rate = validated_rate(deployment['pricing'])
        zone = deployment.get('zone')
        if not isinstance(zone, str) or not re.fullmatch(r'[a-z]+-[a-z0-9]+-[a-z]', zone) or rate['region'] != zone.rsplit('-', 1)[0]:
            raise ValueError('Deployment zone must belong to the region in pricing evidence')
        if rate['accelerator_type'] != deployment['accelerator_type'] or rate['hourly_usd'] != deployment['hourly_usd']:
            raise ValueError('Deployment must match whole-slice pricing evidence')
        if rate['provisioning_model'] != deployment.get('provisioning_model', 'spot'):
            raise ValueError('Deployment provisioning model differs from pricing evidence')
        if value['expected_topology'] != deployment['accelerator_type']:
            raise ValueError('Deployment differs from expected topology')
        for wait_key in ('capacity_wait_seconds', 'startup_wait_seconds'):
            wait_seconds = deployment.get(wait_key)
            if wait_seconds is not None and (type(wait_seconds) is not int or not 1 <= wait_seconds <= deployment.get('attempt_seconds', 1800)):
                label = 'Capacity wait' if wait_key == 'capacity_wait_seconds' else 'Startup wait'
                raise ValueError(f'{label} ({wait_key}) must be a positive integer within the attempt lease')
    for key in ('expected_device_count', 'expected_process_count'):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f'Explicit positive {key} required')
    source = value['source']
    for item in (source, *value.get('artifacts', []), *value.get('overlays', [])):
        if not item['uri'].startswith('gs://') or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Immutable GCS artifact URI and SHA256 required')
        target = Path(item.get('target', 'source'))
        if target.is_absolute() or '..' in target.parts:
            raise ValueError('Artifact target must be contained in the run directory')
    output = value['output_prefix'].rstrip('/')
    if not output.startswith('gs://') or not output.endswith('/' + value['run_id'] + '/' + value['stage_id']):
        raise ValueError('Output prefix must end with unique /run_id/stage_id')
    for key in ('workload', 'setup_checks'):
        argv = value[key]
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError(f'{key} must be a nonempty argv array')
    checks_timeout = value.get('setup_checks_timeout_seconds', 300)
    if type(checks_timeout) is not int or not 30 <= checks_timeout <= 1800:
        raise ValueError('Setup checks timeout must be an integer in 30–1800 seconds')
    meaningful_setup(value['setup_checks'])
    training_output(value)
    if value.get('qualification'):
        contract = value['qualification']
        path = Path(contract['receipt'])
        if contract.get('format', 'bound') not in {'bound', 'test_suite'} or str(path) in {'runtime-receipt.json', 'hardware-receipt.json', 'manifest-identity.txt'}:
            raise ValueError('Invalid/reserved qualification receipt format/path')
        required = contract.get('required_tests')
        if path.is_absolute() or '..' in path.parts or not isinstance(required, list) or not required or not all(isinstance(item, str) and item for item in required) or len(required) != len(set(required)):
            raise ValueError('Qualification requires a contained receipt path and explicit unique required test IDs')
    if (value.get('require_physical_acceptance') is True or training_output(value) is not None) and not value.get('qualification'):
        raise ValueError('Physical training admission requires an explicit qualification receipt contract')
    if not 30 <= value.get('upload_timeout_seconds', 30) <= 120:
        raise ValueError('Upload timeout must be 30–120 seconds')
    return value


def extract_archive(archive, destination, *, allow_internal_links=False):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        for entry in handle.getmembers():
            path = Path(entry.name)
            permitted = entry.isfile() or entry.isdir() or (allow_internal_links and (entry.issym() or entry.islnk()))
            if path.is_absolute() or '..' in path.parts or not permitted:
                raise ValueError(f'Unsafe source archive member: {entry.name}')
            target = destination / path
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Archive target escapes destination: {entry.name}')
            if entry.issym() or entry.islnk():
                link = Path(entry.linkname)
                link_target = (target.parent if entry.issym() else destination) / link
                if link.is_absolute() or not link_target.resolve().is_relative_to(destination):
                    raise ValueError(f'Archive link escapes destination: {entry.name}')
            # Python 3.10 bootstrap may lack data_filter. Explicit containment,
            # file-type and link checks above apply to both supported versions.
            if hasattr(tarfile, 'data_filter'):
                handle.extract(entry, destination, filter='data')
            else:
                handle.extract(entry, destination)


def validate_lock(path):
    pins = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+!_-]+', line):
            raise ValueError(f'Runtime lock requires exact package==version pins: {line}')
        pins.append(line)
    names = [re.split('==', item)[0].lower().replace('_', '-').replace('.', '-') for item in pins]
    if not pins or len(set(names)) != len(names):
        raise ValueError('Empty or duplicate runtime lock')
    return dict(zip(names, [item.split('==')[1] for item in pins], strict=True))


def source_snapshot(source_dir):
    """Bind the actual deployed tree; exclude only generated runtime/output paths."""
    excluded = {'.git', '.venv', '.pixi', '__pycache__', '.pytest_cache', '.ruff_cache', 'artifacts', 'checkpoints'}
    root = Path(source_dir)
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*'))
            if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
            and path.suffix not in {'.pyc', '.log'}}


def qualify_setup_runtime(manifest, root, source_dir, identity, python, inventory, lock, environment):
    """Publish verified runtime identity before checks, acceptance only afterward."""
    from scripts.evaluation_contract import write_atomic
    tree = source_snapshot(source_dir)
    runtime = {'schema_version': 2, 'runtime_verified': True,
        'setup_checks_executed': False, 'physical_acceptance': False, 'qualification': None,
        'manifest_sha256': identity, 'source_tree': tree,
        'runtime_lock_sha256': digest(lock), 'packages': json.loads(inventory),
        'platform': platform.platform(), 'interpreter_sha256': digest(Path(python).resolve()),
        'python': subprocess.check_output([python, '--version'], text=True, timeout=30).strip()}
    # Keep this immutable after checks: their provenance hashes this exact file.
    provisional = root / 'runtime-setup-receipt.json'
    write_atomic(provisional, runtime)
    check_environment = {**environment, 'FLAXCHAT_SETUP_QUALIFICATION': '1',
                         'FLAXCHAT_SOURCE_TREE_SHA256': source_tree_digest(tree)}
    check = [x.replace('{python}', python).replace('{root}', str(root)) for x in manifest['setup_checks']]
    if any('{checkpoint_output}' in item or '{output_prefix}' in item for item in check):
        prefix = manifest['output_prefix'].rstrip('/')
        checkpoint_output = training_output(manifest) or prefix + '/checkpoints'
        check = [item.replace('{output_prefix}', prefix).replace('{checkpoint_output}', checkpoint_output)
                 for item in check]
    subprocess.run(check, cwd=source_dir, env=check_environment, check=True,
                   timeout=manifest.get('setup_checks_timeout_seconds', 300))
    if source_snapshot(source_dir) != tree:
        raise ValueError('Deployed source changed during setup qualification')
    qualification = verify_qualification(manifest, root, identity, tree)
    write_atomic(root / 'runtime-receipt.json', {**runtime, 'setup_checks_executed': True,
        'physical_acceptance': bool(qualification), 'qualification': qualification,
        'setup_runtime_receipt_sha256': digest(provisional)})


def setup(manifest_path, root):
    manifest = load_manifest(manifest_path)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = digest(manifest_path)
    receipt = root / 'manifest-identity.txt'
    if receipt.exists() and receipt.read_text().strip() != identity:
        raise ValueError('Cannot reuse run directory for a different manifest')
    # A failed re-setup must never inherit yesterday's passing qualification.
    prior_receipts = ['runtime-receipt.json', 'runtime-setup-receipt.json', 'hardware-receipt.json']
    if manifest.get('qualification'):
        prior_receipts.append(manifest['qualification']['receipt'])
    for previous in prior_receipts:
        (root / previous).unlink(missing_ok=True)
    receipt.write_text(identity + '\n')
    download_timeout = manifest.get('download_timeout_seconds', 300)
    if not 30 <= download_timeout <= 900:
        raise ValueError('Download timeout must be 30–900 seconds')
    for index, item in enumerate((manifest['source'], *manifest.get('artifacts', []), *manifest.get('overlays', []))):
        temporary = root / f'download-{index}'
        subprocess.run(['gcloud', 'storage', 'cp', item['uri'], str(temporary)], check=True, timeout=download_timeout)
        checked_hash(temporary, item['sha256'])
        target = root / item.get('target', 'source')
        if item.get('archive', index == 0):
            extract_archive(temporary, target, allow_internal_links=index != 0 and item.get('allow_internal_links', False))
            temporary.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(target)
    source_dir = root / manifest['source'].get('target', 'source')
    lock = source_dir / manifest['runtime_lock']['path']
    checked_hash(lock, manifest['runtime_lock']['sha256'])
    pins = validate_lock(lock)
    venv = root / 'venv'
    interpreter = manifest['runtime_lock'].get('python_executable', 'python3.12').replace('{root}', str(root))
    subprocess.run([interpreter, '-m', 'venv', str(venv)], check=True, timeout=120)
    python = str(venv / 'bin/python')
    # --no-deps prevents transitive resolution from silently selecting a floating
    # package. A complete freeze is required and pip check verifies completeness.
    subprocess.run([python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', '--no-index', '--find-links', str(root / manifest['runtime_lock']['wheelhouse_target']), '-r', str(lock)], check=True, timeout=900)
    subprocess.run([python, '-m', 'pip', 'check'], check=True, timeout=120)
    inventory = subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], text=True, timeout=60)
    installed = package_inventory(inventory)
    for package, version in pins.items():
        if installed.get(package) != version:
            raise ValueError(f'Runtime mismatch for {package}')
    unexpected = set(installed) - set(pins) - {'pip', 'setuptools'}
    if unexpected:
        raise ValueError(f'Unexpected runtime packages: {sorted(unexpected)}')
    environment = {**os.environ, 'PYTHONPATH': str(source_dir), 'FLAXCHAT_RUN_ROOT': str(root),
                   'FLAXCHAT_RUN_MANIFEST_SHA256': digest(manifest_path), 'FLAXCHAT_RUNTIME_LOCK_SHA256': manifest['runtime_lock']['sha256']}
    # This executes only on the provisioned worker, never the local controller.
    # Establish physical backend evidence before caller-specific restore tests.
    publication = manifest.get('workload_kind') == 'publication'
    if publication:
        from scripts.representation_workflow import validate_workflow
        kind = manifest.get('workflow_contract', {}).get('kind')
        if kind not in {'publish-flax', 'publish-torch'} or manifest.get('require_physical_acceptance'):
            raise ValueError('Non-model publication requires an explicit publication workflow')
        validate_workflow(manifest, kind)
        hardware = {'backend': 'not-requested', 'publication_only': True}
    else:
        physical_check = "import json,jax; "
        if manifest.get('distributed', False):
            physical_check += "jax.distributed.initialize(); "
        physical_check += "assert jax.default_backend() == 'tpu', 'Physical TPU required'; print(json.dumps(dict(backend=jax.default_backend(), devices=jax.device_count(), local_devices=jax.local_device_count(), processes=jax.process_count(), device_kind=jax.devices()[0].device_kind)))"
        hardware = json.loads(subprocess.check_output([python, '-c', physical_check], cwd=source_dir, env=environment, text=True, timeout=120))
        for key, actual in (('expected_device_count', hardware['devices']), ('expected_process_count', hardware['processes'])):
            if key in manifest and manifest[key] != actual:
                raise ValueError(f'Topology mismatch: {key} expected {manifest[key]}, observed {actual}')
    (root / 'hardware-receipt.json').write_text(json.dumps(hardware, indent=2) + '\n')
    qualify_setup_runtime(manifest, root, source_dir, identity, python, inventory, lock, environment)
    return 0


def run_worker(manifest_path, root):
    manifest = load_manifest(manifest_path)
    root = Path(root).resolve()
    receipt = json.loads((root / 'runtime-receipt.json').read_text())
    if receipt.get('setup_checks_executed') is not True or receipt.get('schema_version') != 2 or receipt['manifest_sha256'] != digest(manifest_path):
        raise ValueError('Setup receipt does not match worker manifest')
    if 'setup_runtime_receipt_sha256' in receipt:
        checked_hash(root / 'runtime-setup-receipt.json', receipt['setup_runtime_receipt_sha256'])
    python = str(root / 'venv/bin/python')
    checkpoint_output = training_output(manifest) or manifest['output_prefix'].rstrip('/') + '/checkpoints'
    argv = [x.replace('{python}', python).replace('{root}', str(root)).replace('{output_prefix}', manifest['output_prefix'].rstrip('/')).replace('{checkpoint_output}', checkpoint_output) for x in manifest['workload']]
    source_dir = root / manifest['source'].get('target', 'source')
    if receipt.get('source_tree') != source_snapshot(source_dir):
        raise ValueError('Deployed source changed since successful setup')
    lock = source_dir / manifest['runtime_lock']['path']
    checked_hash(lock, manifest['runtime_lock']['sha256'])
    if digest(Path(python).resolve()) != receipt.get('interpreter_sha256'):
        raise ValueError('Selected interpreter binary changed since successful setup')
    actual_packages = package_inventory(subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], text=True, timeout=60))
    if actual_packages != package_inventory(receipt['packages']):
        raise ValueError('Installed package inventory changed since successful setup')
    actual_python = subprocess.check_output([python, '--version'], text=True, timeout=30).strip()
    if actual_python != receipt['python']:
        raise ValueError('Selected interpreter changed since successful setup')
    qualification = verify_qualification(manifest, root, digest(manifest_path), receipt['source_tree'])
    if qualification != receipt.get('qualification') or ((manifest.get('require_physical_acceptance') is True or training_output(manifest) is not None) and receipt.get('physical_acceptance') is not True):
        raise ValueError('Physical acceptance changed or is missing before workload admission')
    worker = re.sub('[^a-zA-Z0-9-]', '-', platform.node())
    log_path = root / f'{worker}.log'
    events = []
    def upload():
        try:
            subprocess.run(['gcloud', 'storage', 'cp', str(log_path), manifest['output_prefix'] + '/logs/' + worker + '.log'],
                check=True, timeout=manifest.get('upload_timeout_seconds', 30), stdout=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as error:
            events.append({'event': 'telemetry_upload_failed', 'time_unix': time.time(), 'error': str(error)})
    environment = {**os.environ, 'PYTHONPATH': str(source_dir), 'FLAXCHAT_RUN_ROOT': str(root),
                   'FLAXCHAT_RUN_MANIFEST_SHA256': digest(manifest_path), 'FLAXCHAT_RUNTIME_LOCK_SHA256': manifest['runtime_lock']['sha256']}
    with log_path.open('a') as log:
        process = subprocess.Popen(argv, cwd=source_dir, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def terminate(signum, frame):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass  # The child may have exited while final telemetry is uploading.
        previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                try:
                    status = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    upload()
            # Persist the workload result before any best-effort final telemetry.
            status_path = root / f'{worker}-status.json'
            status_path.write_text(json.dumps({'schema_version': 1, 'manifest_sha256': digest(manifest_path),
                'worker': worker, 'workload_returncode': status, 'telemetry_events': events,
                'finished_unix': time.time()}, indent=2) + '\n')
            upload()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    result = {'schema_version': 1, 'manifest_sha256': digest(manifest_path), 'worker': worker,
              'workload_returncode': status, 'telemetry_events': events, 'finished_unix': time.time()}
    status_path = root / f'{worker}-status.json'
    status_path.write_text(json.dumps(result, indent=2) + '\n')
    try:
        subprocess.run(['gcloud', 'storage', 'cp', str(status_path), manifest['output_prefix'] + '/logs/' + status_path.name],
            check=True, timeout=manifest.get('upload_timeout_seconds', 30))
    except (OSError, subprocess.SubprocessError) as error:
        print(f'Status upload failed (workload status preserved): {error}', flush=True)
    return status if status >= 0 else 128 - status


def supervisor_argv(manifest, manifest_uri, manifest_sha256, output):
    """Generate executable guarded deployment; all mutable bootstrap is hashed."""
    if not manifest_uri.startswith('gs://') or not re.fullmatch('[0-9a-f]{64}', manifest_sha256):
        raise ValueError('Launch requires immutable GCS manifest URI + SHA256')
    deployment = manifest['deployment']
    root = '/tmp/flaxchat-representation-' + manifest['run_id'] + '-' + manifest['stage_id']
    source = manifest['source']
    source_target = source.get('target', 'source')
    bootstrap_extract = "\n".join([
        "import pathlib,tarfile",
        "t=tarfile.open(" + repr(root + '/source.tar.gz') + ")",
        "d=pathlib.Path(" + repr(root + '/' + source_target) + ").resolve()",
        "d.mkdir(parents=True,exist_ok=True)",
        "for e in t.getmembers():",
        " p=pathlib.Path(e.name)",
        " assert not p.is_absolute() and '..' not in p.parts and (e.isfile() or e.isdir()), 'Unsafe bootstrap archive'",
        " assert (d/p).resolve().is_relative_to(d), 'Escaping bootstrap archive'",
        " t.extract(e,d)",
    ])
    bootstrap = f'''set -eu
mkdir -p {shlex.quote(root)}
gcloud storage cp {shlex.quote(manifest_uri)} {shlex.quote(root + '/run.json')}
printf '%s  %s\\n' {shlex.quote(manifest_sha256)} {shlex.quote(root + '/run.json')} | sha256sum -c -
gcloud storage cp {shlex.quote(source['uri'])} {shlex.quote(root + '/source.tar.gz')}
printf '%s  %s\\n' {shlex.quote(source['sha256'])} {shlex.quote(root + '/source.tar.gz')} | sha256sum -c -
python3 -S -c {shlex.quote(bootstrap_extract)}
PYTHONPATH={shlex.quote(root + '/' + source_target)} python3 -m scripts.representation_run setup --manifest {shlex.quote(root + '/run.json')} --root {shlex.quote(root)}
'''
    argv = ['--run-id', manifest['run_id'], '--stage-id', manifest['stage_id'],
            '--output', str(output), '--name', 'flaxchat-validation-' + manifest['run_id'],
            '--setup', json.dumps(['bash', '-c', bootstrap]),
            '--directory', root + '/' + source_target, '--setup-timeout-seconds', '900',
            '--workload', json.dumps(['python3', '-m', 'scripts.representation_run', 'run', '--manifest', root + '/run.json', '--root', root])]
    for key in ('project', 'zone', 'accelerator_type', 'runtime_version', 'provisioning_model', 'hourly_usd', 'budget_usd', 'attempt_seconds', 'capacity_wait_seconds', 'startup_wait_seconds', 'campaign_ledger'):
        if key in deployment:
            argv.extend(['--' + key.replace('_', '-'), str(deployment[key])])
    if deployment.get('allow_long_lease') is True:
        argv.append('--allow-long-lease')
    if deployment.get('tunnel_through_iap'):
        argv.append('--tunnel-through-iap')
    return argv


def seal_manifest(template_path, output, *, upload=False, manifest_uri=None):
    """Seal local deployment inputs before allocation; optional bounded upload."""
    value = json.loads(Path(template_path).read_text())
    local_artifacts = []
    for item in (value['source'], *value.get('artifacts', []), *value.get('overlays', [])):
        local_path = Path(item.pop('local_path'))
        item['sha256'] = digest(local_path)
        local_artifacts.append((local_path, item['uri']))
    lock = value['runtime_lock']
    lock['sha256'] = digest(Path(lock.pop('local_path')))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    load_manifest(output)
    if upload:
        if not manifest_uri or not manifest_uri.startswith('gs://'):
            raise ValueError('Upload requires a GCS manifest URI')
        for path, uri in local_artifacts:
            subprocess.run(['gcloud', 'storage', 'cp', str(path), uri, '--if-generation-match=0'], check=True, timeout=900)
        subprocess.run(['gcloud', 'storage', 'cp', str(output), manifest_uri, '--if-generation-match=0'], check=True, timeout=60)
    return {'manifest_sha256': digest(output), 'uploaded': upload, 'manifest': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'check', 'setup', 'run', 'launch'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--manifest-uri')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--upload', action='store_true', help='Explicitly upload sealed immutable artifacts to their declared fresh GCS URIs')
    args = parser.parse_args(argv)
    if args.action == 'seal':
        if not args.output:
            parser.error('seal requires --output')
        print(json.dumps(seal_manifest(args.manifest, args.output, upload=args.upload, manifest_uri=args.manifest_uri)))
        return 0
    manifest = load_manifest(args.manifest)
    if args.action == 'check':
        print(json.dumps({'valid': True, 'manifest_sha256': digest(args.manifest)}))
        return 0
    if args.action == 'launch':
        if not args.manifest_uri or not args.output:
            parser.error('launch requires --manifest-uri and --output')
        from scripts.gcp_spot_supervisor import main as supervise
        return supervise(supervisor_argv(manifest, args.manifest_uri, digest(args.manifest), args.output))
    if not args.root:
        parser.error('setup/run requires --root')
    return setup(args.manifest, args.root) if args.action == 'setup' else run_worker(args.manifest, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
