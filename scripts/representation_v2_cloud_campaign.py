"""Single-admission cloud controller: prepare, admit, qualify, then continue.

Run only on a provider-expiring GCE host. No user OAuth/Hugging Face token is
needed; gcloud uses the attached service account. Never retries allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import contextlib


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_logged(argv, timeout, log_path, *, capture=False):
    """Keep full diagnostics and kill the whole worker tree on every exit."""
    argv = list(map(str, argv))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_suffix('.stdout')
    with contextlib.ExitStack() as stack:
        log = stack.enter_context(log_path.open('wb'))
        output = stack.enter_context(stdout_path.open('wb')) if capture else log
        process = subprocess.Popen(argv, stdout=output, stderr=log,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    if code:
        with log_path.open('rb') as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 16384))
            tail = log.read().decode('utf-8', errors='replace')
        raise subprocess.CalledProcessError(code, argv, output=tail)
    return subprocess.CompletedProcess(argv, code,
        stdout=stdout_path.read_text(errors='replace') if capture else None)


def bundle_training_data(root, destination):
    """Persist expensive completed preparation before any model admission."""
    with tarfile.open(destination, 'w:gz', compresslevel=1, dereference=True) as archive:
        for name in ('development', 'mixture', 'weights.json', 'registry.json',
                     'heldout-inventory.json', 'final-quarantine.json', 'terminal-data.json'):
            archive.add(Path(root) / name, arcname=name)


def validate_prepared_data(root, spec, source):
    """An immutable archive alone is not proof it belongs to this recipe."""
    from scripts.prepare_representation_v2_training import WEIGHTS
    root = Path(root)
    terminal = json.loads((root / 'terminal-data.json').read_text())
    expected = {'status': 'passed', 'model_execution': False,
                'portable_sha256': spec['portable']['sha256'],
                'original_heldout_sha256': spec['heldout']['sha256'],
                'source_sha256': sha(Path(source) / 'scripts/prepare_representation_v2_training.py')}
    if any(terminal.get(key) != value for key, value in expected.items()):
        raise ValueError('Prepared data terminal does not authenticate this recipe')
    quarantine_path = root / 'final-quarantine.json'
    if terminal.get('final_quarantine_sha256') != sha(quarantine_path):
        raise ValueError('Prepared data quarantine receipt hash mismatch')
    quarantine = json.loads(quarantine_path.read_text())
    if (quarantine.get('status') != 'passed' or quarantine.get('exact_overlaps') != 0
            or set(quarantine.get('sources', {})) != set(WEIGHTS)):
        raise ValueError('Prepared data quarantine is incomplete')
    if json.loads((root / 'weights.json').read_text()) != WEIGHTS:
        raise ValueError('Prepared data mixture weights differ')
    for name, item in quarantine['sources'].items():
        if item.get('train_rows', 0) < 1 or sha(root / 'mixture' / name / 'manifest.json') != item['manifest_sha256']:
            raise ValueError('Prepared data mixture manifest changed: ' + name)
    return terminal


def stage_args(root, output):
    """Same fixed stage on controller preflight and physical trainer."""
    from scripts.prepare_representation_v2_training import WEIGHTS
    root = str(root)
    data = root + '/training-data'
    args = ['--parent-public', root + '/parent', '--parent-manifest', data + '/development/parent-files.json',
            '--source-weights', ','.join(f'{key}={value}' for key, value in WEIGHTS.items()),
            '--output', output, '--steps', '20000', '--batch-size', '128',
            '--learning-rate', '0.00001', '--warmup', '500', '--save-every', '100',
            '--eval-every', '500', '--keep-checkpoints', '3', '--seed', '29',
            '--encoder-chunk-size', '0', '--batch-policy', 'mixed',
            '--weight-quantization', 'none', '--training-scope', 'production',
            '--max-dev-regression', '0.02', '--sts-dev', 'stsb=' + data + '/development/sts']
    for name in WEIGHTS:
        args += ['--data', name + '=' + data + '/mixture/' + name]
    for task in ('retrieval', 'bitext', 'code'):
        args += ['--retrieval-dev', task + '=' + data + '/development/independent/' + task]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Atomic admission persists through attachment loss and service restarts.
    with (root / 'launch-claim.json').open('x') as stream:
        json.dump({'spec_sha256': sha(args.spec), 'started_unix': time.time()}, stream)
    deadline = spec['provider_delete_unix'] - 300
    prefix = spec['output_prefix']
    receipt = {'status': 'running', 'phase': 'bootstrap', 'model_execution_state': 'pending_physical_gate',
               'spec_sha256': sha(args.spec), 'maximum_tpu_attempts': 1}
    os.environ.update(CLOUDSDK_STORAGE_PROCESS_COUNT='1', CLOUDSDK_STORAGE_THREAD_COUNT='1',
                      TOKENIZERS_PARALLELISM='false', HF_DATASETS_DISABLE_PROGRESS_BARS='1')

    command_index = 0

    def run(argv, timeout, *, capture=False, finalizing=False):
        nonlocal command_index
        remaining = (spec['provider_delete_unix'] - 30 if finalizing else deadline) - time.time()
        if remaining <= 30:
            raise TimeoutError('Original provider lease nearly exhausted')
        command_index += 1
        return run_logged(argv, min(timeout, remaining),
                          root / 'controller-logs' / f'command-{command_index:04d}.log', capture=capture)

    def publish(*, finalizing=False):
        save(root / 'campaign.json', receipt)
        run(['gcloud', 'storage', 'cp', root / 'campaign.json', prefix + '/campaign.json'], 45, finalizing=finalizing)

    def phase(name):
        receipt.update(phase=name, updated_unix=time.time())
        publish()
        print(json.dumps({'phase': name}), flush=True)

    def download(item, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        run(['gcloud', 'storage', 'cp', item['uri'], target], 900)
        if sha(target) != item['sha256']:
            raise ValueError(f'Artifact SHA mismatch: {target.name}')

    def immutable_upload(path, uri):
        expected = sha(path)
        run(['gcloud', 'storage', 'cp', path, uri, '--if-generation-match=0'], 1200)
        info = json.loads(run(['gcloud', 'storage', 'objects', 'describe', uri,
                              '--format=json'], 60, capture=True).stdout)
        pinned = uri + '#' + str(info['generation'])
        verify = root / 'verify-download'
        download({'uri': pinned, 'sha256': expected}, verify)
        if verify.stat().st_size != path.stat().st_size:
            raise ValueError('Uploaded byte count mismatch')
        verify.unlink()
        return {'uri': pinned, 'sha256': expected}

    try:
        if shutil.disk_usage(root).free < 60 * 1024**3:
            raise ValueError('At least60GiB free scratch required')
        from scripts.representation_run import extract_archive, load_manifest, supervisor_argv
        base = spec['base_manifest']
        # The controller and TPU must execute exactly the same frozen tree.
        from scripts.representation_run import source_snapshot
        frozen_archive = root / 'controller-source.tar.gz'
        download(spec['source'], frozen_archive)
        frozen_source = root / 'controller-source-verification'
        extract_archive(frozen_archive, frozen_source)
        source = Path(__file__).resolve().parents[1]
        if source_snapshot(source) != source_snapshot(frozen_source):
            raise ValueError('Controller source differs from frozen TPU source')
        receipt['controller_source_sha256'] = spec['source']['sha256']
        shutil.rmtree(frozen_source)
        frozen_archive.unlink()
        for index, item in enumerate(base['artifacts']):
            file = root / ('input-' + str(index))
            download(item, file)
            target = root / item['target']
            if item.get('archive'):
                extract_archive(file, target, allow_internal_links=item.get('allow_internal_links', False))
                file.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                file.replace(target)
        phase('locked-data-runtime')
        interpreter = root / 'cpython/python/bin/python3.12'
        run([interpreter, '-m', 'venv', root / 'venv'], 120)
        python = root / 'venv/bin/python'
        source = Path(__file__).resolve().parents[1]
        if sha(source / base['runtime_lock']['path']) != base['runtime_lock']['sha256']:
            raise ValueError('Controller runtime lock differs from frozen manifest')
        os.environ['PYTHONPATH'] = str(source)
        run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--find-links', root / 'wheels',
             '-r', source / base['runtime_lock']['path']], 900)
        run([python, '-m', 'pip', 'check'], 120)
        prepared = spec.get('prepared_data')
        if prepared:
            if prepared.get('source_sha256') != spec['source']['sha256']:
                raise ValueError('Prepared data source freeze differs from campaign')
            phase('restore-prepared-training-data')
            bundle = root / 'training-data.tar.gz'
            download(prepared, bundle)
            extract_archive(bundle, root / 'training-data')
            validate_prepared_data(root / 'training-data', spec, source)
            data_artifact = {'uri': prepared['uri'], 'sha256': prepared['sha256']}
        else:
            portable, heldout = root / 'portable.tar.gz', root / 'heldout.tar.gz'
            download(spec['portable'], portable)
            download(spec['heldout'], heldout)
            phase('prepare-training-data')
            preparation_seconds = spec.get('preparation_seconds', 7200)
            if type(preparation_seconds) is not int or not 60 <= preparation_seconds <= 21600:
                raise ValueError('preparation_seconds must be an integer from 60 to 21600')
            run([python, '-m', 'scripts.prepare_representation_v2_training',
                 '--portable-archive', portable, '--heldout-archive', heldout,
                 '--heldout-sha256', spec['heldout']['sha256'],
                 '--tokenizer', root / 'parent/tokenizer.json', '--output', root / 'training-data',
                 '--total-seconds', str(preparation_seconds)], preparation_seconds + 60)
            validate_prepared_data(root / 'training-data', spec, source)
            phase('publish-immutable-training-data')
            bundle = root / 'training-data.tar.gz'
            bundle_training_data(root / 'training-data', bundle)
            data_artifact = immutable_upload(bundle, prefix + '/training-data.tar.gz')
        receipt['data_artifact'] = data_artifact
        publish()
        phase('production-data-admission')
        checkpoint = prefix + '/yat-embed-torch-v2-night-1005/representation-continuation/checkpoints'
        run([python, '-m', 'scripts.preflight_yat_embedding_stage', *stage_args(root, checkpoint),
             '--expected-devices', '8', '--expected-processes', '1',
             '--receipt', root / 'stage-preflight.json'], 1800)
        run(['gcloud', 'storage', 'cp', root / 'stage-preflight.json', prefix + '/stage-preflight.json'], 60)
        # One fixed eight-hour lease, within original GCE deadline, no retries.
        if deadline - time.time() < 28800 + 2100:
            raise TimeoutError('Not enough provider lease left for admitted TPU run and cleanup')
        phase('single-tpu-request')
        manifest = json.loads(json.dumps(base))
        manifest.update(run_id='yat-embed-torch-v2-night-1005', stage_id='representation-continuation',
                        source=spec['source'], output_prefix=checkpoint.rsplit('/checkpoints', 1)[0],
                        setup_checks=['{python}', '-m', 'scripts.qualify_representation_v2_tpu', '--root', '{root}'],
                        setup_checks_timeout_seconds=1440,
                        workload=['{python}', '-m', 'scripts.train_yat_embedding_finetune',
                                  *stage_args('{root}', '{checkpoint_output}')],
                        download_timeout_seconds=900)
        nodes = json.loads((source / 'infra/tpu/representation-v2-direct-nodes.json').read_text())
        manifest['qualification'] = {'receipt': 'qualification/summary.json', 'format': 'test_suite',
                                     'required_tests': [node for group in nodes.values() for node in group]}
        manifest['parent_identity']['purpose'] = 'Continue authenticated embedding-v1; fresh optimizer for changed data'
        manifest['artifacts'].append({**data_artifact, 'target': 'training-data', 'archive': True})
        ledger = spec['carried_ledger']
        ledger['budget_usd'] = spec['campaign_budget_usd']
        save(root / 'campaign-ledger.json', ledger)
        manifest['deployment'].update(attempt_seconds=28800, capacity_wait_seconds=1800,
            startup_wait_seconds=600, allow_long_lease=True, budget_usd=spec['campaign_budget_usd'],
            campaign_ledger=str(root / 'campaign-ledger.json'))
        manifest['deployment']['pricing']['observed_at'] = spec['pricing_observed_at']
        manifest_path = root / 'run.json'
        save(manifest_path, manifest)
        load_manifest(manifest_path)
        remote_manifest = immutable_upload(manifest_path, prefix + '/run.json')
        argv = supervisor_argv(manifest, remote_manifest['uri'], remote_manifest['sha256'], root / 'tpu-controller')
        argv[argv.index('--setup-timeout-seconds') + 1] = '1800'
        # The supervisor is invoked once; it arms the independent workflow before allocation.
        receipt['tpu_manifest'] = remote_manifest
        publish()
        run([python, '-m', 'scripts.gcp_spot_supervisor', *argv], 30900)
        controller_receipt = json.loads((root / 'tpu-controller/campaign-receipt.json').read_text())
        if controller_receipt.get('passed') is not True:
            raise ValueError('TPU supervisor did not report a passing campaign')
        if controller_receipt.get('cleanup_required') and not controller_receipt.get('cleanup_verified'):
            raise ValueError('TPU cleanup remains unverified')
        receipt.update(status='completed', phase='finished',
                       model_execution_state=controller_receipt.get('model_execution_state', 'not_inferred'),
                       tpu_cleanup_verified=controller_receipt.get('cleanup_verified', False))
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        if isinstance(error, subprocess.CalledProcessError):
            receipt['worker_error_tail'] = error.output
        raise
    finally:
        receipt['finished_unix'] = time.time()
        controller_receipt_path = root / 'tpu-controller/campaign-receipt.json'
        if controller_receipt_path.exists():
            try:
                controller_receipt = json.loads(controller_receipt_path.read_text())
                receipt['model_execution_state'] = controller_receipt.get('model_execution_state', 'not_inferred')
                receipt['tpu_cleanup_verified'] = controller_receipt.get('cleanup_verified', False)
            except (OSError, ValueError) as error:
                receipt['controller_receipt_error'] = str(error)
        save(root / 'campaign.json', receipt)
        # One archive prevents long command inventories exhausting the reserved
        # diagnostic window before reaching the failing worker's log.
        evidence = [root / 'campaign.json', root / 'campaign-ledger.json',
                    root / 'training-data/terminal-data.json', root / 'training-data/registry-progress.json',
                    root / 'stage-preflight.json', root / 'tpu-controller/campaign-receipt.json',
                    *sorted((root / 'controller-logs').glob('*')),
                    *sorted((root / 'training-data').glob('command-*.log'))]
        diagnostic = root / 'controller-diagnostics.tar.gz'
        try:
            with tarfile.open(diagnostic, 'w:gz') as archive:
                for path in evidence:
                    if path.is_file():
                        archive.add(path, arcname=str(path.relative_to(root)))
            run(['gcloud', 'storage', 'cp', diagnostic,
                 prefix + '/controller-final/controller-diagnostics.tar.gz'], 180, finalizing=True)
        except (OSError, subprocess.SubprocessError, TimeoutError) as error:
            receipt['diagnostic_upload_error'] = str(error)
            save(root / 'campaign.json', receipt)
        try:
            publish(finalizing=True)
        except (OSError, subprocess.SubprocessError, TimeoutError):
            pass


if __name__ == '__main__':
    main()
