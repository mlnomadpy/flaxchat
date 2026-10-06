"""Run the complete cloud controller against a tiny, in-memory fake provider.

Only process/cloud boundaries are faked: extraction, hashes, manifest validation,
manifest generation, receipt publication and diagnostic archives execute for real.
No model, accelerator or network calls are made.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from flaxchat.embedding_stage import add_stage_arguments
from scripts import representation_v2_cloud_campaign as campaign
from scripts.prepare_representation_v2_training import WEIGHTS


def archive_bytes(root):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    return data.getvalue()


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'infra/tpu').mkdir(parents=True)
    (source / 'scripts/prepare_representation_v2_training.py').write_text('frozen fixture recipe')
    (source / 'scripts/representation_v2_cloud_campaign.py').write_text('frozen fixture controller')
    (source / 'runtime.txt').write_text('fixture==1.0\n')
    (source / 'infra/tpu/representation-v2-direct-nodes.json').write_text(json.dumps({
        'tests/physical.py': ['tests.physical::must_run']}))
    cloud = {}

    def artifact(name, content, **extras):
        import hashlib
        uri = 'gs://fixture/inputs/' + name + '#1'
        cloud[uri] = content
        return {'uri': uri, 'sha256': hashlib.sha256(content).hexdigest(), **extras}

    source_artifact = artifact('source.tar.gz', archive_bytes(source))
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    (wheels / 'fixture.whl').write_bytes(b'placeholder; pip process is faked')
    wheel_artifact = artifact('wheels.tar.gz', archive_bytes(wheels), target='wheels', archive=True)
    spec = {'provider_delete_unix': time.time() + 43200,
        'output_prefix': 'gs://fixture/campaign', 'source': source_artifact,
        'portable': artifact('portable', b'portable'), 'heldout': artifact('heldout', b'heldout'),
        'carried_ledger': {'budget_usd': 0, 'attempts': {}}, 'campaign_budget_usd': 100,
        'pricing_observed_at': '2026-10-05T12:00:00+00:00',
        'base_manifest': {'schema_version': 1, 'artifacts': [wheel_artifact],
            'runtime_lock': {'path': 'runtime.txt', 'sha256': campaign.sha(source / 'runtime.txt'),
                             'wheelhouse_target': 'wheels'},
            'parent_identity': {'purpose': 'fixture'}, 'cleanup_owner': 'fixture-owner',
            'expected_topology': 'v5litepod-8', 'expected_device_count': 8, 'expected_process_count': 1,
            'deployment': {'project': 'fixture-project', 'zone': 'us-west4-a',
                'accelerator_type': 'v5litepod-8', 'runtime_version': 'v2-alpha-tpuv5-lite',
                'provisioning_model': 'spot', 'hourly_usd': 9.6,
                'pricing': {'sku': 'fixture', 'region': 'us-west4', 'provisioning_model': 'spot',
                    'source': 'fixture', 'observed_at': '2026-10-05T12:00:00+00:00',
                    'accelerator_type': 'v5litepod-8', 'currency': 'USD',
                    'unit': 'whole_slice_hour', 'hourly_usd': 9.6}}}}
    data = tmp_path / 'prepared'
    data.mkdir()
    quarantine = {'status': 'passed', 'exact_overlaps': 0, 'sources': {}}
    for name in WEIGHTS:
        folder = data / 'mixture' / name
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text('{}')
        quarantine['sources'][name] = {'train_rows': 3, 'manifest_sha256': campaign.sha(folder / 'manifest.json')}
    (data / 'final-quarantine.json').write_text(json.dumps(quarantine))
    (data / 'weights.json').write_text(json.dumps(WEIGHTS))
    terminal = {'status': 'passed', 'model_execution': False,
        'portable_sha256': spec['portable']['sha256'], 'original_heldout_sha256': spec['heldout']['sha256'],
        'source_sha256': campaign.sha(source / 'scripts/prepare_representation_v2_training.py'),
        'final_quarantine_sha256': campaign.sha(data / 'final-quarantine.json')}
    (data / 'terminal-data.json').write_text(json.dumps(terminal))
    spec['prepared_data'] = artifact('prepared.tar.gz', archive_bytes(data),
                                    source_sha256=source_artifact['sha256'])
    root = tmp_path / 'worker'
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))
    state = SimpleNamespace(spec=spec, cloud=cloud, root=root, calls=[], failure=None,
                            source=source, spec_path=spec_path, cleanup=True)

    def fake_command(argv, timeout, log_path, *, capture=False):
        argv = list(map(str, argv))
        state.calls.append(argv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('fixture command: ' + ' '.join(argv) + '\n')
        stdout = None
        if argv[:3] == ['gcloud', 'storage', 'cp']:
            src, dst = argv[3:5]
            if src.startswith('gs://'):
                Path(dst).write_bytes(cloud[src])
            else:
                if '--if-generation-match=0' in argv:
                    assert dst not in cloud, 'must not overwrite immutable artifacts'
                cloud[dst] = Path(src).read_bytes()
                cloud[dst + '#1'] = cloud[dst]
        elif argv[:4] == ['gcloud', 'storage', 'objects', 'describe']:
            assert argv[4] in cloud
            stdout = '{"generation": "1"}'
        elif 'scripts.preflight_yat_embedding_stage' in argv:
            parser = argparse.ArgumentParser()
            add_stage_arguments(parser)
            parser.add_argument('--expected-devices', type=int, required=True)
            parser.add_argument('--expected-processes', type=int, required=True)
            parser.add_argument('--receipt', type=Path, required=True)
            parsed = parser.parse_args(argv[3:])
            assert parsed.expected_devices == 8 and parsed.expected_processes == 1
            assert len(parsed.data) == 11 and parsed.steps == 20000
            assert json.loads(cloud[spec['output_prefix'] + '/campaign.json'])['data_artifact']
            if state.failure == 'preflight':
                log_path.write_text('REAL ORIGINAL PREFLIGHT FAILURE\n')
                raise subprocess.CalledProcessError(19, argv, output='REAL ORIGINAL PREFLIGHT FAILURE')
            parsed.receipt.write_text('{"passed": true}')
        elif 'scripts.gcp_spot_supervisor' in argv:
            assert argv[argv.index('--attempt-seconds') + 1] == '28800'
            assert argv[argv.index('--setup-timeout-seconds') + 1] == '1800'
            assert '--allow-long-lease' in argv
            target = Path(argv[argv.index('--output') + 1])
            target.mkdir(parents=True)
            (target / 'campaign-receipt.json').write_text(json.dumps({
                'passed': True, 'cleanup_required': True, 'cleanup_verified': state.cleanup,
                'model_execution_state': 'not_inferred'}))
        elif '-m' in argv and (argv[2] == 'venv' or argv[2] == 'pip'):
            pass
        else:
            raise AssertionError('Unexpected command: ' + repr(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout)

    monkeypatch.setattr(campaign, '__file__', str(source / 'scripts/representation_v2_cloud_campaign.py'))
    monkeypatch.setattr(campaign, 'run_logged', fake_command)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(sys, 'argv', ['campaign', '--spec', str(spec_path), '--root', str(root)])
    # main mutates process environment; restore it after each metadata test.
    for name in ('PYTHONPATH', 'CLOUDSDK_STORAGE_PROCESS_COUNT', 'CLOUDSDK_STORAGE_THREAD_COUNT',
                 'TOKENIZERS_PARALLELISM', 'HF_DATASETS_DISABLE_PROGRESS_BARS'):
        monkeypatch.setenv(name, '')
    return state


def test_complete_prepared_campaign_one_launch_and_durable_cleanup(scenario):
    campaign.main()
    receipt = json.loads(scenario.cloud['gs://fixture/campaign/campaign.json'])
    assert receipt['status'] == 'completed' and receipt['tpu_cleanup_verified'] is True
    assert receipt['model_execution_state'] == 'not_inferred'
    assert receipt['data_artifact']['uri'] == scenario.spec['prepared_data']['uri']
    assert len([call for call in scenario.calls if 'scripts.gcp_spot_supervisor' in call]) == 1
    assert not any('scripts.prepare_representation_v2_training' in call for call in scenario.calls)
    manifest = json.loads(scenario.cloud['gs://fixture/campaign/run.json'])
    assert manifest['qualification']['required_tests'] == ['tests.physical::must_run']
    assert manifest['artifacts'][-1]['target'] == 'training-data'
    assert manifest['deployment']['campaign_ledger'] == str(scenario.root / 'campaign-ledger.json')
    prior_calls = len(scenario.calls)
    with pytest.raises(FileExistsError):
        campaign.main()
    assert len(scenario.calls) == prior_calls


def test_preflight_failure_preserves_original_diagnostics_and_prepared_data(scenario):
    scenario.failure = 'preflight'
    with pytest.raises(subprocess.CalledProcessError) as error:
        campaign.main()
    assert error.value.returncode == 19
    receipt = json.loads(scenario.cloud['gs://fixture/campaign/campaign.json'])
    assert receipt['status'] == 'failed'
    assert receipt['worker_error_tail'] == 'REAL ORIGINAL PREFLIGHT FAILURE'
    assert receipt['data_artifact']['uri'] in scenario.cloud
    assert not any('scripts.gcp_spot_supervisor' in call for call in scenario.calls)
    diagnostics = scenario.cloud['gs://fixture/campaign/controller-final/controller-diagnostics.tar.gz']
    with tarfile.open(fileobj=io.BytesIO(diagnostics)) as archive:
        logs = [archive.extractfile(item).read() for item in archive if item.name.endswith('.log')]
    assert any(b'REAL ORIGINAL PREFLIGHT FAILURE' in log for log in logs)


def test_unverified_tpu_cleanup_cannot_report_completed(scenario):
    scenario.cleanup = False
    with pytest.raises(ValueError, match='cleanup remains unverified'):
        campaign.main()
    receipt = json.loads(scenario.cloud['gs://fixture/campaign/campaign.json'])
    assert receipt['status'] == 'failed' and receipt['tpu_cleanup_verified'] is False
