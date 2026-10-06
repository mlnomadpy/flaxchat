"""Real provenance ordering with mocked commands; never initialize a model."""
import json
import os
import subprocess
from unittest.mock import patch

import pytest

from scripts import representation_run as runner
from scripts.evaluation_contract import provenance


def context(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    lock = source / 'lock.txt'
    lock.write_text('example==1.0\n')
    python = tmp_path / 'python'
    python.write_text('verified-interpreter-fixture')
    identity = 'a' * 64
    (tmp_path / 'manifest-identity.txt').write_text(identity + '\n')
    (tmp_path / 'hardware-receipt.json').write_text(json.dumps({'backend': 'tpu', 'devices': 8}))
    environment = {'FLAXCHAT_RUN_ROOT': str(tmp_path), 'FLAXCHAT_RUN_MANIFEST_SHA256': identity,
                   'FLAXCHAT_RUNTIME_LOCK_SHA256': runner.digest(lock)}
    arguments = (dict(setup_checks=['{python}', 'qualification'], setup_checks_timeout_seconds=120),
                 tmp_path, source, identity, str(python), '[{"name":"example","version":"1.0"}]', lock, environment)
    return arguments


@pytest.mark.parametrize('failure', [None, 'command', 'acceptance', 'source'])
def test_provisional_provenance_precedes_checks_but_cannot_admit_workload(tmp_path, failure):
    arguments = context(tmp_path)
    captured = {}

    def check(*args, **kwargs):
        assert not (tmp_path / 'runtime-receipt.json').exists()
        provisional = json.loads((tmp_path / 'runtime-setup-receipt.json').read_text())
        assert provisional['runtime_verified'] is True
        assert provisional['setup_checks_executed'] is False
        assert provisional['physical_acceptance'] is False
        with patch.dict(os.environ, kwargs['env'], clear=True):
            captured.update(provenance([], devices=[{'platform': 'tpu'}], batch_size=0))
        # Copying provisional bytes to the final path cannot launch a workload.
        (tmp_path / 'runtime-receipt.json').write_text(json.dumps(provisional))
        with patch.object(runner, 'load_manifest', return_value=arguments[0]), pytest.raises(ValueError, match='Setup receipt'):
            runner.run_worker(tmp_path / 'unused-manifest', tmp_path)
        (tmp_path / 'runtime-receipt.json').unlink()
        if failure == 'command':
            raise subprocess.CalledProcessError(1, ['qualification'])
        if failure == 'source':
            (arguments[2] / 'changed.py').write_text('changed during qualification')

    with patch.object(runner.platform, 'platform', return_value='metadata-test-platform'), \
            patch.object(runner.subprocess, 'check_output', return_value='Python 3.12.12'), \
            patch.object(runner.subprocess, 'run', side_effect=check), \
            patch.object(runner, 'verify_qualification', return_value={'passed': True}) as verify:
        if failure == 'acceptance':
            verify.side_effect = ValueError('physical qualification rejected')
        if failure:
            with pytest.raises((ValueError, subprocess.CalledProcessError)):
                runner.qualify_setup_runtime(*arguments)
            assert not (tmp_path / 'runtime-receipt.json').exists()
        else:
            runner.qualify_setup_runtime(*arguments)
            final = json.loads((tmp_path / 'runtime-receipt.json').read_text())
            assert final['setup_checks_executed'] is True
            assert final['physical_acceptance'] is True
            assert final['setup_runtime_receipt_sha256'] == runner.digest(tmp_path / 'runtime-setup-receipt.json')
    assert captured['deployment_receipts_sha256']['runtime-setup-receipt.json'] == runner.digest(tmp_path / 'runtime-setup-receipt.json')


def test_normal_provenance_never_falls_back_to_provisional_receipt(tmp_path):
    arguments = context(tmp_path)
    with patch.dict(os.environ, arguments[-1], clear=True):
        with pytest.raises(ValueError, match='runtime-receipt.json'):
            provenance([], devices=[], batch_size=0)


def test_setup_provenance_rejects_wrong_manifest_or_false_acceptance(tmp_path):
    arguments = context(tmp_path)
    environment = {**arguments[-1], 'FLAXCHAT_SETUP_QUALIFICATION': '1'}
    record = dict(schema_version=2, runtime_verified=True, setup_checks_executed=False,
                  physical_acceptance=False, qualification=None, manifest_sha256='a' * 64,
                  runtime_lock_sha256=arguments[-1]['FLAXCHAT_RUNTIME_LOCK_SHA256'])
    for changed in ({'manifest_sha256': 'b' * 64}, {'physical_acceptance': True},
                    {'setup_checks_executed': True}, {'runtime_verified': False}):
        (tmp_path / 'runtime-setup-receipt.json').write_text(json.dumps({**record, **changed}))
        with patch.dict(os.environ, environment, clear=True), pytest.raises(ValueError, match='Invalid provisional'):
            provenance([], devices=[], batch_size=0)
