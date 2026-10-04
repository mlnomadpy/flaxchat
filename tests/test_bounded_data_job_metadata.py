"""Controller/process tests only; never imports a model or numerical runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import bounded_data_job as jobs


def spec(tmp_path, code='print("data-only")', timeout=5):
    runner = tmp_path / 'runner.py'
    runner.write_text(code)
    return {'schema_version': 1, 'job_id': 'literal-data-01',
            'root': str(tmp_path / 'flaxchat-data-literal-01'),
            'controller_sha256': jobs.digest(jobs.__file__), 'timeout_seconds': timeout,
            'kill_after_seconds': 1, 'command': [sys.executable, '{input:runner.py}'],
            'inputs': [{'name': 'runner.py', 'path': str(runner), 'bytes': runner.stat().st_size,
                        'sha256': jobs.digest(runner)}]}


def prepared(tmp_path, code='print("data-only")', timeout=5):
    value = spec(tmp_path, code, timeout)
    root = Path(value['root'])
    root.mkdir()
    (root / 'inputs').mkdir()
    (root / 'inputs/runner.py').write_bytes((tmp_path / 'runner.py').read_bytes())
    (root / 'work').mkdir()
    jobs.write_once(root / 'spec.json', value)
    admission = {'spec_sha256': hashlib.sha256(jobs.canonical(value)).hexdigest(),
                 'owner': 'a' * 32, 'job_id': value['job_id']}
    jobs.write_once(root / 'admission.json', admission)
    return root, value, admission


def test_import_has_no_numerical_runtime():
    subprocess.run([sys.executable, '-c', "import sys; import scripts.bounded_data_job; assert not {'jax','flax','numpy','torch'} & set(sys.modules)"], check=True, timeout=10)


def test_literal_command_and_frozen_inventory(tmp_path):
    value = spec(tmp_path)
    assert jobs.validate(value) == Path(value['root'])
    value['command'] = 'python runner.py'
    with pytest.raises(ValueError, match='literal argv'):
        jobs.validate(value)


@pytest.mark.parametrize('key,value', [('timeout_seconds', True), ('timeout_seconds', 0),
                                     ('timeout_seconds', 86401), ('kill_after_seconds', 61)])
def test_finite_bounds(tmp_path, key, value):
    contract = spec(tmp_path)
    contract[key] = value
    with pytest.raises(ValueError, match='Finite'):
        jobs.validate(contract)


def test_controller_and_consumed_source_required(tmp_path):
    value = spec(tmp_path)
    value['controller_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Controller source'):
        jobs.validate(value)
    value['controller_sha256'] = jobs.digest(jobs.__file__)
    value['command'] = [sys.executable, '-c', 'pass']
    with pytest.raises(ValueError, match='consume'):
        jobs.validate(value)


def test_symlinks_and_duplicate_inventory_rejected(tmp_path):
    value = spec(tmp_path)
    link = tmp_path / 'link.py'
    link.symlink_to(tmp_path / 'runner.py')
    value['inputs'][0]['path'] = str(link)
    with pytest.raises(ValueError, match='without symlinks'):
        jobs.validate(value)
    value['inputs'][0]['path'] = str(tmp_path / 'runner.py')
    value['inputs'].append(dict(value['inputs'][0]))
    with pytest.raises(ValueError, match='Unique flat'):
        jobs.validate(value)


def test_receipt_commit_cannot_overwrite(tmp_path):
    path = tmp_path / 'receipt.json'
    jobs.write_once(path, {'first': True})
    with pytest.raises(FileExistsError):
        jobs.write_once(path, {'first': False})
    assert json.loads(path.read_text()) == {'first': True}
    assert list(tmp_path.iterdir()) == [path]


def test_missing_attachment_is_observation_only(tmp_path):
    root, _, admission = prepared(tmp_path)
    before = set(root.iterdir())
    assert jobs.observe(root, admission['spec_sha256']) == {'state': 'startup_unobserved', 'retry_permitted': False}
    assert set(root.iterdir()) == before


def test_reused_pid_not_live_and_foreign_receipt_rejected(tmp_path, monkeypatch):
    root, _, admission = prepared(tmp_path)
    identity = {'pid': 123, 'boot_id': 'boot', 'start_ticks': '10'}
    jobs.write_once(root / 'ownership.json', admission | {'supervisor': identity, 'timeout_process': identity})
    monkeypatch.setattr(jobs, 'process_identity', lambda pid: identity | {'start_ticks': '11'})
    result = jobs.observe(root, admission['spec_sha256'])
    assert result['state'] == 'terminal_receipt_missing' and result['retry_permitted'] is False
    jobs.write_once(root / 'terminal.json', admission | {'owner': 'foreign'})
    with pytest.raises(ValueError, match='Foreign terminal'):
        jobs.observe(root, admission['spec_sha256'])


def test_durable_worker_failure_authoritative_even_if_attachment_succeeded(tmp_path):
    root, _, admission = prepared(tmp_path)
    jobs.write_once(root / 'terminal.json', admission | {'status': 'execution_failed', 'worker_returncode': 17})
    result = jobs.observe(root, admission['spec_sha256'])
    assert result['state'] == 'terminal' and result['worker']['worker_returncode'] == 17
    assert result['retry_permitted'] is False


def test_linux_identity_zombie_and_start_ticks(tmp_path):
    proc = tmp_path / 'proc'
    entry = proc / '123'
    entry.mkdir(parents=True)
    (proc / 'sys/kernel/random').mkdir(parents=True)
    (proc / 'sys/kernel/random/boot_id').write_text('boot1')
    fields = ['S'] + ['0'] * 18 + ['555']
    (entry / 'stat').write_text('123 (name with ) parentheses) ' + ' '.join(fields))
    assert jobs.process_identity(123, proc) == {'pid': 123, 'boot_id': 'boot1', 'start_ticks': '555'}
    fields[0] = 'Z'
    (entry / 'stat').write_text('123 (name) ' + ' '.join(fields))
    assert jobs.process_identity(123, proc) is None


def test_owned_cleanup_requires_both_exact_tags(tmp_path, monkeypatch):
    proc = tmp_path / 'proc'
    (proc / 'sys/kernel/random').mkdir(parents=True)
    (proc / 'sys/kernel/random/boot_id').write_text('boot')
    for pid, env in [(123, b'FLAXCHAT_DATA_JOB_OWNER=owner\0TMPDIR=/owned/work\0'),
                     (124, b'FLAXCHAT_DATA_JOB_OWNER=other\0TMPDIR=/owned/work\0'),
                     (125, b'FLAXCHAT_DATA_JOB_OWNER=owner\0TMPDIR=/foreign/work\0')]:
        entry = proc / str(pid)
        entry.mkdir()
        (entry / 'environ').write_bytes(env)
        (entry / 'stat').write_text(str(pid) + ' (python) ' + ' '.join(['S'] + ['0'] * 18 + ['555']))
    assert [item['pid'] for item in jobs.owned_processes('owner', Path('/owned/work'), proc)] == [123]


@pytest.mark.parametrize('code,expected', [('print("data-only")', 0), ('raise SystemExit(17)', 17),
                                         ('import time; time.sleep(30)', 124)])
def test_real_bounded_data_subprocess_and_terminal_receipt(tmp_path, monkeypatch, code, expected):
    root, _, admission = prepared(tmp_path, code, timeout=1)
    # macOS metadata suite may use GNU timeout, but does not fabricate Linux
    # qualification: process identity/cleanup below are explicit isolated fakes.
    monkeypatch.setattr(jobs, 'process_identity', lambda pid: {'pid': pid, 'boot_id': 'fixture', 'start_ticks': '1'})
    monkeypatch.setattr(jobs, 'owned_processes', lambda owner, scratch: [])
    assert jobs.supervise(root) == (0 if expected == 0 else 1)
    result = jobs.observe(root, admission['spec_sha256'])['worker']
    assert result['worker_returncode'] == expected
    assert result['cleanup_verified'] is True
    assert result['model_execution_qualified'] is False
    with pytest.raises(FileExistsError):
        jobs.supervise(root)


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Real detached ownership requires Linux; not qualified on macOS')
def test_actual_detached_linux_launch_reconnect_and_duplicate_rejection(tmp_path):
    value = spec(tmp_path, 'import time; time.sleep(1); print("finished")', timeout=5)
    jobs.launch(value)
    with pytest.raises(FileExistsError):
        jobs.launch(value)
    root = Path(value['root'])
    admission = {'spec_sha256': hashlib.sha256(jobs.canonical(value)).hexdigest()}
    until = time.monotonic() + 10
    seen_live = False
    while time.monotonic() < until:
        observation = jobs.observe(root, admission['spec_sha256'])
        seen_live |= observation['state'] == 'live'
        if observation['state'] == 'terminal':
            break
        time.sleep(.05)
    assert seen_live
    assert observation['worker']['worker_returncode'] == 0
    assert observation['worker']['cleanup_verified'] is True


def test_external_admission_pin_rejects_rewritten_local_identity(tmp_path):
    root, _, admission = prepared(tmp_path)
    with pytest.raises(ValueError, match='Admission identity'):
        jobs.observe(root, '0' * 64)


def test_cleanup_error_remains_durable_failure(tmp_path, monkeypatch):
    root, _, admission = prepared(tmp_path)
    monkeypatch.setattr(jobs, 'process_identity', lambda pid: {'pid': pid, 'boot_id': 'fixture', 'start_ticks': '1'})
    def unavailable(owner, scratch):
        raise PermissionError('fixture ownership read unavailable')
    monkeypatch.setattr(jobs, 'owned_processes', unavailable)
    assert jobs.supervise(root) == 1
    worker = jobs.observe(root, admission['spec_sha256'])['worker']
    assert worker['worker_returncode'] == 0
    assert worker['cleanup_verified'] is False
    assert worker['cleanup_error_type'] == 'PermissionError'
    assert (root / 'work').is_dir()


def test_mutated_frozen_source_is_rejected_before_worker(tmp_path, monkeypatch):
    root, _, admission = prepared(tmp_path)
    (root / 'inputs/runner.py').write_text('raise SystemExit(0)')
    monkeypatch.setattr(jobs, 'owned_processes', lambda owner, scratch: [])
    assert jobs.supervise(root) == 1
    worker = jobs.observe(root, admission['spec_sha256'])['worker']
    assert worker['error_type'] == 'ValueError'
    assert 'worker_returncode' not in worker
    assert not (root / 'worker.log').exists()
    assert worker['cleanup_verified'] is True


def test_hashing_is_included_in_finite_authentication_deadline(tmp_path, monkeypatch):
    data = tmp_path / 'literal-bytes'
    data.write_bytes(b'source')
    monkeypatch.setattr(jobs.time, 'monotonic', lambda: 10)
    with pytest.raises(TimeoutError, match='authentication deadline'):
        jobs.digest(data, deadline=9)


def test_observe_api_omission_and_invalid_pin_fail_before_reading_root(tmp_path):
    nonexistent = tmp_path / 'no-remote-files'
    with pytest.raises(TypeError):
        jobs.observe(nonexistent)
    for pin in (None, '', 'z' * 64, 'a' * 63, True):
        with pytest.raises(ValueError, match='Independently retained'):
            jobs.observe(nonexistent, pin)
    assert not nonexistent.exists()


def test_observe_cli_omission_fails_before_reading_root(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        jobs.main(['observe', '--root', str(tmp_path / 'missing')])
    assert error.value.code == 2
    assert 'requires --expected-spec-sha256' in capsys.readouterr().err


def test_self_consistent_spec_admission_and_terminal_rewrite_cannot_replace_external_pin(tmp_path):
    root, original, admission = prepared(tmp_path)
    external_pin = admission['spec_sha256']
    changed = dict(original, timeout_seconds=original['timeout_seconds'] + 1)
    changed_pin = hashlib.sha256(jobs.canonical(changed)).hexdigest()
    (root / 'spec.json').write_text(json.dumps(changed))
    (root / 'admission.json').write_text(json.dumps(admission | {'spec_sha256': changed_pin}))
    jobs.write_once(root / 'terminal.json', admission | {'spec_sha256': changed_pin,
                                                      'status': 'execution_succeeded', 'worker_returncode': 0})
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match='Admission identity'):
        jobs.observe(root, external_pin)
    assert {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()} == before
