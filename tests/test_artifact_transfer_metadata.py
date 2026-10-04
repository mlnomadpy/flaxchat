"""Literal subprocess byte streams only; no cloud/model execution."""
import hashlib
import sys
import pytest
from scripts.diagnose_artifact_transfer import stream_transfer, validate_manifest


def manifest(value=b'abc'):
    return {'format': 'flaxchat-artifact-transfer-v1',
            'uri': 'gs://azettaai-yat-eval-0929/embedding-finetune-v1/release/model.safetensors#1790751315402234',
            'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest(),
            'independent_identity_sha256': 'a' * 64}


def literal(value=b'abc', suffix=''):
    return lambda command: [sys.executable, '-c',
                            'import sys;sys.stdout.buffer.write(' + repr(value) + ');sys.stdout.flush();' + suffix]


def run(tmp_path, **options):
    return stream_transfer(manifest(), output=tmp_path / 'artifact', range_bytes=3,
                           phase_seconds=2, total_seconds=40, finalization_margin_seconds=30,
                           **options)


def test_range_bytes_are_reported_but_never_whole_authenticated(tmp_path):
    progress = []
    result = run(tmp_path, command_factory=literal(), progress=progress.append)
    assert result['status'] == 'completed'
    assert result['bytes_received'] == 3
    assert result['observed_stream_sha256'] == manifest()['sha256']
    assert not result['whole_artifact_authenticated']
    assert not result['range_whole_artifact_identity_proven']
    assert not (tmp_path / 'artifact').exists()
    assert result['partial_removed'] and progress


def test_full_bytes_commit_only_after_independent_sha_match(tmp_path):
    result = run(tmp_path, mode='full', command_factory=literal())
    assert result['status'] == 'completed' and result['whole_artifact_authenticated']
    assert (tmp_path / 'artifact').read_bytes() == b'abc'


@pytest.mark.parametrize('payload', [b'ab', b'abcd', b'xyz'])
def test_short_oversized_and_wrongsha_never_commit(tmp_path, payload):
    result = run(tmp_path, mode='full', command_factory=literal(payload))
    assert result['status'] == 'failed'
    assert not result['whole_artifact_authenticated']
    assert not (tmp_path / 'artifact').exists()
    assert not (tmp_path / 'artifact.partial').exists()


def test_client_error_and_stderr_bound_are_observable_without_payload_leak(tmp_path):
    result = run(tmp_path, command_factory=literal(suffix="sys.stderr.write('x'*100000);sys.exit(9)"))
    assert result['status'] == 'failed' and result['client_returncode'] == 9
    assert result['stderr_bytes'] == 100000
    assert result['stderr_truncated'] and result['stderr_retained_bytes'] == 65536
    assert 'x' * 100 not in str(result)


def test_deadline_kills_same_process_group_and_removes_partial(tmp_path):
    result = stream_transfer(manifest(), output=tmp_path / 'artifact', range_bytes=3,
                             phase_seconds=1, total_seconds=40, finalization_margin_seconds=30,
                             command_factory=literal(b'a', 'import time;time.sleep(30)'))
    assert result['status'] == 'failed' and 'deadline' in result['error']
    assert result['bytes_received'] == 1 and result['elapsed_seconds'] < 5
    assert not (tmp_path / 'artifact.partial').exists()


def test_exact_range_command_and_generation(tmp_path):
    commands = []
    def factory(command):
        commands.append(command)
        return literal()(command)
    assert run(tmp_path, command_factory=factory)['status'] == 'completed'
    assert commands == [['gcloud', 'storage', 'cat', manifest()['uri'], '--range=0-2', '--quiet']]


@pytest.mark.parametrize('change', [{'uri': 'gs://bucket/file'}, {'uri': 'gs://bucket/file#0'},
    {'sha256': 'z' * 64}, {'bytes': True}, {'bytes': 2 * 1024**3 + 1},
    {'independent_identity_sha256': ''}])
def test_bad_pin_rejected_before_subprocess(change):
    with pytest.raises(ValueError):
        validate_manifest(manifest() | change)


def test_fresh_destination_and_finite_margin_required(tmp_path):
    (tmp_path / 'artifact').write_bytes(b'protected')
    with pytest.raises(ValueError, match='Fresh'):
        run(tmp_path, command_factory=lambda command: pytest.fail('No execution'))
    (tmp_path / 'artifact').unlink()
    with pytest.raises(ValueError, match='margin'):
        stream_transfer(manifest(), output=tmp_path / 'artifact', range_bytes=3,
                        phase_seconds=15, total_seconds=40, finalization_margin_seconds=30,
                        command_factory=lambda command: pytest.fail('No execution'))
