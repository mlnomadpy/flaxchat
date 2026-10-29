"""Runtime boundary guards exercised without models, checkpoints or cloud calls."""
import json
import pytest

from flaxchat.checkpoint_metadata import MAX_JSON_BYTES, _describe
from scripts.gcp_cleanup_guard import preflight_cleanup_scope


@pytest.mark.parametrize('size', [None, True, -1, 'unknown', MAX_JSON_BYTES + 1])
def test_checkpoint_size_metadata_rejects_invalid_bounds(monkeypatch, size):
    monkeypatch.setattr('flaxchat.checkpoint_metadata.subprocess.check_output',
                        lambda *args, **kwargs: json.dumps({'generation': '123', 'size': size}).encode())
    with pytest.raises(ValueError, match='read bound'):
        _describe('gs://fixture/metadata')


@pytest.mark.parametrize('size', [0, '0', MAX_JSON_BYTES])
def test_checkpoint_size_metadata_accepts_provider_string_or_integer(monkeypatch, size):
    monkeypatch.setattr('flaxchat.checkpoint_metadata.subprocess.check_output',
                        lambda *args, **kwargs: json.dumps({'generation': '123', 'size': size}).encode())
    assert _describe('gs://fixture/metadata') == '123'


def test_cleanup_discovery_rejects_non_object_without_losing_failure_evidence(monkeypatch):
    calls = []

    def read(*args, **kwargs):
        calls.append(args)
        return '[]'

    monkeypatch.setattr('scripts.gcp_cleanup_guard.subprocess.check_output', read)
    with pytest.raises(ValueError, match='JSON object') as failure:
        preflight_cleanup_scope(project='project', zone='us-west4-a',
                                queue='flaxchat-validation-fixture')
    assert len(calls) == 1
    evidence = vars(failure.value)['cleanup_scope_discovery']
    assert evidence['reads'][0]['status'] == 'terminal_failure'
