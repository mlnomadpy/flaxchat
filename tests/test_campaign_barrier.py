import json
from types import SimpleNamespace
import pytest
from scripts import campaign_barrier as barrier


def marker(rank, payload=None, **extra):
    return json.dumps(dict(stage='ready', count=4, rank=rank, payload=payload, **extra))


def test_waits_for_all_hosts_and_returns_rank_order():
    assert barrier.parse_participants(marker(2), stage='ready', count=4) is None
    text = '\n'.join(marker(i, 100-i) for i in [3, 0, 2, 1])
    assert barrier.parse_participants(text, stage='ready', count=4) == [100, 99, 98, 97]


@pytest.mark.parametrize('text', [marker(0)+'\n'+marker(0), marker(4), marker(True), marker(0).replace('ready', 'wrong')])
def test_rejects_duplicate_rank_and_wrong_identity(text):
    with pytest.raises(ValueError):
        barrier.parse_participants(text, stage='ready', count=4)


def test_exchange_waits_and_does_not_hide_permission_failure(tmp_path, monkeypatch):
    replies = iter([SimpleNamespace(returncode=0),
                    SimpleNamespace(returncode=0, stdout=marker(0), stderr=''),
                    SimpleNamespace(returncode=1, stdout='', stderr='PERMISSION_DENIED')])
    monkeypatch.setattr(barrier.subprocess, 'run', lambda *a, **k: next(replies))
    monkeypatch.setattr(barrier.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='PERMISSION_DENIED'):
        barrier.exchange('gs://test/run', tmp_path, 'ready', 0, 4, {}, SimpleNamespace(remaining=lambda: 10))
    assert json.loads((tmp_path/'barrier-ready.json').read_text())['rank'] == 0


def test_exchange_deadline_stops_missing_peer(tmp_path, monkeypatch):
    def expired():
        raise TimeoutError('expired')
    with pytest.raises(TimeoutError, match='expired'):
        barrier.exchange('gs://test/run', tmp_path, 'ready', 0, 4, {}, SimpleNamespace(remaining=expired))


def test_single_host_needs_no_storage(tmp_path):
    assert barrier.exchange('unused', tmp_path, 'ready', 0, 1, 123, None) == [123]
