import os
import json
import subprocess
import sys

import pytest

from scripts.stage_progress import run_stage


@pytest.mark.parametrize("exit_code", [0, 7])
def test_snapshots_before_completion_preserve_exit_status(tmp_path, exit_code):
    snapshots = []

    def publish(path, remaining):
        snapshots.append(path.read_text())
        assert remaining > 0

    code = run_stage(
        [
            sys.executable,
            "-c",
            f"import time; print('started'); time.sleep(.3); raise SystemExit({exit_code})",
        ],
        log=tmp_path / "stage.log",
        env=os.environ,
        timeout=5,
        publish=publish,
        interval=0.05,
    )
    assert code == exit_code
    assert any("started" in snapshot for snapshot in snapshots)


def test_publisher_failure_does_not_kill_workload(tmp_path):
    states = []
    def publish(path, remaining):
        states.append(json.loads(path.with_suffix('.status.json').read_text()))
        raise subprocess.TimeoutExpired("upload", remaining)

    assert (
        run_stage(
            [sys.executable, "-c", "import time; time.sleep(.2); print('done')"],
            log=tmp_path / "stage.log",
            env=os.environ,
            timeout=5,
            publish=publish,
            interval=0.05,
        )
        == 0
    )
    assert states and all(s['state'] == 'publishing' for s in states)
    final = json.loads((tmp_path / 'stage.status.json').read_text())
    assert final['returncode'] == 0
    assert 'TimeoutExpired' in final['publication_error']
    assert "done" in (tmp_path / "stage.log").read_text()


def test_deadline_kills_and_reaps_child(tmp_path):
    log = tmp_path / "stage.log"
    with pytest.raises(subprocess.TimeoutExpired):
        run_stage(
            [
                sys.executable,
                "-c",
                "import os,time; print(os.getpid()); time.sleep(30)",
            ],
            log=log,
            env=os.environ,
            timeout=0.5,
            publish=lambda *_: None,
            interval=0.05,
        )
    assert json.loads(log.with_suffix('.status.json').read_text())['state'] == 'timed_out'
    pid = int(log.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_bounds_rejected_before_launch(tmp_path, value):
    with pytest.raises(ValueError):
        run_stage(
            ["does-not-exist"],
            log=tmp_path / "unused",
            env={},
            timeout=value,
            publish=lambda *_: None,
        )
    assert not (tmp_path / "unused").exists()


def test_monitor_does_not_write_to_transport(tmp_path, monkeypatch):
    def blocked_print(*args, **kwargs):
        pytest.fail('Monitor must not write to SSH stdout')
    monkeypatch.setattr('builtins.print', blocked_print)
    def fail_upload(*args):
        raise OSError('offline')
    assert run_stage([sys.executable, '-c', 'import time; time.sleep(.2)'],
                     log=tmp_path / 'stage.log', env=os.environ, timeout=5,
                     publish=fail_upload, interval=.02) == 0


def test_unexpected_publisher_error_is_recorded(tmp_path):
    def fail(*args):
        raise RuntimeError('unexpected')
    log = tmp_path / 'stage.log'
    with pytest.raises(RuntimeError, match='unexpected'):
        run_stage([sys.executable, '-c', 'import time; time.sleep(30)'],
                  log=log, env=os.environ, timeout=5, publish=fail, interval=.05)
    state = json.loads(log.with_suffix('.status.json').read_text())
    assert state['state'] == 'monitor_failed'
    assert state['error'] == 'RuntimeError: unexpected'
    with pytest.raises(ProcessLookupError):
        os.kill(state['pid'], 0)


def test_wall_deadline_expires_even_if_monotonic_has_time(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('scripts.stage_progress.time.time', lambda: clock[0])
    def advance_wall(*args):
        clock[0] += 100
    log = tmp_path / 'stage.log'
    with pytest.raises(subprocess.TimeoutExpired):
        run_stage([sys.executable, '-c', 'import time; time.sleep(30)'],
                  log=log, env=os.environ, timeout=30,
                  publish=advance_wall, interval=.02)
    assert json.loads(log.with_suffix('.status.json').read_text())['state'] == 'timed_out'
