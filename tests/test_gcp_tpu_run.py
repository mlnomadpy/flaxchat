import shlex

import pytest

from scripts.gcp_tpu_run import worker_command


def test_command_quotes_literal_shell_characters_and_sets_rank():
    result = worker_command(
        ["python", "-c", 'print("$(do-not-run)")'],
        rank=1,
        count=4,
        coordinator="10.0.0.1:1234",
        directory="/tmp/space dir",
        timeout=60,
        distributed=True,
    )
    # Inspect the shell-quoted detached runner, then its quoted workload argv.
    start = result.index("nohup bash -c ")
    runner = shlex.split(result[start:])[3]
    assert "cd '/tmp/space dir'" in runner
    tokens = shlex.split(runner.split(" && ")[1].split("); command_status")[0])
    assert tokens[-1] == 'print("$(do-not-run)")'
    assert "JAX_PROCESS_INDEX=1" in tokens
    assert "FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS=60" in tokens
    assert tokens[:3] == ["timeout", "--signal=KILL", "60"]


def test_setup_does_not_initialize_distributed_runtime():
    result = worker_command(
        ["true"],
        rank=0,
        count=4,
        coordinator="host:123",
        directory="/tmp",
        timeout=60,
        distributed=False,
    )
    assert "JAX_" not in result


@pytest.mark.parametrize("rank,count,timeout", [(4, 4, 60), (0, 0, 60), (0, 4, 0)])
def test_invalid_worker_coordinates(rank, count, timeout):
    with pytest.raises(ValueError):
        worker_command(
            ["true"],
            rank=rank,
            count=count,
            coordinator="host:123",
            directory="/tmp",
            timeout=timeout,
            distributed=True,
        )


def test_remote_failure_is_reported_without_ssh_retry_and_command_is_once(tmp_path):
    import subprocess
    import sys
    import uuid
    from scripts.gcp_tpu_run import remote_returncode

    execution_id = uuid.uuid4().hex
    counter = tmp_path / "counter"
    command = worker_command(
        [
            sys.executable,
            "-c",
            'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(p.read_text()+"x" if p.exists() else "x"); sys.exit(255)',
            str(counter),
        ],
        rank=0,
        count=1,
        coordinator="",
        directory=str(tmp_path),
        timeout=5,
        distributed=False,
        execution_id=execution_id,
    )
    # Test the shell protocol on macOS too, without depending on GNU timeout.
    command = command.replace("timeout --signal=KILL 5 ", "")
    first = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    replay = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert first.returncode == replay.returncode == 0
    assert remote_returncode(first.stdout, execution_id, 0, 0) == 255
    assert remote_returncode(replay.stdout, execution_id, 0, 0) == 255
    assert counter.read_text() == "x"


def test_remote_success_requires_one_matching_completion_marker():
    from scripts.gcp_tpu_run import remote_returncode

    identity = "a" * 32
    marker = f"FLAXCHAT_REMOTE_EXIT_{identity}_2=0\n"
    assert remote_returncode(marker, identity, 2, 0) == 0
    assert remote_returncode("", identity, 2, 0) == 125
    assert remote_returncode(marker * 2, identity, 2, 0) == 0
    assert remote_returncode(marker + marker.replace("=0", "=1"), identity, 2, 0) == 125
    assert remote_returncode(marker, identity, 1, 0) == 125
    assert remote_returncode(marker, identity, 2, 255) == 255


def test_fault_injection_targets_coordination_rank_despite_tpu_reordering():
    from scripts.validate_multihost_interruption import coordinator_process

    assert coordinator_process({"JAX_PROCESS_INDEX": "0"}, runtime_rank=3)
    assert not coordinator_process({"JAX_PROCESS_INDEX": "2"}, runtime_rank=0)
    assert coordinator_process({}, runtime_rank=0)


def test_single_worker_does_not_claim_a_distributed_coordinator():
    result = worker_command(
        ["python", "-m", "pytest"],
        rank=0,
        count=1,
        coordinator="host:123",
        directory="/tmp",
        timeout=60,
        distributed=True,
    )
    assert "JAX_COORDINATOR_ADDRESS=" not in result
    assert "JAX_PROCESS_COUNT=" not in result
    assert "JAX_PROCESS_INDEX=" not in result


def test_cpu_validation_isolates_coordinator_and_splash_precision():
    from scripts.validate_encoder_tpu import stage_environment

    source = dict(
        JAX_COORDINATOR_ADDRESS="host:123",
        JAX_PROCESS_INDEX="0",
        JAX_PROCESS_COUNT="2",
        TPU_WORKER_HOSTNAMES="a,b",
        TPU_WORKER_ID="0",
    )
    env = stage_environment(source, cpu=True)
    assert not set(source) & set(env)
    assert env["JAX_PLATFORMS"] == "cpu"
    assert source["JAX_PROCESS_COUNT"] == "2"
    assert (
        stage_environment({}, matmul_precision="default")[
            "JAX_DEFAULT_MATMUL_PRECISION"
        ]
        == "default"
    )


def test_transport_wall_deadline_survives_host_sleep(tmp_path, monkeypatch):
    import subprocess
    import sys
    import threading
    from scripts import gcp_tpu_run as runner

    launched = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    wall = iter([1000.0, 1100.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(wall))
    with (tmp_path / "worker.log").open("w") as log:
        code = runner.run_transport(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            log,
            30,
            threading.Event(),
        )
    assert code == 124
    assert launched[0].poll() is not None


def test_transport_preserves_failure_and_cancels_promptly(tmp_path):
    import sys
    import threading
    from scripts.gcp_tpu_run import run_transport

    cancelled = threading.Event()
    with (tmp_path / "worker.log").open("w") as log:
        assert (
            run_transport(
                [sys.executable, "-c", "raise SystemExit(7)"], log, 10, cancelled
            )
            == 7
        )
        cancelled.set()
        assert (
            run_transport(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                log,
                10,
                cancelled,
            )
            == 124
        )


def test_iap_transport_is_explicit_and_connection_bounded():
    from scripts.gcp_tpu_run import ssh_command

    for iap in (False, True):
        command = ssh_command(
            "node", "project", "zone", 2, "true", tunnel_through_iap=iap
        )
        assert ("--tunnel-through-iap" in command) == iap
        assert ("alpha" in command) == iap
        assert "--ssh-flag=-o ConnectTimeout=20" in command
        assert command[command.index("--worker") + 1] == "2"


def test_nested_entrypoint_and_child_import_frozen_checkout(tmp_path):
    import os
    import subprocess
    import sys
    import uuid
    from scripts.gcp_tpu_run import remote_returncode

    checkout = tmp_path / "frozen checkout"
    package = checkout / "scripts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "probe.py").write_text('VALUE = "frozen"\n')
    nested = checkout / "artifacts" / "qualification"
    nested.mkdir(parents=True)
    (nested / "run.py").write_text("""import subprocess, sys
from scripts.probe import VALUE
assert VALUE == 'frozen'
subprocess.run([sys.executable, '-c', "from scripts.probe import VALUE; assert VALUE == 'frozen'"], check=True)
print('frozen imports passed')
""")
    stale = tmp_path / "stale"
    (stale / "scripts").mkdir(parents=True)
    (stale / "scripts" / "__init__.py").write_text('raise RuntimeError("stale import")')
    identity = uuid.uuid4().hex
    command = worker_command(
        [sys.executable, "artifacts/qualification/run.py"],
        rank=0,
        count=1,
        coordinator="",
        directory=str(checkout),
        timeout=5,
        distributed=False,
        execution_id=identity,
    ).replace("timeout --signal=KILL 5 ", "")
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(stale)},
        timeout=10,
    )
    assert result.returncode == 0
    assert remote_returncode(result.stdout, identity, 0, 0) == 0, result.stderr
    assert "frozen imports passed" in result.stdout


@pytest.mark.parametrize("cancel_peers", [False, True])
def test_later_rank_failure_cancels_waiting_peers(tmp_path, monkeypatch, cancel_peers):
    import json
    from scripts import gcp_tpu_run as runner

    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            dict(
                state="READY",
                networkEndpoints=[{"ipAddress": "10.0.0.1"}, {"ipAddress": "10.0.0.2"}],
            )
        ),
    )

    def transport(argv, log, timeout, cancelled):
        rank = int(argv[argv.index("--worker") + 1])
        if rank == 1:
            log.write("worker failed before rendezvous\n")
            return 2
        if cancel_peers:
            assert cancelled.wait(2), (
                "A later worker failure must not wait for rank zero"
            )
            return 124
        assert not cancelled.wait(0.05)
        import re

        identity = re.search(r"FLAXCHAT_REMOTE_EXIT_([0-9a-f]{32})_0", argv[-1]).group(
            1
        )
        log.write(f"FLAXCHAT_REMOTE_EXIT_{identity}_0=0\n")
        return 0

    monkeypatch.setattr(runner, "run_transport", transport)
    output = tmp_path / "result"
    assert (
        runner.main(
            [
                *(["--cancel-peers-on-failure"] if cancel_peers else []),
                "--project",
                "fixture",
                "--zone",
                "fixture",
                "--node",
                "fixture",
                "--output",
                str(output),
                "--",
                "true",
            ]
        )
        == 1
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["passed"] is False
    for row in report['workers']:
        assert row['seconds'] >= 0 and row['wall_seconds'] >= 0
        assert row['wall_minus_monotonic_seconds'] == row['wall_seconds'] - row['seconds']
    assert [(row["rank"], row["returncode"]) for row in report["workers"]] == [
        (0, 124 if cancel_peers else 0),
        (1, 2),
    ]


def test_worker_survives_lost_attachment_and_reconnects_once(tmp_path):
    import subprocess
    import sys
    import time
    import uuid
    from scripts.gcp_tpu_run import remote_returncode
    identity = uuid.uuid4().hex
    counter = tmp_path / 'counter'
    cmd = worker_command([sys.executable, '-c',
        'import time,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(p.read_text()+"x" if p.exists() else "x"); time.sleep(2)', str(counter)],
        rank=0,count=1,coordinator='',directory=str(tmp_path),timeout=5,distributed=False,execution_id=identity)
    cmd = cmd.replace('timeout --signal=KILL 5 ', '')
    with (tmp_path/'first.log').open('w') as log:
        first = subprocess.Popen(['bash','-c',cmd],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+5
            while not counter.exists() and time.monotonic()<deadline:
                time.sleep(.02)
            assert counter.exists()
        finally:
            first.kill()
            first.wait(timeout=5)
    result = subprocess.run(['bash','-c',cmd],capture_output=True,text=True,timeout=10)
    assert remote_returncode(result.stdout,identity,0,result.returncode)==0
    assert counter.read_text()=='x'


def test_iap_fallback_reattaches_same_command_with_remaining_wall_budget(monkeypatch):
    import io
    import threading
    from scripts import gcp_tpu_run as runner
    clock = {'mono': 0., 'wall': 1000.}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['mono'])
    monkeypatch.setattr(runner.time, 'time', lambda: clock['wall'])
    calls = []
    def transport(argv, log, timeout, cancelled):
        calls.append((argv, timeout))
        if len(calls) == 1:
            clock['mono'] += 5
            clock['wall'] += 95
            return 255
        return 0
    monkeypatch.setattr(runner, 'run_transport', transport)
    remote = 'the-same-durable-execution-command'
    code, attempts = runner.attach_worker('node', 'project', 'zone', 0, remote,
        io.StringIO(), 100, threading.Event(), iap_fallback=True)
    assert code == 0 and [c[1] for c in calls] == [90, 5]
    assert calls[0][0][-1] == calls[1][0][-1] == remote
    assert '--tunnel-through-iap' not in calls[0][0]
    assert '--tunnel-through-iap' in calls[1][0]
    assert [a['route'] for a in attempts] == ['public', 'iap']


def test_long_running_public_attachment_does_not_switch_to_iap(monkeypatch):
    import io
    import threading
    from scripts import gcp_tpu_run as runner

    clock = {'value': 0.}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['value'])
    monkeypatch.setattr(runner.time, 'time', lambda: clock['value'])
    calls = []

    def transport(argv, log, timeout, cancelled):
        calls.append((argv, timeout))
        log.write('FLAXCHAT_REMOTE_STARTED_fixture_0=1\n')
        if len(calls) == 1:
            clock['value'] += 90
            return 124
        return 0

    monkeypatch.setattr(runner, 'run_transport', transport)
    code, attempts = runner.attach_worker('node', 'project', 'zone', 0,
        'same-command', io.StringIO(), 600, threading.Event(), iap_fallback=True)
    assert code == 0
    assert [a['route'] for a in attempts] == ['public', 'public']
    assert [a['remote_started'] for a in attempts] == [True, True]
    assert [c[1] for c in calls] == [90, 300]


@pytest.mark.parametrize('mode', ['remote_failure', 'cancelled', 'expired', 'iap_only'])
def test_iap_fallback_does_not_retry_unrecoverable_or_completed_attachment(monkeypatch, mode):
    import io
    import threading
    from scripts import gcp_tpu_run as runner
    clock = {'value': 0.}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['value'])
    monkeypatch.setattr(runner.time, 'time', lambda: clock['value'])
    cancelled = threading.Event()
    calls = []
    def transport(argv, log, timeout, event):
        calls.append(argv)
        if mode == 'remote_failure':
            log.write('FLAXCHAT_REMOTE_EXIT_fixture_0=2\n')
            return 0
        if mode == 'cancelled':
            event.set()
        if mode == 'expired':
            clock['value'] = 101
        return 255
    monkeypatch.setattr(runner, 'run_transport', transport)
    runner.attach_worker('node', 'project', 'zone', 0, 'same-command', io.StringIO(),
                         100, cancelled, iap_fallback=True,
                         tunnel_through_iap=mode == 'iap_only')
    assert len(calls) == 1
