"""Controller orchestration checks; no cloud allocation or model execution."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_mlm_data_controller_gce as controller


def arguments(tmp_path):
    spec = {"root": str(tmp_path / "job"), "timeout_seconds": 5}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    token = tmp_path / "private-token"
    token.write_text("fixture")
    return SimpleNamespace(
        spec=path,
        expected_spec_sha256=hashlib.sha256(controller.canonical(spec)).hexdigest(),
        output=tmp_path / "output",
        prefix="gs://fixture/job",
        token_file=token,
        instance="fixture-instance",
        project="fixture-project",
        zone="fixture-zone",
    )


def test_transport_failure_keeps_observing_and_cleans_exact_instance(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    monkeypatch.setattr(controller, "launch", lambda spec: {})
    states = iter(
        [
            {"state": "running"},
            {"state": "terminal", "worker": {"cleanup_verified": True}},
        ]
    )
    monkeypatch.setattr(controller, "observe", lambda *a: next(states))
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "storage":
            raise controller.subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(controller.subprocess, "run", run)
    controller.main(args)
    assert not args.token_file.exists()
    assert calls[-1] == [
        "gcloud",
        "compute",
        "instances",
        "delete",
        "fixture-instance",
        "--project=fixture-project",
        "--zone=fixture-zone",
        "--quiet",
    ]
    assert (
        json.loads((args.output / "observation.json").read_text())["state"]
        == "terminal"
    )


def test_worker_launch_failure_also_erases_token_and_deletes_owned_vm(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)

    def fail(spec):
        raise RuntimeError("fixture admission failure")

    monkeypatch.setattr(controller, "launch", fail)
    calls = []
    monkeypatch.setattr(
        controller, "best_effort", lambda argv, seconds=120: calls.append(argv)
    )
    with pytest.raises(RuntimeError, match="admission failure"):
        controller.main(args)
    assert not args.token_file.exists()
    assert calls[0][4] == "fixture-instance"


def test_terminal_worker_failure_preserves_corpus_before_delete(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.corpus_root = tmp_path / "corpus"
    args.corpus_root.mkdir()
    args.backup_seconds = 900
    monkeypatch.setattr(controller, "launch", lambda spec: {})
    monkeypatch.setattr(
        controller,
        "observe",
        lambda *a: {"state": "terminal", "worker": {"cleanup_verified": True}},
    )
    calls = []

    def transport(argv, seconds=120):
        calls.append((argv, seconds))
        return True

    monkeypatch.setattr(controller, "best_effort", transport)
    controller.main(args)
    backup = [item for item in calls if args.prefix + "/corpus" in item[0]]
    assert backup == [
        (
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                str(args.corpus_root),
                args.prefix + "/corpus",
            ],
            900,
        )
    ]
    assert calls[-1][0][1:4] == ["compute", "instances", "delete"]
    assert json.loads((args.output / "preservation.json").read_text())[
        "corpus_sync_succeeded"
    ]


def test_isolated_launch_uses_distinct_persistent_memory_limited_unit(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    args.workload_unit = "fixture-data-workload"
    args.source = tmp_path
    spec = json.loads(args.spec.read_text())
    root = tmp_path / "job"
    root.mkdir()
    (root / "admission.json").write_text(
        json.dumps({"spec_sha256": args.expected_spec_sha256})
    )
    calls = []
    monkeypatch.setattr(
        controller.subprocess, "run", lambda argv, **kw: calls.append((argv, kw))
    )
    monkeypatch.setattr(controller, "observe", lambda *a: {"state": "live"})
    result = controller.launch_isolated(args, spec)
    argv, kwargs = calls[0]
    assert result["workload_unit"] == args.workload_unit
    assert "--property=RemainAfterExit=yes" in argv
    assert "--property=Type=oneshot" in argv
    assert "--property=MemoryMax=48G" in argv
    assert "--property=OOMPolicy=continue" in argv
    assert "--property=KillMode=control-group" in argv
    assert "--working-directory=" + str(tmp_path) in argv
    assert "--wait" not in argv
    assert argv[-4:] == ["scripts.bounded_data_job", "launch", "--spec", str(args.spec)]
    assert kwargs == {"check": True, "timeout": 180}


def test_isolated_workload_stopped_before_final_backup(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.workload_unit = "fixture-data-workload"
    args.source = tmp_path
    args.corpus_root = tmp_path / "corpus"
    args.corpus_root.mkdir()
    monkeypatch.setattr(controller, "launch_isolated", lambda *a: {})
    monkeypatch.setattr(
        controller,
        "observe",
        lambda *a: {"state": "terminal", "worker": {"cleanup_verified": True}},
    )
    calls = []
    monkeypatch.setattr(
        controller, "best_effort", lambda argv, seconds=120: calls.append(argv) or True
    )
    controller.main(args)
    stop = calls.index(["systemctl", "stop", args.workload_unit])
    backup = next(i for i, argv in enumerate(calls) if args.prefix + "/corpus" in argv)
    assert stop < backup < len(calls) - 1


def test_missing_terminal_after_owned_process_death_triggers_preservation(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    args.corpus_root = tmp_path / "corpus"
    args.corpus_root.mkdir()
    monkeypatch.setattr(controller, "launch", lambda spec: {})
    monkeypatch.setattr(
        controller, "observe", lambda *a: {"state": "terminal_receipt_missing"}
    )
    calls = []
    monkeypatch.setattr(
        controller, "best_effort", lambda argv, seconds=120: calls.append(argv) or True
    )
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: pytest.fail(
            "Dead owned process must not remain in monitor loop"
        ),
    )
    with pytest.raises(RuntimeError, match="without terminal receipt"):
        controller.main(args)
    assert any(args.prefix + "/corpus" in argv for argv in calls)
    assert calls[-1][1:4] == ["compute", "instances", "delete"]
