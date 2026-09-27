import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import validate_encoder_mlm_pair as worker
from scripts.compare_encoder_mlm import MATCHED_FIELDS


def fixture(tmp_path, monkeypatch, fault=None):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("manifest.json", "tokens.npy"):
        (data / name).write_text("fixture")
    common = [
        "python",
        "-m",
        "scripts.evaluate_encoder",
        "--checkpoint",
        "gs://candidate",
        "--checkpoint-step",
        "1972",
        "--max-rows",
        "3073",
        "--seed",
        "2026",
        "--data",
        str(data),
        "--train-data",
        str(data),
        "--output",
        "unused",
    ]
    plan = dict(
        checkpoint="gs://candidate",
        checkpoint_step=1972,
        validation_rows=3073,
        seed=2026,
        commands_in_order={
            "baseline": common + ["--initial-pretrained", "snapshot"],
            "candidate": common,
        },
        input_sha256={str(p): worker.file_hash(p) for p in data.iterdir()},
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        plan=path, prefix="gs://evidence", output=tmp_path / "output"
    )
    calls = []

    def probe(argv, **kwargs):
        assert kwargs["env"]["JAX_PLATFORMS"] == "tpu"
        return json.dumps(
            [
                dict(
                    platform="cpu" if fault == "device" else "tpu",
                    process=0,
                    kind="TPU v5 lite",
                )
            ]
            * 4
        )

    monkeypatch.setattr(worker.subprocess, "check_output", probe)

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "gcloud":
            return SimpleNamespace(returncode=0)
        assert kwargs["env"]["JAX_PLATFORMS"] == "tpu"
        if fault == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        report = {key: "same" for key in MATCHED_FIELDS}
        report.update(
            parameter_source="initial_pretrained"
            if "--initial-pretrained" in argv
            else "checkpoint",
            initial_weights_sha256={"weights": "hash"},
            checkpoint_step=1972,
            seed=2026,
            checkpoint_context="gs://candidate",
            masked_token_loss=1.0,
            masked_tokens=100,
            evaluated_rows=1 if fault == "partial" else 3073,
            backend="tpu",
            devices=["TPU v5 lite"] * 4,
        )
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=1 if fault == "failure" else 0)

    monkeypatch.setattr(worker.subprocess, "run", run)
    monkeypatch.setenv("FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS", "1800")
    return args, plan, calls


def test_pair_worker_runs_both_roles_and_never_claims_production_quality(
    tmp_path, monkeypatch
):
    args, plan, calls = fixture(tmp_path, monkeypatch)
    worker.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["passed"] and not summary["quality_qualified"]
    assert [s["name"] for s in summary["stages"]] == ["baseline", "candidate"]
    assert sum(argv[0] != "gcloud" for argv in calls) == 2
    assert (args.output / "comparison.json").is_file()
    assert json.loads(args.plan.read_text()) == plan


@pytest.mark.parametrize("fault", ["device", "partial", "failure", "timeout"])
def test_pair_failure_retains_failure_evidence(tmp_path, monkeypatch, fault):
    args, _, calls = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
        worker.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert not summary["passed"] and "error" in summary
    assert calls[-1][0] == "gcloud"
    assert not (args.output / "comparison.json").exists()


def test_pair_rejects_changed_data_before_work(tmp_path, monkeypatch):
    args, plan, calls = fixture(tmp_path, monkeypatch)
    Path(next(iter(plan["input_sha256"]))).write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        worker.run(args)
    assert not calls


def test_pair_coordinator_never_imports_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.validate_encoder_mlm_pair; assert 'jax' not in sys.modules; assert 'flaxchat' not in sys.modules",
        ],
        env=os.environ | {"JAX_PLATFORMS": "tpu"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
