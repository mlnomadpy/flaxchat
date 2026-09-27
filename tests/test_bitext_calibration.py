import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.calibrate_encoder_bitext import estimate, run
from scripts import validate_encoder_bitext_pair as worker
from tests.test_encoder_bitext_campaign import prepared


def test_real_calibration_exports_two_subsets_without_scores(tmp_path):
    path, _ = prepared(tmp_path)
    result = run(path, "candidate", tmp_path / "calibration", batch_size=2)
    assert result["subsets"] == 2 and result["pairs"] == 6
    assert len(result["measured_subsets"]) == 2
    assert result["full_export_seconds"] > 0
    assert not result["scores_computed"]
    assert not (tmp_path / "calibration/report.json").exists()


@pytest.mark.parametrize(
    "bad", [None, "order", "coverage", "nan", "batch", "negative_load"]
)
def test_duration_estimate_validates_full_subsets_and_rounds_batches(bad):
    entries = [
        dict(subset="a", pairs=33),
        dict(subset="b", pairs=32),
        dict(subset="c", pairs=1),
    ]
    timings = [
        dict(subset="a", pairs=33, seconds=12),
        dict(subset="b", pairs=32, seconds=2),
    ]
    batch, load = 32, 3
    if bad == "order":
        timings.reverse()
    elif bad == "coverage":
        timings[0]["pairs"] -= 1
    elif bad == "nan":
        timings[1]["seconds"] = float("nan")
    elif bad == "batch":
        batch = 0
    elif bad == "negative_load":
        load = -1
    if bad:
        with pytest.raises(ValueError):
            estimate(
                entries,
                timings,
                batch_size=batch,
                load_seconds=load,
                verification_seconds=1,
            )
    else:
        result = estimate(
            entries,
            timings,
            batch_size=batch,
            load_seconds=load,
            verification_seconds=1,
        )
        assert result["full_export_seconds"] == 3 + 2 + 12 + 4 + 2 + 2
        assert result["pairs"] == 66


def calibration(role, identity):
    return dict(
        role=role,
        plan_sha256=identity,
        subsets=112,
        pairs=88877,
        batch_size=32,
        backend="tpu",
        device_count=4,
        processes=1,
        jax_version="test",
        full_export_seconds=10,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "cpu"),
        ("device_count", 1),
        ("processes", 2),
        ("pairs", 5),
        ("plan_sha256", "wrong"),
        ("full_export_seconds", float("nan")),
    ],
)
def test_pair_estimate_requires_matched_physical_evidence(field, value):
    rows = [calibration(role, "frozen") for role in ("baseline", "candidate")]
    rows[1][field] = value
    with pytest.raises(ValueError):
        worker.pair_estimate(
            rows, plan_sha256="frozen", subsets=112, pairs=88877, batch_size=32
        )


@pytest.mark.parametrize("seconds,passes", [(7200, True), (1000, False)])
def test_worker_only_starts_full_pair_when_it_fits_lease(
    tmp_path, monkeypatch, seconds, passes
):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(subsets=112, pairs=88877)))
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    stages, uploads = [], []
    monkeypatch.setenv("FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS", str(seconds))
    monkeypatch.setenv("JAX_PROCESS_COUNT", "1")

    def fake_stage(command, **kwargs):
        stages.append((command, kwargs))
        if command[2] == "scripts.calibrate_encoder_bitext":
            role = command[command.index("--role") + 1]
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "estimate.json").write_text(
                json.dumps(calibration(role, identity))
            )
        return 0

    monkeypatch.setattr(worker, "run_stage", fake_stage)
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *a, **kw: uploads.append((a, kw))
    )
    args = argparse.Namespace(
        plan=path,
        output=tmp_path / "evidence",
        prefix="gs://test/bitext",
        batch_size=32,
    )
    if passes:
        worker.run(args)
    else:
        with pytest.raises(TimeoutError, match="does not fit"):
            worker.run(args)
    assert len(stages) == (5 if passes else 2)
    assert all(0 < kw["timeout"] < seconds for _, kw in stages)
    if passes:
        assert stages[-1][1]["env"]["JAX_PLATFORMS"] == "cpu"
    assert uploads and 0 < uploads[-1][1]["timeout"] <= 300
    commands = [args[0] for args, _ in uploads]
    assert "--no-clobber" in commands[-1]
    if passes:
        assert [Path(command[-2]).name for command in commands[:-1]] == [
            "baseline",
            "candidate",
        ]
        assert all(
            command[-1] == "gs://test/bitext/evidence/" for command in commands[:-1]
        )
        assert all(0 < kwargs["timeout"] <= 120 for _, kwargs in uploads[:-1])
    else:
        assert len(commands) == 1
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["passed"] is passes and not summary["quality_qualified"]
