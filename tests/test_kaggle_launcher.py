import argparse
import json
import subprocess

import pytest

from flaxchat.launch import LaunchSpec
from scripts.kaggle_tpu_tests import monitor_kernel, validate_revision
from scripts.train_kaggle import build_launch_spec, render_bundle


def test_kaggle_launcher_requires_immutable_full_revision():
    revision = "a" * 40
    assert validate_revision(revision) == revision
    for invalid in ("main", "a" * 7, "A" * 40, "a" * 41):
        with pytest.raises(ValueError, match="full lowercase 40-character Git SHA"):
            validate_revision(invalid)


def test_launch_spec_round_trip_and_secret_values_are_rejected():
    spec = LaunchSpec(
        platform="kaggle",
        accelerator="tpu",
        source_repository="https://example.invalid/repo.git",
        source_revision="a" * 40,
        argv=("python", "-m", "scripts.run_tinystories"),
        secret_names=("HF_TOKEN",),
        recovery=True,
    )
    assert LaunchSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError, match="never values"):
        LaunchSpec(
            platform="kaggle",
            accelerator="tpu",
            source_repository="repo",
            source_revision="a" * 40,
            argv=("python",),
            secret_names=("TOKEN=secret",),
        )


def test_training_bundle_runs_real_pipeline_at_exact_revision(tmp_path):
    args = argparse.Namespace(
        revision="b" * 40,
        repository="https://example.invalid/flaxchat.git",
        accelerator="tpu",
        artifact_dir="artifacts/tinystories",
        layers=2,
        steps=3,
        batch_size=4,
        sequence_length=64,
        secrets=[],
        budget_hours=None,
    )
    spec = build_launch_spec(args)
    render_bundle(spec, "owner/kernel", tmp_path)
    launch = (tmp_path / "launch.py").read_text()
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert "scripts.run_tinystories" in launch
    assert "Training would" not in launch
    assert spec.source_revision in launch
    assert metadata["enable_tpu"] == "true"


def test_monitor_recovers_after_transport_reset(monkeypatch, tmp_path):
    attempts = iter(("reset", "running", "complete"))

    def fake_command(*args, **kwargs):
        state = next(attempts)
        if state == "reset":
            raise subprocess.CalledProcessError(
                1, args, output="", stderr="TLS connection reset"
            )
        return subprocess.CompletedProcess(args, 0, stdout=state, stderr="")

    monkeypatch.setattr("scripts.kaggle_tpu_tests.command", fake_command)
    monkeypatch.setattr("scripts.kaggle_tpu_tests.time.sleep", lambda _: None)
    monkeypatch.setattr("scripts.kaggle_tpu_tests.random.uniform", lambda *_: 0.0)
    monkeypatch.setattr("scripts.kaggle_tpu_tests._download_output", lambda *_: None)
    assert monitor_kernel("owner/kernel", tmp_path, poll_seconds=0) == 0
    state = json.loads((tmp_path / "monitor_state.json").read_text())
    assert state["last_status"] == "complete:artifacts-downloaded"
