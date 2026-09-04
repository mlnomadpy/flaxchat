import argparse
import json
from pathlib import Path
import subprocess

import pytest

from flaxchat.launch import LaunchSpec, execute_launch_spec
from scripts.kaggle_tpu_tests import command, monitor_kernel, validate_revision
from scripts.kaggle_matched_benchmarks import (
    MAXTEXT_REVISION,
    NANOCHAT_REVISION,
    render_bundle as render_matched_bundle,
)
from scripts.train_kaggle import build_launch_spec, render_bundle
from scripts.train_local import (
    build_launch_spec as build_local_launch_spec,
    build_parser as build_local_parser,
)
from scripts.train_tpu import (
    build_launch_spec as build_tpu_launch_spec,
    build_parser as build_tpu_parser,
    run_adapter,
)


def test_acceptance_bundle_installs_all_test_feature_extras():
    source = (
        Path(__file__).resolve().parents[1]
        / "accelerators" / "kaggle" / "launch.py"
    ).read_text()
    assert '".[dev,web,logging,data]"' in source


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


def test_launch_spec_executes_argv_without_shell(monkeypatch):
    spec = LaunchSpec(
        platform="local",
        accelerator="cpu",
        source_repository="local",
        source_revision="a" * 40,
        argv=("python", "-m", "scripts.run_tinystories"),
    )
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("flaxchat.launch.subprocess.run", fake_run)
    assert execute_launch_spec(spec, ("--resume-from-step=3",)).returncode == 0
    assert seen["argv"] == (*spec.argv, "--resume-from-step=3")
    assert "shell" not in seen["kwargs"]


class FakeVM:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == self.fail_at:
                raise RuntimeError(name)
        return call


def _gcp_args(*extra):
    return build_tpu_parser().parse_args(["--name", "test", *extra])


def _gcp_spec(args):
    return LaunchSpec(
        platform="gcp",
        accelerator="v4-8",
        source_repository="local",
        source_revision="a" * 40,
        argv=("python", "-m", "scripts.pretrain"),
        teardown="always" if args.teardown else "never",
    )


@pytest.mark.parametrize("failure", ["up", "run"])
def test_gcp_lifecycle_tears_down_after_failure(failure):
    args = _gcp_args("--teardown")
    vm = FakeVM(fail_at=failure)
    with pytest.raises(RuntimeError, match=failure):
        run_adapter(args, _gcp_spec(args), vm, None)
    assert vm.calls[-1][0] == "down"


def test_gcp_lifecycle_resume_collect_and_teardown():
    args = _gcp_args("--teardown", "--recover", "--gcs", "gs://bucket", "--collect", "run.json")
    vm = FakeVM()
    assert run_adapter(args, _gcp_spec(args), vm, object()) == 0
    names = [call[0] for call in vm.calls]
    assert "run_with_resume" in names
    assert "collect" in names
    assert names[-1] == "down"


def test_platform_dry_runs_share_one_manifest_contract():
    revision = "c" * 40
    local = build_local_launch_spec(
        build_local_parser().parse_args(["--dry-run"]), revision=revision
    )
    gcp_args = _gcp_args("--dry-run")
    gcp = build_tpu_launch_spec(gcp_args, revision=revision)
    kaggle_args = argparse.Namespace(
        revision=revision,
        repository="https://example.invalid/flaxchat.git",
        accelerator="tpu",
        artifact_dir="artifacts/tinystories",
        layers=2,
        steps=100,
        batch_size=4,
        sequence_length=128,
        secrets=[],
        budget_hours=None,
    )
    kaggle = build_launch_spec(kaggle_args)
    for spec in (local, gcp, kaggle):
        restored = LaunchSpec.from_json(spec.to_json())
        assert restored.source_revision == revision
        assert restored.resolved_config
        assert restored.argv[:3] == ("python", "-m", restored.argv[2])
        assert restored.artifacts or restored.platform == "gcp"


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


def test_kaggle_command_bounds_a_hung_cli(monkeypatch):
    monkeypatch.setattr("scripts.kaggle_tpu_tests.kaggle_cli", lambda: ["kaggle"])
    monkeypatch.setattr("scripts.kaggle_tpu_tests.time.sleep", lambda _: None)
    monkeypatch.setattr("scripts.kaggle_tpu_tests.random.uniform", lambda *_: 0.0)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("scripts.kaggle_tpu_tests.subprocess.run", timeout)
    with pytest.raises(subprocess.CalledProcessError) as error:
        command("kernels", "status", "owner/kernel", retries=1, timeout_seconds=7)
    assert error.value.returncode == 124
    assert "timed out after 7s" in error.value.stderr


def test_matched_gpu_bundle_pins_all_three_repositories(tmp_path):
    revision = "d" * 40
    render_matched_bundle("owner/matched", revision, tmp_path)
    source = (tmp_path / "matched.py").read_text()
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert revision in source
    assert NANOCHAT_REVISION in source
    assert MAXTEXT_REVISION in source
    assert "__FLAXCHAT_REVISION__" not in source
    assert 'PYTORCH_REQUIREMENT = "torch==2.5.1+cu118"' in source
    assert 'PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"' in source
    assert '"install-p100-pytorch-runtime"' in source
    assert "'sm_60' in torch.cuda.get_arch_list()" in source
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_tpu"] == "false"


def test_matched_preflight_uses_cpu_before_spending_gpu_quota(tmp_path):
    render_matched_bundle("owner/preflight", "e" * 40, tmp_path, preflight=True)
    source = (tmp_path / "matched.py").read_text()
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert 'MODE = "preflight"' in source
    assert "benchmarks.matched.preflight" in source
    assert 'if MODE == "gpu"' in source
    assert metadata["enable_gpu"] == "false"
    assert metadata["enable_tpu"] == "false"
