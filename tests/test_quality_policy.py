from pathlib import Path
import json
import os
import re
import subprocess

import yaml

from scripts.check_coverage import check_coverage


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def test_third_party_actions_are_immutable():
    for path in WORKFLOWS.iterdir():
        if path.suffix not in {".yml", ".yaml"}:
            continue
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if "uses" in step:
                    assert PINNED_ACTION.fullmatch(step["uses"]), (path, step["uses"])


def test_release_publishes_and_smokes_the_checkpoint_only_on_tags():
    release_path = WORKFLOWS / "release.yml"
    source = release_path.read_text()
    release = yaml.safe_load(source)
    assert release[True] == {"push": {"tags": ["v*"]}}
    assert "flaxchat-tinystories-v0.1.1.tar.gz" in source
    assert "sha256sum --check" in source
    assert "scripts/verify_artifact.py" in source
    assert "scripts/checkpoint_demo.py" in source
    assert len(release["jobs"]) == 1
    assert "matrix" not in source
    assert source.count("pixi run build-package") == 1
    for suffix in ("311", "312", "313"):
        assert f"/tmp/flaxchat-release-{suffix}" in source
    assert "mktemp -d" in source
    assert 'test "$output" = " he4it("' in source
    assert "artifacts/tinystories-smoke/run_manifest.json" not in source
    assert "examples/tinystories-v0.1.1/run_manifest.json" in source
    assert "body_path: docs/RELEASES.md" in source
    assert "cyclonedx-bom==7.3.1" in source
    assert "python -m cyclonedx_py environment" in source
    assert "/tmp/flaxchat-release-313/bin/python" in source
    assert "--output-reproducible" in source
    assert "sha256sum * > ../SHA256SUMS" in source
    assert 'draft: true' in source
    assert 'gh release download "$GITHUB_REF_NAME"' in source
    assert '"$runtime/bin/pip" install "$download"/*.whl' in source
    assert 'gh release edit "$GITHUB_REF_NAME" --draft=false --latest' in source


def test_expensive_workflows_are_opt_in_and_routine_ci_is_linux_only():
    mac = _workflow("macos-compatibility.yml")
    kaggle = _workflow("kaggle-tpu.yml")
    cpu = _workflow("cpu-tests.yml")
    assert set(mac[True]) == {"workflow_dispatch"}
    assert set(kaggle[True]) == {"workflow_dispatch"}
    assert len(cpu["jobs"]) == 1
    assert all(job["runs-on"] == "ubuntu-latest" for job in cpu["jobs"].values())
    assert cpu["concurrency"]["cancel-in-progress"] is True
    assert ".github/**" not in cpu[True]["push"]["paths"]
    for workflow in (
        ".github/workflows/deploy.yaml",
        ".github/workflows/kaggle-tpu.yml",
        ".github/workflows/macos-compatibility.yml",
        ".github/workflows/release.yml",
    ):
        assert workflow in cpu[True]["pull_request"]["paths"]
        assert workflow in cpu[True]["push"]["paths"]
    assert "!benchmarks/results/**" in cpu[True]["pull_request"]["paths"]
    assert "!benchmarks/results/**" in cpu[True]["push"]["paths"]


def test_release_reuses_default_branch_validation_instead_of_retesting():
    release_text = (WORKFLOWS / "release.yml").read_text()
    assert "git merge-base --is-ancestor" in release_text
    assert "actions/workflows/cpu-tests.yml/runs" in release_text
    assert "pixi run test\n" not in release_text
    assert "pixi run test-e2e" not in release_text


def test_pages_uses_default_branch_and_pr_builds_without_deploying():
    pages = _workflow("deploy.yaml")
    assert pages[True]["push"]["branches"] == ["master"]
    assert "pull_request" in pages[True]
    assert pages["jobs"]["deploy"]["if"] == "github.event_name != 'pull_request'"
    upload = next(
        step for step in pages["jobs"]["build"]["steps"]
        if step.get("name") == "Upload artifact"
    )
    assert upload["if"] == "github.event_name != 'pull_request'"


def test_module_coverage_floor_reports_missing_and_low_files():
    report = {"files": {"flaxchat/a.py": {"summary": {"percent_covered": 49}}}}
    assert check_coverage(report, {"flaxchat/a.py": 50, "flaxchat/b.py": 1}) == [
        "flaxchat/a.py: 49.00% is below 50.00%",
        "flaxchat/b.py: missing from coverage report",
    ]


def test_current_tpu_results_share_one_immutable_revision_and_are_linked():
    results = ROOT / "benchmarks" / "results"
    paths = sorted(results.glob("kaggle-tpu-v5e-8-12bfd85-*.json"))
    assert {path.name.removeprefix("kaggle-tpu-v5e-8-12bfd85-") for path in paths} == {
        "pipeline.json",
        "scaling-overhead.json",
        "scaling-strong.json",
        "scaling-weak.json",
        "summary.json",
    }
    records = [json.loads(path.read_text()) for path in paths]
    revisions = {record["source_revision"] for record in records}
    assert len(revisions) == 1
    revision = revisions.pop()
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision.startswith("12bfd85")
    docs = (ROOT / "docs" / "RESULTS.md").read_text()
    assert revision in docs
    for path in paths:
        assert path.name in docs


def test_paid_multihost_launcher_fails_closed_and_uses_bounded_defaults():
    path = ROOT / "infra" / "tpu" / "flexstart.sh"
    launcher = path.read_text()
    assert 'FLAXCHAT_ACCELERATOR:-v5litepod-16' in launcher
    assert 'FLAXCHAT_MAX_RUN:-1h' in launcher
    assert 'FLAXCHAT_VALID_UNTIL:-1h' in launcher
    assert 'FLAXCHAT_APPROVE_PAID_RUN:-' in launcher
    assert 'I_UNDERSTAND_TPU_BILLING' in launcher
    environment = {**os.environ, "PROJECT_ID": "test-project"}
    environment.pop("FLAXCHAT_APPROVE_PAID_RUN", None)
    result = subprocess.run(
        ["bash", str(path), "create"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing paid TPU creation" in result.stderr
