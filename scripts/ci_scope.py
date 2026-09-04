"""Choose the smallest safe GitHub Actions validation scope for a change set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess


FULL_TRIGGERS = {
    "pyproject.toml",
    "pixi.toml",
    "pixi.lock",
    ".github/workflows/cpu-tests.yml",
}

TEST_GROUPS = {
    "benchmarks/": (
        "tests/test_benchmark_compare.py",
        "tests/test_benchmark_protocol.py",
        "tests/test_matched_benchmark.py",
        "tests/test_training_scaling.py",
    ),
    "accelerators/kaggle/": ("tests/test_kaggle_launcher.py",),
    "scripts/kaggle_tpu_tests.py": ("tests/test_kaggle_launcher.py",),
    "scripts/kaggle_matched_benchmarks.py": ("tests/test_kaggle_launcher.py",),
    "scripts/run_matched_benchmarks.py": ("tests/test_matched_benchmark.py",),
    "scripts/check_docs.py": (),
    "scripts/check_coverage.py": ("tests/test_quality_policy.py",),
    "scripts/ci_scope.py": ("tests/test_ci_scope.py",),
    "scripts/checkpoint_demo.py": ("tests/test_published_artifact.py",),
    "scripts/checkpoint_portability.py": ("tests/test_checkpoint_topology.py",),
    "scripts/verify_artifact.py": ("tests/test_published_artifact.py",),
    ".github/workflows/release.yml": ("tests/test_quality_policy.py",),
    ".github/workflows/deploy.yaml": ("tests/test_quality_policy.py",),
    ".github/workflows/kaggle-tpu.yml": ("tests/test_quality_policy.py",),
    ".github/workflows/macos-compatibility.yml": ("tests/test_quality_policy.py",),
}


def _is_test(path: str) -> bool:
    candidate = PurePosixPath(path)
    return candidate.parent == PurePosixPath("tests") and candidate.name.startswith("test_") and candidate.suffix == ".py"


def select_scope(changed_paths: list[str], *, force_full: bool = False) -> dict[str, object]:
    """Return deterministic workflow outputs for the supplied repository paths."""
    paths = sorted({path.strip() for path in changed_paths if path.strip()})
    full = force_full or not paths or any(
        path in FULL_TRIGGERS or path.startswith("flaxchat/") or path.startswith("tasks/")
        for path in paths
    )
    selected: set[str] = set()
    if not full:
        for path in paths:
            if _is_test(path):
                selected.add(path)
            for prefix, tests in TEST_GROUPS.items():
                if path == prefix or path.startswith(prefix):
                    selected.update(tests)
            if path.startswith("scripts/") and path not in TEST_GROUPS:
                selected.update(("tests/test_pipeline.py", "tests/test_stage_functions.py"))

    multidevice = full and any(
        path.startswith(("flaxchat/sharding", "flaxchat/checkpoint"))
        or path in {"tests/test_sharding.py", "tests/test_checkpoint_topology.py"}
        or path in FULL_TRIGGERS
        for path in paths
    )
    e2e = full and any(
        path.startswith(("flaxchat/engine", "flaxchat/gpt", "flaxchat/stages/"))
        or path.startswith("tasks/")
        or path in FULL_TRIGGERS
        for path in paths
    )
    return {
        "mode": "full" if full else "targeted",
        "tests": sorted(selected),
        "run_audit": full and any(path in {"pyproject.toml", "pixi.toml", "pixi.lock"} for path in paths),
        "run_build": full and any(path == "pyproject.toml" or path.startswith("flaxchat/") for path in paths),
        "run_multidevice": multidevice,
        "run_e2e": e2e,
    }


def changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    paths = [] if args.full else changed_paths(args.base, args.head) if args.base else []
    scope = select_scope(paths, force_full=args.full)
    if args.github_output:
        lines = [
            f"mode={scope['mode']}",
            f"tests={json.dumps(scope['tests'], separators=(',', ':'))}",
            *(f"{key}={str(scope[key]).lower()}" for key in (
                "run_audit", "run_build", "run_multidevice", "run_e2e"
            )),
        ]
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")
    else:
        print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    main()
