"""Kaggle accelerator entry point generated with an exact repository revision."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


WORK = Path("/kaggle/working")
SOURCE = WORK / "flaxchat"
RESULTS = WORK / "flaxchat-accelerator-results"
SOURCE_REPOSITORY = "__SOURCE_REPOSITORY__"
SOURCE_REVISION = "__SOURCE_REVISION__"
ACCELERATOR = "__ACCELERATOR__"
JAX_REQUIREMENT = "__JAX_REQUIREMENT__"
MIN_DEVICE_COUNT = __MIN_DEVICE_COUNT__


def run(name: str, command: list[str], cwd: Path | None = None) -> dict[str, object]:
    print(f"\n===== {name} =====", flush=True)
    print("$", " ".join(command), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    log_path = RESULTS / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    return_code = process.wait()
    elapsed = time.monotonic() - started
    print(f"===== {name}: rc={return_code}, elapsed={elapsed:.1f}s =====", flush=True)
    return {"name": name, "return_code": return_code, "elapsed_seconds": elapsed}


RESULTS.mkdir(parents=True, exist_ok=True)
if SOURCE.exists():
    shutil.rmtree(SOURCE)
checkout = run("clone", ["git", "clone", "--filter=blob:none", SOURCE_REPOSITORY, str(SOURCE)])
if checkout["return_code"] == 0:
    checkout = run("checkout-revision", ["git", "checkout", SOURCE_REVISION], cwd=SOURCE)

install = (
    run(
        "install",
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--find-links",
            "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
            JAX_REQUIREMENT, ".[dev]",
        ],
        cwd=SOURCE,
    )
    if checkout["return_code"] == 0
    else {"name": "install", "return_code": 1, "elapsed_seconds": 0.0}
)

checks: list[dict[str, object]] = [checkout, install]
if install["return_code"] == 0:
    checks.append(run(
        "device-check",
        [
            sys.executable, "-c",
            "import json,jax; d=jax.devices(); expected=" + repr(ACCELERATOR) + "; "
            "print(json.dumps({'backend':jax.default_backend(),'devices':[str(x) for x in d]},indent=2)); "
            "assert jax.default_backend()==expected and len(d)>=" + str(MIN_DEVICE_COUNT),
        ],
        cwd=SOURCE,
    ))
    checks.append(run(
        "pytest-all",
        [
            sys.executable, "-m", "pytest", "tests", "-v", "--durations=25",
            f"--junitxml={RESULTS / 'pytest.xml'}",
        ],
        cwd=SOURCE,
    ))
    checks.append(run(
        "pretrain-smoke",
        [sys.executable, "-m", "scripts.pretrain", "--cpu-smoke"],
        cwd=SOURCE,
    ))
    checks.append(run(
        "tinystories-e2e",
        [
            sys.executable, "-m", "scripts.run_tinystories",
            "--output-dir", str(RESULTS / "tinystories"),
            "--max-train-stories", "128",
            "--max-validation-stories", "32",
            "--sequence-length", "128",
            "--embedding-dim", "32",
            "--batch-size", "8",
            "--pretrain-steps", "2",
            "--sft-steps", "1",
            "--rl-steps", "1",
            "--max-new-tokens", "4",
        ],
        cwd=SOURCE,
    ))
    attention_command = [
        sys.executable, "-m", "benchmarks.attention",
        "--backend", "splash" if ACCELERATOR == "tpu" else "xla",
        "--sequence-lengths",
        *(["1024", "2048", "4096", "8192"] if ACCELERATOR == "tpu" else ["1024", "2048"]),
        "--heads", "2", "--head-dim", "16",
        "--warmup", "1", "--iterations", "3",
    ]
    checks.append(run("attention-benchmark", attention_command, cwd=SOURCE))
    if ACCELERATOR == "tpu":
        checks.append(run(
            "training-scaling",
            [
                sys.executable, "-m", "benchmarks.training_scaling",
                "--device-counts", "1", "2", "4", "8",
                "--global-batch-size", "32",
                "--warmup", "2", "--iterations", "5",
                "--output", str(RESULTS / "training-scaling.json"),
            ],
            cwd=SOURCE,
        ))
    checks.append(run(
        "speculative-benchmark",
        [
            sys.executable, "-m", "benchmarks.speculative",
            "--max-tokens", "16", "--draft-steps", "4", "--iterations", "3",
        ],
        cwd=SOURCE,
    ))

summary = {
    "source_repository": SOURCE_REPOSITORY,
    "source_revision": SOURCE_REVISION,
    "accelerator": ACCELERATOR,
    "checks": checks,
    "passed": bool(checks) and all(item["return_code"] == 0 for item in checks),
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
shutil.rmtree(SOURCE, ignore_errors=True)
print("\n===== FINAL SUMMARY =====", flush=True)
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if summary["passed"] else 1)
