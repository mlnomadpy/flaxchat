"""Generated one-kernel entry point for the matched trainer comparison."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


WORK = Path("/kaggle/working")
RESULTS = WORK / "flaxchat-matched-results"
FLAXCHAT = WORK / "flaxchat"
NANOCHAT = WORK / "nanochat"
MAXTEXT = WORK / "maxtext"
FLAXCHAT_REVISION = "__FLAXCHAT_REVISION__"
NANOCHAT_REVISION = "__NANOCHAT_REVISION__"
MAXTEXT_REVISION = "__MAXTEXT_REVISION__"
MODE = "__MODE__"
JAX_REQUIREMENT = "__JAX_REQUIREMENT__"


def run(name: str, argv: list[str], cwd: Path | None = None) -> dict[str, object]:
    print(f"\n===== {name} =====", flush=True)
    print("$", " ".join(argv), flush=True)
    log_path = RESULTS / f"bootstrap-{name}.log"
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    return_code = process.wait()
    return {
        "name": name,
        "returncode": return_code,
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
    }


RESULTS.mkdir(parents=True, exist_ok=True)
for path in (FLAXCHAT, NANOCHAT, MAXTEXT):
    shutil.rmtree(path, ignore_errors=True)
sources = (
    ("flaxchat", "https://github.com/mlnomadpy/flaxchat.git", FLAXCHAT, FLAXCHAT_REVISION),
    ("nanochat", "https://github.com/karpathy/nanochat.git", NANOCHAT, NANOCHAT_REVISION),
    ("maxtext", "https://github.com/AI-Hypercomputer/maxtext.git", MAXTEXT, MAXTEXT_REVISION),
)
checks: list[dict[str, object]] = []
for name, repository, path, revision in sources:
    clone = run(f"clone-{name}", ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(path)])
    checks.append(clone)
    if clone["returncode"] == 0:
        checks.append(run(f"checkout-{name}", ["git", "checkout", revision], cwd=path))

if all(check["returncode"] == 0 for check in checks):
    checks.append(run(
        "install-flaxchat",
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            JAX_REQUIREMENT, ".[dev,data]",
        ],
        cwd=FLAXCHAT,
    ))
if all(check["returncode"] == 0 for check in checks):
    checks.append(run(
        "install-maxtext-package",
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", ".",
        ],
        cwd=MAXTEXT,
    ))
if all(check["returncode"] == 0 for check in checks):
    checks.append(run(
        "install-maxtext-runtime",
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "omegaconf", "etils", "ml-collections", "clu", "tensorboardX", "grain",
            "jaxtyping", "qwix", "aqtp", "einops", "pathwaysutils", "tokamax",
        ],
        cwd=MAXTEXT,
    ))
if MODE == "gpu" and all(check["returncode"] == 0 for check in checks):
    checks.append(run(
        "device-check",
        [
            sys.executable, "-c",
            "import jax; d=jax.devices(); print(d); assert jax.default_backend()=='gpu' and len(d)==1 and 'P100' in d[0].device_kind.upper()",
        ],
        cwd=FLAXCHAT,
    ))
if MODE == "gpu" and all(check["returncode"] == 0 for check in checks):
    checks.append(run(
        "matched-suite",
        [
            sys.executable, "-m", "scripts.run_matched_benchmarks",
            "--flaxchat-source", str(FLAXCHAT),
            "--flaxchat-revision", FLAXCHAT_REVISION,
            "--nanochat-source", str(NANOCHAT),
            "--nanochat-revision", NANOCHAT_REVISION,
            "--maxtext-source", str(MAXTEXT),
            "--maxtext-revision", MAXTEXT_REVISION,
            "--output-dir", str(RESULTS / "records"),
        ],
        cwd=FLAXCHAT,
    ))
if MODE == "preflight" and all(check["returncode"] == 0 for check in checks):
    checks.append(run(
        "native-model-preflight",
        [
            sys.executable, "-m", "benchmarks.matched.preflight",
            "--nanochat-source", str(NANOCHAT),
            "--maxtext-source", str(MAXTEXT),
            "--output", str(RESULTS / "preflight.json"),
        ],
        cwd=FLAXCHAT,
    ))

summary = {
    "format_version": 1,
    "source_revisions": {
        "flaxchat": FLAXCHAT_REVISION,
        "nanochat": NANOCHAT_REVISION,
        "MaxText": MAXTEXT_REVISION,
    },
    "mode": MODE,
    "checks": checks,
    "passed": bool(checks) and all(check["returncode"] == 0 for check in checks),
}
(RESULTS / "bootstrap-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if summary["passed"] else 1)
