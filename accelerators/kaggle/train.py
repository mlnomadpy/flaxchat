"""Kaggle training entry point rendered from a :class:`LaunchSpec`."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


WORK = Path("/kaggle/working")
SOURCE = WORK / "flaxchat"
RESULTS = WORK / "flaxchat-training-results"
SPEC = json.loads("__LAUNCH_SPEC_JSON__")


def run(name: str, argv: list[str], cwd: Path | None = None) -> dict[str, object]:
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with (RESULTS / f"{name}.log").open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    return {
        "name": name,
        "return_code": process.wait(),
        "elapsed_seconds": time.monotonic() - started,
    }


RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "launch_spec.json").write_text(
    json.dumps(SPEC, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
shutil.rmtree(SOURCE, ignore_errors=True)
clone = run("clone", ["git", "clone", "--filter=blob:none", SPEC["source_repository"], str(SOURCE)])
checkout = (
    run("checkout", ["git", "checkout", "--detach", SPEC["source_revision"]], SOURCE)
    if clone["return_code"] == 0
    else {"name": "checkout", "return_code": 1, "elapsed_seconds": 0.0}
)
install = (
    run("install", [sys.executable, "-m", "pip", "install", "--quiet", ".[dev]"], SOURCE)
    if checkout["return_code"] == 0
    else {"name": "install", "return_code": 1, "elapsed_seconds": 0.0}
)
training = (
    run("training", list(SPEC["argv"]), SOURCE)
    if install["return_code"] == 0
    else {"name": "training", "return_code": 1, "elapsed_seconds": 0.0}
)
for artifact in SPEC.get("artifacts", []):
    source = SOURCE / artifact
    destination = RESULTS / artifact
    if source.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
summary = {
    "source_revision": SPEC["source_revision"],
    "accelerator": SPEC["accelerator"],
    "checks": [clone, checkout, install, training],
    "passed": training["return_code"] == 0,
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if summary["passed"] else 1)
