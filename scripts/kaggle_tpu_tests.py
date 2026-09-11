"""Submit and optionally monitor the complete accelerator suite via Kaggle CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import random
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "accelerators" / "kaggle" / "launch.py"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_KERNEL_VERSION = re.compile(r"\bversion\s+(\d+)\b", re.IGNORECASE)


def validate_revision(revision: str) -> str:
    """Require an immutable, full Git object ID for remote execution."""
    if not _GIT_SHA.fullmatch(revision):
        raise ValueError("revision must be a full lowercase 40-character Git SHA")
    return revision


def kaggle_cli() -> list[str] | None:
    """Locate Kaggle whether installed as a console script or Python module."""
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    candidates = [sys.executable, shutil.which("python3"), shutil.which("python")]
    for python in dict.fromkeys(candidate for candidate in candidates if candidate):
        result = subprocess.run(
            [
                python,
                "-c",
                "import importlib.util; raise SystemExit(importlib.util.find_spec('kaggle') is None)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return [python, "-m", "kaggle"]
    return None


def command(
    *args: str,
    capture: bool = False,
    retries: int = 3,
    timeout_seconds: int = 120,
) -> subprocess.CompletedProcess:
    cli = kaggle_cli()
    if cli is None:
        raise RuntimeError("Kaggle CLI not found")
    full_command = [*cli, *args]
    for attempt in range(retries):
        try:
            result = subprocess.run(
                full_command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stdout = (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout or ""
            )
            result = subprocess.CompletedProcess(
                full_command,
                124,
                stdout=stdout,
                stderr=f"Kaggle CLI connection timed out after {timeout_seconds}s",
            )
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            if not capture:
                print(combined, end="")
            return result
        transient = any(
            marker in combined.lower()
            for marker in (
                "connecttimeout",
                "connection timed out",
                "connectionerror",
                "connection reset",
                "connection aborted",
                "max retries exceeded",
                "temporarily unavailable",
            )
        )
        if transient and attempt < retries - 1:
            delay = min(30.0, 2 ** attempt) + random.uniform(0.0, 0.5)
            print(f"Kaggle API transport failed; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue
        if not capture:
            print(combined, end="")
        raise subprocess.CalledProcessError(
            result.returncode,
            full_command,
            output=result.stdout,
            stderr=result.stderr,
        )
    raise AssertionError("unreachable")


def _kernel_version(text: str) -> int | None:
    match = _KERNEL_VERSION.search(text)
    return int(match.group(1)) if match else None


def _read_monitor_state(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_monitor_state(
    path: Path,
    kernel_id: str,
    status: str,
    *,
    kernel_version: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = _read_monitor_state(path)
    payload = {
        "kernel_id": kernel_id,
        "kernel_version": kernel_version or previous.get("kernel_version"),
        "last_status": status,
        "updated_at": time.time(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _download_output(kernel_id: str, output_dir: Path) -> None:
    """Download and atomically replace output so partial data is never published."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as directory:
        workspace = Path(directory)
        downloaded = workspace / "downloaded"
        downloaded.mkdir()
        command("kernels", "output", kernel_id, "-p", str(downloaded), retries=5)
        state_path = output_dir / "monitor_state.json"
        if state_path.exists():
            shutil.copy2(state_path, downloaded / state_path.name)
        backup = workspace / "previous"
        if output_dir.exists():
            output_dir.replace(backup)
        try:
            downloaded.replace(output_dir)
        except BaseException:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise


def monitor_kernel(
    kernel_id: str,
    output_dir: Path,
    *,
    poll_seconds: int = 30,
    outage_budget_seconds: int = 900,
) -> int:
    """Reconnectable monitor that distinguishes transport loss from remote failure."""
    state_path = output_dir / "monitor_state.json"
    kernel_version = _read_monitor_state(state_path).get("kernel_version")
    outage_started: float | None = None
    outage_attempt = 0
    while True:
        try:
            result = command("kernels", "status", kernel_id, capture=True, retries=1)
            status = (result.stdout + result.stderr).strip()
            kernel_version = _kernel_version(status) or kernel_version
            outage_started = None
            outage_attempt = 0
        except subprocess.CalledProcessError as error:
            combined = ((error.stdout or "") + (error.stderr or "")).lower()
            transient = any(marker in combined for marker in (
                "connecttimeout", "connection timed out", "connectionerror",
                "connection reset", "connection aborted", "max retries exceeded",
                "temporarily unavailable", "tls", "ssl",
            ))
            if not transient:
                raise
            now = time.monotonic()
            if outage_started is None:
                outage_started = now
            elapsed = now - outage_started
            _write_monitor_state(
                state_path,
                kernel_id,
                f"transport-outage:{elapsed:.1f}s",
                kernel_version=kernel_version,
            )
            if elapsed >= outage_budget_seconds:
                print("Kaggle API outage budget exhausted; rerun with --resume-monitor", flush=True)
                return 2
            delay = min(60.0, 2 ** min(outage_attempt, 6)) + random.uniform(0.0, 1.0)
            outage_attempt += 1
            time.sleep(delay)
            continue
        print(status, flush=True)
        _write_monitor_state(
            state_path, kernel_id, status, kernel_version=kernel_version
        )
        normalized = status.lower()
        if "complete" in normalized:
            _download_output(kernel_id, output_dir)
            _write_monitor_state(
                state_path,
                kernel_id,
                "complete:artifacts-downloaded",
                kernel_version=kernel_version,
            )
            return 0
        if any(word in normalized for word in ("error", "failed", "cancel")):
            _download_output(kernel_id, output_dir)
            return 1
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True, help="owner/kernel-slug")
    parser.add_argument("--accelerator", choices=("tpu", "gpu"), default="tpu")
    parser.add_argument("--revision", help="git revision to test; defaults to HEAD")
    parser.add_argument("--repository", default="https://github.com/mlnomadpy/flaxchat.git")
    parser.add_argument("--wait", action="store_true", help="poll until complete or failed")
    parser.add_argument(
        "--resume-monitor", "--status-only", action="store_true",
        help="monitor an existing kernel without submitting a new version",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--outage-budget-seconds", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "kaggle")
    args = parser.parse_args(argv)

    if kaggle_cli() is None:
        parser.error("Kaggle CLI not found; install with `python -m pip install kaggle`")
    owner, separator, slug = args.kernel_id.partition("/")
    if not separator or not owner or not slug:
        parser.error("--kernel-id must use owner/kernel-slug")

    if not args.resume_monitor:
        revision = args.revision or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        try:
            revision = validate_revision(revision)
        except ValueError as error:
            parser.error(str(error))
        with tempfile.TemporaryDirectory(prefix="flaxchat-kaggle-") as directory:
            bundle = Path(directory)
            source = TEMPLATE.read_text(encoding="utf-8")
            source = source.replace('"__SOURCE_REPOSITORY__"', json.dumps(args.repository))
            source = source.replace('"__SOURCE_REVISION__"', json.dumps(revision))
            source = source.replace('"__ACCELERATOR__"', json.dumps(args.accelerator))
            source = source.replace(
                '"__JAX_REQUIREMENT__"',
                json.dumps("jax[tpu]>=0.9.0" if args.accelerator == "tpu" else "jax[cuda12]>=0.9.0"),
            )
            source = source.replace(
                "__MIN_DEVICE_COUNT__", "8" if args.accelerator == "tpu" else "1"
            )
            (bundle / "launch.py").write_text(source, encoding="utf-8")
            metadata = {
                "id": args.kernel_id,
                "title": slug.replace("-", " ").title(),
                "code_file": "launch.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": str(args.accelerator == "gpu").lower(),
                "enable_tpu": str(args.accelerator == "tpu").lower(),
                "enable_internet": "true",
                "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [], "model_sources": [],
            }
            (bundle / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            push = command("kernels", "push", "-p", str(bundle), capture=True)
        version = _kernel_version((push.stdout or "") + (push.stderr or ""))
        _write_monitor_state(
            args.output_dir / "monitor_state.json",
            args.kernel_id,
            "submitted",
            kernel_version=version,
        )
        print(f"Submitted {args.kernel_id} at exact revision {revision}")
        if not args.wait:
            return 0
    return monitor_kernel(
        args.kernel_id,
        args.output_dir,
        poll_seconds=args.poll_seconds,
        outage_budget_seconds=args.outage_budget_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
