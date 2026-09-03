"""Submit and optionally monitor the complete suite through the Kaggle CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "accelerators" / "kaggle" / "launch.py"


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


def command(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cli = kaggle_cli()
    if cli is None:
        raise RuntimeError("Kaggle CLI not found")
    full_command = [*cli, *args]
    for attempt in range(3):
        result = subprocess.run(full_command, text=True, capture_output=True)
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
                "max retries exceeded",
                "temporarily unavailable",
            )
        )
        if transient and attempt < 2:
            delay = 2 ** attempt
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True, help="owner/kernel-slug")
    parser.add_argument("--revision", help="git revision to test; defaults to HEAD")
    parser.add_argument("--repository", default="https://github.com/mlnomadpy/flaxchat.git")
    parser.add_argument("--wait", action="store_true", help="poll until complete or failed")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "kaggle")
    args = parser.parse_args()

    if kaggle_cli() is None:
        parser.error("Kaggle CLI not found; install with `python -m pip install kaggle`")
    revision = args.revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    owner, separator, slug = args.kernel_id.partition("/")
    if not separator or not owner or not slug:
        parser.error("--kernel-id must use owner/kernel-slug")

    with tempfile.TemporaryDirectory(prefix="flaxchat-kaggle-") as directory:
        bundle = Path(directory)
        source = TEMPLATE.read_text(encoding="utf-8")
        source = source.replace('"__SOURCE_REPOSITORY__"', json.dumps(args.repository))
        source = source.replace('"__SOURCE_REVISION__"', json.dumps(revision))
        (bundle / "launch.py").write_text(source, encoding="utf-8")
        metadata = {
            "id": args.kernel_id,
            "title": slug.replace("-", " ").title(),
            "code_file": "launch.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": [],
        }
        (bundle / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        command("kernels", "push", "-p", str(bundle))

    print(f"Submitted {args.kernel_id} at exact revision {revision}")
    if not args.wait:
        return 0
    while True:
        result = command("kernels", "status", args.kernel_id, capture=True)
        status = (result.stdout + result.stderr).strip()
        print(status, flush=True)
        normalized = status.lower()
        if "complete" in normalized:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            command("kernels", "output", args.kernel_id, "-p", str(args.output_dir))
            return 0
        if any(word in normalized for word in ("error", "failed", "cancel")):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            command("kernels", "output", args.kernel_id, "-p", str(args.output_dir))
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
