"""Run all matched trainer adapters sequentially and retain every raw log."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


SHA40 = re.compile(r"[0-9a-f]{40}")


def require_checkout(path: Path, expected_revision: str) -> None:
    if not SHA40.fullmatch(expected_revision):
        raise ValueError("source revisions must be full lowercase 40-character SHAs")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected_revision:
        raise RuntimeError(f"{path} is {actual}, expected {expected_revision}")


def run_logged(name: str, argv: list[str], output_dir: Path, *, cwd: Path) -> dict:
    log_path = output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(log_path.read_text(encoding="utf-8"), end="", flush=True)
    return {
        "name": name,
        "returncode": result.returncode,
        "started_at": started,
        "finished_at": time.time(),
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flaxchat-source", type=Path, default=Path.cwd())
    parser.add_argument("--flaxchat-revision", required=True)
    parser.add_argument("--nanochat-source", required=True, type=Path)
    parser.add_argument("--nanochat-revision", required=True)
    parser.add_argument("--maxtext-source", required=True, type=Path)
    parser.add_argument("--maxtext-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sources = {
        "flaxchat": (args.flaxchat_source.resolve(), args.flaxchat_revision),
        "nanochat": (args.nanochat_source.resolve(), args.nanochat_revision),
        "MaxText": (args.maxtext_source.resolve(), args.maxtext_revision),
    }
    for source, revision in sources.values():
        require_checkout(source, revision)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = args.flaxchat_source.resolve() / "benchmarks" / "protocol.yaml"
    data = output_dir / "matched_batches.npz"
    stages = [run_logged(
        "prepare-data",
        [sys.executable, "-m", "benchmarks.matched.prepare_data", "--output", str(data)],
        output_dir,
        cwd=args.flaxchat_source,
    )]
    runners = (
        ("flaxchat", "benchmarks.matched.flaxchat_runner", args.flaxchat_source, args.flaxchat_revision),
        ("nanochat", "benchmarks.matched.nanochat_runner", args.nanochat_source, args.nanochat_revision),
        ("MaxText", "benchmarks.matched.maxtext_runner", args.maxtext_source, args.maxtext_revision),
    )
    if stages[0]["returncode"] == 0:
        for name, module, source, revision in runners:
            argv = [
                sys.executable, "-m", module,
                "--data", str(data),
                "--protocol", str(protocol),
                "--revision", revision,
                "--checkpoint-dir", str(output_dir / "checkpoints" / name.lower()),
                "--output", str(output_dir / f"{name.lower()}.json"),
            ]
            if name != "flaxchat":
                argv.extend(("--source", str(source)))
            stages.append(run_logged(name, argv, output_dir, cwd=args.flaxchat_source))
    record_paths = [output_dir / f"{name.lower()}.json" for name, *_ in runners]
    if all(stage["returncode"] == 0 for stage in stages) and all(path.exists() for path in record_paths):
        comparison_path = output_dir / "comparison.json"
        with comparison_path.open("w", encoding="utf-8") as comparison:
            result = subprocess.run(
                [sys.executable, "-m", "benchmarks.compare", "--protocol", str(protocol), *(str(path) for path in record_paths)],
                cwd=args.flaxchat_source,
                stdout=comparison,
                stderr=subprocess.STDOUT,
                text=True,
            )
        stages.append({
            "name": "compare",
            "returncode": result.returncode,
            "log": str(comparison_path),
        })
    summary = {
        "format_version": 1,
        "sources": {name: revision for name, (_, revision) in sources.items()},
        "stages": stages,
        "success": bool(stages) and all(stage["returncode"] == 0 for stage in stages) and len(stages) == 5,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
