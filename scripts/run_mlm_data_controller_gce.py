"""Launch one authenticated bounded data job and observe that same execution."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import re
import sys
import time

from scripts.bounded_data_job import canonical, launch, observe


def best_effort(argv, seconds=120):
    """Receipt transport must not kill a healthy data process group."""
    try:
        subprocess.run(argv, check=True, timeout=seconds)
        return True
    except (subprocess.SubprocessError, OSError) as error:
        print(
            json.dumps({"transport_error": type(error).__name__, "command": argv[:3]}),
            flush=True,
        )
        return False


def launch_isolated(args, spec):
    """Place the workload below its own memory limit, outside the observer."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,120}", args.workload_unit):
        raise ValueError("Explicit safe workload unit name required")
    if args.source is None or not args.source.is_absolute():
        raise ValueError(
            "Absolute source working directory required for isolated workload"
        )
    subprocess.run(
        [
            "systemd-run",
            "--unit=" + args.workload_unit,
            "--property=Type=oneshot",
            "--property=RemainAfterExit=yes",
            "--property=MemoryMax=48G",
            "--property=OOMPolicy=continue",
            "--property=KillMode=control-group",
            "--working-directory=" + str(args.source),
            sys.executable,
            "-m",
            "scripts.bounded_data_job",
            "launch",
            "--spec",
            str(args.spec),
        ],
        check=True,
        timeout=180,
    )
    deadline = time.monotonic() + 60
    root = Path(spec["root"])
    while time.monotonic() < deadline:
        if (root / "admission.json").exists():
            state = observe(spec["root"], args.expected_spec_sha256)
            if state["state"] != "startup_unobserved":
                admission = json.loads((root / "admission.json").read_text())
                return {
                    **admission,
                    "workload_unit": args.workload_unit,
                    "status": "isolated workload admitted; observe terminal receipts",
                }
        if (root / "admission-failure.json").exists():
            raise RuntimeError(
                "Isolated workload admission failed; never retry this root"
            )
        time.sleep(1)
    raise TimeoutError("Isolated workload ownership not ready within 60 seconds")


def main(args):
    spec = json.loads(args.spec.read_text())
    if hashlib.sha256(canonical(spec)).hexdigest() != args.expected_spec_sha256:
        raise ValueError(
            "Independently pinned data specification differs before launch"
        )
    args.output.mkdir(exist_ok=True)
    try:
        admission = (
            launch_isolated(args, spec)
            if getattr(args, "workload_unit", None)
            else launch(spec)
        )
        (args.output / "launch.json").write_text(json.dumps(admission, indent=2))
        deadline = time.monotonic() + spec["timeout_seconds"] + 180
        while time.monotonic() < deadline:
            state = observe(spec["root"], args.expected_spec_sha256)
            path = args.output / "observation.json"
            path.write_text(json.dumps(state, indent=2))
            best_effort(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(path),
                    args.prefix + "/observation.json",
                ]
            )
            if state["state"] == "terminal":
                best_effort(
                    [
                        "gcloud",
                        "storage",
                        "rsync",
                        "--recursive",
                        spec["root"],
                        args.prefix + "/controller",
                    ],
                    180,
                )
                if not state["worker"]["cleanup_verified"]:
                    raise RuntimeError(
                        "Owned data processes remain unresolved; deleting owned VM"
                    )
                return
            if state["state"] == "terminal_receipt_missing":
                raise RuntimeError(
                    "Owned workload process died without terminal receipt; preserving corpus"
                )
            time.sleep(120)
        raise TimeoutError("Data observer deadline")
    finally:
        if getattr(args, "workload_unit", None):
            best_effort(["systemctl", "stop", args.workload_unit], 90)
        # This observer has its own systemd unit/OOM score, so it can preserve
        # committed progress even when the worker is killed by the kernel.
        # Never include credentials or the enclosing release root in this copy.
        corpus = getattr(args, "corpus_root", None)
        if corpus is not None and corpus.is_dir():
            preserved = best_effort(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    str(corpus),
                    args.prefix + "/corpus",
                ],
                getattr(args, "backup_seconds", 3600),
            )
            receipt = args.output / "preservation.json"
            receipt.write_text(
                json.dumps(
                    {
                        "corpus_sync_succeeded": preserved,
                        "corpus_prefix": args.prefix + "/corpus",
                        "training_launched": False,
                    },
                    indent=2,
                )
            )
            best_effort(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(receipt),
                    args.prefix + "/preservation.json",
                ]
            )
        args.token_file.unlink(missing_ok=True)
        # Provider DELETE lease is the independent fallback if transport fails here.
        best_effort(
            [
                "gcloud",
                "compute",
                "instances",
                "delete",
                args.instance,
                "--project=" + args.project,
                "--zone=" + args.zone,
                "--quiet",
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--workload-unit")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--backup-seconds", type=int, default=3600)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--zone", required=True)
    main(parser.parse_args())
