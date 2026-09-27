"""Calibrate and, only if time permits, run full matched bitext on one leased TPU host.

The stdlib-only parent does not initialize JAX or contend with its child workers.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts.stage_progress import run_stage
from scripts.workload_deadline import WorkloadDeadline


def pair_estimate(estimates, *, plan_sha256, subsets, pairs, batch_size):
    if len(estimates) != 2 or [e["role"] for e in estimates] != [
        "baseline",
        "candidate",
    ]:
        raise ValueError("Both model calibrations required")
    for item in estimates:
        if any(
            item.get(k) != v
            for k, v in dict(
                plan_sha256=plan_sha256,
                subsets=subsets,
                pairs=pairs,
                batch_size=batch_size,
                backend="tpu",
                device_count=4,
                processes=1,
            ).items()
        ):
            raise ValueError("Calibration identity or physical topology mismatch")
        if (
            not math.isfinite(item["full_export_seconds"])
            or item["full_export_seconds"] <= 0
        ):
            raise ValueError("Invalid full campaign duration estimate")
    if estimates[0]["jax_version"] != estimates[1]["jax_version"]:
        raise ValueError("Unmatched calibration runtime")
    return 1.5 * sum(e["full_export_seconds"] for e in estimates) + 600


def run(args):
    if (
        not args.prefix.startswith("gs://")
        or int(os.environ.get("JAX_PROCESS_COUNT", "1")) != 1
    ):
        raise ValueError("Single-host worker and persistent GCS evidence required")
    if args.batch_size < 1:
        raise ValueError("Positive batch size required")
    frozen = json.loads(args.plan.read_text())
    identity = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    deadline = WorkloadDeadline(
        min(6500, int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ | {
        "JAX_PLATFORMS": "tpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    }
    summary: dict[str, Any] = dict(
        passed=False, quality_qualified=False, plan_sha256=identity, stages=[]
    )

    def publish(path, seconds):
        subprocess.run(
            ["gcloud", "storage", "cp", str(path), args.prefix + "/progress/"],
            check=True,
            timeout=min(30, seconds),
            capture_output=True,
        )

    def stage(name, module, *argv, maximum=None, cpu=False):
        seconds = deadline.remaining() - 360
        if maximum is not None:
            seconds = min(seconds, maximum)
        if seconds <= 0:
            raise TimeoutError("Insufficient worker time including evidence reserve")
        command = [sys.executable, "-m", module, *map(str, argv)]
        code = run_stage(
            command,
            log=args.output / f"{name}.log",
            env=env | ({"JAX_PLATFORMS": "cpu"} if cpu else {}),
            timeout=seconds,
            publish=publish,
        )
        summary["stages"].append(dict(name=name, returncode=code))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if code:
            raise RuntimeError(f"{name} exited {code}")

    try:
        estimates = []
        for role in ("baseline", "candidate"):
            output = args.output / (role + "-calibration")
            stage(
                role + "-calibration",
                "scripts.calibrate_encoder_bitext",
                "--plan",
                args.plan,
                "--role",
                role,
                "--output",
                output,
                "--batch-size",
                args.batch_size,
                "--max-seconds",
                "1200",
                maximum=1200,
            )
            estimates.append(json.loads((output / "estimate.json").read_text()))
        required = pair_estimate(
            estimates,
            plan_sha256=identity,
            subsets=frozen["subsets"],
            pairs=frozen["pairs"],
            batch_size=args.batch_size,
        )
        summary["estimated_pair_seconds_with_margin_and_scoring"] = required
        summary["remaining_seconds_before_full_run"] = deadline.remaining()
        if required + 360 > deadline.remaining():
            raise TimeoutError(
                "Complete matched bitext pair does not fit remaining lease"
            )
        for role in ("baseline", "candidate"):
            stage(
                role,
                "scripts.export_encoder_bitext_campaign",
                "--plan",
                args.plan,
                "--role",
                role,
                "--output",
                args.output / role,
                "--batch-size",
                args.batch_size,
                "--max-seconds",
                str(deadline.remaining() - 360),
            )
            # A completed model is immutable. Persist it before starting the
            # next model so Spot loss cannot erase an already finished export.
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    "--recursive",
                    str(args.output / role),
                    args.prefix + "/" + args.output.name + "/",
                ],
                check=True,
                timeout=deadline.remaining(120),
            )
        stage(
            "comparison",
            "scripts.compare_encoder_bitext_campaign",
            "--plan",
            args.plan,
            "--baseline",
            args.output / "baseline",
            "--candidate",
            args.output / "candidate",
            "--output",
            args.output / "comparison.json",
            cpu=True,
        )
        summary["passed"] = True
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                "--recursive",
                "--no-clobber",
                str(args.output),
                args.prefix + "/",
            ],
            check=True,
            timeout=deadline.remaining(300),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/encoder-bitext-evidence")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
