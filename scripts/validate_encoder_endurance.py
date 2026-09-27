"""Calibrate and run sustained recovery without assuming SSH/JAX rank equivalence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from scripts.plan_encoder_endurance import plan, select_calibration_log
from scripts.workload_deadline import WorkloadDeadline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/encoder-endurance"))
    args = parser.parse_args(argv)
    rank = int(os.environ["JAX_PROCESS_INDEX"])
    if (
        not args.prefix.startswith("gs://")
        or not 0 <= rank < args.processes
        or args.processes < 2
    ):
        parser.error("Require a GCS prefix and valid multi-host launcher coordinates")
    root = args.output
    root.mkdir(exist_ok=False, parents=True)
    deadline = WorkloadDeadline(
        min(6500, int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90)
    )
    env = os.environ | {"JAX_DEFAULT_MATMUL_PRECISION": "highest"}
    calibration = root / "calibration.log"
    command = [
        sys.executable,
        "-m",
        "scripts.train_encoder",
        "--config",
        "artifacts/mmbert-base/config.json",
        "--pretrained",
        "artifacts/mmbert-base",
        "--data",
        "artifacts/encoder-scale-0923/fixture/data-512",
        "--steps",
        "100",
        "--batch-size",
        "64",
        "--save-every",
        "100",
        "--dtype",
        "bfloat16",
        "--residual-dtype",
        "float32",
        "--mlm-projection",
        "masked",
        "--mlm-loss-backend",
        "pallas",
        "--output",
        args.prefix + "/calibration",
    ]
    with calibration.open("w") as log:
        code = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=deadline.remaining(600),
        ).returncode
    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            str(calibration),
            args.prefix + f"/calibration-evidence/worker-{rank}.log",
        ],
        check=True,
        timeout=deadline.remaining(60),
    )
    if code:
        return code
    if rank == 0:
        logs = []
        for worker in range(args.processes):
            local = root / f"worker-{worker}.log"
            while True:
                result = subprocess.run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        args.prefix + f"/calibration-evidence/worker-{worker}.log",
                        str(local),
                    ],
                    capture_output=True,
                    timeout=deadline.remaining(30),
                )
                if result.returncode == 0:
                    break
                time.sleep(min(3, deadline.remaining()))
            logs.append(local)
        selected, updates = select_calibration_log(
            logs, devices=args.devices, processes=args.processes
        )
        frozen: dict[str, Any] = plan(
            updates, minimum_seconds=1800, remaining_seconds=deadline.remaining() - 180
        )
        frozen.update(
            devices=args.devices,
            processes=args.processes,
            backend="pallas",
            batch_size=64,
            calibration_primary_worker_log=selected.name,
        )
        (root / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(root / "plan.json"),
                args.prefix + "/plan.json",
            ],
            check=True,
            timeout=deadline.remaining(60),
        )
    else:
        while True:
            result = subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    args.prefix + "/plan.json",
                    str(root / "plan.json"),
                ],
                capture_output=True,
                timeout=deadline.remaining(30),
            )
            if result.returncode == 0:
                break
            time.sleep(min(3, deadline.remaining()))
    selected = json.loads((root / "plan.json").read_text())
    if (
        selected["devices"] != args.devices
        or selected["processes"] != args.processes
        or selected["minimum_seconds"] != 1800
    ):
        raise ValueError("Shared plan topology or duration mismatch")
    remaining = int(deadline.remaining() - 65)
    if selected["estimated_total_seconds"] > remaining:
        raise RuntimeError("Lease no longer fits frozen sustained plan")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_encoder_scale",
            "--prefix",
            args.prefix + "/scale",
            "--fixture",
            "artifacts/encoder-scale-0923/fixture",
            "--expected-devices",
            str(args.devices),
            "--expected-processes",
            str(args.processes),
            "--steps",
            str(selected["steps"]),
            "--batch-size",
            "64",
            "--minimum-seconds",
            "1800",
            "--timeout-seconds",
            str(min(6600, remaining)),
            "--mlm-loss-backend",
            "pallas",
        ],
        env=env,
        timeout=deadline.remaining(),
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
