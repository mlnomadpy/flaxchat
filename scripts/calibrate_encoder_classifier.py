"""Bounded physical calibration of a frozen full-schedule classifier recipe.

No provisioning; parent uses only the standard library and never claims quality.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.plan_encoder_classification import plan as estimate
from scripts.stage_progress import run_stage
from scripts.validate_encoder_quality_pilot import file_hash
from scripts.workload_deadline import WorkloadDeadline


def calibration_command(plan, key):
    argv = list(plan["commands"][key]["train"])
    if argv[1:3] != ["-m", "scripts.finetune_encoder_classifier"]:
        raise ValueError("Expected classifier trainer")
    if "--resume" in argv or "--stop-after" in argv:
        raise ValueError("Calibration requires a fresh full recipe")
    for flag, value in (
        ("--steps", plan["steps"]),
        ("--batch-size", plan["batch_size"]),
        ("--warmup-steps", plan["warmup_steps"]),
    ):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != str(value):
            raise ValueError("Command differs from frozen full schedule")
    if plan["steps"] <= 100:
        raise ValueError("Full horizon must exceed calibration")
    if argv.count("--output") != 1 or not argv[argv.index("--output") + 1].startswith(
        "gs://"
    ):
        raise ValueError("Require unique persistent checkpoint output")
    argv[0] = sys.executable
    return argv + ["--stop-after", "100"]


def review(records, frozen, *, devices, lease_seconds):
    configs = [r for r in records if r.get("event") == "classifier_run_config"]
    if len(configs) != 1:
        raise ValueError("Require unique classifier runtime evidence")
    config = configs[0]
    if (config.get("backend"), config.get("devices"), config.get("processes")) != (
        "tpu",
        devices,
        1,
    ):
        raise ValueError("Wrong physical calibration topology")
    for key in ("steps", "batch_size", "warmup_steps"):
        if config["recipe"].get(key) != frozen[key]:
            raise ValueError("Runtime schedule differs from frozen plan")
    updates = [r for r in records if r.get("event") == "classifier_train_step"]
    if len(updates) != 100:
        raise ValueError("Require all 100 calibration updates")
    return estimate(
        records,
        steps=frozen["steps"],
        batch_size=frozen["batch_size"],
        evaluation_examples=frozen["evaluation_examples"],
        remaining_seconds=lease_seconds,
    )


def run(args):
    if (
        not args.prefix.startswith("gs://")
        or int(os.environ.get("JAX_PROCESS_COUNT", "1")) != 1
    ):
        raise ValueError("Require single-host worker and GCS evidence")
    frozen = json.loads(args.plan.read_text())
    command = calibration_command(frozen, args.key)
    for field in ("input_sha256", "source_sha256"):
        if not frozen.get(field):
            raise ValueError("Missing frozen provenance")
        for path, digest in frozen[field].items():
            if file_hash(path) != digest:
                raise ValueError(f"Frozen file changed: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = WorkloadDeadline(
        min(1800, int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90)
    )
    env = os.environ | {
        "JAX_PLATFORMS": "tpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    }
    summary = dict(
        passed=False,
        quality_qualified=False,
        plan_sha256=file_hash(args.plan),
        worker_sha256=file_hash(__file__),
        key=args.key,
    )

    def publish(path, seconds):
        subprocess.run(
            ["gcloud", "storage", "cp", str(path), args.prefix + "/progress/"],
            check=True,
            timeout=min(30, seconds),
            capture_output=True,
        )

    try:
        log = args.output / "calibration.log"
        code = run_stage(
            command,
            log=log,
            env=env,
            timeout=deadline.remaining() - 65,
            publish=publish,
        )
        if code:
            raise RuntimeError(f"Classifier calibration exited {code}")
        records = [
            json.loads(line)
            for line in log.read_text().splitlines()
            if line.startswith("{")
        ]
        result = review(
            records,
            frozen,
            devices=args.devices,
            lease_seconds=args.full_run_lease_seconds,
        )
        checkpoint = command[command.index("--output") + 1]
        # Read committed metadata in a separate CPU process, avoiding parent TPU ownership.
        probe = "import json,sys; from flaxchat.checkpoint import load_checkpoint_metadata; print(json.dumps(load_checkpoint_metadata(sys.argv[1],step=100)))"
        metadata = subprocess.check_output(
            [sys.executable, "-c", probe, checkpoint],
            env=env | {"JAX_PLATFORMS": "cpu"},
            text=True,
            timeout=deadline.remaining(60),
        )
        committed = json.loads(metadata)
        if (
            committed["step"] != 100
            or committed["resolved_config"]["steps"] != frozen["steps"]
        ):
            raise ValueError("Missing committed full-schedule calibration checkpoint")
        (args.output / "checkpoint.json").write_text(
            json.dumps(committed, indent=2) + "\n"
        )
        (args.output / "estimate.json").write_text(json.dumps(result, indent=2) + "\n")
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
                *map(str, args.output.iterdir()),
                args.prefix + "/evidence/",
            ],
            check=True,
            timeout=deadline.remaining(60),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--key", default="baseline-seed-17")
    p.add_argument("--prefix", required=True)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--full-run-lease-seconds", type=float, default=6500)
    p.add_argument(
        "--output", type=Path, default=Path("/tmp/encoder-classifier-calibration")
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
