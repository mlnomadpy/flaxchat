"""Bounded physical encoder endurance and exact-recovery campaign.

Run on every worker through gcp_spot_supervisor; never provisions resources.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from scripts.validate_encoder_tpu import stage_environment, validate_hardware
from scripts.workload_deadline import WorkloadDeadline
from scripts.stage_progress import run_stage


def validate_updates(path, *, first, last, batch, length, minimum_seconds=0):
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.startswith("{")
    ]
    steps = [r for r in records if r.get("event") == "train_step"]
    if [r["step"] for r in steps] != list(range(first, last + 1)):
        raise ValueError("Incomplete or unordered update horizon")
    for row in steps:
        if (
            type(row["step"]) is not int
            or row["updated"] is not True
            or row["projection_dense_fallback"] is not False
            or row["tokens"] != batch * length
            or not math.isfinite(row["loss"])
            or not 0 < row["masked_tokens"] <= row["tokens"]
            or not math.isfinite(row["seconds"])
            or row["seconds"] <= 0
            or row["includes_compilation"] is not (row["step"] == first)
            or type(row.get("includes_profiling", False)) is not bool
        ):
            raise ValueError("Invalid update evidence")
    measured = [r for r in steps if not r["includes_compilation"] and not r.get("includes_profiling", False)]
    # Legacy logs can omit useful-token counts, but partial or invalid counts
    # must not produce a deceptively complete throughput measurement.
    has_nonpadding = any("nonpadding_tokens" in r for r in steps)
    if has_nonpadding and any(
        type(r.get("nonpadding_tokens")) is not int
        or not r["masked_tokens"] <= r["nonpadding_tokens"] <= r["tokens"]
        for r in steps
    ):
        raise ValueError("Invalid or incomplete nonpadding token evidence")
    seconds = sum(r["seconds"] for r in measured)
    if seconds < minimum_seconds:
        raise ValueError("Insufficient measured training duration")
    checkpoints = [r for r in records if r.get("event") == "checkpoint"]
    if (
        not checkpoints
        or type(checkpoints[-1]["step"]) is not int
        or checkpoints[-1]["step"] != last
        or any(not math.isfinite(row["seconds"]) or row["seconds"] <= 0
               for row in checkpoints)
    ):
        raise ValueError("Missing final committed checkpoint")
    return dict(
        steps=len(steps),
        measured_steps=len(measured),
        measured_seconds=seconds,
        median_measured_step_seconds=statistics.median(r['seconds'] for r in measured)
        if measured else None,
        checkpoint_seconds=sum(r['seconds'] for r in checkpoints),
        checkpoint_events=len(checkpoints),
        compilation_inclusive_step_seconds=sum(r['seconds'] for r in steps if r['includes_compilation']),
        profiling_inclusive_step_seconds=sum(r['seconds'] for r in steps if r.get('includes_profiling', False)),
        timing_scope='Step and checkpoint event durations; not total wall time. Compilation/profiling categories may overlap.',
        tokens_per_second=sum(r["tokens"] for r in measured) / seconds
        if seconds
        else None,
        tokens_per_second_basis="padded_input_positions",
        nonpadding_tokens_per_second=(
            sum(r["nonpadding_tokens"] for r in measured) / seconds
            if has_nonpadding and seconds
            else None
        ),
        nonpadding_fraction=(
            sum(r["nonpadding_tokens"] for r in measured)
            / sum(r["tokens"] for r in measured)
            if has_nonpadding and measured
            else None
        ),
        masked_targets_per_second=sum(r["masked_tokens"] for r in measured) / seconds
        if seconds
        else None,
        final_loss=steps[-1]["loss"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--expected-devices", type=int, required=True)
    p.add_argument("--expected-processes", type=int, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--minimum-seconds", type=float, default=120)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--snapshot", default="artifacts/mmbert-base")
    p.add_argument(
        "--mlm-loss-backend",
        choices=["xla", "xla_full", "xla_local", "pallas"],
        default="xla",
    )
    a = p.parse_args()
    if (
        not a.prefix.startswith("gs://")
        or a.steps < 100
        or a.steps % 2
        or not 60 <= a.timeout_seconds <= 6600
        or not math.isfinite(a.minimum_seconds)
        or a.minimum_seconds < 0
        or a.batch_size <= 0
    ):
        p.error("Require bounded, even training horizon and GCS output")
    rank = int(os.environ.get("JAX_PROCESS_INDEX", "0"))
    output = Path(a.fixture) / f"results-scale-{a.expected_devices}-{rank}"
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        passed=False,
        quality_qualified=False,
        stages=[],
        expected_devices=a.expected_devices,
        expected_processes=a.expected_processes,
        steps=a.steps,
        batch_size=a.batch_size,
        minimum_seconds=a.minimum_seconds,
        mlm_loss_backend=a.mlm_loss_backend,
    )
    deadline = WorkloadDeadline(a.timeout_seconds)
    env = stage_environment(os.environ)

    def persist():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                *map(str, output.iterdir()),
                f"{a.prefix}/worker-{rank}/",
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )

    def stage(name, argv, expected=0, marker=None):
        remaining = deadline.remaining() - 65
        if remaining <= 0:
            raise TimeoutError("Campaign deadline exhausted")
        log = output / f"{name}.log"

        def publish_progress(path, seconds):
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(path),
                    str(path.with_suffix('.status.json')),
                    f"{a.prefix}/worker-{rank}/progress/",
                ],
                check=True,
                timeout=min(30, seconds),
                capture_output=True,
            )

        code = run_stage(
            argv, log=log, env=env, timeout=remaining, publish=publish_progress
        )
        passed = code == expected and (marker is None or marker in log.read_text())
        report["stages"].append(dict(name=name, passed=passed, returncode=code))
        persist()
        if not passed:
            raise RuntimeError(f"{name} failed")

    try:
        probe = "import json,jax,flaxchat.common; print(json.dumps(dict(backend=jax.default_backend(),process_count=jax.process_count(),device_count=jax.device_count(),process_index=jax.process_index())))"
        stage("hardware", [sys.executable, "-c", probe])
        hardware = json.loads(
            next(
                x
                for x in (output / "hardware.log").read_text().splitlines()
                if x.startswith("{")
            )
        )
        validate_hardware(
            hardware,
            "multi" if a.expected_processes > 1 else "single",
            a.expected_processes,
            a.expected_devices,
        )
        report["hardware"] = hardware
        base = [
            "--config",
            a.snapshot + "/config.json",
            "--pretrained",
            a.snapshot,
            "--data",
            a.fixture + "/data-512",
            "--steps",
            str(a.steps),
            "--batch-size",
            str(a.batch_size),
            "--save-every",
            str(a.steps // 2),
            "--dtype",
            "bfloat16",
            "--residual-dtype",
            "float32",
            "--mlm-projection",
            "masked",
            "--mlm-loss-backend",
            a.mlm_loss_backend,
            "--loss-chunk-size",
            "128",
        ]
        baseline = a.prefix + "/baseline"
        recovery = a.prefix + "/recovery"
        stage(
            "baseline",
            [
                sys.executable,
                "-m",
                "scripts.train_encoder",
                *base,
                "--output",
                baseline,
            ],
        )
        # The shorter fault/recovery path verifies the complete sustained horizon.
        stage(
            "interrupt",
            [
                sys.executable,
                "-m",
                "scripts.validate_encoder_interruption",
                *base,
                "--output",
                recovery,
                "--fault-step",
                str(a.steps // 2),
            ],
            expected=-9,
            marker=f"FAULT_INJECTION: encoder SIGKILL after committed step {a.steps // 2}",
        )
        stage(
            "resume",
            [
                sys.executable,
                "-m",
                "scripts.train_encoder",
                *base,
                "--output",
                recovery,
                "--resume",
            ],
        )
        manifests = []
        for name, uri in [("baseline", baseline), ("recovery", recovery)]:
            raw = subprocess.check_output(
                ["gcloud", "storage", "cat", f"{uri}/{a.steps}/manifest/metadata"],
                text=True,
                timeout=60,
            )
            (output / f"{name}-manifest.json").write_text(raw)
            manifests.append(json.loads(raw))
        comparisons = {
            k: manifests[0][k] == manifests[1][k]
            for k in ("model_state", "optimizer_state", "training_state")
        }
        if not all(comparisons.values()):
            raise ValueError("Final model, optimizer or cursor mismatch")
        report["recovery"] = comparisons
        if hardware["process_index"] == 0:
            report["baseline"] = validate_updates(
                output / "baseline.log",
                first=1,
                last=a.steps,
                batch=a.batch_size,
                length=512,
                minimum_seconds=a.minimum_seconds,
            )
            report["resume"] = validate_updates(
                output / "resume.log",
                first=a.steps // 2 + 1,
                last=a.steps,
                batch=a.batch_size,
                length=512,
            )
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        persist()
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
