"""Independently verify raw worker evidence for a bounded endurance campaign."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.validate_encoder_scale import validate_updates
from scripts.validate_encoder_tpu import validate_hardware


def summarize(
    root,
    controller,
    *,
    processes,
    devices,
    steps,
    batch,
    minimum_seconds,
    backend="xla",
):
    if (
        any(type(value) is not int for value in (processes, devices, steps, batch))
        or processes < 1
        or devices < processes
        or devices % processes
        or steps < 4
        or steps % 2
        or batch < 1
        or batch % devices
        or not math.isfinite(minimum_seconds)
        or minimum_seconds <= 0
    ):
        raise ValueError("Invalid endurance topology, horizon, batch or duration")
    failures, ranks, manifests, evidence = [], [], [], []
    performance = None
    control = json.loads(Path(controller).read_text())
    workers = control.get("workers", [])
    controller_ok = not (
        control.get("passed") is not True
        or len(workers) != processes
        or any(type(w.get("rank")) is not int for w in workers)
        or sorted(w["rank"] for w in workers) != list(range(processes))
        or any(
            type(w.get("returncode")) is not int or w["returncode"] != 0
            for w in workers
        )
    )
    for directory in sorted(Path(root).glob("worker-*")):
        try:
            report = json.loads((directory / "summary.json").read_text())
            hardware = json.loads(
                next(
                    line
                    for line in (directory / "hardware.log").read_text().splitlines()
                    if line.startswith("{")
                )
            )
            validate_hardware(
                hardware, "multi" if processes > 1 else "single", processes, devices
            )
            rank = hardware["process_index"]
            if type(rank) is not int or report["hardware"] != hardware:
                raise ValueError("Invalid or inconsistent physical rank")
            ranks.append(rank)
            if (
                report["passed"] is not True
                or report["steps"] != steps
                or report["batch_size"] != batch
                or report["minimum_seconds"] != minimum_seconds
                or report["mlm_loss_backend"] != backend
            ):
                raise ValueError("Worker failed or changed the declared protocol")
            stages = report["stages"]
            if [(s["name"], s["returncode"], s["passed"]) for s in stages] != [
                ("hardware", 0, True),
                ("baseline", 0, True),
                ("interrupt", -9, True),
                ("resume", 0, True),
            ]:
                raise ValueError("Incomplete or unexpected stage history")
            if (
                f"FAULT_INJECTION: encoder SIGKILL after committed step {steps // 2}"
                not in (directory / "interrupt.log").read_text()
            ):
                raise ValueError("Missing committed interruption evidence")
            left = json.loads((directory / "baseline-manifest.json").read_text())
            right = json.loads((directory / "recovery-manifest.json").read_text())
            if any(
                type(m.get("step")) is not int or m["step"] != steps
                for m in (left, right)
            ):
                raise ValueError("Raw checkpoint manifest has the wrong final step")
            for key in ("model_state", "optimizer_state", "training_state"):
                if not left[key] or left[key] != right[key]:
                    raise ValueError("Raw final state mismatch or empty inventory")
            manifests.append(left)
            if rank == 0:
                performance = validate_updates(
                    directory / "baseline.log",
                    first=1,
                    last=steps,
                    batch=batch,
                    length=512,
                    minimum_seconds=minimum_seconds,
                )
                validate_updates(
                    directory / "resume.log",
                    first=steps // 2 + 1,
                    last=steps,
                    batch=batch,
                    length=512,
                )
            else:
                for stage in ("baseline", "resume"):
                    for line in (directory / f"{stage}.log").read_text().splitlines():
                        if (
                            line.startswith("{")
                            and json.loads(line).get("event") == "train_step"
                        ):
                            raise ValueError(
                                "Training updates logged by non-primary JAX rank"
                            )
            evidence.append(
                {
                    str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in directory.iterdir()
                    if f.is_file()
                }
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            failures.append(f"{directory}: {exc}")
    if sorted(ranks) != list(range(processes)):
        failures.append("Missing or duplicate physical workers")
    if any(m != manifests[0] for m in manifests[1:]):
        failures.append("Workers read different final manifests")
    return dict(
        passed=controller_ok and not failures,
        controller_passed=controller_ok,
        raw_worker_evidence_passed=not failures,
        scope="bounded_encoder_endurance_and_exact_recovery",
        production_quality_qualified=False,
        failures=(
            ["Controller did not complete successfully on every worker"]
            if not controller_ok
            else []
        )
        + failures,
        evidence_sha256=evidence,
        devices=devices,
        processes=processes,
        steps=steps,
        minimum_seconds=minimum_seconds,
        mlm_loss_backend=backend,
        baseline_performance=performance,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--controller", required=True)
    p.add_argument("--processes", type=int, required=True)
    p.add_argument("--devices", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--minimum-seconds", type=float, default=120)
    p.add_argument(
        "--backend", choices=["xla", "xla_full", "xla_local", "pallas"], default="xla"
    )
    p.add_argument("--output", required=True)
    args = vars(p.parse_args())
    output = args.pop("output")
    result = summarize(**args)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
