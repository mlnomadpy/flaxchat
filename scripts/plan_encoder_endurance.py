"""Freeze an endurance horizon from measured calibration before qualification."""

import math
import statistics


def plan(steps, *, minimum_seconds, remaining_seconds, reserve_seconds=900):
    if (
        not math.isfinite(minimum_seconds)
        or minimum_seconds <= 0
        or not math.isfinite(remaining_seconds)
        or remaining_seconds <= 0
        or not math.isfinite(reserve_seconds)
        or reserve_seconds < 0
    ):
        raise ValueError("Require finite positive durations and a nonnegative reserve")
    measured = [s for s in steps if not s["includes_compilation"]]
    if len(measured) < 20:
        raise ValueError("At least 20 measured calibration updates required")
    if any(
        s["updated"] is not True
        or s["projection_dense_fallback"] is not False
        or not math.isfinite(s["loss"])
        or not math.isfinite(s["seconds"])
        or s["seconds"] <= 0
        for s in measured
    ):
        raise ValueError("Calibration contains failed, nonfinite or fallback updates")
    seconds = statistics.median(s["seconds"] for s in measured)
    horizon = max(100, 2 * math.ceil(minimum_seconds * 1.25 / seconds / 2))
    estimated = 2 * horizon * seconds + reserve_seconds
    if estimated > remaining_seconds:
        raise ValueError("Insufficient lease for sustained baseline and exact recovery")
    return dict(
        steps=horizon,
        minimum_seconds=minimum_seconds,
        calibrated_median_seconds=seconds,
        calibration_updates=len(measured),
        estimated_total_seconds=estimated,
        reserve_seconds=reserve_seconds,
        duration_margin=1.25,
        measured_duration_still_required=True,
    )


def select_calibration_log(
    paths, *, devices, processes, steps=100, batch=64, length=512, backend="pallas"
):
    """Select the actual primary process evidence, independent of SSH ordering."""
    import json
    from pathlib import Path
    from scripts.validate_encoder_scale import validate_updates

    paths = [Path(path) for path in paths]
    if len(paths) != processes or len({p.resolve() for p in paths}) != processes:
        raise ValueError("Exactly one calibration log per physical worker required")
    candidates = []
    for path in paths:
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.startswith("{")
        ]
        if any(row.get("event") == "train_step" for row in records):
            candidates.append((path, records))
    if len(candidates) != 1:
        raise ValueError("Exactly one runtime primary calibration log required")
    selected, records = candidates[0]
    configs = [row for row in records if row.get("event") == "run_config"]
    if (
        len(configs) != 1
        or configs[0].get("backend") != "tpu"
        or configs[0].get("devices") != devices
        or configs[0].get("processes") != processes
        or configs[0].get("mlm_loss_backend") != backend
    ):
        raise ValueError("Calibration topology/backend mismatch")
    validate_updates(selected, first=1, last=steps, batch=batch, length=length)
    return selected, [row for row in records if row.get("event") == "train_step"]
