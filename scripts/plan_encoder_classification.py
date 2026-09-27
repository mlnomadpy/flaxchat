"""Estimate a fixed classification recipe from complete physical calibration logs."""

import math
import statistics


def plan(
    records,
    *,
    steps,
    batch_size,
    evaluation_examples,
    remaining_seconds,
    reserve_seconds=600,
):
    if (
        type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(evaluation_examples) is not int
        or evaluation_examples < 1
        or not math.isfinite(remaining_seconds)
        or remaining_seconds <= 0
        or not math.isfinite(reserve_seconds)
        or reserve_seconds < 0
    ):
        raise ValueError("Invalid classification planning dimensions or lease")
    rows = [r for r in records if r.get("event") == "classifier_train_step"]
    if len(rows) < 21 or [r.get("step") for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Require at least 21 consecutive calibration updates")
    for i, row in enumerate(rows):
        if (
            type(row["step"]) is not int
            or row.get("updated") is not True
            or row.get("examples") != batch_size
            or row.get("includes_compilation") is not (i == 0)
            or not math.isfinite(row["loss"])
            or not math.isfinite(row["seconds"])
            or row["seconds"] <= 0
        ):
            raise ValueError("Invalid classification calibration evidence")
    # Use the slower 90th percentile with 25% headroom; evaluation is charged
    # at training speed, plus an explicit checkpoint/compile/upload allowance.
    timings = sorted(row["seconds"] for row in rows[1:])
    seconds = timings[math.ceil(0.9 * len(timings)) - 1]
    estimated = (
        steps + math.ceil(evaluation_examples / batch_size)
    ) * seconds * 1.25 + reserve_seconds
    if estimated > remaining_seconds:
        raise ValueError("Full fixed recipe and evaluation do not fit remaining lease")
    return dict(
        steps=steps,
        batch_size=batch_size,
        evaluation_examples=evaluation_examples,
        calibration_updates=len(rows),
        median_step_seconds=statistics.median(timings),
        p90_step_seconds=seconds,
        duration_margin=1.25,
        estimated_total_seconds=estimated,
        reserve_seconds=reserve_seconds,
        quality_qualified=False,
        full_recipe_execution_still_required=True,
    )
