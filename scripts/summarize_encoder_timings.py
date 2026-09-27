"""Summarize recorded encoder steps without discarding latency outliers."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(rows):
    """Keep raw warm samples; throughput is total tokens / total duration."""
    steps = [row for row in rows if row.get("event") == "train_step"]
    if not steps:
        raise ValueError("No training steps")
    numbers = [row["step"] for row in steps]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("Training steps must be ordered and contiguous")
    for row in steps:
        for key in ("includes_compilation", "includes_profiling", "updated"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"Missing or invalid {key}")
        if not row["updated"]:
            raise ValueError("Skipped training update")
        for key in ("seconds", "loss", "gradient_norm_before_clip"):
            if not math.isfinite(row[key]):
                raise ValueError(f"Nonfinite {key}")
        if row["seconds"] <= 0:
            raise ValueError("Nonpositive step duration")
        for key in ("tokens", "masked_tokens", "nonpadding_tokens"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"Invalid {key}")
        if not row["masked_tokens"] <= row["nonpadding_tokens"] <= row["tokens"]:
            raise ValueError("Inconsistent token counts")
    warm = [r for r in steps if not r["includes_compilation"] and not r["includes_profiling"]]
    if not warm:
        raise ValueError("No unprofiled warm steps")
    seconds = [r["seconds"] for r in warm]
    duration = math.fsum(seconds)
    checkpoints = [r["seconds"] for r in rows if r.get("event") == "checkpoint"]
    if any(not math.isfinite(s) or s <= 0 for s in checkpoints):
        raise ValueError("Invalid checkpoint duration")
    phase_keys = ("batch_preparation_seconds", "compilation_diagnostics_seconds",
                  "synchronized_execution_and_profile_seconds")
    phase_timings = None
    if any(key in row for row in steps for key in phase_keys):
        for row in steps:
            if any(key not in row or not math.isfinite(row[key]) or row[key] < 0
                   for key in phase_keys):
                raise ValueError("Missing or invalid timing phase")
            if not math.isclose(math.fsum(row[key] for key in phase_keys),
                                row["seconds"], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("Timing phases do not partition the step duration")
        phase_timings = {
            key: {"warm_total_seconds": math.fsum(row[key] for row in warm),
                  "warm_median_seconds": statistics.median(row[key] for row in warm)}
            for key in phase_keys
        }
    gc_timings = None
    if any(row.get("host_gc_seconds") is not None for row in steps):
        for row in steps:
            seconds_gc = row.get("host_gc_seconds")
            count_gc = row.get("host_gc_collections")
            if (seconds_gc is None or not math.isfinite(seconds_gc) or seconds_gc < 0
                    or type(count_gc) is not int or count_gc < 0):
                raise ValueError("Missing or invalid GC observation")
        gc_timings = {
            "warm_total_seconds": math.fsum(row["host_gc_seconds"] for row in warm),
            "warm_collections": sum(row["host_gc_collections"] for row in warm),
            "warm_steps_with_collection": sum(row["host_gc_collections"] > 0 for row in warm),
            "slowest_warm_step": max(warm, key=lambda row: row["seconds"]),
            "scope": "Observed Python GC overlapping step measurement; retained in all throughput figures. Does not partition TPU device time.",
        }
    return {
        "warm_samples": warm,
        "measured_steps": len(warm),
        "median_seconds": statistics.median(seconds),
        "p95_seconds": sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1],
        "p95_method": "nearest rank; descriptive only for small samples",
        "max_seconds": max(seconds),
        "aggregate_tokens_per_second": sum(r["tokens"] for r in warm) / duration,
        "aggregate_nonpadding_tokens_per_second": sum(r["nonpadding_tokens"] for r in warm) / duration,
        "aggregate_masked_targets_per_second": sum(r["masked_tokens"] for r in warm) / duration,
        "checkpoint_seconds": checkpoints,
        "recorded_overhead": {
            "all_step_seconds": math.fsum(row["seconds"] for row in steps),
            "compilation_flagged_step_seconds": math.fsum(
                row["seconds"] for row in steps if row["includes_compilation"]),
            "profiling_flagged_step_seconds": math.fsum(
                row["seconds"] for row in steps if row["includes_profiling"]),
            "checkpoint_seconds": math.fsum(checkpoints),
            "scope": "Recorded host intervals only, not job wall time or billing. Flagged intervals include training and may overlap each other; do not add them to all_step_seconds. Initialization, restore and shutdown are not covered.",
        },
        "warm_phase_timings": phase_timings,
        "warm_gc_timings": gc_timings,
        "phase_timing_scope": "Host wall intervals; asynchronous batch transfers can finish during synchronized execution. Medians are not additive. Null means legacy logs without this instrumentation.",
        "compiled_memory": [r for r in rows if r.get("event") == "compiled_memory"],
        "device_memory": [r for r in rows if r.get("event") == "device_memory"],
        "scope": "Unprofiled warm training steps; excludes compile, profiling and checkpoint overhead. No outliers removed. Compiler memory and runtime memory are separate observations.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    planned = report["planned_cases"]
    observed = list(report["batches"])
    if len(set(planned)) != len(planned) or observed != planned[:len(observed)]:
        raise ValueError("Observed cases must be an ordered prefix of the plan")
    if report["complete"] and observed != planned:
        raise ValueError("Complete report is missing planned cases")
    for rows in report["batches"].values():
        steps = [row["step"] for row in rows if row.get("event") == "train_step"]
        if steps != list(range(1, report["horizon"] + 1)):
            raise ValueError("A retained case has not completed its planned horizon")
        configs = [row for row in rows if row.get("event") == "run_config"]
        if len(configs) != 1 or configs[0]["backend"] != report["backend"]:
            raise ValueError("Case backend does not match report")
    result = {
        "complete": report["complete"],
        "backend": report["backend"],
        "planned_cases": report["planned_cases"],
        "cases": {name: summarize(rows) for name, rows in report["batches"].items()},
        "production_qualified": False,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
