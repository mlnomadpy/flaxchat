"""Validate and compare apples-to-apples trainer benchmark records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


REQUIRED_FIELDS = {
    "framework", "source_revision", "hardware", "device_count", "precision",
    "model_parameters", "target_model_parameters", "parameter_tolerance_fraction",
    "sequence_length", "global_batch_size",
    "steady_tokens_per_second", "compile_seconds", "peak_memory_bytes",
    "model_flops_utilization", "checkpoint_seconds", "scaling_efficiency",
    "validation_metric", "validation_value", "limitations", "protocol_sha256",
    "dataset", "dataset_revision", "optimizer", "seed", "warmup_steps",
    "measured_steps", "data_sha256",
}


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_record(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"{path} is missing benchmark fields: {sorted(missing)}")
    if not _SHA40.fullmatch(record["source_revision"]):
        raise ValueError(f"{path}: source_revision must be an exact 40-character git SHA")
    if not _SHA256.fullmatch(record["protocol_sha256"]):
        raise ValueError(f"{path}: protocol_sha256 must be a 64-character SHA-256")
    if not _SHA256.fullmatch(record["data_sha256"]):
        raise ValueError(f"{path}: data_sha256 must be a 64-character SHA-256")
    for name in (
        "device_count", "model_parameters", "target_model_parameters",
        "sequence_length", "global_batch_size", "peak_memory_bytes", "measured_steps",
    ):
        if not isinstance(record[name], int) or isinstance(record[name], bool) or record[name] <= 0:
            raise ValueError(f"{path}: {name} must be a positive integer")
    for name in ("steady_tokens_per_second", "compile_seconds", "checkpoint_seconds"):
        if not _is_finite_number(record[name]) or record[name] <= 0:
            raise ValueError(f"{path}: {name} must be finite and positive")
    if (
        not isinstance(record["warmup_steps"], int)
        or isinstance(record["warmup_steps"], bool)
        or record["warmup_steps"] < 0
    ):
        raise ValueError(f"{path}: warmup_steps must be a non-negative integer")
    tolerance = record["parameter_tolerance_fraction"]
    if not _is_finite_number(tolerance) or not 0 <= tolerance < 1:
        raise ValueError(
            f"{path}: parameter_tolerance_fraction must be finite and in [0, 1)"
        )
    relative_error = abs(
        record["model_parameters"] - record["target_model_parameters"]
    ) / record["target_model_parameters"]
    if relative_error > tolerance:
        raise ValueError(
            f"{path}: model_parameters is {relative_error:.2%} from the target, "
            f"exceeding the {tolerance:.2%} tolerance"
        )
    if not _is_finite_number(record["model_flops_utilization"]) or not 0 <= record["model_flops_utilization"] <= 1:
        raise ValueError(f"{path}: model_flops_utilization must be between zero and one")
    if not _is_finite_number(record["scaling_efficiency"]) or record["scaling_efficiency"] < 0:
        raise ValueError(f"{path}: scaling_efficiency must be finite and non-negative")
    if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
        raise ValueError(f"{path}: seed must be an integer")
    if not _is_finite_number(record["validation_value"]):
        raise ValueError(f"{path}: validation_value must be finite")
    if not isinstance(record["limitations"], str) or not record["limitations"].strip():
        raise ValueError(f"{path}: limitations must be a non-empty string")
    return record


def compare(records: list[dict]) -> dict:
    if len(records) < 2:
        raise ValueError("comparison requires at least two benchmark records")
    frameworks = [record["framework"] for record in records]
    if len(set(frameworks)) != len(frameworks):
        raise ValueError("comparison requires one record per framework")
    controlled = (
        "hardware", "device_count", "precision", "target_model_parameters",
        "parameter_tolerance_fraction",
        "sequence_length", "global_batch_size", "validation_metric",
        "protocol_sha256", "dataset", "dataset_revision", "optimizer", "seed",
        "warmup_steps", "measured_steps", "data_sha256",
    )
    reference = records[0]
    mismatches = {
        field: [record[field] for record in records]
        for field in controlled
        if any(record[field] != reference[field] for record in records[1:])
    }
    if mismatches:
        raise ValueError(f"benchmark controls differ: {mismatches}")
    baseline = reference["steady_tokens_per_second"]
    return {
        "controls": {field: reference[field] for field in controlled},
        "results": [
            {
                **record,
                "throughput_relative_to_first": record["steady_tokens_per_second"] / baseline,
                "parameters_relative_to_target": (
                    record["model_parameters"] / record["target_model_parameters"]
                ),
            }
            for record in records
        ],
    }


def protocol_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    records = [load_record(path) for path in args.records]
    expected_hash = protocol_sha256(args.protocol)
    unexpected = {
        record["protocol_sha256"] for record in records
        if record["protocol_sha256"] != expected_hash
    }
    if unexpected:
        raise ValueError(
            f"record protocol hashes do not match {args.protocol}: "
            f"expected {expected_hash}, received {sorted(unexpected)}"
        )
    print(json.dumps(compare(records), indent=2))


if __name__ == "__main__":
    main()
