"""Validate and compare apples-to-apples trainer benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "framework", "source_revision", "hardware", "device_count", "precision",
    "model_parameters", "sequence_length", "global_batch_size",
    "steady_tokens_per_second", "compile_seconds", "peak_memory_bytes",
    "model_flops_utilization", "checkpoint_seconds", "scaling_efficiency",
    "validation_metric", "validation_value", "limitations",
}


def load_record(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"{path} is missing benchmark fields: {sorted(missing)}")
    for name in (
        "device_count", "model_parameters", "sequence_length", "global_batch_size",
        "steady_tokens_per_second", "compile_seconds", "peak_memory_bytes",
        "checkpoint_seconds",
    ):
        if record[name] <= 0:
            raise ValueError(f"{path}: {name} must be positive")
    for name in ("model_flops_utilization", "scaling_efficiency"):
        if not 0 <= record[name] <= 1:
            raise ValueError(f"{path}: {name} must be between zero and one")
    if not isinstance(record["limitations"], str) or not record["limitations"].strip():
        raise ValueError(f"{path}: limitations must be a non-empty string")
    return record


def compare(records: list[dict]) -> dict:
    if len(records) < 2:
        raise ValueError("comparison requires at least two benchmark records")
    controlled = (
        "hardware", "device_count", "precision", "model_parameters",
        "sequence_length", "global_batch_size", "validation_metric",
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
            }
            for record in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    args = parser.parse_args()
    print(json.dumps(compare([load_record(path) for path in args.records]), indent=2))


if __name__ == "__main__":
    main()
