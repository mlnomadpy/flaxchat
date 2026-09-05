"""Shared controls and record helpers for matched trainer benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DATASET = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TARGET_PARAMETERS = 600_000
PARAMETER_TOLERANCE = 0.05
SEQUENCE_LENGTH = 256
GLOBAL_BATCH_SIZE = 8
SEED = 42
WARMUP_STEPS = 5
MEASURED_STEPS = 20
EXPECTED_HARDWARE = "Tesla P100-PCIE-16GB"
P100_FP32_PEAK_FLOPS = 9.3e12


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_batches(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        batches = {name: archive[name].copy() for name in archive.files}
    expected = {
        "train_inputs": (WARMUP_STEPS + MEASURED_STEPS + 1, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH),
        "train_targets": (WARMUP_STEPS + MEASURED_STEPS + 1, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH),
        "validation_inputs": (GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH),
        "validation_targets": (GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH),
    }
    if {name: value.shape for name, value in batches.items()} != expected:
        raise ValueError(f"matched batch shapes differ from protocol: expected {expected}")
    if any(value.dtype != np.int32 for value in batches.values()):
        raise ValueError("matched token batches must be int32")
    return batches


def validate_hardware(detected: str) -> None:
    normalized = detected.upper().replace("-", " ")
    if "P100" not in normalized:
        raise RuntimeError(f"matched protocol requires {EXPECTED_HARDWARE}; detected {detected}")


def make_record(
    *,
    framework: str,
    source_revision: str,
    model_parameters: int,
    steady_tokens_per_second: float,
    compile_seconds: float,
    peak_memory_bytes: int,
    checkpoint_seconds: float,
    validation_value: float,
    protocol_sha256: str,
    data_sha256: str,
    software: dict[str, str],
    limitations: str,
) -> dict:
    throughput = float(steady_tokens_per_second)
    return {
        "framework": framework,
        "source_revision": source_revision,
        "hardware": EXPECTED_HARDWARE,
        "device_count": 1,
        "precision": "float32",
        "model_parameters": int(model_parameters),
        "target_model_parameters": TARGET_PARAMETERS,
        "parameter_tolerance_fraction": PARAMETER_TOLERANCE,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "steady_tokens_per_second": throughput,
        "compile_seconds": float(compile_seconds),
        "peak_memory_bytes": max(1, int(peak_memory_bytes)),
        "model_flops_utilization": min(1.0, 6.0 * model_parameters * throughput / P100_FP32_PEAK_FLOPS),
        "checkpoint_seconds": float(checkpoint_seconds),
        "scaling_efficiency": 1.0,
        "validation_metric": "held_out_cross_entropy",
        "validation_value": float(validation_value),
        "limitations": limitations,
        "protocol_sha256": protocol_sha256,
        "data_sha256": data_sha256,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "optimizer": "adamw",
        "seed": SEED,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "software": software,
    }


def write_record(path: str | Path, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
