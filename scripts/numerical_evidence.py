"""Preserve actual device values for reproducible numerical investigations."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def write_numerical_evidence(directory, arrays, *, metadata):
    """Write a new case directory, never overwrite an existing observation.

    Materialize JAX arrays before serialization. NumPy's NPZ representation of
    nonstandard BF16 dtypes can lose the dtype on reload; FP32 represents every
    BF16 value exactly. Record both original and stored dtypes explicitly.
    Inputs, outputs and gradients should all be supplied by the caller.
    """
    directory = Path(directory)
    stored, descriptions = {}, {}
    for name, value in arrays.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid numerical evidence array name: {name}")
        array = np.asarray(value)
        original_dtype = array.dtype.name
        if original_dtype == "bfloat16":
            array = array.astype(np.float32)
        if array.dtype.kind not in "biufc":
            raise ValueError(f"Unsupported evidence dtype: {original_dtype}")
        stored[name] = array
        descriptions[name] = dict(shape=list(array.shape), dtype=original_dtype,
                                  stored_dtype=array.dtype.name)
    if not stored:
        raise ValueError("At least one numerical array is required")
    # Validate metadata before creating a case, and forbid nonstandard JSON NaNs.
    json.dumps(metadata, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "arrays.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **stored)
    report = dict(format="flaxchat-numerical-evidence-v1", arrays=descriptions,
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  metadata=metadata)
    # The manifest is the completion marker; partial writes are not valid cases.
    with (directory / "manifest.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def read_numerical_evidence(directory):
    """Verify integrity and return NumPy values plus original-dtype metadata."""
    directory = Path(directory)
    report = json.loads((directory / "manifest.json").read_text())
    if report.get("format") != "flaxchat-numerical-evidence-v1":
        raise ValueError("Unknown numerical evidence format")
    archive = directory / "arrays.npz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != report["archive_sha256"]:
        raise ValueError("Numerical evidence checksum mismatch")
    with np.load(archive, allow_pickle=False) as loaded:
        if set(loaded.files) != set(report["arrays"]):
            raise ValueError("Numerical evidence array names mismatch")
        arrays = {name: loaded[name] for name in loaded.files}
    for name, array in arrays.items():
        description = report["arrays"][name]
        if (list(array.shape) != description["shape"]
                or array.dtype.name != description["stored_dtype"]):
            raise ValueError(f"Numerical evidence shape/dtype mismatch: {name}")
    return arrays, report
