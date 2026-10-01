"""Read committed Orbax JSON admission evidence without Orbax or JAX.

This authenticates metadata and its recorded model manifest, not tensor bytes.
Physical restore must still verify the state manifest before mutating a model.
Only an explicit committed step is supported: no mkdir, cleanup, or latest-step
discovery runs against a namespace that may have a concurrent writer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

MAX_JSON_BYTES = 8 * 1024 * 1024


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _json(payload):
    def invalid(value):
        raise ValueError(f"Nonfinite checkpoint JSON value: {value}")

    value = json.loads(payload, object_pairs_hook=_object, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint JSON must be an object")
    return value


def _describe(uri):
    value = _json(
        subprocess.check_output(
            ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
            timeout=60,
        )
    )
    generation = str(value.get("generation", ""))
    size = value.get("size")
    if not generation.isdecimal() or int(generation) <= 0:
        raise ValueError("Missing GCS metadata object generation")
    if (
        isinstance(size, bool)
        or not str(size).isdecimal()
        or not 0 <= int(size) <= MAX_JSON_BYTES
    ):
        raise ValueError("Checkpoint metadata object exceeds read bound or lacks size")
    return generation


def read_committed_metadata(checkpoint_dir, step, *, include_receipt=False):
    """Read a stable explicit step and authenticate commit/identity JSON.

    Local reads are bounded and checked again after all required files are read.
    GCS reads select exact generations and then reject any namespace changes.
    No cleanup-enabled checkpoint manager or numerical import is involved.
    """
    if type(step) is not int or step < 0:
        raise ValueError("An explicit nonnegative integer parent step is required")
    root = str(checkpoint_dir).rstrip("/")
    cloud = root.startswith("gs://")
    if cloud:
        if not re.fullmatch(
            r"gs://[a-z0-9][a-z0-9._-]*(?:/[^?#]+)?", root
        ) or ".." in root.split("/"):
            raise ValueError("Invalid GCS checkpoint namespace")
    else:
        root = str(Path(root).expanduser().resolve(strict=True))
        if not Path(root).is_dir():
            raise ValueError("Checkpoint namespace must be an existing directory")
    prefix = root + "/" + str(step)
    payloads, versions = {}, {}

    def read(relative):
        location = prefix + "/" + relative
        if cloud:
            generation = _describe(location)
            payload = subprocess.check_output(
                ["gcloud", "storage", "cat", location + "#" + generation], timeout=60
            )
            versions[location] = generation
        else:
            path = Path(location)
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > MAX_JSON_BYTES
            ):
                raise ValueError(f"Missing or oversized committed metadata: {relative}")
            with path.open("rb") as handle:
                payload = handle.read(MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            raise ValueError("Checkpoint metadata exceeds bounded read size")
        payloads[location] = payload
        return payload

    try:
        commit = _json(read("_CHECKPOINT_METADATA"))
        timestamp = commit.get("commit_timestamp_nsecs")
        if type(timestamp) is not int or timestamp <= 0:
            raise ValueError("Checkpoint step is not committed")
        for marker in (
            "commit_success.txt",
            "metadata/commit_success.txt",
            "manifest/commit_success.txt",
        ):
            read(marker)
        metadata = _json(read("metadata/metadata"))
        manifest = _json(read("manifest/metadata"))
        if manifest.get("format_version") not in (2, 3):
            raise ValueError("Unsupported checkpoint format")
        if (
            type(manifest.get("step")) is not int
            or manifest["step"] != step
            or metadata.get("step", step) != step
        ):
            raise ValueError("Checkpoint selected step and metadata/manifest disagree")
        if _hash(metadata) != manifest.get("metadata_sha256"):
            raise ValueError("Checkpoint metadata integrity mismatch")
        identity = {
            "resolved_config": metadata.get(
                "resolved_config", metadata.get("model_config")
            ),
            "tokenizer": metadata.get("tokenizer_identity", "unavailable"),
            "data_manifest": metadata.get("data_manifest_identity", "unavailable"),
            "source_revision": metadata.get("source_revision", "unavailable"),
            "source_python_sha256": metadata.get("source_python_sha256", "unavailable"),
        }
        if manifest.get("identity") != identity or manifest.get(
            "identity_sha256"
        ) != _hash(identity):
            raise ValueError("Checkpoint semantic identity integrity mismatch")
        model = manifest.get("model_state")
        if not isinstance(model, dict) or not model:
            raise ValueError("Missing recorded checkpoint model state")
        for leaf in model.values():
            if (
                not isinstance(leaf, dict)
                or not re.fullmatch("[0-9a-f]{64}", str(leaf.get("sha256", "")))
                or not isinstance(leaf.get("shape"), list)
                or any(type(n) is not int or n < 0 for n in leaf["shape"])
                or not isinstance(leaf.get("dtype"), str)
                or not leaf["dtype"]
            ):
                raise ValueError("Invalid recorded model-state identity")
        for location, payload in payloads.items():
            if cloud:
                changed = _describe(location) != versions[location]
            else:
                path = Path(location)
                with path.open("rb") as handle:
                    changed = (
                        path.is_symlink() or handle.read(MAX_JSON_BYTES + 1) != payload
                    )
            if changed:
                raise ValueError("Checkpoint changed during read-only admission")
        result = {**metadata, "step": step}
        if include_receipt:
            result["committed_receipt"] = {
                "step": step,
                "manifest_sha256": _hash(manifest),
                "metadata_sha256": manifest["metadata_sha256"],
                "model_state": model,
            }
        return result
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as error:
        raise ValueError(
            f"Committed checkpoint metadata admission failed: {error}"
        ) from error
