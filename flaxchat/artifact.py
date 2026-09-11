"""Validation and path resolution for published flaxchat artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


CHECKSUM_FILE = "SHA256SUMS.json"
MANIFEST_FILE = "run_manifest.json"
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def artifact_checksums(directory: str | Path) -> dict[str, str]:
    """Return canonical SHA-256 identities for every artifact payload file."""
    root = Path(directory)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact contains a symbolic link: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative != CHECKSUM_FILE:
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_artifact(directory: str | Path) -> dict[str, Any]:
    """Verify an artifact completely, then return its validated manifest."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {directory}")
    expected = json.loads((root / CHECKSUM_FILE).read_text(encoding="utf-8"))
    if (
        not isinstance(expected, dict)
        or not all(
            isinstance(path, str)
            and isinstance(digest, str)
            and _SHA256.fullmatch(digest)
            for path, digest in expected.items()
        )
        or expected != artifact_checksums(root)
    ):
        raise ValueError("artifact checksum mismatch")
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    required = {
        "artifacts",
        "format_version",
        "model_config",
        "release_compatibility",
        "resolved_config",
        "source_revision",
        "tokenizer_sha256",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError("artifact manifest is missing required publication metadata")
    if manifest["format_version"] != 1:
        raise ValueError("unsupported artifact manifest format")
    if not isinstance(manifest["source_revision"], str) or not _SHA40.fullmatch(
        manifest["source_revision"]
    ):
        raise ValueError("artifact source revision must be an exact git SHA")
    return manifest


def resolve_artifact_path(directory: str | Path, relative: str) -> Path:
    """Resolve a manifest path while preventing escape from the artifact root."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be a non-empty relative path")
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes its root: {relative}")
    if not path.exists():
        raise ValueError(f"artifact path does not exist: {relative}")
    return path
