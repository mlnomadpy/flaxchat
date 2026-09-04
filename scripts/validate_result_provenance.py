"""Validate the canonical records behind published performance claims."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "benchmarks" / "results" / "provenance-index.json"
_SHA40 = re.compile(r"[0-9a-f]{40}")


def _resolve_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointers must start with '/'")
    value = document
    for encoded in pointer[1:].split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def validate(index_path: Path = DEFAULT_INDEX, root: Path = ROOT) -> list[str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    records: dict[str, Any] = {}
    result_dir = index_path.parent
    revision = index.get("current_source_revision")
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        errors.append("index: current_source_revision must be an exact git SHA")

    for name, identity in index.get("records", {}).items():
        path = result_dir / identity["path"]
        if not path.is_file():
            errors.append(f"{name}: record is missing")
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
            errors.append(f"{name}: record digest diverged")
        record = json.loads(payload)
        records[name] = record
        if record.get("source_revision") != revision:
            errors.append(f"{name}: stale source revision")

    for name, locator in index.get("identity", {}).items():
        try:
            value = _resolve_pointer(records[locator["record"]], locator["pointer"])
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"{name}: identity evidence is missing")
        else:
            if not value:
                errors.append(f"{name}: identity evidence is empty")

    for name, claim in index.get("claims", {}).items():
        try:
            actual = _resolve_pointer(records[claim["record"]], claim["pointer"])
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"{name}: canonical value is missing")
        else:
            if actual != claim["value"]:
                errors.append(f"{name}: claim value diverged from canonical record")

    for publication, link in index.get("publications", {}).items():
        path = root / publication
        if not path.is_file() or link not in path.read_text(encoding="utf-8"):
            errors.append(f"{publication}: missing provenance index link")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("\n".join(errors))
        return 1
    print("published result provenance is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
