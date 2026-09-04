"""Create or verify SHA-256 checksums for a model artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHECKSUM_FILE = "SHA256SUMS.json"


def checksums(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = str(path.relative_to(directory))
        if relative != CHECKSUM_FILE:
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    path = args.directory / CHECKSUM_FILE
    actual = checksums(args.directory)
    if args.write:
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        return 0
    expected = json.loads(path.read_text())
    if expected != actual:
        parser.exit(1, "artifact checksum mismatch\n")
    print(f"Verified {len(actual)} artifact files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
