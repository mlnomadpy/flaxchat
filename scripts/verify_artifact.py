"""Create or verify SHA-256 checksums for a model artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flaxchat.artifact import CHECKSUM_FILE, artifact_checksums


checksums = artifact_checksums


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
    expected = json.loads(path.read_text(encoding="utf-8"))
    if expected != actual:
        parser.exit(1, "artifact checksum mismatch\n")
    print(f"Verified {len(actual)} artifact files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
