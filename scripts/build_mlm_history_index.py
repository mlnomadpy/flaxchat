"""Derive a compact exclusion index from authenticated prepared MLM provenance."""

import argparse
import gzip
import json
from pathlib import Path

from flaxchat.encoder_data import file_hash
from scripts.prepare_yat_mlm_continuation import historical_digest


def build(directories, output):
    receipts, identities = [], set()
    for directory in directories:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        documents = directory / "documents.jsonl"
        expected = manifest["documents_sha256"]
        if file_hash(documents) != expected:
            raise ValueError(f"Historical provenance checksum differs: {documents}")
        count = 0
        with documents.open() as stream:
            for line in stream:
                identities.add(historical_digest(json.loads(line)))
                count += 1
        if count != manifest["documents"] or file_hash(documents) != expected:
            raise ValueError("Historical provenance count or identity changed")
        receipts.append(
            dict(
                path=str(directory),
                manifest_sha256=file_hash(manifest_path),
                documents_sha256=expected,
                documents=count,
            )
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        with gzip.GzipFile(
            filename="", fileobj=destination, mode="wb", mtime=0
        ) as stream:
            for identity in sorted(identities):
                stream.write((json.dumps({"text_sha256": identity}) + "\n").encode())
    receipt = dict(
        format="flaxchat-derived-mlm-history-v1",
        inputs=receipts,
        unique_documents=len(identities),
        sha256=file_hash(output),
        scope="Exact normalized documents in supplied MLM provenance; no semantic exclusion claim",
    )
    output.with_suffix(output.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.prepared, args.output)))
