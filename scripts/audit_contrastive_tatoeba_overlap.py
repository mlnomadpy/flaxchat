"""Audit exact normalized-text overlap between contrastive training and sealed Tatoeba.

This checks both sentences of every training pair and every pinned Tatoeba
test pair. It never loads a model or uses benchmark labels for model selection.
"""

import argparse
import json
from pathlib import Path

from scripts.prepare_encoder_contrastive import file_hash, inputs, rows, text_hash


def audit(prepared, sealed_manifest):
    prepared = Path(prepared)
    sealed_manifest = Path(sealed_manifest)
    manifest_path = prepared / "manifest.json"
    manifest_hash = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sealed_manifest_sha256") != file_hash(sealed_manifest):
        raise ValueError("Prepared training does not pin this sealed inventory")
    train_path = prepared / "train.jsonl"
    train_record = manifest["files"]["train.jsonl"]
    if file_hash(train_path) != train_record["sha256"]:
        raise ValueError("Prepared training file changed")
    train_hashes = set()
    train_pairs = 0
    for _, row in rows(train_path):
        if row.get("split") != "train":
            raise ValueError("Nontraining row in prepared training file")
        hashes = [text_hash(row[name]) for name in ("sentence1", "sentence2")]
        if hashes != row.get("text_hashes"):
            raise ValueError("Persisted text hash differs from normalized training text")
        train_hashes.update(hashes)
        train_pairs += 1
    if train_pairs != train_record["rows"]:
        raise ValueError("Prepared training pair count changed")
    sealed_files = inputs(sealed_manifest, manifest["sealed_manifest_sha256"], "test")
    if len(sealed_files) != 112:
        raise ValueError("Require all 112 pinned Tatoeba shards")
    sealed_hashes = set()
    sealed_pairs = 0
    overlaps = set()
    per_subset = {}
    for source, record in sealed_files:
        pair_count = 0
        subset_overlaps = set()
        for _, row in rows(source):
            hashes = {text_hash(row[name]) for name in ("sentence1", "sentence2")}
            sealed_hashes.update(hashes)
            subset_overlaps.update(hashes & train_hashes)
            pair_count += 1
        per_subset[source.name.removesuffix(".jsonl.gz")] = dict(
            pairs=pair_count, overlapping_unique_texts=len(subset_overlaps),
            source_sha256=record["sha256"],
        )
        sealed_pairs += pair_count
        overlaps.update(subset_overlaps)
    if file_hash(train_path) != train_record["sha256"] or file_hash(manifest_path) != manifest_hash:
        raise ValueError("Prepared training inputs changed during audit")
    return dict(
        prepared_manifest_sha256=manifest_hash,
        train_jsonl_sha256=train_record["sha256"],
        sealed_manifest_sha256=manifest["sealed_manifest_sha256"],
        train_pairs=train_pairs, train_unique_texts=len(train_hashes),
        sealed_shards=len(sealed_files), sealed_pairs=sealed_pairs,
        sealed_unique_texts=len(sealed_hashes),
        overlapping_unique_texts=len(overlaps), passed=not overlaps,
        per_subset=per_subset,
        limitations=[
            "Only exact NFKC-casefold-whitespace normalized text overlap is checked.",
            "Paraphrases, partial text, translated semantic duplicates and imported-base pretraining exposure remain unaudited.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an overlap audit")
    result = audit(args.prepared, args.sealed_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_subset"}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
