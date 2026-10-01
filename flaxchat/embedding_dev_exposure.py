"""Recompute declared known-stage exposure from authenticated historical inputs.

Completeness covers the declared inventory, not undisclosed prior training.
No models, backends or external calls.
"""

import hashlib
import json
from pathlib import Path
import re
import time
from flaxchat.embedding_historical_rows import historical_row_view


def parent_exposure(path, index, parent_hashes, *, timeout_seconds=600):
    if not 1 <= timeout_seconds <= 1800:
        raise ValueError("Bounded parent exposure deadline required")
    deadline = time.monotonic() + timeout_seconds

    def bounded_hash(file):
        value = hashlib.sha256()
        with Path(file).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Parent exposure audit budget exhausted")
                value.update(chunk)
        return value.hexdigest()

    def bounded_json(file):
        if Path(file).stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Parent exposure metadata exceeds bounded size")
        return json.loads(Path(file).read_text())

    spec = bounded_json(path)
    expected = spec.get("expected_stage_identities")
    if (
        spec.get("format") != "flaxchat-known-contrastive-exposure-input-v1"
        or spec.get("parent_files_sha256") != parent_hashes
        or spec.get("known_contrastive_inventory_complete") is not True
        or not isinstance(expected, list)
        or not expected
        or len(set(expected)) != len(expected)
        or any(not re.fullmatch("[0-9a-f]{64}", stage) for stage in expected)
    ):
        raise ValueError(
            "Complete immutable parent known-stage exposure inventory required"
        )
    stages = spec.get("stages", [])
    if {stage.get("stage_identity_sha256") for stage in stages} != set(expected) or len(
        stages
    ) != len(expected):
        raise ValueError("Parent exposure stage coverage mismatch")
    proof = {
        "format": "flaxchat-known-contrastive-exposure-proof-v1",
        "input_sha256": bounded_hash(path),
        "parent_files_sha256": parent_hashes,
        "candidate_identity": index.identity,
        "complete": False,
        "checked_stages": {},
        "overlap_rows": 0,
        "unresolved_exposure": [
            "inherited MLM/pretraining exposure",
            "semantic/near-duplicate/translation exposure",
            "completeness of historical contrastive-stage lineage beyond declared inventory",
        ],
    }
    for stage in stages:
        metadata_path = (Path(path).parent / stage["checkpoint_metadata"]).resolve()
        if bounded_hash(metadata_path) != stage["stage_identity_sha256"]:
            raise ValueError("Historical stage metadata identity mismatch")
        metadata = bounded_json(metadata_path)
        declared = metadata.get("resolved_config", {}).get("data_manifests")
        sources = stage.get("sources", [])
        if (
            not declared
            or {entry["name"] for entry in sources} != set(declared)
            or len(sources) != len(declared)
        ):
            raise ValueError("Historical stage source inventory incomplete")
        checked = {
            "metadata_sha256": stage["stage_identity_sha256"],
            "sources": {},
            "overlap_rows": 0,
        }
        for entry in sources:
            directory = (Path(path).parent / entry["directory"]).resolve()
            manifest_path = directory / "manifest.json"
            if bounded_hash(manifest_path) != declared[entry["name"]]:
                raise ValueError(
                    "Historical prepared source differs from committed stage"
                )
            manifest = bounded_json(manifest_path)
            if manifest.get("source") != entry["name"]:
                raise ValueError("Historical prepared source name mismatch")
            identity = manifest.get("source_identity", {})
            if not identity.get("repo") or not re.fullmatch(
                "[0-9a-f]{40}", identity.get("revision", "")
            ):
                raise ValueError("Historical source immutable provenance missing")
            expected_rows = manifest.get("rows", {}).get("train")
            if type(expected_rows) is not int or expected_rows < 1:
                raise ValueError("Historical train source must contain real rows")
            raw = directory / "train.jsonl"
            expected_hash = manifest["raw_files"]["train.jsonl"]
            if bounded_hash(raw) != expected_hash:
                raise ValueError("Historical raw training checksum mismatch")
            count, exposed = 0, 0
            with raw.open() as stream:
                while True:
                    line = stream.readline(4 * 1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 4 * 1024 * 1024:
                        raise ValueError(
                            "Historical parent raw row exceeds bounded size"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Parent exposure audit budget exhausted")
                    count += 1
                    view = historical_row_view(json.loads(line), manifest['source_identity'],
                        entry['name'], manifest, producer_policy=entry.get('historical_row_policy'))
                    if index.matches(view, manifest["source_identity"], entry["name"]):
                        exposed += 1
            if (
                count != expected_rows
                or bounded_hash(raw) != expected_hash
                or bounded_hash(manifest_path) != declared[entry["name"]]
            ):
                raise ValueError("Historical raw training coverage changed/incomplete")
            checked["sources"][entry["name"]] = {
                "manifest_sha256": declared[entry["name"]],
                "raw_sha256": expected_hash,
                "checked_rows": count,
                "overlap_rows": exposed,
                "historical_semantics": {"policy": "immutable-source-read-only-view-v1",
                    "producer_policy": entry.get('historical_row_policy', manifest.get('historical_row_policy'))},
            }
            checked["overlap_rows"] += exposed
        if bounded_hash(metadata_path) != stage["stage_identity_sha256"]:
            raise ValueError("Stage metadata changed during exposure audit")
        proof["checked_stages"][stage["stage_identity_sha256"]] = checked
        proof["overlap_rows"] += checked["overlap_rows"]
    if proof["overlap_rows"]:
        raise ValueError(
            "Candidate probes exposed to known contrastive parent training; select fresh independent data"
        )
    if bounded_hash(path) != proof["input_sha256"]:
        raise ValueError("Parent exposure inventory changed during audit")
    proof["complete"] = True
    return proof
