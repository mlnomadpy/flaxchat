"""Build bounded parent exposure inputs from actual retained metadata/raw files.

No downloads or models. Supplied stages define scope; undisclosed historical
lineage remains unresolved. Hash-match directories instead of handcrafting JSON.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from flaxchat.embedding_data import row_identities
from flaxchat.embedding_historical_rows import (
    historical_row_view,
    MIRACL,
    VERIFIED_LEGACY_PRODUCERS,
)
from scripts.evaluation_contract import write_atomic
from scripts.release_contract import validate_export


def build(
    parent,
    stage_metadata,
    prepared_directories,
    output,
    *,
    timeout_seconds=600,
    max_rows_per_file=5_000_000,
    max_stages=64,
    max_directories=256,
    require_parent_lineage=False,
    producer_policies=None,
):
    parent, output = Path(parent), Path(output)
    if (
        output.exists()
        or not 1 <= timeout_seconds <= 1800
        or max_rows_per_file < 1
        or not 1 <= len(stage_metadata) <= max_stages <= 256
        or not 1 <= len(prepared_directories) <= max_directories <= 1024
    ):
        raise ValueError(
            "Fresh output and bounded stage/directory/time/row budgets required"
        )
    deadline = time.monotonic() + timeout_seconds

    def remaining():
        if time.monotonic() >= deadline:
            raise TimeoutError("Parent inventory build budget exhausted")

    def sha(path):
        value = hashlib.sha256()
        remaining()
        with Path(path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                remaining()
                value.update(chunk)
        return value.hexdigest()

    def read(path):
        remaining()
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Metadata exceeds16MiB bound")
        content = path.read_bytes()
        if sha(path) != hashlib.sha256(content).hexdigest():
            raise ValueError("Metadata changed while reading")
        return json.loads(content), hashlib.sha256(content).hexdigest()

    result = {
        "format": "flaxchat-known-contrastive-exposure-input-v1",
        "known_contrastive_inventory_complete": False,
        "expected_stage_identities": [],
        "stages": [],
        "builder_status": "running",
        "scope": "supplied-known-contrastive-stage-inventory-only",
        "parent_checkpoint_metadata_bound": False,
        "unresolved_exposure": [
            "undisclosed contrastive-stage lineage",
            "inherited MLM/pretraining",
            "semantic/near-duplicate/translation exposure",
        ],
        "limits": {
            "timeout_seconds": timeout_seconds,
            "max_rows_per_file": max_rows_per_file,
            "max_stages": max_stages,
            "max_directories": max_directories,
        },
    }
    write_atomic(output, result)
    try:
        parent_hashes = {
            name: sha(parent / name)
            for name in ("model.safetensors", "config.json", "tokenizer.json")
        }
        result["parent_files_sha256"] = parent_hashes
        parent_metadata_sha = None
        if (parent / "export.json").exists():
            export, _ = read(parent / "export.json")
            validate_export(parent, export)
            parent_metadata_sha = sha(parent / "checkpoint-metadata.json")
            result["parent_checkpoint_metadata_bound"] = True
            result["parent_export_sha256"] = sha(parent / "export.json")
        elif require_parent_lineage:
            raise ValueError("Authenticated parent export/checkpoint lineage required")
        sources = {}
        seen_directories = set()
        for folder in prepared_directories:
            directory = Path(folder).resolve()
            if directory in seen_directories:
                raise ValueError("Duplicate prepared directory")
            seen_directories.add(directory)
            manifest, identity = read(directory / "manifest.json")
            if identity in sources:
                raise ValueError("Ambiguous duplicated prepared manifest identity")
            source = manifest.get("source")
            upstream = manifest.get("source_identity", {})
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(upstream.get("repo"), str)
                or not upstream["repo"]
                or not re.fullmatch("[0-9a-f]{40}", upstream.get("revision", ""))
            ):
                raise ValueError("Prepared source requires immutable upstream identity")
            rows = manifest.get("rows", {}).get("train")
            if type(rows) is not int or not 1 <= rows <= max_rows_per_file:
                raise ValueError(
                    "Historical train rows missing or outside declared row budget"
                )
            expected = manifest.get("raw_files", {}).get("train.jsonl")
            if not isinstance(expected, str) or not re.fullmatch(
                "[0-9a-f]{64}", expected
            ):
                raise ValueError("Historical train raw hash required")
            sources[identity] = {
                "directory": directory,
                "manifest": manifest,
                "name": source,
            }
        policies = {}
        policy_path = None
        if producer_policies:
            policy_path = Path(producer_policies).resolve()
            policies, policy_sha = read(policy_path)
            names = {source["name"] for source in sources.values()}
            if not isinstance(policies, dict) or not set(policies) <= names:
                raise ValueError("Producer policies must map supplied source names")
            for name, policy in policies.items():
                known = (
                    VERIFIED_LEGACY_PRODUCERS.get(policy.get("policy"))
                    if isinstance(policy, dict)
                    else None
                )
                if known is None or policy != {"policy": policy["policy"], **known}:
                    raise ValueError("Unverified historical producer policy")
                for found in sources.values():
                    upstream = found["manifest"]["source_identity"]
                    if (
                        found["name"] == name
                        and (upstream["repo"], upstream["revision"]) != MIRACL
                    ):
                        raise ValueError(
                            "Historical producer policy does not match pinned MIRACL source"
                        )
            result["producer_policies"] = {
                "path": str(policy_path),
                "sha256": policy_sha,
            }
        seen_stages = set()
        used_sources = set()
        for file in stage_metadata:
            metadata_path = Path(file).resolve()
            metadata, identity = read(metadata_path)
            if identity in seen_stages:
                raise ValueError("Duplicate stage metadata identity")
            seen_stages.add(identity)
            declared = metadata.get("resolved_config", {}).get("data_manifests")
            if not isinstance(declared, dict) or not declared:
                raise ValueError(
                    "Stage lacks committed resolved_config.data_manifests; recover actual metadata"
                )
            entries = []
            for name, digest in sorted(declared.items()):
                remaining()
                if (
                    not isinstance(digest, str)
                    or not re.fullmatch("[0-9a-f]{64}", digest)
                    or digest not in sources
                ):
                    raise ValueError(
                        f"Missing authenticated prepared source for stage: {name}"
                    )
                found = sources[digest]
                if found["name"] != name:
                    raise ValueError(
                        "Committed source name differs from matched manifest"
                    )
                directory, manifest = found["directory"], found["manifest"]
                raw = directory / "train.jsonl"
                expected_hash = manifest["raw_files"]["train.jsonl"]
                count = 0
                if sha(raw) != expected_hash:
                    raise ValueError("Historical raw checksum mismatch")
                with raw.open(encoding="utf-8") as stream:
                    while True:
                        remaining()
                        line = stream.readline(4 * 1024 * 1024 + 1)
                        if not line:
                            break
                        if len(line) > 4 * 1024 * 1024:
                            raise ValueError("Historical raw row exceeds4MiB bound")
                        count += 1
                        if count > manifest["rows"]["train"]:
                            raise ValueError("Historical raw has extra rows")
                        row = historical_row_view(
                            json.loads(line),
                            manifest["source_identity"],
                            name,
                            manifest,
                            producer_policy=policies.get(name),
                        )
                        row_identities(row, name)
                if (
                    count != manifest["rows"]["train"]
                    or sha(raw) != expected_hash
                    or sha(directory / "manifest.json") != digest
                ):
                    raise ValueError(
                        "Historical raw/source coverage changed or incomplete"
                    )
                entries.append(
                    {
                        "name": name,
                        "directory": str(directory),
                        "manifest_sha256": digest,
                        "train_raw_sha256": expected_hash,
                        "checked_rows": count,
                        **(
                            {"historical_row_policy": policies[name]}
                            if name in policies
                            else {}
                        ),
                    }
                )
                used_sources.add(digest)
            if sha(metadata_path) != identity:
                raise ValueError("Stage metadata changed during build")
            result["stages"].append(
                {
                    "stage_identity_sha256": identity,
                    "checkpoint_metadata": str(metadata_path),
                    "sources": entries,
                }
            )
        if parent_metadata_sha is not None and parent_metadata_sha not in seen_stages:
            raise ValueError(
                "Current parent checkpoint metadata is missing from declared stage inventory"
            )
        if any(sha(parent / name) != value for name, value in parent_hashes.items()):
            raise ValueError("Parent files changed during inventory build")
        if policy_path and sha(policy_path) != policy_sha:
            raise ValueError("Historical producer policy changed during build")
        result["expected_stage_identities"] = sorted(seen_stages)
        result["unused_prepared_manifests"] = sorted(set(sources) - used_sources)
        result["known_contrastive_inventory_complete"] = True
        result["builder_status"] = "complete"
        result["builder_sha256"] = sha(Path(__file__))
        result["exposure_checked"] = False
        result["remaining_checks"] = [
            "candidate exact/aligned exposure scan",
            "actual independent gate admission",
            "physical model/quality validation",
        ]
        write_atomic(output, result)
        return result
    except Exception as error:
        result.update(
            builder_status="failed",
            known_contrastive_inventory_complete=False,
            error=f"{type(error).__name__}: {error}",
        )
        write_atomic(output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-directory", type=Path, required=True)
    parser.add_argument("--stage-metadata", type=Path, action="append", required=True)
    parser.add_argument(
        "--prepared-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-rows-per-file", type=int, default=5_000_000)
    parser.add_argument("--max-stages", type=int, default=64)
    parser.add_argument("--max-directories", type=int, default=256)
    parser.add_argument("--require-parent-lineage", action="store_true")
    parser.add_argument("--producer-policies", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.parent_directory,
                args.stage_metadata,
                args.prepared_directory,
                args.output,
                timeout_seconds=args.timeout_seconds,
                max_rows_per_file=args.max_rows_per_file,
                max_stages=args.max_stages,
                max_directories=args.max_directories,
                require_parent_lineage=args.require_parent_lineage,
                producer_policies=args.producer_policies,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
