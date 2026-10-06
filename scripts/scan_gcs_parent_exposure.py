"""Generation-pinned, bounded historical raw scan; run on a cloud host near GCS.

A receipt never substitutes for raw-file authentication in development admission.
Optional materialization feeds the existing local inventory builder unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

from flaxchat.embedding_data import IDENTITY_POLICY, row_identities
from flaxchat.embedding_historical_rows import historical_row_view, historical_stage_sources, historical_manifest_view
from flaxchat.embedding_development_quarantine import load_candidate_exclusions
from scripts.evaluation_contract import write_atomic


MAX_METADATA = 16 * 1024 * 1024
MAX_LINE = 4 * 1024 * 1024


def scan(
    stage_metadata,
    prepared_directories,
    object_receipt,
    output,
    *,
    materialize_directory=None,
    development_exclusions=None,
    producer_policies=None,
    overlap_details_output=None,
    max_overlap_text_identities=10000,
    timeout_seconds=1800,
    max_total_bytes=4 * 1024**3,
    max_rows_per_file=5_000_000,
    _command_prefix=None,
):
    output = Path(output)
    if output.exists() or not 1 <= timeout_seconds <= 3600:
        raise ValueError("Fresh output and deadline of1..3600 seconds required")
    if not 1 <= len(stage_metadata) <= 64 or not 1 <= len(prepared_directories) <= 256:
        raise ValueError("Bounded nonempty stage/source inventory required")
    if (
        not 1 <= max_total_bytes <= 64 * 1024**3
        or not 1 <= max_rows_per_file <= 10_000_000
    ):
        raise ValueError("Finite byte/row budgets required")
    details_path = Path(overlap_details_output) if overlap_details_output else None
    if (type(max_overlap_text_identities) is not int or
            not 1 <= max_overlap_text_identities <= 100000):
        raise ValueError("Finite overlap text identity cap required")
    if details_path and (not development_exclusions or details_path.exists() or
                         details_path.is_symlink() or details_path.resolve() == output.resolve()):
        raise ValueError("Fresh separate overlap details output and candidate required")
    deadline = time.monotonic() + timeout_seconds
    result = {
        "format": "flaxchat-gcs-parent-raw-scan-v1",
        "status": "running",
        "coverage_complete": False,
        "identity_policy": IDENTITY_POLICY,
        "sources": [],
        "scope": "supplied-known-contrastive-stage-raw-inputs-only",
        "admission_ready": False,
        "unresolved_exposure": [
            "undisclosed contrastive lineage",
            "inherited MLM/pretraining",
            "semantic/near-duplicate/translation exposure",
        ],
        "limits": {
            "timeout_seconds": timeout_seconds,
            "max_total_bytes": max_total_bytes,
            "max_rows_per_file": max_rows_per_file,
        },
    }
    write_atomic(output, result)
    matched_texts = set()
    details_truncated = False

    def write_details(complete):
        if details_path is None:
            return
        details = {
            "format": "flaxchat-gcs-candidate-overlap-details-v1",
            "status": result["status"],
            "scope": result["scope"],
            "identity_policy": IDENTITY_POLICY,
            "candidate_identity": result.get("candidate_identity"),
            "coverage_complete": complete,
            "exact_text_identity_details_complete": complete and not details_truncated,
            "details_truncated": details_truncated,
            "max_overlap_text_identities": max_overlap_text_identities,
            "matched_text_sha256": sorted(matched_texts),
            "matched_identity_kind": "exact-text-only",
            "exact_aligned_overlap_rows": result.get("exact_aligned_overlap_rows"),
            "sources": result["sources"],
            "stage_metadata_sha256": result.get("stage_metadata_sha256"),
            "object_receipt_sha256": result.get("object_receipt_sha256"),
            "producer_policies_sha256": result.get("producer_policies_sha256"),
            "admission_ready": False,
            "remaining_checks": result.get("remaining_checks", result["unresolved_exposure"]),
        }
        write_atomic(details_path, details)
        result["overlap_details"] = {
            "path": str(details_path),
            "sha256": hashlib.sha256(details_path.read_bytes()).hexdigest(),
            "exact_text_identity_details_complete": details["exact_text_identity_details_complete"],
            "details_truncated": details_truncated,
        }

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("GCS historical scan deadline exhausted")
        return left

    stable = {}

    def read(path):
        remaining()
        path = Path(path).resolve()
        with path.open("rb") as stream:
            raw = stream.read(MAX_METADATA + 1)
        if len(raw) > MAX_METADATA:
            raise ValueError("Metadata exceeds16MiB bound")
        digest = hashlib.sha256(raw).hexdigest()
        if path in stable:
            raise ValueError("Duplicate input metadata path")
        stable[path] = digest
        return json.loads(raw), digest, raw

    try:
        objects, object_hash, _ = read(object_receipt)
        records = objects.get("objects")
        if not isinstance(records, list) or not 1 <= len(records) <= 256:
            raise ValueError("Bounded object inventory required")
        object_map = {}
        total_expected = 0
        for item in records:
            uri, size = item.get("uri"), item.get("bytes")
            if not isinstance(uri, str) or not re.fullmatch(
                r"gs://[a-z0-9][a-z0-9._-]*/[^#\s]+/train\.jsonl#[1-9][0-9]*", uri
            ):
                raise ValueError("Explicit immutable GCS train generation required")
            name = item.get("source", uri.split("/")[-2])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in object_map:
                raise ValueError("Ambiguous/unsafe source object inventory")
            if type(size) is not int or size < 1:
                raise ValueError("Positive object byte inventory required")
            total_expected += size
            object_map[name] = item
        if total_expected > max_total_bytes:
            raise ValueError("Object inventory exceeds total byte budget")
        sources = {}
        for directory in prepared_directories:
            manifest, digest, raw = read(Path(directory) / "manifest.json")
            manifest = historical_manifest_view(manifest, digest)
            name = manifest.get("source")
            upstream = manifest.get("source_identity", {})
            rows = manifest.get("rows", {}).get("train")
            expected = manifest.get("raw_files", {}).get("train.jsonl")
            if (
                name not in object_map
                or name in sources
                or not isinstance(upstream.get("repo"), str)
                or not upstream["repo"]
                or not re.fullmatch(r"[0-9a-f]{40}", upstream.get("revision", ""))
            ):
                raise ValueError("Unique pinned upstream/source mapping required")
            if (
                type(rows) is not int
                or not 1 <= rows <= max_rows_per_file
                or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)
            ):
                raise ValueError(
                    "Committed raw hash and bounded row inventory required"
                )
            sources[name] = (manifest, digest, raw)
        committed = {}
        stage_hashes = []
        for path in stage_metadata:
            metadata, digest, _ = read(path)
            stage_hashes.append(digest)
            declared = historical_stage_sources(metadata)
            if not isinstance(declared, dict) or not declared:
                raise ValueError("Actual committed stage data_manifests required")
            for name, identity in declared.items():
                if name not in sources or sources[name][1] != identity:
                    raise ValueError("Stage/manifest identity mismatch")
                if name in committed and committed[name] != identity:
                    raise ValueError(
                        "Distinct source versions require separate campaigns"
                    )
                committed[name] = identity
        if set(committed) != set(sources) or set(sources) != set(object_map):
            raise ValueError("Exact stage/source/object coverage required")
        policies = {}
        if producer_policies:
            policies, policy_hash, _ = read(producer_policies)
            if (
                not isinstance(policies, dict)
                or not set(policies) <= set(sources)
                or any(not isinstance(value, dict) for value in policies.values())
            ):
                raise ValueError(
                    "Producer policies must map authenticated source names"
                )
            result["producer_policies_sha256"] = policy_hash
        result.update(
            stage_metadata_sha256=sorted(stage_hashes),
            object_receipt_sha256=object_hash,
        )
        index = (
            load_candidate_exclusions(development_exclusions)
            if development_exclusions
            else None
        )
        result["candidate_identity"] = index.identity if index else None
        destination = (
            Path(materialize_directory).resolve() if materialize_directory else None
        )
        if destination:
            destination.mkdir(parents=True, exist_ok=False)
            result["materialize_directory"] = str(destination)
        prefix = (
            list(_command_prefix) if _command_prefix else ["gcloud", "storage", "cat"]
        )
        for name in sorted(sources):
            remaining()
            manifest, digest, raw_manifest = sources[name]
            item = object_map[name]
            folder = destination / name if destination else None
            if folder:
                folder.mkdir()
                (folder / "manifest.json").write_bytes(raw_manifest)
            partial = folder / "train.jsonl.partial" if folder else None
            handle = partial.open("wb") if partial else None
            checked = byte_count = overlap = 0
            overlap_kinds = {}
            value = hashlib.sha256()
            pending = bytearray()
            process = None
            try:
                with selectors.DefaultSelector() as selector:
                    process = subprocess.Popen(
                        prefix + [item["uri"]],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    selector.register(process.stdout, selectors.EVENT_READ)

                    def consume(
                        line, manifest=manifest, name=name, overlap_kinds=overlap_kinds
                    ):
                        nonlocal checked, overlap, details_truncated
                        checked += 1
                        if checked > manifest["rows"]["train"]:
                            raise ValueError("Raw object contains excess rows")
                        row = json.loads(line)
                        try:
                            row = historical_row_view(
                                row,
                                manifest["source_identity"],
                                name,
                                manifest,
                                producer_policy=policies.get(name),
                            )
                            identities = row_identities(row, name)
                        except (KeyError, AttributeError, TypeError) as error:
                            raise ValueError("Malformed historical row") from error
                        if index:
                            if details_path:
                                for identity in sorted(set(identities.values()) & index.text_hashes):
                                    if identity not in matched_texts:
                                        if len(matched_texts) < max_overlap_text_identities:
                                            matched_texts.add(identity)
                                        else:
                                            details_truncated = True
                            match = index.matches(
                                row, manifest["source_identity"], name
                            )
                            if match:
                                overlap += 1
                                overlap_kinds[match] = overlap_kinds.get(match, 0) + 1

                    while True:
                        events = selector.select(min(remaining(), 1))
                        if not events:
                            continue
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        byte_count += len(chunk)
                        if byte_count > item["bytes"]:
                            raise ValueError("Raw object exceeds pinned byte inventory")
                        value.update(chunk)
                        if handle:
                            handle.write(chunk)
                        pending.extend(chunk)
                        while True:
                            newline = pending.find(b"\n")
                            if newline < 0:
                                break
                            if newline > MAX_LINE:
                                raise ValueError("Raw row exceeds4MiB bound")
                            consume(bytes(pending[:newline]))
                            del pending[: newline + 1]
                        if len(pending) > MAX_LINE:
                            raise ValueError("Raw row exceeds4MiB bound")
                    if pending:
                        consume(bytes(pending))
                    code = process.wait(timeout=remaining())
                    if code != 0:
                        raise RuntimeError(f"GCS stream failed with exit status {code}")
                if (
                    byte_count != item["bytes"]
                    or checked != manifest["rows"]["train"]
                    or value.hexdigest() != manifest["raw_files"]["train.jsonl"]
                ):
                    raise ValueError("Incomplete or unauthenticated raw object")
                if handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    handle = None
                    partial.rename(folder / "train.jsonl")
                result["sources"].append(
                    {
                        "name": name,
                        "uri": item["uri"],
                        "manifest_sha256": digest,
                        "train_raw_sha256": value.hexdigest(),
                        "checked_rows": checked,
                        "bytes": byte_count,
                        "overlap_rows": overlap if index else None,
                        "overlap_kinds": overlap_kinds,
                        "directory": str(folder) if folder else None,
                    }
                )
                write_atomic(output, result)
            finally:
                if process:
                    # Kill the entire stream process group, including inherited descendants.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        # Restricted host sandboxes may forbid group signals.
                        if process.poll() is None:
                            process.kill()
                    process.wait(timeout=5)
                    process.stdout.close()
                if handle:
                    handle.close()
                if partial and partial.exists():
                    partial.unlink()
        for path, expected in stable.items():
            remaining()
            with path.open("rb") as stream:
                raw = stream.read(MAX_METADATA + 1)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Metadata changed during scan")
        if (
            index
            and load_candidate_exclusions(development_exclusions).identity
            != index.identity
        ):
            raise ValueError("Candidate identity changed during scan")
        result.update(
            status="complete",
            coverage_complete=True,
            candidate_exposure_checked=index is not None,
            exact_aligned_overlap_rows=sum(s["overlap_rows"] for s in result["sources"])
            if index
            else None,
        )
        result["remaining_checks"] = [
            "authenticated parent export and full known-stage inventory builder",
            "independent development admission recomputation",
            "physical model/quality validation",
        ]
        write_details(True)
        write_atomic(output, result)
        return result
    except Exception as error:
        result.update(
            status="failed",
            coverage_complete=False,
            error=f"{type(error).__name__}: {error}",
        )
        write_details(False)
        write_atomic(output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-metadata", action="append", required=True)
    parser.add_argument("--prepared-directory", action="append", required=True)
    parser.add_argument("--object-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--materialize-directory")
    parser.add_argument("--development-exclusions")
    parser.add_argument("--producer-policies")
    parser.add_argument("--overlap-details-output")
    parser.add_argument("--max-overlap-text-identities", type=int, default=10000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--max-total-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--max-rows-per-file", type=int, default=5_000_000)
    args = vars(parser.parse_args())
    stage_metadata = args.pop("stage_metadata")
    directories = args.pop("prepared_directory")
    receipt = scan(stage_metadata, directories, **args)
    # No text rows, credentials or provider stderr in user-visible output.
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "coverage_complete": receipt["coverage_complete"],
                "overlap_rows": receipt["exact_aligned_overlap_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
