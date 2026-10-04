"""Bounded exact overlap audit of prepared raw data and supplied pinned benchmarks.

No downloads, model execution, quarantine mutation or semantic-cleanliness claim.
Uses the same modality identity as training; SQLite bounds resident index memory.
"""

from __future__ import annotations
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from flaxchat.embedding_data import IDENTITY_POLICY, text_identity
from scripts.evaluation_contract import write_atomic

UNIT_KEYS = ("task", "split", "subset", "role")


def unit_key(unit):
    if any(not isinstance(unit.get(key), str) or not unit[key] for key in UNIT_KEYS):
        raise ValueError("Expected task/split/subset/role identity required")
    if unit["role"] not in ("query", "corpus"):
        raise ValueError("Benchmark role must be query or corpus")
    return tuple(unit[key] for key in UNIT_KEYS)


def audit(
    prepared,
    benchmark_manifest,
    output,
    *,
    max_rows=1_000_000,
    timeout_seconds=600,
    max_samples=100,
):
    if (
        max_rows < 1
        or not 1 <= timeout_seconds <= 1800
        or not 0 <= max_samples <= 10_000
    ):
        raise ValueError("Invalid bounded row/time/sample budget")
    deadline = time.monotonic() + timeout_seconds
    report = {
        "format": "flaxchat-exact-benchmark-overlap-v1",
        "identity_policy": IDENTITY_POLICY,
        "complete": False,
        "state": "running",
        "scope": "supplied_prepared_raw_and_pinned_benchmark_files_only",
        "limits": {
            "max_rows_per_file": max_rows,
            "timeout_seconds": timeout_seconds,
            "max_samples": max_samples,
        },
        "inputs": [],
        "prepared_coverage": [],
        "benchmark_coverage": [],
        "missing_units": [],
        "overlap_counts": {},
        "samples": [],
        "empty_content_policy": "exclude-empty-string/retain-whitespace-v1",
        "ignored_empty_content": {},
        "physical_tpu_qualified": False,
        "unresolved_exposure": [
            "semantic/near-duplicate/translation overlap",
            "parent pretraining and previous-stage exposure",
            "unsupplied benchmark tasks/corpora/query splits",
            "remote repository provenance beyond supplied immutable revision",
        ],
    }
    write_atomic(Path(output), report)
    counts = Counter()

    def remaining():
        if time.monotonic() >= deadline:
            raise TimeoutError("Overlap audit wall-time budget exhausted")

    def authenticated(path, expected=None, *, record=True):
        remaining()
        h = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                remaining()
                h.update(chunk)
        observed = h.hexdigest()
        if record:
            report["inputs"].append({"path": str(path.resolve()), "sha256": observed})
        if expected is not None and (
            not isinstance(expected, str)
            or not re.fullmatch("[0-9a-f]{64}", expected)
            or observed != expected
        ):
            raise ValueError(f"Input hash mismatch: {path}")
        return observed

    def load_manifest(path):
        path = Path(path)
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Manifest exceeds4MiB bound")
        sha = authenticated(path)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Manifest changed during audit")
        return json.loads(content)

    def rows(path, expected_rows, coverage):
        with path.open(encoding="utf-8") as stream:
            index = 0
            while True:
                remaining()
                line = stream.readline(4 * 1024 * 1024 + 1)
                if not line:
                    return
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError("Raw JSONL row exceeds4MiB bound")
                if index >= expected_rows:
                    raise ValueError("Raw file contains more rows than declared")
                if index >= max_rows:
                    coverage["row_limit_reached"] = True
                    return
                yield index, json.loads(line)
                index += 1

    try:
        manifest = load_manifest(benchmark_manifest)
        if manifest.get("format") != "embedding-benchmark-overlap-input-v1":
            raise ValueError("Unsupported benchmark input manifest")
        expected = {}
        for unit in manifest["expected_units"]:
            key = unit_key(unit)
            if key in expected or type(unit.get("rows")) is not int or unit["rows"] < 1:
                raise ValueError(
                    "Unique positive expected unit row inventories required"
                )
            expected[key] = unit["rows"]
        if not expected:
            raise ValueError("Expected benchmark inventory must be nonempty")
        benchmark_rows = Counter()
        declared_rows = Counter()
        seen_paths = set()
        files = manifest["files"]
        for entry in files:
            key = unit_key(entry)
            if (
                key not in expected
                or entry.get("modality") not in ("code", "text")
                or not isinstance(entry.get("text_field"), str)
                or not entry["text_field"]
            ):
                raise ValueError("Invalid benchmark file unit/modality/text field")
            if (
                not isinstance(entry.get("dataset_repo"), str)
                or not entry["dataset_repo"]
                or not re.fullmatch("[0-9a-f]{40}", entry.get("dataset_revision", ""))
            ):
                raise ValueError(
                    "Immutable benchmark dataset revision and repository required"
                )
            if type(entry.get("rows")) is not int or entry["rows"] < 1:
                raise ValueError("Positive file row inventory required")
            path = (Path(benchmark_manifest).parent / entry["path"]).resolve()
            if path in seen_paths:
                raise ValueError("Duplicate benchmark input path")
            seen_paths.add(path)
            declared_rows[key] += entry["rows"]
        if any(value > expected[key] for key, value in declared_rows.items()):
            raise ValueError("Declared benchmark file rows exceed expected unit")
        with (
            tempfile.TemporaryDirectory(prefix="flaxchat-overlap-") as temporary,
            closing(
                sqlite3.connect(str(Path(temporary) / "identities.sqlite"))
            ) as connection,
        ):
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute(
                "CREATE TABLE identity(hash TEXT, source TEXT, split TEXT, field TEXT, rownum INTEGER)"
            )
            connection.execute("CREATE INDEX lookup ON identity(hash)")
            names = set()
            for specification in prepared:
                source, separator, directory = specification.partition("=")
                if not separator or not source or source in names:
                    raise ValueError("Unique prepared SOURCE=DIR required")
                names.add(source)
                root = Path(directory)
                prepared_manifest = load_manifest(root / "manifest.json")
                if (
                    prepared_manifest.get("source") != source
                    or prepared_manifest.get("identity_policy") != IDENTITY_POLICY
                ):
                    raise ValueError(
                        "Prepared source/identity policy differs from current shared identity"
                    )
                identity = prepared_manifest.get("source_identity", {})
                if (
                    not isinstance(identity.get("repo"), str)
                    or not identity["repo"]
                    or not re.fullmatch("[0-9a-f]{40}", identity.get("revision", ""))
                ):
                    raise ValueError(
                        "Prepared source must declare immutable data revision"
                    )
                for split in ("train", "dev"):
                    path = root / f"{split}.jsonl"
                    authenticated(
                        path, prepared_manifest["raw_files"][f"{split}.jsonl"]
                    )
                    total = prepared_manifest["rows"][split]
                    if type(total) is not int or total < 1:
                        raise ValueError("Positive prepared row count required")
                    coverage = {
                        "source": source,
                        "split": split,
                        "dataset_repo": identity["repo"],
                        "dataset_revision": identity["revision"],
                        "expected_rows": total,
                        "scanned_rows": 0,
                        "complete": False,
                    }
                    report["prepared_coverage"].append(coverage)
                    for index, row in rows(path, total, coverage):
                        if not isinstance(row, dict) or not isinstance(
                            row.get("modalities", {}), dict
                        ):
                            raise ValueError(
                                "Prepared row and modalities must be objects"
                            )
                        for field in ("query", "positive", "negative"):
                            value = row.get(field)
                            modality = row.get("modalities", {}).get(
                                field,
                                "code"
                                if source == "code" and field != "query"
                                else "text",
                            )
                            if modality not in ("text", "code"):
                                raise ValueError("Unknown text modality")
                            if value is None:
                                if field != "negative":
                                    raise ValueError(
                                        "Prepared query/positive is absent"
                                    )
                                continue
                            content_identity = text_identity(value, modality)
                            if value:
                                connection.execute(
                                    "INSERT INTO identity VALUES(?,?,?,?,?)",
                                    (content_identity, source, split, field, index),
                                )
                            else:
                                key = "/".join((source, split, field))
                                report["ignored_empty_content"][key] = (
                                    report["ignored_empty_content"].get(key, 0) + 1
                                )
                        coverage["scanned_rows"] += 1
                    coverage["complete"] = coverage["scanned_rows"] == total
                    if (
                        coverage["scanned_rows"] < min(total, max_rows)
                        or coverage["scanned_rows"] > total
                    ):
                        raise ValueError(
                            "Prepared raw row inventory differs from manifest"
                        )
                    authenticated(
                        path,
                        prepared_manifest["raw_files"][f"{split}.jsonl"],
                        record=False,
                    )
                    connection.commit()
            if not names:
                raise ValueError("At least one prepared source required")
            for entry in files:
                key = unit_key(entry)
                path = (Path(benchmark_manifest).parent / entry["path"]).resolve()
                authenticated(path, entry["sha256"])
                coverage = {
                    **dict(zip(UNIT_KEYS, key, strict=True)),
                    "path": str(path),
                    "dataset_repo": entry["dataset_repo"],
                    "dataset_revision": entry["dataset_revision"],
                    "expected_rows": entry["rows"],
                    "scanned_rows": 0,
                    "complete": False,
                }
                report["benchmark_coverage"].append(coverage)
                for index, row in rows(path, entry["rows"], coverage):
                    value = row[entry["text_field"]]
                    if not isinstance(value, str):
                        raise ValueError("Benchmark text/code content must be a string")
                    if not value:
                        empty_key = "/".join(key)
                        report["ignored_empty_content"][empty_key] = (
                            report["ignored_empty_content"].get(empty_key, 0) + 1
                        )
                    matches = (
                        connection.execute(
                            "SELECT source,split,field,COUNT(*) FROM identity WHERE hash=? GROUP BY source,split,field",
                            (text_identity(value, entry["modality"]),),
                        ).fetchall()
                        if value
                        else []
                    )
                    for source, split, field, presentations in matches:
                        counter_key = "/".join((*key, source, split, field))
                        counts[counter_key] += 1
                        if len(report["samples"]) < max_samples:
                            report["samples"].append(
                                {
                                    "benchmark_unit": dict(
                                        zip(UNIT_KEYS, key, strict=True)
                                    ),
                                    "benchmark_row": index,
                                    "content_sha256": text_identity(
                                        value, entry["modality"]
                                    ),
                                    "prepared_source": source,
                                    "prepared_split": split,
                                    "prepared_field": field,
                                    "matching_prepared_presentations": presentations,
                                }
                            )
                    coverage["scanned_rows"] += 1
                coverage["complete"] = coverage["scanned_rows"] == entry["rows"]
                if (
                    coverage["scanned_rows"] < min(entry["rows"], max_rows)
                    or coverage["scanned_rows"] > entry["rows"]
                ):
                    raise ValueError(
                        "Benchmark raw row inventory differs from manifest"
                    )
                authenticated(path, entry["sha256"], record=False)
                benchmark_rows[key] += coverage["scanned_rows"]
        report["missing_units"] = [
            {
                **dict(zip(UNIT_KEYS, key, strict=True)),
                "expected_rows": total,
                "scanned_rows": benchmark_rows[key],
            }
            for key, total in expected.items()
            if benchmark_rows[key] != total
        ]
        report["complete"] = (
            bool(report["prepared_coverage"])
            and all(
                c["complete"]
                for c in report["prepared_coverage"] + report["benchmark_coverage"]
            )
            and not report["missing_units"]
        )
    except TimeoutError as error:
        report["partial_reason"] = str(error)
    except Exception as error:
        report.update(
            state="failed",
            complete=False,
            error=f"{type(error).__name__}: {error}",
            zero_exact_overlap_established=False,
        )
        write_atomic(Path(output), report)
        raise
    report["state"] = "complete" if report["complete"] else "partial"
    report["overlap_counts"] = dict(sorted(counts.items()))
    report["exact_overlap_found"] = bool(counts)
    report["zero_exact_overlap_established"] = report["complete"] and not counts
    report["identity_implementation_sha256"] = hashlib.sha256(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("flaxchat/embedding_data.py")
        .read_bytes()
    ).hexdigest()
    report["audit_implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    write_atomic(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        action="append",
        required=True,
        help="SOURCE=DIR with authenticated train/dev JSONL",
    )
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    report = audit(
        args.prepared,
        args.benchmark_manifest,
        args.output,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        max_samples=args.max_samples,
    )
    return 2 if not report["complete"] else (1 if report["exact_overlap_found"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
