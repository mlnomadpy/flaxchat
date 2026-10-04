"""Exact overlap tools on raw fixtures only; no model or dataset downloads."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.audit_embedding_overlap import audit
from flaxchat.embedding_data import IDENTITY_POLICY


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    prepared = root / "prepared"
    prepared.mkdir()
    train = [
        {
            "query": "Find Foo",
            "positive": 'def Foo():\n    return "X"',
            "negative": None,
        },
        {
            "query": "Find other",
            "positive": "def bar():\n    return 2",
            "negative": "def negative(): pass",
        },
    ]
    dev = [{"query": "Held out", "positive": "def held(): pass", "negative": None}]
    raw = {
        f"{split}.jsonl": write_jsonl(prepared / f"{split}.jsonl", rows)
        for split, rows in (("train", train), ("dev", dev))
    }
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "source": "code",
                "identity_policy": IDENTITY_POLICY,
                "source_identity": {"repo": "example/code", "revision": "a" * 40},
                "raw_files": raw,
                "rows": {"train": 2, "dev": 1},
            }
        )
    )
    units = [
        {
            "task": "CodeRetrieval",
            "split": "test",
            "subset": "python",
            "role": "corpus",
            "rows": 3,
        },
        {
            "task": "CodeRetrieval",
            "split": "test",
            "subset": "python",
            "role": "query",
            "rows": 1,
        },
    ]
    files = []
    for unit, rows in (
        (
            units[0],
            [
                {"content": train[0]["positive"].replace("\n", "\r\n")},
                {"content": 'def foo():\n    return "x"'},
                {"content": dev[0]["positive"]},
            ],
        ),
        (units[1], [{"content": train[0]["query"]}]),
    ):
        name = unit["role"] + ".jsonl"
        sha = write_jsonl(root / name, rows)
        files.append(
            {
                **unit,
                "path": name,
                "sha256": sha,
                "dataset_repo": "example/benchmark",
                "dataset_revision": "b" * 40,
                "text_field": "content",
                "modality": "code" if unit["role"] == "corpus" else "text",
            }
        )
    manifest = {
        "format": "embedding-benchmark-overlap-input-v1",
        "expected_units": units,
        "files": files,
    }
    path = root / "benchmark.json"
    path.write_text(json.dumps(manifest))
    return prepared, path, manifest


class OverlapTests(unittest.TestCase):
    def test_shared_exact_code_identity_train_dev_and_query_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, manifest, _ = fixture(root)
            result = audit([f"code={prepared}"], manifest, root / "report.json")
            self.assertTrue(result["complete"])
            self.assertTrue(result["exact_overlap_found"])
            self.assertFalse(result["zero_exact_overlap_established"])
            self.assertEqual(sum(result["overlap_counts"].values()), 3)
            self.assertEqual(len(result["inputs"]), 6)
            self.assertEqual(
                {row["prepared_split"] for row in result["samples"]}, {"train", "dev"}
            )
            self.assertEqual(result["identity_policy"], IDENTITY_POLICY)
            self.assertIn(
                "parent pretraining and previous-stage exposure",
                result["unresolved_exposure"],
            )

    def test_partial_row_budget_and_missing_units_never_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, path, manifest = fixture(root)
            result = audit([f"code={prepared}"], path, root / "report.json", max_rows=1)
            self.assertFalse(result["complete"])
            self.assertFalse(result["zero_exact_overlap_established"])
            self.assertTrue(result["missing_units"])
            manifest["files"].pop()
            path.write_text(json.dumps(manifest))
            result = audit([f"code={prepared}"], path, root / "report.json")
            self.assertFalse(result["complete"])
            self.assertEqual(result["missing_units"][0]["role"], "query")

    def test_tamper_mutable_revision_and_hidden_extra_row_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, path, manifest = fixture(root)
            (root / "corpus.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                audit([f"code={prepared}"], path, root / "report.json")
            self.assertEqual(
                json.loads((root / "report.json").read_text())["state"], "failed"
            )
            manifest["files"][0]["dataset_revision"] = "main"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit([f"code={prepared}"], path, root / "report.json")
            manifest["files"][0]["dataset_revision"] = "b" * 40
            manifest["files"][0]["sha256"] = write_jsonl(
                root / "corpus.jsonl", [{"content": "new"}] * 4
            )
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit([f"code={prepared}"], path, root / "report.json", max_rows=3)

    def test_malformed_prepared_values_fail_and_close_database(self):
        import sqlite3

        real_connect = sqlite3.connect
        for bad in (
            {"query": []},
            {"positive": {}},
            {"negative": 0},
            {"positive": False},
            {"query": "", "modalities": {"query": "image"}},
            {"modalities": []},
            {"negative": None, "modalities": {"negative": "invalid"}},
        ):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepared, path, _ = fixture(root)
                rows = [
                    json.loads(line)
                    for line in (prepared / "train.jsonl").read_text().splitlines()
                ]
                rows[0].update(bad)
                manifest_path = prepared / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["raw_files"]["train.jsonl"] = write_jsonl(
                    prepared / "train.jsonl", rows
                )
                manifest_path.write_text(json.dumps(manifest))
                connections = []

                def connect(*args, connections=connections, **kwargs):
                    connection = real_connect(*args, **kwargs)
                    connections.append(connection)
                    return connection

                with patch(
                    "scripts.audit_embedding_overlap.sqlite3.connect",
                    side_effect=connect,
                ):
                    with self.assertRaises(ValueError):
                        audit([f"code={prepared}"], path, root / "report.json")
                report = json.loads((root / "report.json").read_text())
                self.assertEqual(report["state"], "failed")
                self.assertFalse(report["complete"])
                self.assertFalse(report["zero_exact_overlap_established"])
                self.assertEqual(len(connections), 1)
                with self.assertRaises(sqlite3.ProgrammingError):
                    connections[0].execute("SELECT1")

    def test_timeout_after_database_open_closes_it_and_retains_partial_receipt(self):
        import sqlite3

        real_connect = sqlite3.connect
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, path, _ = fixture(root)
            connections = []

            def connect(*args, **kwargs):
                connection = real_connect(*args, **kwargs)
                connections.append(connection)
                return connection

            with (
                patch(
                    "scripts.audit_embedding_overlap.sqlite3.connect",
                    side_effect=connect,
                ),
                patch(
                    "scripts.audit_embedding_overlap.time.monotonic",
                    side_effect=lambda: 2 if connections else 0,
                ),
            ):
                report = audit(
                    [f"code={prepared}"], path, root / "report.json", timeout_seconds=1
                )
            self.assertEqual(report["state"], "partial")
            self.assertFalse(report["complete"])
            self.assertFalse(report["zero_exact_overlap_established"])
            self.assertEqual(len(connections), 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                connections[0].execute("SELECT1")

    def test_complete_zero_exact_overlap_does_not_claim_semantic_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, path, manifest = fixture(root)
            for entry in manifest["files"]:
                entry["sha256"] = write_jsonl(
                    root / entry["path"],
                    [{"content": f"unseen{index}"} for index in range(entry["rows"])],
                )
            path.write_text(json.dumps(manifest))
            result = audit(
                [f"code={prepared}"], path, root / "report.json", max_samples=0
            )
            self.assertTrue(result["zero_exact_overlap_established"])
            self.assertTrue(result["unresolved_exposure"])
            self.assertFalse(result["physical_tpu_qualified"])

    def test_expired_budget_persists_explicit_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, path, _ = fixture(root)
            with patch(
                "scripts.audit_embedding_overlap.time.monotonic", side_effect=[0, 2]
            ):
                result = audit(
                    [f"code={prepared}"], path, root / "report.json", timeout_seconds=1
                )
            self.assertFalse(result["complete"])
            self.assertIn("budget exhausted", result["partial_reason"])
            self.assertFalse(result["zero_exact_overlap_established"])
