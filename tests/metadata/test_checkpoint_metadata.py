"""Committed admission reads: no numerical imports, tensor loads or cleanup."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from flaxchat.checkpoint_metadata import read_committed_metadata


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixture():
    metadata = {
        "step": 4,
        "resolved_config": {"encoder": {"epsilon": 0.01}},
        "tokenizer_identity": "a" * 64,
        "source_python_sha256": "b" * 64,
    }
    identity = {
        "resolved_config": metadata["resolved_config"],
        "tokenizer": metadata["tokenizer_identity"],
        "data_manifest": "unavailable",
        "source_revision": "unavailable",
        "source_python_sha256": metadata["source_python_sha256"],
    }
    manifest = {
        "format_version": 3,
        "step": 4,
        "metadata_sha256": digest(metadata),
        "identity": identity,
        "identity_sha256": digest(identity),
        "model_state": {
            "kernel": {"shape": [2, 2], "dtype": "bfloat16", "sha256": "c" * 64}
        },
    }
    return {
        "_CHECKPOINT_METADATA": json.dumps({"commit_timestamp_nsecs": 123}).encode(),
        "commit_success.txt": b"",
        "metadata/commit_success.txt": b"",
        "manifest/commit_success.txt": b"",
        "metadata/metadata": json.dumps(metadata).encode(),
        "manifest/metadata": json.dumps(manifest).encode(),
    }


def put(root, files):
    for name, content in files.items():
        path = root / "4" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes())
        for p in root.rglob("*")
        if p.is_file()
    }


class CommittedMetadataTests(unittest.TestCase):
    def test_optional_checkpoint_preflight_uses_the_nonmutating_reader(self):
        from scripts import preflight_yat_embedding_stage as preflight

        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "parent"
            put(root, fixture())
            before = snapshot(root)
            receipt = Path(name) / "admission.json"
            args = SimpleNamespace(
                parent_checkpoint=str(root),
                parent_step=4,
                expected_devices=8,
                expected_processes=1,
                resume=False,
                receipt=receipt,
            )
            admitted = {
                "parent_hashes": {"tokenizer.json": "a" * 64},
                "encoder_config": {"epsilon": 0.01},
                "mixture_receipt": {},
                "admission_receipt": {},
                "data_items": [],
            }
            with (
                patch.object(
                    preflight.argparse.ArgumentParser, "parse_args", return_value=args
                ),
                patch.object(preflight, "prepare_stage", return_value=admitted),
            ):
                preflight.main()
            result = json.loads(receipt.read_text())
            self.assertTrue(result["passed"])
            self.assertFalse(result["physical_tpu_qualified"])
            self.assertEqual(result["parent"]["committed_artifact"]["step"], 4)
            self.assertEqual(before, snapshot(root))

    def test_valid_read_preserves_writer_and_receipt_without_numerical_imports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            put(root, fixture())
            pending = root / "5.orbax-checkpoint-tmp-0"
            pending.mkdir()
            (pending / "writer").write_bytes(b"active")
            before = snapshot(root)
            result = read_committed_metadata(root, 4, include_receipt=True)
            self.assertEqual(result["step"], 4)
            self.assertEqual(
                result["committed_receipt"]["metadata_sha256"],
                digest(json.loads(fixture()["metadata/metadata"])),
            )
            self.assertEqual(before, snapshot(root))
            script = 'import sys; from flaxchat.checkpoint_metadata import read_committed_metadata; read_committed_metadata(sys.argv[1],4); assert not any(n in sys.modules for n in ("jax","flax","orbax"))'
            subprocess.run(
                [sys.executable, "-S", "-c", script, str(root)], check=True, timeout=15
            )

    def test_absent_namespace_and_missing_commit_never_create_or_clean(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "absent"
            with self.assertRaises((ValueError, FileNotFoundError)):
                read_committed_metadata(root, 4)
            self.assertFalse(root.exists())
            for missing in fixture():
                with (
                    self.subTest(missing=missing),
                    tempfile.TemporaryDirectory() as trial,
                ):
                    root = Path(trial)
                    files = fixture()
                    files.pop(missing)
                    put(root, files)
                    before = snapshot(root)
                    with self.assertRaises(ValueError):
                        read_committed_metadata(root, 4)
                    self.assertEqual(before, snapshot(root))

    def test_corrupt_uncommitted_and_semantically_inconsistent_metadata_fail(self):
        changes = [
            ("_CHECKPOINT_METADATA", b'{"commit_timestamp_nsecs":null}'),
            ("metadata/metadata", b'{"step":4,"step":4}'),
            ("metadata/metadata", b'{"step":NaN}'),
            ("metadata/metadata", b"{}"),
        ]
        for field in (
            "step",
            "identity_sha256",
            "identity",
            "model_state",
            "format_version",
        ):
            manifest = json.loads(fixture()["manifest/metadata"])
            manifest[field] = None
            changes.append(("manifest/metadata", json.dumps(manifest).encode()))
        for filename, content in changes:
            with (
                self.subTest(filename=filename, content=content),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                files = fixture()
                files[filename] = content
                put(root, files)
                with self.assertRaises(ValueError):
                    read_committed_metadata(root, 4)

    def test_cloud_reads_exact_generations_then_rejects_changed_namespace(self):
        files = fixture()
        prefix = "gs://bucket/parents/4/"
        describes = {}

        def command(argv, **kwargs):
            self.assertEqual(argv[:2], ["gcloud", "storage"])
            self.assertEqual(kwargs["timeout"], 60)
            if argv[2:4] == ["objects", "describe"]:
                uri = argv[4]
                relative = uri.removeprefix(prefix)
                self.assertIn(relative, files)
                describes[uri] = describes.get(uri, 0) + 1
                generation = (
                    "8"
                    if changing
                    and relative == "metadata/metadata"
                    and describes[uri] > 1
                    else "7"
                )
                return json.dumps(
                    {"generation": generation, "size": len(files[relative])}
                ).encode()
            self.assertEqual(argv[2], "cat")
            self.assertTrue(argv[3].endswith("#7"))
            return files[argv[3].removeprefix(prefix).removesuffix("#7")]

        changing = False
        with patch(
            "flaxchat.checkpoint_metadata.subprocess.check_output", side_effect=command
        ):
            self.assertEqual(
                read_committed_metadata("gs://bucket/parents", 4)["step"], 4
            )
        changing = True
        describes.clear()
        with patch(
            "flaxchat.checkpoint_metadata.subprocess.check_output", side_effect=command
        ):
            with self.assertRaisesRegex(ValueError, "changed during"):
                read_committed_metadata("gs://bucket/parents", 4)

    def test_reader_bound_and_explicit_step(self):
        for step in (None, True, -1, 1.0):
            with self.assertRaises(ValueError):
                read_committed_metadata("/missing", step)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            put(root, fixture())
            with patch("flaxchat.checkpoint_metadata.MAX_JSON_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "oversized"):
                    read_committed_metadata(root, 4)


if __name__ == "__main__":
    unittest.main()
