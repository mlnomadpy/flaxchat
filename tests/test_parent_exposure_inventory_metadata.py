"""Retained-stage inventory discovery and proof consumption; no models/network."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from flaxchat.encoder_data import file_hash
from flaxchat.embedding_dev_exposure import parent_exposure
from flaxchat.embedding_development_quarantine import load_candidate_exclusions
from scripts.build_parent_exposure_inventory import build
from scripts.release_contract import (
    artifact_hashes,
    canonical_hash,
    checkpoint_identity,
)
from tests.test_independent_retrieval_dev_metadata import fixture


class InventoryBuilderTests(unittest.TestCase):
    def test_hash_matching_builder_feeds_actual_exposure_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, independent, _, hashes, _ = fixture(root)
            output = root / "built.json"
            receipt = build(
                args.parent_public,
                [root / "historical-checkpoint-metadata.json"],
                [root / "historical-data"],
                output,
            )
            self.assertTrue(receipt["known_contrastive_inventory_complete"])
            self.assertFalse(receipt["parent_checkpoint_metadata_bound"])
            self.assertFalse(receipt["exposure_checked"])
            self.assertEqual(receipt["stages"][0]["sources"][0]["checked_rows"], 8)
            self.assertTrue(
                Path(receipt["stages"][0]["sources"][0]["directory"]).is_absolute()
            )
            proof = parent_exposure(
                output, load_candidate_exclusions(independent / "candidate"), hashes
            )
            self.assertTrue(proof["complete"])
            self.assertEqual(proof["overlap_rows"], 0)
            with self.assertRaises(ValueError):
                build(
                    args.parent_public,
                    [root / "historical-checkpoint-metadata.json"],
                    [root / "historical-data"],
                    output,
                )

    def test_legacy_miracl_stream_builder_and_admission_preserve_original_bytes(self):
        from tests import test_gcs_parent_exposure_metadata as stream_fixture
        from scripts.scan_gcs_parent_exposure import scan
        from flaxchat.embedding_historical_rows import MIRACL, VERIFIED_LEGACY_PRODUCERS
        import sys

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, independent, hashes, objects, cli = (
                stream_fixture.GCSExposureScanTests().setup_case(root)
            )
            raw = root / "historical-data/train.jsonl"
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            for i, row in enumerate(rows):
                row.update(group=f"miracl:en:{100 + i}", coordinate=f"en:{i}")
                row.pop("upstream_group", None)
                row.pop("language", None)
            raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            manifest_path = root / "historical-data/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_identity"] = {"repo": MIRACL[0], "revision": MIRACL[1]}
            manifest["raw_files"]["train.jsonl"] = file_hash(raw)
            manifest_path.write_text(json.dumps(manifest))
            metadata = root / "historical-checkpoint-metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "resolved_config": {
                            "data_manifests": {"pairs": file_hash(manifest_path)}
                        }
                    }
                )
            )
            object_data = json.loads(objects.read_text())
            object_data["objects"][0]["bytes"] = raw.stat().st_size
            objects.write_text(json.dumps(object_data))
            policy_name = "yat-embedding-src-0929-v1"
            policy = {"policy": policy_name, **VERIFIED_LEGACY_PRODUCERS[policy_name]}
            policies = root / "policies.json"
            policies.write_text(json.dumps({"pairs": policy}))
            scan(
                [metadata],
                [root / "historical-data"],
                objects,
                root / "scan.json",
                materialize_directory=root / "cloud-raw",
                producer_policies=policies,
                _command_prefix=[sys.executable, str(cli)],
            )
            materialized = root / "cloud-raw/pairs"
            self.assertEqual(
                raw.read_bytes(), (materialized / "train.jsonl").read_bytes()
            )
            self.assertEqual(
                manifest_path.read_bytes(),
                (materialized / "manifest.json").read_bytes(),
            )
            with self.assertRaises(ValueError):
                build(
                    args.parent_public,
                    [metadata],
                    [materialized],
                    root / "missing-policy.json",
                )
            inventory = root / "inventory.json"
            built = build(
                args.parent_public,
                [metadata],
                [materialized],
                inventory,
                producer_policies=policies,
            )
            self.assertEqual(built["producer_policies"]["sha256"], file_hash(policies))
            self.assertEqual(
                built["stages"][0]["sources"][0]["historical_row_policy"], policy
            )
            proof = parent_exposure(
                inventory, load_candidate_exclusions(independent / "candidate"), hashes
            )
            self.assertTrue(proof["complete"])
            self.assertEqual(proof["overlap_rows"], 0)
            policy["producer_sha256"] = "0" * 64
            policies.write_text(json.dumps({"pairs": policy}))
            with self.assertRaises(ValueError):
                build(
                    args.parent_public,
                    [metadata],
                    [materialized],
                    root / "forged-policy.json",
                    producer_policies=policies,
                )

    def test_missing_tampered_duplicate_and_bound_inputs_rejected(self):
        for mode in (
            "missing",
            "raw-tamper",
            "metadata-mismatch",
            "duplicate-stage",
            "row-budget",
            "strict-lineage",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, _, _, _, _, _ = fixture(root)
                metadata = root / "historical-checkpoint-metadata.json"
                sources = [root / "historical-data"]
                stages = [metadata]
                kwargs = {}
                if mode == "missing":
                    sources = [root / "data"]
                    (sources[0] / "manifest.json").write_text("{}")
                elif mode == "raw-tamper":
                    (sources[0] / "train.jsonl").write_text("{}\n")
                elif mode == "metadata-mismatch":
                    metadata.write_text(
                        json.dumps(
                            {"resolved_config": {"data_manifests": {"pairs": "a" * 64}}}
                        )
                    )
                elif mode == "duplicate-stage":
                    stages.append(metadata)
                elif mode == "row-budget":
                    kwargs["max_rows_per_file"] = 2
                elif mode == "strict-lineage":
                    kwargs["require_parent_lineage"] = True
                output = root / "built.json"
                with self.assertRaises(ValueError):
                    build(args.parent_public, stages, sources, output, **kwargs)
                receipt = json.loads(output.read_text())
                self.assertEqual(receipt["builder_status"], "failed")
                self.assertFalse(receipt["known_contrastive_inventory_complete"])

    def test_actual_parent_export_requires_its_committed_stage_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _, config, hashes, _ = fixture(root)
            parent = Path(args.parent_public)
            metadata = {
                "model_family": "yat_embedding_finetune",
                "resolved_config": {
                    "encoder": config,
                    "data_manifests": {
                        "pairs": file_hash(root / "historical-data/manifest.json")
                    },
                },
                "tokenizer_identity": hashes["tokenizer.json"],
            }
            (parent / "checkpoint-metadata.json").write_text(json.dumps(metadata))
            manifest = {
                "step": 14000,
                "model_state": {"placeholder": {}},
                "metadata_sha256": canonical_hash(metadata),
                "identity": checkpoint_identity(metadata),
            }
            manifest["identity_sha256"] = canonical_hash(manifest["identity"])
            (parent / "checkpoint-manifest.json").write_text(json.dumps(manifest))
            export = {
                "source_model_family": "yat_embedding_finetune",
                "source_checkpoint_step": 14000,
                "tensors": 1,
                "source_checkpoint_metadata_sha256": manifest["metadata_sha256"],
                "source_checkpoint_manifest_identity_sha256": manifest[
                    "identity_sha256"
                ],
                "tokenizer_identity": hashes["tokenizer.json"],
                "sha256": hashes["model.safetensors"],
                "bytes": (parent / "model.safetensors").stat().st_size,
                "artifacts_sha256": artifact_hashes(
                    parent,
                    (
                        "config.json",
                        "tokenizer.json",
                        "model.safetensors",
                        "checkpoint-metadata.json",
                        "checkpoint-manifest.json",
                    ),
                ),
            }
            (parent / "export.json").write_text(json.dumps(export))
            with self.assertRaisesRegex(
                ValueError, "Current parent checkpoint metadata is missing"
            ):
                build(
                    parent,
                    [root / "historical-checkpoint-metadata.json"],
                    [root / "historical-data"],
                    root / "missing-current.json",
                )
            result = build(
                parent,
                [parent / "checkpoint-metadata.json"],
                [root / "historical-data"],
                root / "valid.json",
                require_parent_lineage=True,
            )
            self.assertTrue(result["parent_checkpoint_metadata_bound"])
            self.assertEqual(
                result["expected_stage_identities"],
                [file_hash(parent / "checkpoint-metadata.json")],
            )

    def test_expired_budget_leaves_failed_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _, _, _, _ = fixture(root)
            output = root / "built.json"
            with patch(
                "scripts.build_parent_exposure_inventory.time.monotonic",
                side_effect=[0, 2],
            ):
                with self.assertRaises(TimeoutError):
                    build(
                        args.parent_public,
                        [root / "historical-checkpoint-metadata.json"],
                        [root / "historical-data"],
                        output,
                        timeout_seconds=1,
                    )
            self.assertFalse(
                json.loads(output.read_text())["known_contrastive_inventory_complete"]
            )
