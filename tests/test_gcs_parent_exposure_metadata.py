"""Bounded fake CLI streams; no GCS network, cloud allocation or model calls."""

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import time
import unittest

from flaxchat.embedding_development_quarantine import load_candidate_exclusions
from flaxchat.embedding_dev_exposure import parent_exposure
from scripts.build_parent_exposure_inventory import build
from scripts.scan_gcs_parent_exposure import scan
from tests.test_independent_retrieval_dev_metadata import fixture


class GCSExposureScanTests(unittest.TestCase):
    def setup_case(self, root, mode="valid"):
        args, _, independent, _, hashes, _ = fixture(root)
        raw = root / "historical-data/train.jsonl"
        objects = root / "objects.json"
        objects.write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "uri": "gs://fixture/embedding-data/pairs/train.jsonl#123456",
                            "bytes": raw.stat().st_size,
                        }
                    ]
                }
            )
        )
        cli = root / "fake_gcloud.py"
        cli.write_text(
            "import pathlib,sys,time\n"
            f"mode={mode!r}\n"
            "assert sys.argv[-1].endswith('#123456')\n"
            "if mode=='hang':\n"
            " pathlib.Path(__file__).with_suffix('.pid').write_text(str(__import__('os').getpid()))\n"
            " time.sleep(60)\n"
            f"data=pathlib.Path({str(raw)!r}).read_bytes()\n"
            "if mode=='short': data=data[:-10]\n"
            "if mode=='tamper': data=data.replace(b'positive',b'positivx',1)\n"
            "sys.stdout.buffer.write(data);sys.stdout.buffer.flush()\n"
            "if mode=='exit': sys.exit(7)\n"
        )
        return args, independent, hashes, objects, cli

    def run_scan(self, root, objects, cli, **kwargs):
        return scan(
            [root / "historical-checkpoint-metadata.json"],
            [root / "historical-data"],
            objects,
            root / "scan.json",
            _command_prefix=[sys.executable, str(cli)],
            **kwargs,
        )

    def test_stream_materialization_feeds_existing_builder_and_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, independent, hashes, objects, cli = self.setup_case(root)
            receipt = self.run_scan(
                root,
                objects,
                cli,
                materialize_directory=root / "cloud-raw",
                development_exclusions=independent / "candidate",
            )
            self.assertTrue(receipt["coverage_complete"])
            self.assertFalse(receipt["admission_ready"])
            self.assertEqual(receipt["exact_aligned_overlap_rows"], 0)
            self.assertEqual(receipt["sources"][0]["checked_rows"], 8)
            self.assertEqual(
                (root / "cloud-raw/pairs/train.jsonl").read_bytes(),
                (root / "historical-data/train.jsonl").read_bytes(),
            )
            inventory = root / "inventory.json"
            build(
                args.parent_public,
                [root / "historical-checkpoint-metadata.json"],
                [root / "cloud-raw/pairs"],
                inventory,
            )
            proof = parent_exposure(
                inventory, load_candidate_exclusions(independent / "candidate"), hashes
            )
            self.assertTrue(proof["complete"])
            self.assertEqual(proof["overlap_rows"], 0)

    def test_stream_failure_and_identity_mismatch_leave_failed_receipt(self):
        for mode in (
            "short",
            "tamper",
            "exit",
            "unpinned",
            "byte-budget",
            "stage-mismatch",
            "extra-object",
            "row-budget",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, _, objects, cli = self.setup_case(root, mode)
                kwargs = {"materialize_directory": root / "cloud-raw"}
                if mode == "unpinned":
                    objects.write_text(objects.read_text().replace("#123456", ""))
                elif mode == "byte-budget":
                    kwargs["max_total_bytes"] = 1
                elif mode == "row-budget":
                    kwargs["max_rows_per_file"] = 1
                elif mode == "stage-mismatch":
                    (root / "historical-checkpoint-metadata.json").write_text(
                        json.dumps(
                            {"resolved_config": {"data_manifests": {"pairs": "a" * 64}}}
                        )
                    )
                elif mode == "extra-object":
                    content = json.loads(objects.read_text())
                    content["objects"].append(
                        {"uri": "gs://fixture/data/other/train.jsonl#2", "bytes": 1}
                    )
                    objects.write_text(json.dumps(content))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_scan(root, objects, cli, **kwargs)
                receipt = json.loads((root / "scan.json").read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse(receipt["coverage_complete"])
                self.assertFalse((root / "cloud-raw/pairs/train.jsonl").exists())
                self.assertFalse(
                    (root / "cloud-raw/pairs/train.jsonl.partial").exists()
                )

    def test_deadline_kills_stream_and_closes_pipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, objects, cli = self.setup_case(root, "hang")
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                self.run_scan(
                    root,
                    objects,
                    cli,
                    timeout_seconds=1,
                    materialize_directory=root / "cloud-raw",
                )
            self.assertLess(time.monotonic() - started, 4)
            pid = int(cli.with_suffix(".pid").read_text())
            with self.assertRaises(ProcessLookupError):
                __import__("os").kill(pid, 0)
            self.assertEqual(
                json.loads((root / "scan.json").read_text())["status"], "failed"
            )

    def test_authenticated_overlap_is_reported_without_clean_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, independent, _, objects, cli = self.setup_case(root)
            candidate = next((independent / "candidate").glob("*.jsonl"))
            exposed = json.loads(candidate.read_text().splitlines()[0])
            raw = root / "historical-data/train.jsonl"
            rows = raw.read_text().splitlines()
            rows[0] = json.dumps(exposed)
            raw.write_text("\n".join(rows) + "\n")
            manifest_path = root / "historical-data/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["raw_files"]["train.jsonl"] = hashlib.sha256(
                raw.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            metadata = root / "historical-checkpoint-metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "resolved_config": {
                            "data_manifests": {
                                "pairs": hashlib.sha256(
                                    manifest_path.read_bytes()
                                ).hexdigest()
                            }
                        }
                    }
                )
            )
            object_data = json.loads(objects.read_text())
            object_data["objects"][0]["bytes"] = raw.stat().st_size
            objects.write_text(json.dumps(object_data))
            receipt = self.run_scan(
                root, objects, cli, development_exclusions=independent / "candidate",
                overlap_details_output=root / "details.json",
            )
            self.assertTrue(receipt["coverage_complete"])
            self.assertGreater(receipt["exact_aligned_overlap_rows"], 0)
            self.assertFalse(receipt["admission_ready"])
            details = json.loads((root / "details.json").read_text())
            from flaxchat.embedding_data import row_identities
            expected = set(row_identities(exposed, 'pairs').values())
            self.assertEqual(set(details['matched_text_sha256']), expected)
            self.assertTrue(details['exact_text_identity_details_complete'])
            self.assertFalse(details['details_truncated'])
            self.assertEqual(details['candidate_identity'], receipt['candidate_identity'])
            self.assertEqual(receipt['overlap_details']['sha256'], hashlib.sha256(
                (root / 'details.json').read_bytes()).hexdigest())
            self.assertNotIn(exposed['query'], (root / 'details.json').read_text())
            self.assertFalse(details['admission_ready'])
            # A separately requested capped report preserves full raw coverage,
            # but cannot claim its matched-identity list is complete.
            (root / 'scan.json').unlink()
            capped = self.run_scan(root, objects, cli,
                development_exclusions=independent / 'candidate',
                overlap_details_output=root / 'capped-details.json',
                max_overlap_text_identities=1)
            capped_details = json.loads((root / 'capped-details.json').read_text())
            self.assertTrue(capped['coverage_complete'])
            self.assertEqual(capped['exact_aligned_overlap_rows'], receipt['exact_aligned_overlap_rows'])
            self.assertEqual(len(capped_details['matched_text_sha256']), 1)
            self.assertTrue(capped_details['details_truncated'])
            self.assertFalse(capped_details['exact_text_identity_details_complete'])

    def test_failed_or_unbound_details_never_claim_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, independent, _, objects, cli = self.setup_case(root, 'short')
            with self.assertRaises(ValueError):
                self.run_scan(root, objects, cli, overlap_details_output=root / 'details.json')
            self.assertFalse((root / 'details.json').exists())
            with self.assertRaises(ValueError):
                self.run_scan(root, objects, cli,
                    development_exclusions=independent / 'candidate',
                    overlap_details_output=root / 'details.json', max_overlap_text_identities=True)
            with self.assertRaises((ValueError, RuntimeError)):
                self.run_scan(root, objects, cli,
                    development_exclusions=independent / 'candidate',
                    overlap_details_output=root / 'details.json')
            details = json.loads((root / 'details.json').read_text())
            self.assertEqual(details['status'], 'failed')
            self.assertFalse(details['coverage_complete'])
            self.assertFalse(details['exact_text_identity_details_complete'])
            self.assertFalse(details['admission_ready'])

    def test_scan_without_candidate_never_claims_clean_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, objects, cli = self.setup_case(root)
            receipt = self.run_scan(root, objects, cli)
            self.assertTrue(receipt["coverage_complete"])
            self.assertFalse(receipt["candidate_exposure_checked"])
            self.assertIsNone(receipt["exact_aligned_overlap_rows"])
            self.assertIsNone(receipt["sources"][0]["directory"])


if __name__ == "__main__":
    unittest.main()
