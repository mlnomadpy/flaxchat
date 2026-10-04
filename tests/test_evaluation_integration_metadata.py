"""Evaluation/Hub integration fixtures without model execution or network."""

import ast
import copy
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.evaluation_contract import aggregate
from scripts.merge_yat_mteb_splits import merge as merge_splits
from scripts.merge_yat_mteb_subsets import merge as merge_subsets
from scripts.assemble_yat_embedding_full_mteb import assemble_receipts
from scripts.release_contract import (
    artifact_hashes,
    canonical_hash,
    checkpoint_identity,
    digest,
)
from tests.test_evaluation_contract import receipt


class IntegrationTests(unittest.TestCase):
    def test_subset_merge_covers_every_split_and_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = receipt()
            spec = full["inventory"]["a"]
            spec["eval_splits"] = ["test", "dev"]
            spec["expected_rows"] += [["dev", "en"], ["dev", "fr"]]
            full["results"]["a"]["result"]["scores"]["dev"] = copy.deepcopy(
                full["results"]["a"]["result"]["scores"]["test"]
            )
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps(full))
            paths = []
            for subset in ("en", "fr"):
                selected_spec = {
                    **copy.deepcopy(spec),
                    "hf_subsets": [subset],
                    "expected_rows": [[split, subset] for split in spec["eval_splits"]],
                }
                raw = copy.deepcopy(full["results"]["a"]["result"])
                raw["scores"] = {
                    split: [row for row in rows if row["hf_subset"] == subset]
                    for split, rows in raw["scores"].items()
                }
                part = {
                    "identity": full["identity"],
                    "task_name": "a",
                    "dataset_revision": spec["dataset_revision"],
                    "full_subsets": ["en", "fr"],
                    "subsets": [subset],
                    "task_inventory": selected_spec,
                    "result": raw,
                }
                path = root / f"{subset}.json"
                path.write_text(json.dumps(part))
                paths.append(path)
            merged = merge_subsets(paths, inventory, root / "merged.json")
            self.assertTrue(aggregate(merged)["complete"])
            with self.assertRaises(ValueError):
                merge_subsets([paths[0], paths[0]], inventory, root / "duplicate.json")

    def test_split_merge_and_cached_assembler_recheck_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = receipt()
            spec = full["inventory"]["a"]
            spec["eval_splits"] = ["test", "dev"]
            spec["expected_rows"] += [["dev", "en"], ["dev", "fr"]]
            paths = []
            for split in ("test", "dev"):
                selected_spec = {
                    **copy.deepcopy(spec),
                    "eval_splits": [split],
                    "expected_rows": [[split, subset] for subset in spec["hf_subsets"]],
                }
                raw = copy.deepcopy(full["results"]["a"]["result"])
                raw["scores"] = {split: raw["scores"]["test"]}
                part = {
                    "identity": full["identity"],
                    "task_name": "a",
                    "dataset_revision": spec["dataset_revision"],
                    "full_splits": spec["eval_splits"],
                    "split": split,
                    "task_inventory": selected_spec,
                    "full_task_inventory": spec,
                    "result": raw,
                }
                path = root / f"{split}.json"
                path.write_text(json.dumps(part))
                paths.append(path)
            merged = merge_splits(paths, root / "split-complete.json")
            full["results"]["a"]["result"] = merged["result"]
            shard = root / "shard.json"
            shard.write_text(json.dumps(full))
            output = root / "summary.json"
            first = assemble_receipts({"suite": [shard]}, output)
            self.assertEqual(first, assemble_receipts({"suite": [shard]}, output))
            altered = copy.deepcopy(full)
            altered["results"]["a"]["result"]["scores"]["test"][0]["main_score"] = 0.9
            shard.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                assemble_receipts({"suite": [shard]}, output)

    def test_backend_receipt_retains_effective_runtime(self):
        from scripts.validate_yat_torch_parity import _run_receipt
        from tests.test_release_contract import case

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensors", "tokenizer.json"):
                (root / name).write_text(name)
            (root / "config.json").write_text('{"num_hidden_layers":2}')
            inputs = root / "inputs.npy"
            inputs.write_bytes(b"inputs")
            output = root / "out.npz"
            output.write_bytes(b"output")
            Path(str(inputs) + ".json").write_text(
                json.dumps(
                    {"inputs_sha256": digest(inputs), "sequence_length": 32, "rows": 72}
                )
            )
            runtime = case({}, "fixture")["runs"]["jax"]["runtime"]
            runtime["numeric_environment"] = {"JAX_DEFAULT_MATMUL_PRECISION": "highest"}
            runtime["effective_jax_config"] = {
                "jax_default_matmul_precision": "highest"
            }
            with patch(
                "scripts.validate_yat_torch_parity.current_identity",
                return_value={
                    "runtime": runtime,
                    "source_sha256": {"implementation": "frozen"},
                },
            ):
                _run_receipt("jax", root, inputs, output, 1, [{"platform": "tpu"}])
            record = json.loads(Path(str(output) + ".json").read_text())
            self.assertEqual(record["runtime"], runtime)
            self.assertEqual(record["scope"]["intermediate_indices"], [0])

    def test_actual_pinned_task_metadata_receipt(self):
        from scripts.evaluation_contract import validate_inventory_spec, validate_result

        record = json.loads(
            Path(
                "tests/fixtures/mteb-2.21.8-task-metadata.json"
            ).read_text()
        )
        self.assertEqual(record["mteb_version"], "2.21.8")
        self.assertFalse(record["model_executed"])
        self.assertFalse(record["datasets_loaded"])
        expected = {
            "STSBenchmark": 1,
            "MIRACLRetrievalHardNegatives": 18,
            "WebLINXCandidatesReranking": 6,
        }
        self.assertEqual(
            {
                name: len(spec["expected_rows"])
                for name, spec in record["inventory"].items()
            },
            expected,
        )
        for name, spec in record["inventory"].items():
            validate_inventory_spec(spec)
            rows = {
                split: [
                    {"hf_subset": subset, "main_score": 0.5}
                    for subset in spec["hf_subsets"]
                ]
                for split in spec["eval_splits"]
            }
            item = {
                "category": spec["category"],
                "result": {
                    "task_name": name,
                    "dataset_revision": spec["dataset_revision"],
                    "scores": rows,
                },
            }
            validate_result(name, item, spec)
            rows[spec["eval_splits"][0]].pop()
            with self.assertRaises(ValueError):
                validate_result(name, item, spec)

    def test_campaign_resumes_only_missing_cases_and_rejects_identity_or_raw_changes(
        self,
    ):
        from scripts.run_yat_parity_campaign import run_campaign
        from tests.test_release_contract import case
        from scripts.evaluation_contract import write_atomic

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            evidence = root / "evidence"
            for folder in (source, target):
                for name in ("model.safetensors", "tokenizer.json"):
                    (folder / name).write_text(name)
                (folder / "config.json").write_text(
                    '{"num_hidden_layers":22,"max_position_embeddings":512,"hidden_size":4}'
                )
            (target / "yat_encoder.py").write_text("implementation")
            (target / "conversion.json").write_text("{}")
            artifacts = artifact_hashes(target)
            fixture = case(artifacts, digest(target / "conversion.json"))
            identities = {
                backend: {
                    key: fixture["runs"][backend][key]
                    for key in ("runtime", "source_sha256", "model_sha256")
                }
                for backend in ("jax", "torch")
            }

            def probe(argv, **kwargs):
                write_atomic(Path(argv[-1]), identities[argv[4]])

            def fake_case(source, target, jp, tp, evidence, length, batch, timeout):
                stem = f"length-{length}-batch-{batch}"
                report = case(
                    artifacts, digest(target / "conversion.json"), length, batch
                )
                inputs = evidence / f"{stem}-inputs.npy"
                inputs.write_bytes(b"fixture")
                report["scope"]["inputs_sha256"] = digest(inputs)
                for backend in ("jax", "torch"):
                    raw = evidence / f"{stem}-{backend}.npz"
                    raw.write_bytes(b"output")
                    report["runs"][backend].update(
                        output_sha256=digest(raw), inputs_sha256=digest(inputs)
                    )
                    write_atomic(Path(str(raw) + ".json"), report["runs"][backend])
                write_atomic(evidence / f"{stem}-report.json", report)

            with (
                patch(
                    "scripts.run_yat_parity_campaign.subprocess.run", side_effect=probe
                ),
                patch(
                    "scripts.run_yat_parity_campaign.run_case", side_effect=fake_case
                ) as worker,
                patch("scripts.run_yat_parity_campaign.validate_release") as release,
            ):
                budgets = dict(
                    host_ram_budget_bytes=16 * 2**30,
                    scratch_budget_bytes=16 * 2**30,
                    evidence_budget_bytes=16 * 2**30,
                )
                partial = run_campaign(
                    source, target, "jax", "torch", evidence, 120, 60, 1, **budgets
                )
                self.assertFalse(partial["complete"])
                self.assertEqual(worker.call_count, 1)
                release.assert_not_called()
                complete = run_campaign(
                    source, target, "jax", "torch", evidence, 120, 60, 5, **budgets
                )
                self.assertTrue(complete["complete"])
                self.assertEqual(worker.call_count, 6)
                release.assert_called_once()
                run_campaign(
                    source, target, "jax", "torch", evidence, 120, 60, 1, **budgets
                )
                self.assertEqual(worker.call_count, 6)
                identities["jax"]["runtime"]["numeric_environment"] = {
                    "XLA_USE_BF16": "1"
                }
                with self.assertRaises(ValueError):
                    run_campaign(
                        source, target, "jax", "torch", evidence, 120, 60, 1, **budgets
                    )
                identities["jax"]["runtime"]["numeric_environment"] = {}
                (evidence / "length-128-batch-1-jax.npz").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    run_campaign(
                        source, target, "jax", "torch", evidence, 120, 60, 1, **budgets
                    )

    def test_parity_case_deadline_preserves_failure_receipt(self):
        import subprocess
        from scripts.run_yat_parity_case import run_case

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "conversion.json").write_text("{}")
            with patch(
                "scripts.run_yat_parity_case.subprocess.run",
                side_effect=subprocess.TimeoutExpired("mock", 60),
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_case(
                        root / "source",
                        target,
                        "jax-python",
                        "torch-python",
                        root / "evidence",
                        128,
                        8,
                        60,
                    )
            status = json.loads(
                (root / "evidence/length-128-batch-8-status.json").read_text()
            )
            self.assertEqual(status["state"], "failed")
            self.assertFalse(status["full_matrix_qualified"])

    def test_conversion_mapping_without_importing_torch(self):
        tree = ast.parse(Path("torch_port/convert_yat_encoder.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "map_name"
        )
        namespace = {"re": re}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "map_name", "exec"),
            namespace,
        )
        self.assertEqual(
            namespace["map_name"]("['layers']['2']['qkv']['kernel']"),
            ("layers.2.qkv.weight", True),
        )
        self.assertEqual(
            namespace["map_name"]("['layers']['2']['yat_alpha']"),
            ("layers.2.yat_alpha", False),
        )
        with self.assertRaises(ValueError):
            namespace["map_name"]("['layers']['2']['unknown']")

    def test_new_stage_publisher_verifies_remote_and_cleans_token(self):
        from scripts import publish_yat_embedding_from_gcp as publisher

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            for name in ("model.safetensors", "tokenizer.json"):
                (release / name).write_text(name)
            (release / "config.json").write_text("{}")
            (release / "README.md").write_text("candidate model")
            metadata = {
                "model_family": "yat_embedding_finetune",
                "resolved_config": {"encoder": {}},
                "tokenizer_identity": digest(release / "tokenizer.json"),
            }
            manifest = {
                "step": 17000,
                "model_state": {"weight": {}},
                "identity": checkpoint_identity(metadata),
                "metadata_sha256": canonical_hash(metadata),
            }
            manifest["identity_sha256"] = canonical_hash(manifest["identity"])
            (release / "checkpoint-metadata.json").write_text(json.dumps(metadata))
            (release / "checkpoint-manifest.json").write_text(json.dumps(manifest))
            export = {
                "source_model_family": "yat_embedding_finetune",
                "source_checkpoint_step": 17000,
                "source_checkpoint_metadata_sha256": manifest["metadata_sha256"],
                "source_checkpoint_manifest_identity_sha256": manifest[
                    "identity_sha256"
                ],
                "tokenizer_identity": metadata["tokenizer_identity"],
                "tensors": 1,
                "sha256": digest(release / "model.safetensors"),
                "bytes": (release / "model.safetensors").stat().st_size,
                "artifacts_sha256": artifact_hashes(
                    release,
                    (
                        "model.safetensors",
                        "config.json",
                        "tokenizer.json",
                        "checkpoint-metadata.json",
                        "checkpoint-manifest.json",
                    ),
                ),
            }
            (release / "export.json").write_text(json.dumps(export))
            remote = root / "remote"
            remote.mkdir()
            created = []

            class Api:
                def __init__(self, token):
                    self.token = token

                def repo_exists(self, *args, **kwargs):
                    return False

                def create_repo(self, *args, **kwargs):
                    created.append(args[0])

                def upload_folder(self, **kwargs):
                    for path in release.iterdir():
                        shutil.copy(path, remote / path.name)
                    (remote / "config.json").write_text('{"corrupt":true}')

                def model_info(self, *args, **kwargs):
                    return SimpleNamespace(
                        sha="a" * 40,
                        siblings=[
                            SimpleNamespace(rfilename=p.name) for p in remote.iterdir()
                        ],
                    )

            def download(repo, name, **kwargs):
                self.assertEqual(kwargs["revision"], "a" * 40)
                return str(remote / name)

            token = root / "token"
            token.write_text("temporary-placeholder")
            transport = SimpleNamespace(HfApi=Api, hf_hub_download=download)
            with (
                patch.dict(sys.modules, {"huggingface_hub": transport}),
                patch.object(
                    sys,
                    "argv",
                    ["publish", str(release), str(token), "--repo", "owner/new-stage"],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    publisher.main()
            self.assertEqual(created, ["owner/new-stage"])
            self.assertFalse(token.exists())


if __name__ == "__main__":
    unittest.main()
