"""Model-free PyTorch release identity and finite-metric validation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.release_contract import (
    ARTIFACTS,
    VERSION,
    artifact_hashes,
    digest,
    canonical_hash,
    checkpoint_identity,
    validate_release,
    validate_metrics,
    validate_export,
    intermediate_scope,
    validate_runtime_identity,
)


def case(artifacts, conversion_hash, length=128, batch=1):
    metrics = {
        name: {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p99_abs": 0.0,
            "min_vector_cosine": 1.0,
            "mean_vector_cosine": 1.0,
            "shape": [72, 4] if name == "pool" else [72, length, 4],
        }
        for name in ("embedding", "layer_0", "layer_10", "hidden", "pool")
    }
    scope = {
        **intermediate_scope({"num_hidden_layers": 22}),
        "rows": 72,
        "sequence_length": length,
        "batch_size": batch,
        "retrieval_protocol": "8-query/64-corpus-v1",
        "inputs_sha256": "a" * 64,
        "fixture": "multilingual/code/long/empty/all-padding-v2",
    }
    run = {
        "model_sha256": {
            key: value for key, value in artifacts.items() if key != "yat_encoder.py"
        },
        "format": VERSION,
        "scope": scope,
        "output_sha256": "b" * 64,
        "inputs_sha256": scope["inputs_sha256"],
        "source_sha256": {"implementation": "pinned"},
        "runtime": {
            "runtime_packages": {"jax": "pinned"},
            "interpreter": {
                "version": "3.12",
                "implementation": "CPython",
                "build": ["pinned"],
                "executable_sha256": "pinned",
            },
            "numeric_environment": {},
            "effective_jax_config": {},
            "effective_torch_config": {},
            "deployment_receipts_sha256": {
                name: "pinned"
                for name in (
                    "manifest-identity.txt",
                    "runtime-receipt.json",
                    "hardware-receipt.json",
                )
            },
        },
        "devices": ["TPU"],
    }
    return {
        "format": VERSION,
        "artifacts_sha256": artifacts,
        "conversion_sha256": conversion_hash,
        "jax_backend": "physical TPU",
        "torch_backend": "torch_xla TPU",
        "metrics": metrics,
        "retrieval": {
            "top_1_agreement": 1.0,
            "top_3_set_agreement": 1.0,
            "max_cosine_score_drift": 0.0,
        },
        "scope": scope,
        "runs": {
            "jax": {**run, "backend": "jax"},
            "torch": {**run, "backend": "torch", "model_sha256": artifacts},
        },
    }


class ReleaseContractTests(unittest.TestCase):
    def test_reduced_scope_and_output_receipts_fail_closed(self):
        valid = case({}, "unused")
        validate_metrics(valid)
        for key in ("rows", "fixture", "batch_size", "inputs_sha256"):
            changed = copy.deepcopy(valid)
            changed["scope"].pop(key)
            with self.subTest(scope=key), self.assertRaises(ValueError):
                validate_metrics(changed)
        for key in ("shape",):
            changed = copy.deepcopy(valid)
            changed["metrics"]["pool"].pop(key)
            with self.assertRaises(ValueError):
                validate_metrics(changed)
        for key in ("scope", "output_sha256", "backend", "format"):
            changed = copy.deepcopy(valid)
            changed["runs"]["jax"].pop(key)
            with self.subTest(run=key), self.assertRaises(ValueError):
                validate_metrics(changed)
        changed = copy.deepcopy(valid)
        changed["metrics"]["pool"]["shape"] = [8, 4]
        with self.assertRaises(ValueError):
            validate_metrics(changed)

    def test_publication_recomputes_durable_evidence_and_rejects_tampering(self):
        import numpy as np
        from scripts.parity_evidence import validate_durable_parity
        from scripts import validate_yat_torch_parity as evaluator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            root.mkdir()
            evidence = Path(directory) / "private-evidence"
            evidence.mkdir()
            config = {
                "num_hidden_layers": 22,
                "hidden_size": 4,
                "max_position_embeddings": 512,
                "pad_token_id": 0,
            }
            for name in ARTIFACTS:
                (root / name).write_text(
                    json.dumps(config) if name == "config.json" else name
                )
            metadata = {
                "model_family": "yat_embedding_finetune",
                "resolved_config": {"encoder": config},
            }
            manifest = {
                "step": 17000,
                "metadata_sha256": canonical_hash(metadata),
                "identity": checkpoint_identity(metadata),
            }
            manifest["identity_sha256"] = canonical_hash(manifest["identity"])
            (root / "checkpoint-metadata.json").write_text(json.dumps(metadata))
            (root / "checkpoint-manifest.json").write_text(json.dumps(manifest))
            hashes = artifact_hashes(root)
            export = {
                "source_checkpoint_step": 17000,
                "source_model_family": "yat_embedding_finetune",
                "tensors": 19,
                "sha256": hashes["model.safetensors"],
                "source_checkpoint_metadata_sha256": manifest["metadata_sha256"],
                "source_checkpoint_manifest_identity_sha256": manifest[
                    "identity_sha256"
                ],
                "artifacts_sha256": artifact_hashes(
                    root,
                    (
                        "config.json",
                        "tokenizer.json",
                        "checkpoint-metadata.json",
                        "checkpoint-manifest.json",
                    ),
                ),
            }
            (root / "source-export.json").write_text(json.dumps(export))
            conversion = {
                "format": "flaxchat-yat-pytorch-v2",
                "source_model_family": "yat_embedding_finetune",
                "source_checkpoint_step": 17000,
                "tensor_count": 19,
                "source_export_sha256": digest(root / "source-export.json"),
                "artifacts_sha256": hashes,
                "source_weights_sha256": hashes["model.safetensors"],
                "torch_weights_sha256": hashes["model.safetensors"],
                "tokenizer_sha256": hashes["tokenizer.json"],
            }
            (root / "conversion.json").write_text(json.dumps(conversion))
            cases = []
            for length in (128, 256, 512):
                for batch in (1, 8):
                    item = case(hashes, digest(root / "conversion.json"), length, batch)
                    stem = f"length-{length}-batch-{batch}"
                    tokens = np.ones((72, length), np.int32)
                    tokens[-1] = 0
                    inputs = evidence / f"{stem}-inputs.npy"
                    np.save(inputs, tokens)
                    item["scope"]["inputs_sha256"] = digest(inputs)
                    fixture = {
                        key: item["scope"][key]
                        for key in (
                            "sequence_length",
                            "rows",
                            "retrieval_protocol",
                            "inputs_sha256",
                            "fixture",
                        )
                    }
                    Path(str(inputs) + ".json").write_text(json.dumps(fixture))
                    arrays = {
                        name: np.ones(metric["shape"], np.float32)
                        for name, metric in item["metrics"].items()
                    }
                    for array in arrays.values():
                        array[-1] = 0
                    for backend in ("jax", "torch"):
                        raw = evidence / f"{stem}-{backend}.npz"
                        np.savez(raw, **arrays)
                        run = item["runs"][backend]
                        run.update(
                            inputs_sha256=digest(inputs),
                            output_sha256=digest(raw),
                            source_sha256={
                                str(Path(evaluator.__file__)): digest(
                                    Path(evaluator.__file__)
                                )
                            },
                        )
                        if backend == "torch":
                            run["conversion_sha256"] = digest(root / "conversion.json")
                        Path(str(raw) + ".json").write_text(json.dumps(run))
                    report_path = evidence / f"{stem}-report.json"
                    cases.append(
                        evaluator.compare(
                            evidence / f"{stem}-jax.npz",
                            evidence / f"{stem}-torch.npz",
                            report_path,
                        )
                    )
            report = {
                "format": VERSION,
                "cases": cases,
                "conversion_sha256": digest(root / "conversion.json"),
                "artifacts_sha256": hashes,
            }
            budgets = dict(
                host_ram_budget_bytes=16 * 2**30, evidence_budget_bytes=16 * 2**30
            )
            accepted = validate_durable_parity(
                root, conversion, report, evidence, **budgets
            )
            self.assertEqual(len(accepted["files_sha256"]), 42)
            for insufficient in (
                {"host_ram_budget_bytes": 1, "evidence_budget_bytes": 16 * 2**30},
                {"host_ram_budget_bytes": 16 * 2**30, "evidence_budget_bytes": 1},
            ):
                with self.subTest(budgets=insufficient), self.assertRaises(ValueError):
                    validate_durable_parity(
                        root, conversion, report, evidence, **insufficient
                    )
            raw = evidence / "length-128-batch-1-torch.npz"
            original = raw.read_bytes()
            raw.write_bytes(b"forged")
            with self.assertRaises(ValueError):
                validate_durable_parity(root, conversion, report, evidence, **budgets)
            raw.write_bytes(original)
            # A reduced raw tensor with consistently rewritten hashes/sidecars
            # must still fail its claimed72-row coverage.
            reduced = copy.deepcopy(report)
            arrays = {
                name: np.ones(metric["shape"], np.float32)[:8]
                for name, metric in reduced["cases"][0]["metrics"].items()
            }
            np.savez(raw, **arrays)
            reduced["cases"][0]["runs"]["torch"]["output_sha256"] = digest(raw)
            sidecar = Path(str(raw) + ".json")
            old_sidecar = sidecar.read_text()
            sidecar.write_text(json.dumps(reduced["cases"][0]["runs"]["torch"]))
            path = evidence / "length-128-batch-1-report.json"
            original_report = path.read_text()
            path.write_text(json.dumps(reduced["cases"][0]))
            with self.assertRaises(ValueError):
                validate_durable_parity(root, conversion, reduced, evidence, **budgets)
            raw.write_bytes(original)
            sidecar.write_text(old_sidecar)
            path.write_text(original_report)
            # A tiny ZIP member claiming an enormous array must be rejected
            # from its bounded NPY header before np.load can allocate it.
            import io
            import zipfile
            from unittest.mock import patch

            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {"descr": "<f4", "fortran_order": False, "shape": (72, 2**40, 4)},
            )
            with zipfile.ZipFile(io.BytesIO(original)) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            members["embedding.npy"] = header.getvalue()
            with zipfile.ZipFile(raw, "w") as archive:
                for name, payload in members.items():
                    archive.writestr(name, payload)
            malicious = copy.deepcopy(report)
            malicious["cases"][0]["runs"]["torch"]["output_sha256"] = digest(raw)
            sidecar.write_text(json.dumps(malicious["cases"][0]["runs"]["torch"]))
            path.write_text(json.dumps(malicious["cases"][0]))
            with patch(
                "scripts.parity_evidence.np.load",
                side_effect=AssertionError("must reject before array load"),
            ):
                with self.assertRaises(ValueError):
                    validate_durable_parity(
                        root, conversion, malicious, evidence, **budgets
                    )
            raw.write_bytes(original)
            sidecar.write_text(old_sidecar)
            path.write_text(original_report)
            # Alter both published and durable claims: only raw recomputation catches this.
            changed = copy.deepcopy(report)
            changed["cases"][0]["metrics"]["pool"]["max_abs"] = 0.1
            path.write_text(json.dumps(changed["cases"][0]))
            with self.assertRaises(ValueError):
                validate_durable_parity(root, conversion, changed, evidence, **budgets)
            path.write_text(original_report)
            with self.assertRaises(ValueError):
                validate_durable_parity(
                    root, conversion, report, root / "raw", **budgets
                )

    def test_depth_and_complete_runtime_scope(self):
        self.assertEqual(
            intermediate_scope({"num_hidden_layers": 1})["intermediate_indices"], [0]
        )
        self.assertEqual(
            intermediate_scope({"num_hidden_layers": 2})["intermediate_indices"], [0]
        )
        self.assertEqual(
            intermediate_scope({"num_hidden_layers": 4})["intermediate_indices"], [0, 1]
        )
        runtime = case({}, "unused")["runs"]["jax"]["runtime"]
        validate_runtime_identity(runtime)
        for key in runtime:
            changed = copy.deepcopy(runtime)
            changed.pop(key)
            with self.assertRaises(ValueError):
                validate_runtime_identity(changed)
        report = case({}, "unused")
        report["scope"] = {
            **report["scope"],
            **intermediate_scope({"num_hidden_layers": 1}),
        }
        for run in report["runs"].values():
            run["scope"] = report["scope"]
        report["metrics"].pop("layer_10")
        validate_metrics(report)

    def test_zero_output_and_shape_fail_closed(self):
        import numpy as np
        from scripts.validate_yat_torch_parity import validate_output_arrays

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {
                name: np.ones((72, 32, 4), np.float32)
                for name in ("embedding", "layer_0", "layer_10", "hidden")
            }
            arrays["pool"] = np.ones((72, 4), np.float32)
            for array in arrays.values():
                array[-1] = 0
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, **arrays)
            np.savez(right, **arrays)
            with np.load(left) as reference, np.load(right) as candidate:
                validate_output_arrays(
                    reference,
                    candidate,
                    {
                        "rows": 72,
                        "sequence_length": 32,
                        **intermediate_scope({"num_hidden_layers": 22}),
                    },
                )
            invalid = {name: value.copy() for name, value in arrays.items()}
            invalid["pool"][0] = 0
            np.savez(right, **invalid)
            with np.load(left) as reference, np.load(right) as candidate:
                with self.assertRaises(ValueError):
                    validate_output_arrays(
                        reference,
                        candidate,
                        {
                            "rows": 72,
                            "sequence_length": 32,
                            **intermediate_scope({"num_hidden_layers": 22}),
                        },
                    )
            collapsed = {name: np.zeros_like(value) for name, value in arrays.items()}
            np.savez(left, **collapsed)
            np.savez(right, **collapsed)
            with np.load(left) as reference, np.load(right) as candidate:
                with self.assertRaises(ValueError):
                    validate_output_arrays(
                        reference,
                        candidate,
                        {
                            "rows": 72,
                            "sequence_length": 32,
                            **intermediate_scope({"num_hidden_layers": 22}),
                        },
                    )

    def test_authenticated_export_allows_new_stage_and_rejects_lineage_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensors", "tokenizer.json"):
                (root / name).write_text(name)
            (root / "config.json").write_text("{}")
            metadata = {
                "model_family": "yat_embedding_finetune",
                "resolved_config": {"encoder": {}},
                "tokenizer_identity": digest(root / "tokenizer.json"),
            }
            manifest = {
                "step": 17000,
                "model_state": {"weight": {}},
                "identity": checkpoint_identity(metadata),
                "metadata_sha256": canonical_hash(metadata),
            }
            manifest["identity_sha256"] = canonical_hash(manifest["identity"])
            (root / "checkpoint-metadata.json").write_text(json.dumps(metadata))
            (root / "checkpoint-manifest.json").write_text(json.dumps(manifest))
            export = {
                "source_model_family": "yat_embedding_finetune",
                "source_checkpoint_step": 17000,
                "source_checkpoint_metadata_sha256": manifest["metadata_sha256"],
                "source_checkpoint_manifest_identity_sha256": manifest[
                    "identity_sha256"
                ],
                "tokenizer_identity": metadata["tokenizer_identity"],
                "tensors": 1,
                "sha256": digest(root / "model.safetensors"),
                "bytes": (root / "model.safetensors").stat().st_size,
                "artifacts_sha256": artifact_hashes(
                    root,
                    (
                        "model.safetensors",
                        "config.json",
                        "tokenizer.json",
                        "checkpoint-metadata.json",
                        "checkpoint-manifest.json",
                    ),
                ),
            }
            validate_export(root, export)
            bad = copy.deepcopy(export)
            bad["source_checkpoint_step"] += 1
            with self.assertRaises(ValueError):
                validate_export(root, bad)
            (root / "config.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                validate_export(root, export)

    def test_nan_inf_missing_metrics_fail(self):
        for bad in (float("nan"), float("inf"), None):
            report = case({}, "unused")
            report["metrics"]["pool"]["min_vector_cosine"] = bad
            with self.assertRaises(ValueError):
                validate_metrics(report)
        report = case({}, "unused")
        report["metrics"].pop("hidden")
        with self.assertRaises(ValueError):
            validate_metrics(report)

    def test_artifact_stale_receipt_and_scope_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"num_hidden_layers": 22, "hidden_size": 4}
            for name in ARTIFACTS:
                (root / name).write_text(
                    json.dumps(config) if name == "config.json" else name
                )
            metadata = {
                "model_family": "yat_embedding_finetune",
                "resolved_config": {"encoder": config},
            }
            manifest = {
                "step": 17000,
                "metadata_sha256": canonical_hash(metadata),
                "identity": checkpoint_identity(metadata),
            }
            manifest["identity_sha256"] = canonical_hash(manifest["identity"])
            source_export = {
                "source_checkpoint_step": 17000,
                "source_model_family": "yat_embedding_finetune",
                "tensors": 19,
                "sha256": digest(root / "model.safetensors"),
            }
            (root / "source-export.json").write_text(json.dumps(source_export))
            (root / "checkpoint-metadata.json").write_text(json.dumps(metadata))
            (root / "checkpoint-manifest.json").write_text(json.dumps(manifest))
            source_export.update(
                source_checkpoint_metadata_sha256=manifest["metadata_sha256"],
                source_checkpoint_manifest_identity_sha256=manifest["identity_sha256"],
                artifacts_sha256=artifact_hashes(
                    root,
                    (
                        "config.json",
                        "tokenizer.json",
                        "checkpoint-metadata.json",
                        "checkpoint-manifest.json",
                    ),
                ),
            )
            (root / "source-export.json").write_text(json.dumps(source_export))
            hashes = artifact_hashes(root)
            conversion = {
                "format": "flaxchat-yat-pytorch-v2",
                "source_model_family": "yat_embedding_finetune",
                "source_checkpoint_step": 17000,
                "tensor_count": 19,
                "source_export_sha256": digest(root / "source-export.json"),
                "artifacts_sha256": hashes,
                "source_weights_sha256": hashes["model.safetensors"],
                "torch_weights_sha256": hashes["model.safetensors"],
                "tokenizer_sha256": hashes["tokenizer.json"],
            }
            (root / "conversion.json").write_text(json.dumps(conversion))
            sha = digest(root / "conversion.json")
            report = {
                "format": VERSION,
                "artifacts_sha256": hashes,
                "conversion_sha256": sha,
                "cases": [
                    case(hashes, sha, length, batch)
                    for length in (128, 256, 512)
                    for batch in (1, 8)
                ],
            }
            validate_release(root, conversion, report)
            for key, value in (
                ("numeric_environment", {"XLA_USE_BF16": "1"}),
                ("effective_jax_config", {"jax_enable_x64": True}),
                ("interpreter", {"version": "different"}),
                ("deployment_receipts_sha256", {}),
            ):
                changed = copy.deepcopy(report)
                changed["cases"][0]["runs"]["jax"]["runtime"][key] = value
                with self.assertRaises(ValueError):
                    validate_release(root, conversion, changed)
            for name in ARTIFACTS:
                original = (root / name).read_bytes()
                (root / name).write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    validate_release(root, conversion, report)
                (root / name).write_bytes(original)
            old = copy.deepcopy(report)
            old["conversion_sha256"] = "stale"
            with self.assertRaises(ValueError):
                validate_release(root, conversion, old)
            old = copy.deepcopy(report)
            old["cases"].pop()
            with self.assertRaises(ValueError):
                validate_release(root, conversion, old)
            old = copy.deepcopy(report)
            old["cases"][0]["runs"]["jax"]["inputs_sha256"] = "different"
            with self.assertRaises(ValueError):
                validate_release(root, conversion, old)


if __name__ == "__main__":
    unittest.main()
