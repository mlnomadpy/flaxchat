"""Real candidate→tokens→admission interfaces on injected raw data, no model."""

import json
import shutil
from pathlib import Path
import tempfile
import unittest
import numpy as np
from flaxchat.encoder_data import file_hash
from flaxchat.embedding_quality import load_retrieval_dev, paired_dev_ids
from flaxchat.embedding_development_quarantine import load_candidate_exclusions
from scripts.prepare_embedding_retrieval_dev import prepare
from tests.metadata import test_embedding_stage as stage_fixtures
from tests.test_development_quarantine_metadata import candidate


def fixture(root):
    import tokenizers

    args = stage_fixtures.StageAdmissionTests().fixture(root)
    parent = Path(args.parent_public)
    config = json.loads((parent / "config.json").read_text())
    vocab = {
        word: index
        for index, word in enumerate(
            ["[PAD]", "[UNK]", "en", "ar", "query", "positive", "negative"]
            + [str(i) for i in range(9)]
        )
    }
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(parent / "tokenizer.json"))
    hashes = {
        name: file_hash(parent / name)
        for name in ("model.safetensors", "config.json", "tokenizer.json")
    }
    Path(args.parent_manifest).write_text(json.dumps(hashes))
    candidates = candidate(root)
    index = load_candidate_exclusions(candidates)
    data = Path(args.data[0].split("=", 1)[1])
    manifest_path = data / "manifest.json"
    data_manifest = json.loads(manifest_path.read_text())
    data_manifest.update(
        tokenizer_sha256=hashes["tokenizer.json"],
        source_identity={"repo": "fixture/clean", "revision": "a" * 40},
        development_quarantine={"identity": index.identity},
    )
    manifest_path.write_text(json.dumps(data_manifest))
    historical_data = root / "historical-data"
    shutil.copytree(data, historical_data)
    metadata_path = root / "historical-checkpoint-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {"resolved_config": {"data_manifests": {"pairs": file_hash(manifest_path)}}}
        )
    )
    identity = file_hash(metadata_path)
    exposure = root / "exposure.json"
    exposure.write_text(
        json.dumps(
            {
                "format": "flaxchat-known-contrastive-exposure-input-v1",
                "parent_files_sha256": hashes,
                "known_contrastive_inventory_complete": True,
                "expected_stage_identities": [identity],
                "stages": [
                    {
                        "stage_identity_sha256": identity,
                        "checkpoint_metadata": metadata_path.name,
                        "sources": [
                            {"name": "pairs", "directory": historical_data.name}
                        ],
                    }
                ],
            }
        )
    )
    output = root / "independent"
    names = list(json.loads((candidates / "exclusions.json").read_text())["sources"])
    manifest = prepare(
        candidates,
        names,
        exposure,
        Path(args.parent_manifest),
        parent / "tokenizer.json",
        parent / "config.json",
        output,
        query_length=4,
        document_length=4,
    )
    return args, data, output, config, hashes, manifest


class IndependentDevelopmentTests(unittest.TestCase):
    def test_production_rejects_valid_foreign_sts_not_in_parent_checked_candidate(self):
        from flaxchat.embedding_stage import prepare_stage
        from scripts.prepare_embedding_sts_dev import prepare as prepare_sts
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _data, output, _config, hashes, _manifest = fixture(root)
            parent = Path(args.parent_public)
            raw = root / 'foreign-sts.jsonl'
            raw.write_text(''.join(json.dumps({'sentence1': f'foreign q{i}', 'sentence2': f'foreign p{i}',
                'score': float(i), 'language': 'en'}) + '\n' for i in range(3)))
            folder = root / 'foreign-sts'
            prepare_sts(raw, file_hash(raw), parent / 'tokenizer.json', parent / 'config.json', folder,
                source_split='validation', length=4, source_repo='foreign/valid', source_revision='b' * 40)
            args.sts_dev = [f'semantic={folder}']
            args.retrieval_dev = [f'multilingual={output}']
            args.training_scope = 'production'
            args.dev_max_rows = 8
            with self.assertRaisesRegex(ValueError, 'actual authenticated candidate pair/score'):
                prepare_stage(args, device_count=4, process_count=1)

    def test_actual_preparation_and_full_shared_stage_admission(self):
        from flaxchat.embedding_stage import prepare_stage

        with tempfile.TemporaryDirectory() as directory:
            args, data, output, config, hashes, manifest = fixture(Path(directory))
            arrays, loaded, digest = load_retrieval_dev(
                output,
                config,
                hashes["tokenizer.json"],
                training_directories=[data],
                parent_hashes=hashes,
            )
            self.assertEqual(len(arrays["query_tokens"]), 8)
            self.assertEqual(set(arrays["languages"]), {"en", "ar"})
            self.assertEqual(loaded, manifest)
            self.assertEqual(digest, file_hash(output / "manifest.json"))
            self.assertIn(
                "completeness of historical contrastive-stage lineage beyond declared inventory",
                manifest["parent_exposure"]["unresolved_exposure"],
            )
            args.retrieval_dev = [f"multilingual={output}"]
            args.dev_max_rows = 8
            stage = prepare_stage(args, device_count=4, process_count=1)
            self.assertEqual(len(stage["retrieval_data"]["multilingual"][1]), 8)
            receipt = stage["admission_receipt"]["independent_retrieval_development"][
                "multilingual"
            ]
            self.assertEqual(
                receipt["candidate_identity"], manifest["candidate_identity"]
            )
            self.assertFalse(stage["admission_receipt"]["physical_tpu_qualified"])
            args.training_scope = 'production'
            with self.assertRaisesRegex(ValueError, 'Production quality coverage'):
                prepare_stage(args, device_count=4, process_count=1)

    def test_changed_historical_raw_and_forged_complete_proof_rejected(self):
        for mode in ("raw", "proof", "missing-stage"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, data, output, config, hashes, manifest = fixture(Path(directory))
                if mode == "raw":
                    path = Path(directory) / "historical-data/train.jsonl"
                    path.write_text(path.read_text() + "{}\n")
                elif mode == "proof":
                    path = output / "parent-exposure-proof.json"
                    proof = json.loads(path.read_text())
                    proof["checked_stages"][next(iter(proof["checked_stages"]))][
                        "sources"
                    ]["pairs"]["checked_rows"] = 1
                    path.write_text(json.dumps(proof))
                    manifest["files"][path.name] = file_hash(path)
                    (output / "manifest.json").write_text(json.dumps(manifest))
                else:
                    path = output / "parent-exposure-input.json"
                    spec = json.loads(path.read_text())
                    spec["stages"] = []
                    path.write_text(json.dumps(spec))
                    manifest["files"][path.name] = file_hash(path)
                    (output / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_retrieval_dev(
                        output,
                        config,
                        hashes["tokenizer.json"],
                        training_directories=[data],
                        parent_hashes=hashes,
                    )

    def test_known_parent_exposure_and_current_training_quarantine_are_not_booleans(
        self,
    ):
        from flaxchat.embedding_dev_exposure import parent_exposure

        with tempfile.TemporaryDirectory() as directory:
            args, data, output, config, hashes, manifest = fixture(Path(directory))
            candidate_row = json.loads(
                (output / "raw.jsonl").read_text().splitlines()[0]
            )
            historical_data = Path(directory) / "historical-data"
            raw = historical_data / "train.jsonl"
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            rows[0] = candidate_row
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            training_manifest = json.loads(
                (historical_data / "manifest.json").read_text()
            )
            training_manifest["raw_files"]["train.jsonl"] = file_hash(raw)
            (historical_data / "manifest.json").write_text(
                json.dumps(training_manifest)
            )
            historical = Path(directory) / "historical-checkpoint-metadata.json"
            historical.write_text(
                json.dumps(
                    {
                        "resolved_config": {
                            "data_manifests": {
                                "pairs": file_hash(historical_data / "manifest.json")
                            }
                        }
                    }
                )
            )
            spec = json.loads((output / "parent-exposure-input.json").read_text())
            identity = file_hash(historical)
            spec["expected_stage_identities"] = [identity]
            spec["stages"][0]["stage_identity_sha256"] = identity
            path = Path(directory) / "exposed.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "exposed to known contrastive"):
                parent_exposure(
                    path, load_candidate_exclusions(output / "candidate"), hashes
                )
            # Restore the authenticated past before targeting the future train guard.
            shutil.rmtree(historical_data)
            shutil.copytree(data, historical_data)
            historical.write_text(
                json.dumps(
                    {
                        "resolved_config": {
                            "data_manifests": {
                                "pairs": file_hash(historical_data / "manifest.json")
                            }
                        }
                    }
                )
            )
            training_manifest = json.loads((data / "manifest.json").read_text())
            training_manifest["development_quarantine"] = {
                "parent_exposure_checked": True
            }
            (data / "manifest.json").write_text(json.dumps(training_manifest))
            with self.assertRaises(ValueError):
                load_retrieval_dev(
                    output,
                    config,
                    hashes["tokenizer.json"],
                    training_directories=[data],
                    parent_hashes=hashes,
                )

    def test_future_training_exposure_and_raw_token_disagreement_fail_closed(self):
        for mode in ("future-raw", "tokens", "task", "duplicate-source"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, data, output, config, hashes, manifest = fixture(Path(directory))
                if mode == "future-raw":
                    candidate_row = json.loads(
                        (output / "raw.jsonl").read_text().splitlines()[0]
                    )
                    path = data / "train.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0] = candidate_row
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    prepared_manifest = json.loads((data / "manifest.json").read_text())
                    prepared_manifest["raw_files"]["train.jsonl"] = file_hash(path)
                    (data / "manifest.json").write_text(json.dumps(prepared_manifest))
                elif mode == "tokens":
                    path = output / "query_tokens.npy"
                    values = np.load(path)
                    values[0, 0] = 15
                    np.save(path, values)
                    manifest["files"][path.name] = file_hash(path)
                    (output / "manifest.json").write_text(json.dumps(manifest))
                elif mode == "task":
                    manifest["task"] = "code"
                    (output / "manifest.json").write_text(json.dumps(manifest))
                else:
                    manifest["candidate_sources"] *= 2
                    (output / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_retrieval_dev(
                        output,
                        config,
                        hashes["tokenizer.json"],
                        training_directories=[data],
                        parent_hashes=hashes,
                    )

    def test_query_group_transitive_relevance_and_code_case(self):
        rows = [
            dict(
                query="q",
                positive="Foo",
                group="a",
                language="en",
                modalities={"positive": "code"},
            ),
            dict(
                query="q",
                positive="foo",
                group="b",
                language="en",
                modalities={"positive": "code"},
            ),
            dict(
                query="different",
                positive="bar",
                group="b",
                language="en",
                modalities={"positive": "code"},
            ),
        ]
        ids = paired_dev_ids(rows)
        self.assertEqual(len(set(ids["positive_group_ids"])), 1)
        self.assertNotEqual(ids["positive_text_ids"][0], ids["positive_text_ids"][1])
