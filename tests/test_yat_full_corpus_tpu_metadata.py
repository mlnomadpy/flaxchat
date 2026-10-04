"""Frozen physical integration admission and literal token batching; no models."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_yat_full_corpus_tpu as evaluator
from scripts.release_contract import (
    artifact_hashes,
    canonical_hash,
    checkpoint_identity,
    digest,
)
from tests.test_production_parent_tpu_metadata import export_fixture


def fixture(tmp_path):
    parent = tmp_path / "parent"
    export_fixture(parent)
    config = {
        "yat_bias": 1,
        "yat_epsilon": 0.01,
        "yat_alpha_trainable": True,
        "ffn_type": "yat_glu",
        "attention_score": "yat_softmax",
        "max_position_embeddings": 8192,
    }
    (parent / "config.json").write_text(json.dumps(config))
    metadata = json.loads((parent / "checkpoint-metadata.json").read_text())
    metadata["resolved_config"]["encoder"] = config
    (parent / "checkpoint-metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((parent / "checkpoint-manifest.json").read_text())
    manifest.update(
        metadata_sha256=canonical_hash(metadata), identity=checkpoint_identity(metadata)
    )
    manifest["identity_sha256"] = canonical_hash(manifest["identity"])
    (parent / "checkpoint-manifest.json").write_text(json.dumps(manifest))
    export = json.loads((parent / "export.json").read_text())
    export.update(
        source_checkpoint_metadata_sha256=manifest["metadata_sha256"],
        source_checkpoint_manifest_identity_sha256=manifest["identity_sha256"],
        artifacts_sha256=artifact_hashes(
            parent, evaluator.parent_preflight.__globals__["FILES"][:-1]
        ),
    )
    (parent / "export.json").write_text(json.dumps(export))
    contract = {
        "format": "flaxchat-yat-full-corpus-tpu-v1",
        "purpose": "reporting_only",
        "corpus_scope": "full_source_corpus",
        "source_split": "dev",
        "source_identity": {"repo": "literal/source", "revision": "a" * 40},
        "source_files_sha256": evaluator.source_inventory(),
        "sequence_length": 128,
        "encoder_batch_size": 16,
        "document_block_size": 512,
        "query_block_size": 64,
        "top_k": 10,
        "max_documents": 1000,
        "max_queries": 100,
        "max_qrels": 100,
        "execution_seconds": 120,
        "max_retained_topk_bytes": 10000,
        "max_host_results_bytes": 10000,
        "max_host_working_bytes": 16 * 1024**2,
        "max_document_block_host_bytes": 1024**2,
        "prompts": {"query": "", "document": ""},
        "pooling": "mean-nonpadding-including-special-tokens",
        "normalization": "L2-FP32",
        "padding": "per-batch-power-of-two-32-to-sequence-length",
        "truncation": "right-token-id-truncation",
        "protocol": {
            "format": "flaxchat-full-corpus-metrics-v1",
            "cutoffs": [1, 10],
            "ndcg_gain": "linear",
            "tie_break": "descending-score-ascending-document-id",
            "relevant_threshold": 0,
        },
        "runtime": {
            "python": "literal-version",
            "packages": {name: "literal-version" for name in evaluator.PACKAGES},
        },
        "parent": {
            "directory": str(parent),
            "step": 14000,
            "metadata_sha256": canonical_hash(metadata),
            "manifest_sha256": canonical_hash(manifest),
            "leaves": 1,
            "artifacts_sha256": artifact_hashes(
                parent, evaluator.parent_preflight.__globals__["FILES"]
            ),
        },
        "files": {},
    }
    rows = {
        "corpus": [
            {"id": "d1", "text": "first doc"},
            {"id": "d2", "text": "second doc"},
        ],
        "queries": [{"id": "q1", "text": "query"}],
        "qrels": [{"query_id": "q1", "document_id": "d2", "relevance": 1}],
    }
    for role, items in rows.items():
        target = tmp_path / (role + ".jsonl")
        target.write_text("".join(json.dumps(row) + "\n" for row in items))
        contract["files"][role] = {
            "path": target.name,
            "bytes": target.stat().st_size,
            "sha256": digest(target),
        }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path, contract


def freeze(path, contract):
    path.write_text(json.dumps(contract))
    return digest(path)


def test_integration_import_does_not_initialize_numerical_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.evaluate_yat_full_corpus_tpu; assert not {'jax','numpy','torch','flax'} & set(sys.modules)",
        ],
        check=True,
        timeout=10,
    )


def test_real_snapshot_parent_and_source_authentication_are_metadata_only(tmp_path):
    path, _ = fixture(tmp_path)
    contract, receipt, docs, queries = evaluator.admit(path, digest(path))
    assert docs == ["d1", "d2"] and len(queries) == 1
    assert receipt["model_execution"] is False
    assert receipt["retained_topk_bytes"] == 80
    assert receipt["parent"]["expected_leaves"] == 1
    assert contract["parent"]["artifacts_sha256"]["model.safetensors"] == digest(
        tmp_path / "parent/model.safetensors"
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("sequence_length", True, "Finite"),
        ("top_k", 1, "cutoff"),
        ("max_queries", 10000, "Finite"),
        ("max_retained_topk_bytes", 79, "memory"),
        ("prompts", {"query": "prefix", "document": ""}, "no-prompt"),
        ("normalization", "none", "protocol"),
        ("purpose", "selection", "reporting-only"),
    ],
)
def test_frozen_geometry_protocol_memory_and_scope(tmp_path, field, value, match):
    path, contract = fixture(tmp_path)
    contract[field] = value
    with pytest.raises(ValueError, match=match):
        evaluator.admit(path, freeze(path, contract))


def test_snapshot_mutation_and_source_substitution_fail(tmp_path):
    path, contract = fixture(tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(corpus.read_bytes().replace(b"first doc", b"other doc"))
    with pytest.raises(ValueError, match="SHA256"):
        evaluator.admit(path, digest(path))
    contract["source_files_sha256"]["flaxchat/encoder.py"] = "0" * 64
    with pytest.raises(ValueError, match="source freeze"):
        evaluator.admit(path, freeze(path, contract))


def test_parent_hash_and_missing_qrels_cannot_be_self_certified(tmp_path):
    path, contract = fixture(tmp_path)
    contract["parent"]["artifacts_sha256"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="Independently pinned"):
        evaluator.admit(path, freeze(path, contract))


def test_cpu_backend_refused_before_model_import_or_output(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(default_backend=lambda: "cpu", process_count=lambda: 1),
    )
    with pytest.raises(RuntimeError, match="physical single-host TPU"):
        evaluator.run(path, digest(path), tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_literal_token_batch_preserves_bucketing_right_truncation_and_padding_counts():
    rows, exposure = evaluator.token_batch(
        [[1, 2, 0], list(range(1, 42))],
        batch_size=4,
        length=40,
        pad_id=0,
        vocab_size=100,
    )
    assert len(rows) == 4 and all(len(row) == 40 for row in rows)
    assert rows[0][:4] == [1, 2, 0, 0] and rows[1] == list(range(1, 41))
    assert rows[2] == [0] * 40
    assert exposure == {
        "texts": 2,
        "useful_tokens": 42,
        "processed_tokens": 160,
        "truncated_texts": 1,
    }
    short, _ = evaluator.token_batch(
        [[2]], batch_size=1, length=128, pad_id=0, vocab_size=100
    )
    assert len(short[0]) == 32


@pytest.mark.parametrize("ids", [[], [[]], [[-1]], [[100]], [[True]]])
def test_invalid_literal_token_ids_never_reach_the_model(ids):
    with pytest.raises(ValueError):
        evaluator.token_batch(ids, batch_size=1, length=128, pad_id=0, vocab_size=100)


def test_runtime_pins_checked_before_model_construction(tmp_path, monkeypatch):
    _, contract = fixture(tmp_path)
    monkeypatch.setattr(evaluator.platform, "python_version", lambda: "literal-version")
    monkeypatch.setattr(
        evaluator.importlib.metadata, "version", lambda name: "different-version"
    )
    with pytest.raises(ValueError, match="runtime differs"):
        evaluator.verify_runtime(contract)


def test_full_host_memory_bound_is_checked_during_metadata_growth(tmp_path):
    path, contract = fixture(tmp_path)
    contract['max_host_working_bytes'] = 1
    with pytest.raises(ValueError, match='Host working-memory'):
        evaluator.admit(path, freeze(path, contract))


def test_actual_linux_host_headroom_metadata_is_required(tmp_path):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemTotal: 1000 kB\nMemAvailable: 100 kB\n')
    assert evaluator.verify_host_headroom(102400, meminfo)['available_bytes'] == 102400
    with pytest.raises(ValueError, match='headroom insufficient'):
        evaluator.verify_host_headroom(102401, meminfo)
    meminfo.write_text('MemTotal: 1000 kB\n')
    with pytest.raises(ValueError, match='headroom insufficient'):
        evaluator.verify_host_headroom(1, meminfo)


def test_document_blocks_bound_text_memory_even_below_row_limit():
    rows = [{'id': 'a', 'text': 'x'*1000}, {'id': 'b', 'text': 'y'*1000}, {'id': 'c', 'text': 'z'}]
    limit = evaluator.snapshot_row_host_bytes(rows[0]) + 1
    batches = list(evaluator.bounded_document_batches(iter(rows), maximum_rows=100, maximum_host_bytes=limit))
    assert [[row['id'] for row in batch] for batch in batches] == [['a'], ['b'], ['c']]
    with pytest.raises(ValueError, match='Document text exceeds'):
        list(evaluator.bounded_document_batches(iter(rows), maximum_rows=100, maximum_host_bytes=1))
