"""Metadata and cursor contract only; no model is allocated or run."""

import numpy as np
import pytest

from scripts.train_encoder_contrastive import (
    PARENT_METADATA_SHA256, _same_path, checked_resume_cursor,
    contrastive_parent_receipt, expected_checkpoint_identity,
    load_contrastive_parent_for_preflight, load_parent_for_preflight,
    metadata_digest, schedule, source_identity,
)


def test_stage_identity_pins_parent_data_tokenizer_and_source():
    recipe = {"parent": {"step": 48236, "metadata_sha256": "parent"}, "steps": 500}
    identity = expected_checkpoint_identity(recipe, "tokenizer", "data", "source")
    assert identity["resolved_config"]["parent"]["step"] == 48236
    assert identity["tokenizer"] == "tokenizer"
    assert identity["data_manifest"] == "data"
    assert identity["source_python_sha256"] == "source"
    assert identity != expected_checkpoint_identity(recipe, "tokenizer", "different-data", "source")
    assert identity != expected_checkpoint_identity(recipe | {"steps": 501}, "tokenizer", "data", "source")
    assert len(source_identity()) == 64


def test_exact_resume_cursor_rejects_skipped_or_out_of_horizon_updates():
    state = dict(completed_steps=np.array([25], np.int32),
                 optimizer_updates=np.array([25], np.int32))
    assert checked_resume_cursor(state, steps=500, stop=50) == 25
    with pytest.raises(ValueError):
        checked_resume_cursor(state | {"optimizer_updates": np.array([24], np.int32)}, steps=500, stop=50)
    with pytest.raises(ValueError):
        checked_resume_cursor(state, steps=500, stop=24)
    with pytest.raises(ValueError):
        checked_resume_cursor(state, steps=20, stop=20)


def test_parent_output_is_distinct_and_schedule_is_bounded():
    assert _same_path("gs://bucket/parent/", "gs://bucket/parent")
    assert not _same_path("gs://bucket/parent", "gs://bucket/contrastive")
    schedule(2e-5, 25, 10, .1)
    schedule(2e-5, 5001, 10, .1)
    with pytest.raises(ValueError):
        schedule(2e-5, 100001, 10, .1)


def test_new_stage_pins_contrastive_parent_without_claiming_exact_resume():
    encoder = {"ffn_type": "yat_glu"}
    metadata = {
        "model_family": "modernbert_contrastive_encoder", "step": 500,
        "tokenizer_identity": "tokenizer", "data_manifest_identity": "old-data",
        "resolved_config": {
            "encoder": encoder, "data_manifest_sha256": "old-data",
            "parent": {"step": 48236, "metadata_sha256": PARENT_METADATA_SHA256},
        },
    }
    receipt = contrastive_parent_receipt(metadata, "gs://bucket/old", 500,
                                          "tokenizer", encoder)
    assert receipt["step"] == 500
    assert receipt["data_manifest_sha256"] == "old-data"
    assert "weights_only" in receipt["policy"]
    with pytest.raises(ValueError):
        contrastive_parent_receipt(metadata, "gs://bucket/old", 500,
                                   "different-tokenizer", encoder)
    with pytest.raises(ValueError):
        contrastive_parent_receipt(metadata | {"data_manifest_identity": "changed"},
                                   "gs://bucket/old", 500, "tokenizer", encoder)


def test_offline_contrastive_parent_requires_committed_manifest(tmp_path):
    import json

    metadata = {"model_family": "modernbert_contrastive_encoder", "tokenizer_identity": "tok"}
    metadata_file = tmp_path / "metadata.json"
    manifest_file = tmp_path / "manifest.json"
    metadata_file.write_text(json.dumps(metadata))
    manifest_file.write_text(json.dumps({"step": 500, "metadata_sha256": metadata_digest(metadata)}))
    assert load_contrastive_parent_for_preflight(metadata_file, manifest_file, 500)["step"] == 500
    with pytest.raises(ValueError):
        load_contrastive_parent_for_preflight(metadata_file, manifest_file, 499)
    metadata_file.write_text(json.dumps(metadata | {"tokenizer_identity": "changed"}))
    with pytest.raises(ValueError):
        load_contrastive_parent_for_preflight(metadata_file, manifest_file, 500)


def test_saved_parent_metadata_matches_committed_manifest():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "artifacts/yat-mmbert-base-mlm-48236"
    parent = load_parent_for_preflight(root / "checkpoint-metadata.json",
                                       root / "checkpoint-manifest.json")
    assert parent["model_family"] == "modernbert"
    assert parent["resolved_config"]["encoder"]["ffn_type"] == "yat_glu"
