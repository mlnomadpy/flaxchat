"""Metadata and cursor contract only; no model is allocated or run."""

import numpy as np
import pytest

from scripts.train_encoder_contrastive import (
    _same_path, checked_resume_cursor, expected_checkpoint_identity, load_parent_for_preflight,
    schedule, source_identity,
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
    with pytest.raises(ValueError):
        schedule(2e-5, 5001, 10, .1)


def test_saved_parent_metadata_matches_committed_manifest():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "artifacts/yat-mmbert-base-mlm-48236"
    parent = load_parent_for_preflight(root / "checkpoint-metadata.json",
                                       root / "checkpoint-manifest.json")
    assert parent["model_family"] == "modernbert"
    assert parent["resolved_config"]["encoder"]["ffn_type"] == "yat_glu"
