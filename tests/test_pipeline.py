"""Integration coverage for the canonical reproducible training pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from flaxchat.pipeline import PipelineConfig, load_fixture_stories, run_pipeline


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"


def test_pipeline_config_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least 24"):
        PipelineConfig(embedding_dim=16)
    with pytest.raises(ValueError, match="divisible"):
        PipelineConfig(embedding_dim=25, heads=2)
    with pytest.raises(ValueError, match="every training stage"):
        PipelineConfig(rl_steps=0)


def test_fixture_split_is_deterministic():
    first = load_fixture_stories(FIXTURE)
    second = load_fixture_stories(FIXTURE)
    assert first == second
    assert first[0] and first[1]
    expected = [line for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(first[0]) + len(first[1]) == len(expected)


@pytest.mark.integration
def test_complete_pipeline_emits_restorable_artifacts(tmp_path):
    train, validation = load_fixture_stories(FIXTURE)
    config = PipelineConfig(
        sequence_length=32,
        embedding_dim=24,
        heads=2,
        vocab_size=320,
        batch_size=2,
        pretrain_steps=2,
        sft_steps=1,
        rl_steps=1,
        max_new_tokens=1,
    )
    manifest = run_pipeline(train, validation, tmp_path, config)

    assert manifest["status"] == "complete"
    assert manifest["stages"] == [
        "tokenizer", "pretrain", "sft", "rl", "eval", "inference"
    ]
    assert manifest["hardware"]["device_count"] >= 1
    assert len(manifest["protocol_sha256"]) == 64
    assert len(manifest["checkpoint_manifest_sha256"]) == 64
    assert len(manifest["sample"]["token_ids"]) > 1
    for name in ("pretrain_loss", "sft_loss", "rl_loss"):
        assert manifest["metrics"][name]
        assert all(math.isfinite(value) for value in manifest["metrics"][name])
    assert math.isfinite(manifest["metrics"]["validation_loss"])

    saved = json.loads((tmp_path / "run_manifest.json").read_text())
    assert saved["protocol_sha256"] == manifest["protocol_sha256"]
    assert (tmp_path / saved["artifacts"]["tokenizer"]).is_file()
    checkpoint = tmp_path / saved["artifacts"]["checkpoint"]
    assert (checkpoint / "4" / "manifest" / "metadata").is_file()
