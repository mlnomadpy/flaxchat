"""New-stage orchestration/data contracts only; no model or device execution."""

import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from scripts import run_mlm_checkpoint_stage as stage


def arguments(root):
    encoder = {
        "attention_score": "goat_input",
        "ffn_type": "yat_glu",
        "yat_bias": 1,
        "yat_epsilon": 0.01,
        "yat_alpha_trainable": True,
        "yat_alpha": 1.0,
        "yat_attention_alpha": 0.1,
        "compute_dtype": "bfloat16",
        "residual_dtype": "float32",
        "attention_backend": "xla",
        "yat_compute_mode": "bf16",
        "yat_ffn_compute_mode": "bf16_adaptive",
        "yat_attention_implementation": "centered_fp32_scores",
        "yat_attention_block_size": 64,
        "yat_global_attention_block_size": 64,
        "mlm_projection": "masked",
        "mlm_loss_backend": "xla_full",
        "mlm_vocab_tile": 1024,
        "loss_chunk_size": 128,
        "use_remat": True,
    }
    stage.write(root / "stage-encoder.json", encoder)
    return SimpleNamespace(
        root=root,
        parent_checkpoint="gs://bucket/parent/checkpoints",
        parent_step=100000,
        parent_metadata_sha256="a" * 64,
        parent_manifest_sha256="b" * 64,
        steps=500000,
        output="gs://bucket/new/checkpoints",
        max_seconds=10000,
        evidence_prefix="gs://bucket/new/evidence",
    )


def test_trained_initialization_and_exact_newstage_resume_are_separate(tmp_path):
    args = arguments(tmp_path)
    fresh = stage.training_argv(args, args.output, 4, save_every=4)
    resumed = stage.training_argv(args, args.output, 8, resume=True, save_every=4)
    assert fresh[fresh.index("--initialize-step") + 1] == "100000"
    assert "--resume" not in fresh and "--initialize-from-checkpoint" not in resumed
    assert (
        "--resume" in resumed
        and "--random-init" not in fresh
        and "--migrate-to-goat" not in fresh
    )
    assert "--local-gradient-accumulation" in fresh
    for option, value in [
        ("--attention-score", "goat_input"),
        ("--yat-epsilon", "0.01"),
        ("--residual-dtype", "float32"),
        ("--yat-compute-mode", "bf16"),
        ("--yat-ffn-compute-mode", "bf16_adaptive"),
        ("--batch-size", "64"),
        ("--accumulation-steps", "4"),
        ("--learning-rate", "0.0001"),
        ("--steps", "500000"),
    ]:
        assert (
            fresh[fresh.index(option) + 1]
            == resumed[resumed.index(option) + 1]
            == value
        )


def test_parent_pin_and_architecture_cannot_be_replaced_by_boolean(tmp_path):
    args = arguments(tmp_path)
    metadata = {
        "model_family": "modernbert",
        "resolved_config": {"encoder": stage.read(tmp_path / "stage-encoder.json")},
        "committed_receipt": {"metadata_sha256": "a" * 64, "manifest_sha256": "b" * 64},
    }
    with patch(
        "flaxchat.checkpoint_metadata.read_committed_metadata", return_value=metadata
    ):
        parent, receipt = stage.authenticated_parent(args)
    assert parent["resolved_config"]["encoder"]["attention_score"] == "goat_input"
    assert receipt["manifest_sha256"] == "b" * 64
    metadata["committed_receipt"] = {
        "metadata_sha256": "a" * 64,
        "manifest_sha256": "c" * 64,
    }
    with (
        patch(
            "flaxchat.checkpoint_metadata.read_committed_metadata",
            return_value=metadata,
        ),
        pytest.raises(ValueError, match="identity differs"),
    ):
        stage.authenticated_parent(args)


def test_bounded_selection_scans_train_once_excludes_exact_duplicates_and_balances(
    tmp_path,
):
    validation = np.array([[i, 0, 0] for i in range(1, 9)] + [[4, 0, 0]], np.int32)
    calls = []

    class Training:
        def __iter__(self):
            for row in [np.array([1, 0, 0]), np.array([8, 0, 0])]:
                calls.append(row[0])
                yield row

    docs = tmp_path / "documents.jsonl"
    docs.write_text(
        json.dumps({"first_row": 0, "end_row": 4, "language_script": "en"})
        + "\n"
        + json.dumps({"first_row": 4, "end_row": 9, "language_script": "ar"})
        + "\n"
    )
    selected, languages, excluded = stage.balanced_selection(
        Training(), validation, docs, pad=0, seed=2026, count=4
    )
    assert calls == [1, 8] and excluded == 2 and languages == ["ar", "en", "ar", "en"]
    assert not {0, 7} & set(selected.tolist())
    assert len({stage.row_digest(validation[i], 0) for i in selected}) == 4
    with pytest.raises(ValueError, match="Insufficient"):
        stage.balanced_selection([], validation, docs, pad=0, seed=2026, count=9)


def test_gate_rejects_nonfinite_wrong_masks_or_regressed_parent_comparison():
    baseline = {
        "backend": "tpu",
        "devices": 8,
        "processes": 1,
        "masked_tokens": 1234,
        "evaluation_batch_size": 16,
        "evaluated_rows": 512,
        "selection_sha256": "a" * 64,
        "selected_rows_sha256": "b" * 64,
        "protocol": "fixed",
        "masked_token_loss": 2.0,
    }
    stage.validate_report(baseline, baseline)
    for change in [
        {"masked_token_loss": float("nan")},
        {"masked_token_loss": 2.2},
        {"masked_tokens": 1233},
        {"backend": "cpu"},
        {"evaluated_rows": 8},
        {"masked_token_loss": True},
    ]:
        with pytest.raises(ValueError):
            stage.validate_report({**baseline, **change}, baseline)


def test_required_updates_applied_contiguous_finite(tmp_path):
    log = tmp_path / "train.log"
    events = [
        dict(
            event="train_step",
            step=i,
            updated=True,
            loss=2.0,
            masked_tokens=10,
            gradient_norm_before_clip=1.0,
        )
        for i in range(5, 9)
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in events))
    assert len(stage.validate_updates(log, 4, 8)) == 4
    events[-1]["updated"] = False
    log.write_text("".join(json.dumps(row) + "\n" for row in events))
    with pytest.raises(ValueError, match="optimizer update"):
        stage.validate_updates(log, 4, 8)


def test_lease_fit_stops_cleanly_or_uses_checkpoint_multiple():
    assert stage.segment_boundary(0, 500000, 1700) is None
    assert stage.segment_boundary(0, 500000, 10000) == 250
    assert stage.segment_boundary(250, 500000, 30000, 600, 250) == 5250
    assert stage.segment_boundary(250, 500000, 1000, 600, 250) is None
    boundary = stage.segment_boundary(250, 500000, 4000, 600, 250)
    assert boundary > 250 and boundary % 250 == 0


def test_owned_deadline_child_is_reaped_without_claiming_latest_save(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        stage.execute(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "sleep.log",
            time.monotonic() + 0.05,
            graceful_seconds=1,
        )
    assert time.monotonic() - started < 5


def test_physical_test_optin_is_explicit_and_does_not_change_global_environment(
    monkeypatch,
):
    monkeypatch.delenv("FLAXCHAT_PHYSICAL_TPU", raising=False)
    assert stage.physical_test_environment()["FLAXCHAT_PHYSICAL_TPU"] == "1"
    import os

    assert "FLAXCHAT_PHYSICAL_TPU" not in os.environ
    stage.validate_device_inventory("tpu", ["TPU v5 lite"] * 8, 1)
    for backend, kinds, processes in [
        ("cpu", ["cpu"] * 8, 1),
        ("tpu", ["TPU v6e"] * 8, 1),
        ("tpu", ["TPU v5 lite"] * 4, 1),
        ("tpu", ["TPU v5 lite"] * 8, 2),
    ]:
        with pytest.raises(RuntimeError, match="physical eight-device v5e"):
            stage.validate_device_inventory(backend, kinds, processes)


def test_cache_identity_does_not_rehash_entire_corpus_per_eval(tmp_path):
    args = arguments(tmp_path)
    cache = tmp_path / "new-stage-evaluation"
    cache.mkdir()
    corpus = tmp_path / "corpus"
    (corpus / "train").mkdir(parents=True)
    (corpus / "validation").mkdir()
    for relative in [
        "COMPLETE.json",
        "recipe.json",
        "train/manifest.json",
        "validation/manifest.json",
    ]:
        (corpus / relative).write_text("{}")
    np.save(cache / "inputs.npy", np.ones((2, 2), np.int32))
    selection = {
        "files": {"inputs.npy": stage.sha(cache / "inputs.npy")},
        "corpus_complete_sha256": stage.sha(corpus / "COMPLETE.json"),
        "recipe_sha256": stage.sha(corpus / "recipe.json"),
        "train_manifest_sha256": stage.sha(corpus / "train/manifest.json"),
        "validation_manifest_sha256": stage.sha(corpus / "validation/manifest.json"),
    }
    stage.write(cache / "selection.json", selection)
    assert stage.authenticate_selection(args) == selection
    (cache / "inputs.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="heldout token/mask"):
        stage.authenticate_selection(args)
