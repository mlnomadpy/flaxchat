from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.matched.common import load_batches, make_record, validate_hardware
from benchmarks.matched.prepare_data import encode_documents, make_sequences
from benchmarks.matched.preflight import within_budget
from benchmarks.matched.nanochat_runner import token_tensor


def test_framework_neutral_byte_encoding_is_deterministic():
    tokens = encode_documents([{"text": "A"}, {"text": "é"}], 7)
    assert tokens.tolist() == [1, 68, 2, 1, 198, 172, 2]


def test_sequences_use_next_token_targets(monkeypatch):
    monkeypatch.setattr("benchmarks.matched.prepare_data.SEQUENCE_LENGTH", 3)
    inputs, targets = make_sequences(np.arange(8, dtype=np.int32), 2)
    assert inputs.tolist() == [[0, 1, 2], [4, 5, 6]]
    assert targets.tolist() == [[1, 2, 3], [5, 6, 7]]


def test_batch_loader_fails_closed_on_shape_drift(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, train_inputs=np.ones((1, 1, 1), dtype=np.int32))
    with pytest.raises(ValueError, match="shapes differ"):
        load_batches(path)


def test_record_carries_shared_data_identity():
    record = make_record(
        framework="flaxchat",
        source_revision="a" * 40,
        model_parameters=600_000,
        steady_tokens_per_second=100.0,
        compile_seconds=1.0,
        peak_memory_bytes=100,
        checkpoint_seconds=1.0,
        validation_value=2.0,
        protocol_sha256="b" * 64,
        data_sha256="c" * 64,
        software={"jax": "1"},
        limitations="fixture",
    )
    assert record["data_sha256"] == "c" * 64
    assert record["model_flops_utilization"] > 0


def test_hardware_guard_rejects_unmatched_accelerators():
    validate_hardware("Tesla P100-PCIE-16GB")
    with pytest.raises(RuntimeError, match="requires Tesla P100"):
        validate_hardware("Tesla T4")


def test_parameter_budget_is_inclusive_at_five_percent():
    assert within_budget(570_000)
    assert within_budget(630_000)
    assert not within_budget(569_999)


def test_nanochat_token_batches_are_promoted_to_int64():
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    converted = token_tensor(values, torch.device("cpu"))
    assert converted.dtype == torch.int64
    assert converted.tolist() == values.tolist()
