import json

import pytest

from benchmarks.compare import compare, load_record


def _record(framework="flaxchat", throughput=100.0):
    return {
        "framework": framework, "source_revision": "abc123",
        "hardware": "TPU v5e", "device_count": 8, "precision": "bfloat16",
        "model_parameters": 90_000_000, "sequence_length": 1024,
        "global_batch_size": 256, "steady_tokens_per_second": throughput,
        "compile_seconds": 12.0, "peak_memory_bytes": 1024,
        "validation_metric": "loss", "validation_value": 2.5,
    }


def test_compare_requires_identical_controls():
    changed = _record("MaxText")
    changed["global_batch_size"] = 128
    with pytest.raises(ValueError, match="controls differ"):
        compare([_record(), changed])


def test_compare_reports_normalized_throughput(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_record()))
    first = load_record(path)
    result = compare([first, _record("nanochat", 125.0)])
    assert result["results"][1]["throughput_relative_to_first"] == 1.25


def test_record_schema_fails_closed(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="missing benchmark fields"):
        load_record(path)
