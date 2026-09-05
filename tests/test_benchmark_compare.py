import json

import pytest

from benchmarks.compare import compare, load_record, protocol_sha256


def _record(framework="flaxchat", throughput=100.0):
    return {
        "framework": framework, "source_revision": "b" * 40,
        "hardware": "TPU v5e", "device_count": 8, "precision": "bfloat16",
        "model_parameters": 90_000_000, "target_model_parameters": 90_000_000,
        "parameter_tolerance_fraction": 0.05, "sequence_length": 1024,
        "global_batch_size": 256, "steady_tokens_per_second": throughput,
        "compile_seconds": 12.0, "peak_memory_bytes": 1024,
        "model_flops_utilization": 0.3, "checkpoint_seconds": 1.5,
        "scaling_efficiency": 0.9, "limitations": "Synthetic fixture only.",
        "validation_metric": "loss", "validation_value": 2.5,
        "protocol_sha256": "a" * 64,
        "data_sha256": "d" * 64,
        "dataset": "roneneldan/TinyStories", "dataset_revision": "c" * 40,
        "optimizer": "adamw", "seed": 42, "warmup_steps": 2,
        "measured_steps": 10,
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


def test_record_rejects_placeholder_revision_and_nonfinite_values(tmp_path):
    record = _record()
    record["source_revision"] = "REQUIRED"
    path = tmp_path / "placeholder.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="exact 40-character git SHA"):
        load_record(path)

    record["source_revision"] = "b" * 40
    record["validation_value"] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="validation_value must be finite"):
        load_record(path)


def test_record_rejects_boolean_and_fractional_integer_fields(tmp_path):
    record = _record()
    record["device_count"] = True
    path = tmp_path / "bad-integer.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="device_count must be a positive integer"):
        load_record(path)

    record["device_count"] = 1.5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="device_count must be a positive integer"):
        load_record(path)

def test_comparison_rejects_duplicate_frameworks():
    with pytest.raises(ValueError, match="one record per framework"):
        compare([_record(), _record()])


def test_superlinear_efficiency_is_valid(tmp_path):
    record = _record()
    record["scaling_efficiency"] = 1.05
    path = tmp_path / "superlinear.json"
    path.write_text(json.dumps(record))
    assert load_record(path)["scaling_efficiency"] == 1.05


def test_parameter_counts_may_differ_inside_declared_tolerance(tmp_path):
    first = _record()
    second = _record("MaxText")
    second["model_parameters"] = 94_000_000
    result = compare([first, second])
    assert result["results"][1]["parameters_relative_to_target"] == pytest.approx(94 / 90)

    second["model_parameters"] = 95_000_000
    path = tmp_path / "outside-tolerance.json"
    path.write_text(json.dumps(second))
    with pytest.raises(ValueError, match="exceeding the 5.00% tolerance"):
        load_record(path)


def test_protocol_hash_uses_exact_file_bytes(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_bytes(b"seed: 42\n")
    assert protocol_sha256(protocol) == "1094bf3e273cc9ac0e931e1904eca62f06ecad395a1ceb093092a66c0e839a9c"
