import pytest

from benchmarks.training_scaling import add_scaling_efficiency, validate_device_counts


def test_device_counts_are_sorted_unique_and_bounded():
    assert validate_device_counts([8, 1, 4, 4], 8) == [1, 4, 8]
    with pytest.raises(ValueError, match="one-device baseline"):
        validate_device_counts([2, 4], 8)
    with pytest.raises(ValueError, match="between 1 and 8"):
        validate_device_counts([1, 16], 8)


def test_strong_scaling_efficiency_uses_one_device_baseline():
    measurements = [
        {"device_count": 1, "steady_tokens_per_second": 100.0},
        {"device_count": 2, "steady_tokens_per_second": 180.0},
        {"device_count": 4, "steady_tokens_per_second": 320.0},
    ]
    result = add_scaling_efficiency(measurements)
    assert result[0]["scaling_efficiency"] == 1.0
    assert result[1]["scaling_efficiency"] == 0.9
    assert result[2]["scaling_efficiency"] == 0.8


def test_efficiency_rejects_nonbaseline_first_record():
    with pytest.raises(ValueError, match="one-device baseline"):
        add_scaling_efficiency([
            {"device_count": 2, "steady_tokens_per_second": 100.0}
        ])
