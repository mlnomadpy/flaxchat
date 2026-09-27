import numpy as np
import pytest
from scripts.validate_encoder_reference import compare_arrays


def test_reference_gate_checks_every_element_not_only_global_norm():
    expected = np.ones(10000)
    candidate = expected.copy()
    candidate[-1] += 0.1
    report = compare_arrays(expected, candidate, atol=0.002, rtol=0.001)
    assert not report["passed"] and report["relative_l2"] < 0.002
    assert compare_arrays(expected, expected + 0.001, atol=0.002, rtol=0.001)["passed"]


@pytest.mark.parametrize(
    "left,right",
    [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])],
)
def test_reference_gate_rejects_missing_and_nonfinite(left, right):
    with pytest.raises(ValueError):
        compare_arrays(left, right, atol=0.002, rtol=0.001)
