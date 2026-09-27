"""The edge gate must reject null-space drift even below general tolerances."""
import json

import numpy as np
import pytest

from scripts.yat_attention_edge_suite import check_values


def test_exact_null_gate_rejects_small_cancellation_residual():
    expected = [np.zeros((2,)) for _ in range(5)]
    actual = [x.copy() for x in expected]
    actual[1][0] = 1e-10
    ordinary = check_values(actual, expected, 1., kind="random")
    invariant = check_values(actual, expected, 1., kind="segment_keys")
    assert ordinary["q"]["passed"]
    assert not invariant["q"]["passed"]
    assert not invariant["q"]["null_exact"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_fail_and_report_remains_serializable(bad):
    expected = [np.ones((2,)) for _ in range(5)]
    actual = [x.copy() for x in expected]
    actual[3][0] = bad
    checks = check_values(actual, expected, 1., kind="random")
    assert not checks["v"]["passed"]
    assert checks["v"]["error"] is None
    json.dumps(checks, allow_nan=False)


def test_shape_mismatch_cannot_pass_by_numpy_broadcasting():
    expected = [np.ones((2,)) for _ in range(5)]
    actual = [x.copy() for x in expected]
    actual[0] = np.ones((1, 2))
    with pytest.raises(ValueError, match="Unexpected output shape"):
        check_values(actual, expected, 1., kind="random")
