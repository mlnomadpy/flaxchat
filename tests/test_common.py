"""
Tests for common utilities.
"""

import pytest
from flaxchat.common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    get_base_dir, get_peak_flops, DummyWandb,
)
import jax.numpy as jnp


class TestComputeDtype:
    def test_dtype_is_valid(self):
        assert COMPUTE_DTYPE in (jnp.bfloat16, jnp.float16, jnp.float32)

    def test_reason_is_string(self):
        assert isinstance(COMPUTE_DTYPE_REASON, str)
        assert len(COMPUTE_DTYPE_REASON) > 0


class TestGetBaseDir:
    def test_returns_string(self):
        base_dir = get_base_dir()
        assert isinstance(base_dir, str)
        assert len(base_dir) > 0


class TestGetPeakFlops:
    def test_known_tpu(self):
        assert get_peak_flops("TPU v4") == 275e12
        assert get_peak_flops("TPU v5e") == 197e12
        assert get_peak_flops("TPU v5p") == 459e12

    def test_unknown_device(self):
        flops = get_peak_flops("unknown_device_xyz")
        assert flops == float('inf')

    def test_case_insensitive(self):
        assert get_peak_flops("tpu v4") == 275e12


class TestDummyWandb:
    def test_log_noop(self):
        wb = DummyWandb()
        wb.log({"loss": 1.0})  # should not raise
        wb.finish()  # should not raise
