"""
Tests for common utilities.
"""

from unittest.mock import patch

import pytest
from flaxchat.common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    compute_init, get_base_dir, get_peak_flops, DummyWandb,
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


class TestComputeInit:
    def test_distributed_initialization_precedes_topology_queries(self, monkeypatch):
        events = []
        monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "coordinator:1234")
        with (
            patch("flaxchat.common.jax.distributed.is_initialized", return_value=False),
            patch(
                "flaxchat.common.jax.distributed.initialize",
                side_effect=lambda: events.append("initialize"),
            ),
            patch(
                "flaxchat.common.jax.process_count",
                side_effect=lambda: events.append("process_count") or 2,
            ),
            patch("flaxchat.common.jax.device_count", return_value=8),
            patch("flaxchat.common.jax.local_device_count", return_value=4),
            patch("flaxchat.common.jax.default_backend", return_value="tpu"),
            patch("flaxchat.common.setup_mesh", return_value="mesh"),
        ):
            assert compute_init() == "mesh"
        assert events[0] == "initialize"

    def test_initialized_launcher_is_not_initialized_twice(self, monkeypatch):
        monkeypatch.setenv("TPU_WORKER_ID", "0")
        with (
            patch("flaxchat.common.jax.distributed.is_initialized", return_value=True),
            patch("flaxchat.common.jax.distributed.initialize") as initialize,
            patch("flaxchat.common.jax.process_count", return_value=2),
            patch("flaxchat.common.jax.device_count", return_value=8),
            patch("flaxchat.common.jax.local_device_count", return_value=4),
            patch("flaxchat.common.jax.default_backend", return_value="tpu"),
            patch("flaxchat.common.setup_mesh", return_value="mesh"),
        ):
            compute_init()
        initialize.assert_not_called()
