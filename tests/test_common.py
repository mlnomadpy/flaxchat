"""
Tests for common utilities.
"""

from unittest.mock import patch

import pytest
from flaxchat.common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    _initialize_runtime, _is_multi_process_environment, compute_init,
    get_base_dir, get_mesh, get_peak_flops, setup_mesh, DummyWandb,
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
    def test_setup_mesh_registers_global_mesh(self):
        mesh = setup_mesh()
        assert get_mesh() is mesh

    def test_runtime_initializes_before_backend_discovery(self, monkeypatch):
        events = []
        monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "coordinator:1234")
        monkeypatch.delenv("FLAXCHAT_DTYPE", raising=False)
        with (
            patch("flaxchat.common.jax.distributed.is_initialized", return_value=False),
            patch(
                "flaxchat.common.jax.distributed.initialize",
                side_effect=lambda: events.append("initialize"),
            ),
            patch(
                "flaxchat.common.jax.default_backend",
                side_effect=lambda: events.append("default_backend") or "tpu",
            ),
        ):
            dtype, _ = _initialize_runtime()
        assert dtype == jnp.bfloat16
        assert events[0] == "initialize"

    @pytest.mark.parametrize(
        "environment",
        [
            {"JAX_COORDINATOR_ADDRESS": "coordinator:1234"},
            {"JAX_PROCESS_COUNT": "2"},
            {"SLURM_NTASKS": "4"},
            {"OMPI_COMM_WORLD_SIZE": "8"},
            {"PMI_SIZE": "2"},
            {"TPU_WORKER_HOSTNAMES": "worker-0,worker-1"},
        ],
    )
    def test_multi_process_launchers_are_detected(self, environment):
        assert _is_multi_process_environment(environment)

    @pytest.mark.parametrize(
        "environment",
        [
            {},
            {"CLOUD_TPU_TASK_ID": "0"},
            {"TPU_WORKER_ID": "0"},
            {"JAX_PROCESS_COUNT": "1", "JAX_PROCESS_INDEX": "0"},
            {"TPU_WORKER_HOSTNAMES": "worker-0"},
            {"SLURM_NTASKS": "not-an-integer"},
        ],
    )
    def test_single_process_markers_are_not_distributed(self, environment):
        assert not _is_multi_process_environment(environment)

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
        monkeypatch.setenv("JAX_PROCESS_COUNT", "2")
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

    def test_kaggle_single_host_does_not_initialize_distributed(self, monkeypatch):
        monkeypatch.delenv("JAX_COORDINATOR_ADDRESS", raising=False)
        for key in ("JAX_PROCESS_COUNT", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("CLOUD_TPU_TASK_ID", "0")
        monkeypatch.setenv("TPU_WORKER_ID", "0")
        monkeypatch.setenv("TPU_WORKER_HOSTNAMES", "localhost")
        with (
            patch("flaxchat.common.jax.distributed.is_initialized", return_value=False),
            patch("flaxchat.common.jax.distributed.initialize") as initialize,
            patch("flaxchat.common.jax.process_count", return_value=1),
            patch("flaxchat.common.jax.device_count", return_value=8),
            patch("flaxchat.common.jax.local_device_count", return_value=8),
            patch("flaxchat.common.jax.default_backend", return_value="tpu"),
            patch("flaxchat.common.setup_mesh", return_value="mesh"),
        ):
            assert compute_init() == "mesh"
        initialize.assert_not_called()
