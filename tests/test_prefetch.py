"""
Tests for flaxchat/prefetch.py — background data prefetcher.

Uses CPU-only JAX (no TPU needed) with a trivial mesh and sharding.
"""

import time
import threading

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

from flaxchat.prefetch import BackgroundPrefetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_mesh():
    """Single-device CPU mesh for testing."""
    devices = jax.local_devices()[:1]
    return Mesh(np.array(devices), axis_names=("data",))


def _make_counter_fn(total, shape=(4, 8)):
    """Return a data_fn that yields `total` batches then raises StopIteration."""
    state = {"count": 0}

    def data_fn():
        if state["count"] >= total:
            raise StopIteration
        i = state["count"]
        state["count"] += 1
        inp = np.full(shape, i, dtype=np.int32)
        tgt = np.full(shape, i + 100, dtype=np.int32)
        return inp, tgt

    return data_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBackgroundPrefetcher:
    """Core prefetcher tests."""

    def test_basic_iteration(self):
        """Prefetcher yields the correct number of batches with correct values."""
        mesh = _cpu_mesh()
        sharding = NamedSharding(mesh, P())
        total = 5
        data_fn = _make_counter_fn(total)

        pf = BackgroundPrefetcher(data_fn, mesh, sharding, prefetch_count=2)
        results = list(pf)
        pf.stop()

        assert len(results) == total
        for i, (inp, tgt) in enumerate(results):
            assert int(inp.flatten()[0]) == i
            assert int(tgt.flatten()[0]) == i + 100

    def test_next_protocol(self):
        """next(prefetcher) works and raises StopIteration at the end."""
        mesh = _cpu_mesh()
        sharding = NamedSharding(mesh, P())
        data_fn = _make_counter_fn(3)

        pf = BackgroundPrefetcher(data_fn, mesh, sharding, prefetch_count=2)
        for _ in range(3):
            inp, tgt = next(pf)
            assert inp.shape == (4, 8)

        with pytest.raises(StopIteration):
            next(pf)
        pf.stop()

    def test_prefetch_ahead(self):
        """Queue fills up to prefetch_count before main thread consumes."""
        mesh = _cpu_mesh()
        sharding = NamedSharding(mesh, P())

        call_log = []
        total = 10

        def data_fn():
            if len(call_log) >= total:
                raise StopIteration
            call_log.append(time.monotonic())
            return np.zeros((2, 4), dtype=np.int32), np.ones((2, 4), dtype=np.int32)

        pf = BackgroundPrefetcher(data_fn, mesh, sharding, prefetch_count=4)

        # Give the worker time to fill the queue.
        time.sleep(0.3)

        # The worker should have called data_fn at least prefetch_count times
        # (it may have produced prefetch_count items sitting in the queue,
        # plus possibly one more that is being put).
        assert len(call_log) >= 4, (
            f"Expected at least 4 prefetched calls, got {len(call_log)}"
        )

        # Consume everything.
        results = list(pf)
        pf.stop()
        assert len(results) == total

    def test_stop_cleanup(self):
        """stop() terminates the background thread even if data is infinite."""
        mesh = _cpu_mesh()
        sharding = NamedSharding(mesh, P())

        def infinite_data():
            return np.zeros((2, 4), dtype=np.int32), np.ones((2, 4), dtype=np.int32)

        pf = BackgroundPrefetcher(infinite_data, mesh, sharding, prefetch_count=2)

        # Consume a couple of items to prove it works.
        _ = next(pf)
        _ = next(pf)

        pf.stop()
        assert not pf._thread.is_alive(), "Worker thread should be dead after stop()"

    def test_empty_data(self):
        """Prefetcher handles a data_fn that immediately raises StopIteration."""
        mesh = _cpu_mesh()
        sharding = NamedSharding(mesh, P())

        def empty_fn():
            raise StopIteration

        pf = BackgroundPrefetcher(empty_fn, mesh, sharding, prefetch_count=2)
        results = list(pf)
        pf.stop()

        assert len(results) == 0

    def test_arrays_are_jax(self):
        """Returned arrays are JAX arrays, not numpy."""
        mesh = _cpu_mesh()
        sharding = NamedSharding(mesh, P())
        data_fn = _make_counter_fn(1)

        pf = BackgroundPrefetcher(data_fn, mesh, sharding, prefetch_count=1)
        inp, tgt = next(pf)
        pf.stop()

        assert isinstance(inp, jax.Array)
        assert isinstance(tgt, jax.Array)
