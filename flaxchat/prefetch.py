"""
Background data prefetcher for overlapping data loading with TPU compute.

Runs a background thread that calls a user-supplied data function and places
the resulting arrays onto devices via jax.device_put, so the main training
loop never blocks on host-to-device transfer.

Usage:
    prefetcher = BackgroundPrefetcher(get_batch, mesh, batch_sharding)
    for step in range(num_steps):
        inputs, targets = next(prefetcher)  # already on TPU
        loss = train_step(model, optimizer, inputs, targets)
    prefetcher.stop()
"""

import threading
import queue

import jax
import jax.numpy as jnp


class BackgroundPrefetcher:
    """Prefetch batches in a background thread while TPU computes."""

    def __init__(self, data_fn, mesh, batch_sharding, prefetch_count=2):
        """
        Args:
            data_fn: callable that returns (inputs, targets) numpy arrays.
                     Raise StopIteration to signal end of data.
            mesh: JAX mesh for sharding (kept for reference, sharding is
                  specified via batch_sharding).
            batch_sharding: NamedSharding (or any jax.Sharding) to use when
                            placing arrays on devices.
            prefetch_count: number of batches to prefetch ahead.
        """
        self.data_fn = data_fn
        self.mesh = mesh
        self.batch_sharding = batch_sharding
        self.prefetch_count = prefetch_count

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
        self._stop_event = threading.Event()
        self._exhausted = False

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── background worker ────────────────────────────────────────────────
    def _worker(self):
        try:
            while not self._stop_event.is_set():
                try:
                    arrays = self.data_fn()
                except StopIteration:
                    self._queue.put(_SENTINEL)
                    return

                # Place on device in the background thread.
                on_device = tuple(
                    jax.device_put(jnp.asarray(a), self.batch_sharding)
                    for a in arrays
                )

                # Block until there is room in the queue, but check the stop
                # event periodically so we can shut down cleanly.
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(on_device, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception:
            # If the worker dies unexpectedly, send a sentinel so the main
            # thread doesn't hang forever.
            try:
                self._queue.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    # ── iterator protocol ────────────────────────────────────────────────
    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration

        item = self._queue.get()
        if item is _SENTINEL:
            self._exhausted = True
            raise StopIteration
        return item

    # ── cleanup ──────────────────────────────────────────────────────────
    def stop(self):
        """Signal the background thread to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        # Drain any remaining items so the worker isn't stuck on a full queue.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# Sentinel object to signal end-of-data through the queue.
_SENTINEL = object()
