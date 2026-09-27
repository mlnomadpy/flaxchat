"""Bounded, per-process JAX traces of warmed-up training steps."""
from contextlib import contextmanager
import gc
from pathlib import Path
import time

import jax
from jax.experimental import multihost_utils


class HostGCTiming:
    """Optional GC observation; never collect, disable, or retune the collector."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.seconds = 0.0
        self.collections = 0
        self.started = {}
        self.callback = self._observe
        if enabled:
            gc.callbacks.append(self.callback)

    def _observe(self, phase, info):
        generation = info['generation']
        if phase == 'start':
            self.started[generation] = time.monotonic()
        elif phase == 'stop':
            start = self.started.pop(generation, None)
            if start is not None:
                self.seconds += time.monotonic() - start
                self.collections += 1

    def snapshot(self):
        return self.seconds, self.collections

    def close(self):
        if self.enabled:
            gc.callbacks.remove(self.callback)
            self.enabled = False


class TrainingTrace:
    """Profile a relative step window; never start a public profiler server."""

    def __init__(self, directory=None, *, skip=2, steps=1):
        if skip < 1 or steps < 1:
            raise ValueError('Profiling requires at least one warmup and one captured step')
        self.directory = directory
        self.skip = skip
        self.steps = steps
        self.active = False

    @contextmanager
    def step(self, index):
        if self.directory and index == self.skip:
            path = str(Path(self.directory) / f'process-{jax.process_index()}')
            options = jax.profiler.ProfileOptions()
            # Python tracing is on by default in our pinned JAX. Full-model
            # NNX tracing can produce a very large host event stream.
            options.python_tracer_level = 0
            options.host_tracer_level = 1
            jax.profiler.start_trace(path, create_perfetto_link=False, profiler_options=options)
            self.active = True
        completed = False
        try:
            with jax.profiler.StepTraceAnnotation('train', step_num=index):
                yield
            completed = True
        finally:
            if self.active and index + 1 >= self.skip + self.steps:
                self.close()
                # Export duration differs across hosts. Keep the resulting wait
                # inside the profiled step instead of the next timed collective.
                # Do not introduce a collective on an exception/cleanup path.
                if completed and jax.process_count() > 1:
                    multihost_utils.sync_global_devices(f'training-trace-export-{index}')

    def close(self):
        if self.active:
            self.active = False
            jax.profiler.stop_trace()
