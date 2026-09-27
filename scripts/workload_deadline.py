"""Bound child work and evidence uploads by one campaign deadline."""
import math
import time


class WorkloadDeadline:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Campaign duration must be finite and positive')
        self.monotonic_end = time.monotonic() + seconds
        self.wall_end = time.time() + seconds

    def remaining(self, maximum=None):
        remaining = min(self.monotonic_end - time.monotonic(), self.wall_end - time.time())
        if remaining <= 0:
            raise TimeoutError('Campaign workload deadline exhausted')
        return min(remaining, maximum) if maximum is not None else remaining
