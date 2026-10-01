"""Host-metadata accounting; never executes a model or changes recovery state."""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

# All hosts gather this exact vector at every accepted update.
STEP_FIELDS = ('pairs', 'processed_tokens', 'useful_tokens', 'input_seconds',
               'update_seconds', 'step_seconds', 'first_variant_call')
PHASES = ('setup', 'input', 'compile_and_first_update', 'steady_update',
          'evaluation', 'best_checkpoint', 'recovery_checkpoint', 'finalization')


def reduce_step(host_rows):
    rows = [list(row) for row in host_rows]
    if not rows or any(len(row) != len(STEP_FIELDS) for row in rows):
        raise ValueError('Incomplete host telemetry vector')
    if any(not math.isfinite(float(v)) or v < 0 for row in rows for v in row):
        raise ValueError('Telemetry must be finite and nonnegative')
    for row in rows:
        if any(v != int(v) or v > 2**53 for v in row[:3]):
            raise ValueError('Exposure counts must be exactly representable integers')
        if row[2] > row[1] or row[6] not in (0, 1):
            raise ValueError('Invalid useful exposure or compilation marker')
    if len({row[6] for row in rows}) != 1:
        raise ValueError('Hosts disagree on compiled variant cadence')
    result = {f'{name}_global': sum(int(row[i]) for row in rows)
              for i, name in enumerate(STEP_FIELDS[:3])}
    result.update({f'{name}_slowest_host': max(float(row[i]) for row in rows)
                   for i, name in enumerate(STEP_FIELDS[3:6], 3)})
    elapsed = result['step_seconds_slowest_host']
    result['useful_tokens_per_second_global'] = result['useful_tokens_global'] / elapsed if elapsed else None
    result['first_variant_call'] = bool(rows[0][6])
    result['timing_scope'] = 'accepted-update-including-input-and-first-call-compilation-before-telemetry-gather'
    result['useful_exposure_definition'] = 'query-positive-and-valid-negative-nonpadding-tokens; repeated-exposures-included'
    return result


class InvocationTelemetry:
    """Invocation wall-time accounting, deliberately separate from checkpoint state."""
    def __init__(self, started=None, clock=time.monotonic):
        self.clock = clock
        self.started = clock() if started is None else started
        self.seconds = dict.fromkeys(PHASES, 0.0)
        self.pairs = self.processed_tokens = self.useful_tokens = self.accepted_steps = 0

    @contextmanager
    def phase(self, name):
        if name not in self.seconds:
            raise ValueError('Unknown telemetry phase')
        started = self.clock()
        try:
            yield
        finally:
            self.seconds[name] += self.clock() - started

    def record_step(self, step):
        self.accepted_steps += 1
        self.pairs += step['pairs_global']
        self.processed_tokens += step['processed_tokens_global']
        self.useful_tokens += step['useful_tokens_global']
        self.seconds['input'] += step['input_seconds_slowest_host']
        phase = 'compile_and_first_update' if step['first_variant_call'] else 'steady_update'
        self.seconds[phase] += step['update_seconds_slowest_host']

    def report(self, host_rows):
        rows = [list(row) for row in host_rows]
        if not rows or any(len(row) != len(PHASES) + 1 for row in rows):
            raise ValueError('Incomplete stage timing vector')
        if any(not math.isfinite(float(v)) or v < 0 for row in rows for v in row):
            raise ValueError('Stage timing must be finite and nonnegative')
        seconds = max(float(row[-1]) for row in rows)
        return {'scope': 'successful-invocation-from-run-entry-through-manager-close',
                'completed_steps_this_invocation': self.accepted_steps,
                'pairs_global_this_invocation': self.pairs,
                'processed_tokens_global_this_invocation': self.processed_tokens,
                'useful_tokens_global_this_invocation': self.useful_tokens,
                'wall_seconds_slowest_host': seconds,
                'phase_seconds_slowest_host': {name: max(float(row[i]) for row in rows)
                                              for i, name in enumerate(PHASES)},
                'phase_timing_note': 'Per-phase host maxima need not sum to wall time; first-call compile is not isolated.',
                'whole_invocation_useful_tokens_per_second_global': self.useful_tokens / seconds if seconds else None,
                'posted_cost_usd': None}

    def timing_vector(self):
        return [self.seconds[name] for name in PHASES] + [self.clock() - self.started]
