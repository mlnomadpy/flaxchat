"""Literal host metadata only; no numerical backend or model execution."""
import pytest

from flaxchat.embedding_telemetry import InvocationTelemetry, PHASES, reduce_step


def test_global_exposure_sums_actual_unequal_hosts_and_uses_slowest_time():
    step = reduce_step([[2, 31, 25, 0.1, 1, 1.1, 1], [2, 49, 30, 0.2, 2, 2.2, 1]])
    assert step['pairs_global'] == 4
    assert step['processed_tokens_global'] == 80
    assert step['useful_tokens_global'] == 55
    assert step['useful_tokens_per_second_global'] == pytest.approx(25)
    assert step['update_seconds_slowest_host'] == 2


@pytest.mark.parametrize('rows', [[], [[1]], [[1, 2, 3, 0, 0, 0, 0]],
                                   [[1, 2, 2, float('nan'), 0, 0, 0]],
                                   [[1.5, 2, 2, 0, 0, 0, 0]],
                                   [[1, 2, 2, 0, 0, 0, 0], [1, 2, 2, 0, 0, 0, 1]]])
def test_rejects_invalid_or_divergent_vectors(rows):
    with pytest.raises(ValueError):
        reduce_step(rows)


def test_invocation_includes_setup_evaluation_checkpoint_close_without_lifetime_claim():
    clock = iter([10., 12., 15., 18., 22.])
    meter = InvocationTelemetry(started=0, clock=lambda: next(clock))
    with meter.phase('evaluation'):
        pass
    with meter.phase('best_checkpoint'):
        pass
    meter.record_step(reduce_step([[2, 90, 60, 1, 4, 5, 1]]))
    vector = meter.timing_vector()
    report = meter.report([vector, [0.] * len(PHASES) + [30.]])
    assert report['wall_seconds_slowest_host'] == 30
    assert report['whole_invocation_useful_tokens_per_second_global'] == 2
    assert report['phase_seconds_slowest_host']['evaluation'] == 2
    assert report['phase_seconds_slowest_host']['best_checkpoint'] == 3
    assert report['phase_seconds_slowest_host']['compile_and_first_update'] == 4
    assert report['posted_cost_usd'] is None
    assert report['completed_steps_this_invocation'] == 1


def test_zero_exposure_completed_resume_has_no_rate_claim():
    meter = InvocationTelemetry(started=0, clock=lambda: 0)
    assert meter.report([meter.timing_vector()])['whole_invocation_useful_tokens_per_second_global'] is None


def test_phase_records_failure_without_changing_recovery_state():
    times = iter([0., 1., 4.])
    meter = InvocationTelemetry(clock=lambda: next(times))
    with pytest.raises(RuntimeError), meter.phase('recovery_checkpoint'):
        raise RuntimeError('save failed')
    assert meter.seconds['recovery_checkpoint'] == 3
    assert meter.accepted_steps == 0


def test_large_integer_exposure_is_not_rounded_through_float32():
    count = 2**24 + 1
    report = reduce_step([[1, count, count, 0, 1, 1, 0], [1, count + 2, count + 2, 0, 1, 1, 0]])
    assert report['useful_tokens_global'] == 2 * count + 2


@pytest.mark.parametrize('rows', [[], [[0]], [[float('inf')] * (len(PHASES) + 1)]])
def test_rejects_invalid_stage_timing(rows):
    with pytest.raises(ValueError):
        InvocationTelemetry().report(rows)
