import pytest
from flaxchat.profiling import TrainingTrace


def test_window_and_error_cleanup(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr('jax.process_index', lambda: 3)
    monkeypatch.setattr('jax.profiler.start_trace', lambda path, **kw: events.append(('start', path, kw)))
    monkeypatch.setattr('jax.profiler.stop_trace', lambda: events.append(('stop',)))
    trace = TrainingTrace(str(tmp_path), skip=2, steps=2)
    for i in range(6):
        with trace.step(i):
            events.append(('step', i))
    trace.close()
    assert events[2][:2] == ('start', str(tmp_path / 'process-3'))
    options = events[2][2]['profiler_options']
    assert options.python_tracer_level == 0 and options.host_tracer_level == 1
    assert events[2][2]['create_perfetto_link'] is False
    assert events[5] == ('stop',)
    assert sum(e[0] == 'start' for e in events) == 1
    assert sum(e[0] == 'stop' for e in events) == 1
    trace = TrainingTrace(str(tmp_path), skip=1, steps=4)
    try:
        with trace.step(1):
            raise RuntimeError('training failure')
    except RuntimeError:
        trace.close()
    assert events[-1] == ('stop',)


def test_disabled_does_not_start_profiler(monkeypatch):
    monkeypatch.setattr('jax.profiler.start_trace', lambda *a, **kw: pytest.fail('unexpected trace'))
    trace = TrainingTrace()
    for i in range(8):
        with trace.step(i):
            pass
    trace.close()


@pytest.mark.parametrize('fails', [False, True])
def test_multihost_export_barrier_only_after_success(monkeypatch, tmp_path, fails):
    events = []
    monkeypatch.setattr('jax.process_index', lambda: 0)
    monkeypatch.setattr('jax.process_count', lambda: 4)
    monkeypatch.setattr('jax.profiler.start_trace', lambda *a, **kw: events.append('start'))
    monkeypatch.setattr('jax.profiler.stop_trace', lambda: events.append('stop'))
    monkeypatch.setattr('flaxchat.profiling.multihost_utils.sync_global_devices',
                        lambda name: events.append(name))
    trace = TrainingTrace(str(tmp_path), skip=1, steps=1)
    try:
        with trace.step(1):
            events.append('execution')
            if fails:
                raise RuntimeError('execution failed')
    except RuntimeError:
        assert fails
    trace.close()
    expected = ['start', 'execution', 'stop']
    if not fails:
        expected.append('training-trace-export-1')
    assert events == expected


@pytest.mark.parametrize('skip,steps', [(0, 3), (-1, 3), (2, 0)])
def test_invalid_window(skip, steps):
    with pytest.raises(ValueError):
        TrainingTrace(skip=skip, steps=steps)


def test_real_training_trace(tmp_path, monkeypatch):
    from tests.test_encoder_training import test_preparation_training_and_exact_resume
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'float32', 'float32', 'dense', 'xla', 'profile')
    assert list((tmp_path / 'traces' / 'process-0').rglob('*.xplane.pb'))


def test_donated_training_state_preserves_exact_resume(tmp_path, monkeypatch, capsys):
    from tests.test_encoder_training import test_preparation_training_and_exact_resume
    import re
    test_preparation_training_and_exact_resume(
        tmp_path, monkeypatch, 'bfloat16', 'bfloat16', 'masked', 'xla', 'donation')
    aliases = [int(x) for x in re.findall(r'(?<!\w)alias_size_in_bytes=(\d+)', capsys.readouterr().out)]
    assert aliases and all(n > 0 for n in aliases)
