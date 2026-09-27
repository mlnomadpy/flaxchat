import pytest
from scripts.workload_deadline import WorkloadDeadline


def test_deadline_bounds_children_and_includes_system_sleep(monkeypatch):
    from scripts import workload_deadline as module
    clock = {'mono': 100., 'wall': 1000.}
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock['mono'])
    monkeypatch.setattr(module.time, 'time', lambda: clock['wall'])
    deadline = WorkloadDeadline(20)
    assert deadline.remaining(5) == 5
    clock['mono'] += 8
    assert deadline.remaining(60) == 12
    clock['wall'] += 21
    with pytest.raises(TimeoutError):
        deadline.remaining()


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf')])
def test_deadline_rejects_invalid_budgets(seconds):
    with pytest.raises(ValueError):
        WorkloadDeadline(seconds)


def test_projection_campaign_bounds_hung_evidence_upload(tmp_path, monkeypatch):
    from argparse import Namespace
    import subprocess
    from scripts import benchmark_encoder_projection as campaign
    monkeypatch.setattr(campaign, 'preflight_cases', lambda *args: None)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert 0 < kwargs['timeout'] <= 2
        if argv[0] == 'gcloud':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return Namespace(returncode=1)
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    args = Namespace(max_seconds=2, cases=[], data_root='unused', output=str(tmp_path),
                     prefix='gs://unused/run', hourly_usd=1, kernel_rows=[128], tiles=[])
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.campaign(args)
    assert (tmp_path / 'summary.json').is_file()
    assert any(argv[0] == 'gcloud' for argv in calls)
