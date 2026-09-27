import pytest
from scripts.plan_encoder_endurance import plan


def rows(seconds=0.1):
    return [
        dict(
            includes_compilation=i == 0,
            updated=True,
            projection_dense_fallback=False,
            loss=2.0,
            seconds=seconds,
        )
        for i in range(101)
    ]


@pytest.mark.parametrize("seconds", [0.03, 0.1, 0.7, 2.0])
def test_horizon_scales_with_physical_calibration_and_reserves_recovery(seconds):
    result = plan(rows(seconds), minimum_seconds=1800, remaining_seconds=6000)
    assert result["steps"] % 2 == 0
    assert result["steps"] * seconds >= 2250
    assert result["estimated_total_seconds"] <= 6000
    assert result["measured_duration_still_required"]


@pytest.mark.parametrize("defect", ["short", "nan", "fallback", "rejected", "lease"])
def test_endurance_plan_rejects_unqualified_or_unaffordable_calibration(defect):
    evidence = rows()
    if defect == "short":
        evidence = evidence[:10]
    if defect == "nan":
        evidence[-1]["loss"] = float("nan")
    if defect == "fallback":
        evidence[-1]["projection_dense_fallback"] = True
    if defect == "rejected":
        evidence[-1]["updated"] = False
    with pytest.raises(ValueError):
        plan(
            evidence,
            minimum_seconds=1800,
            remaining_seconds=1000 if defect == "lease" else 6000,
        )


@pytest.mark.parametrize("primary", [0, 2])
def test_calibration_primary_is_independent_of_ssh_order(tmp_path, primary):
    import json
    from scripts.plan_encoder_endurance import select_calibration_log

    logs = [tmp_path / f"worker-{i}.log" for i in range(4)]
    for path in logs:
        path.write_text("non-primary runtime log\n")
    records = [
        dict(
            event="run_config",
            backend="tpu",
            devices=16,
            processes=4,
            mlm_loss_backend="pallas",
        )
    ]
    records += [
        dict(row, event="train_step", step=i + 1, tokens=64 * 512, masked_tokens=300)
        for i, row in enumerate(rows()[:100])
    ]
    records += [dict(event="checkpoint", step=100, seconds=1.0)]
    logs[primary].write_text("\n".join(map(json.dumps, records)))
    selected, updates = select_calibration_log(logs, devices=16, processes=4)
    assert selected == logs[primary] and len(updates) == 100
    with pytest.raises(ValueError, match="topology"):
        select_calibration_log(logs, devices=8, processes=4)
    logs[(primary + 1) % 4].write_text(logs[primary].read_text())
    with pytest.raises(ValueError, match="Exactly one runtime"):
        select_calibration_log(logs, devices=16, processes=4)


def test_endurance_coordinator_downloads_all_workers_before_planning(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from scripts import validate_encoder_endurance as runner

    monkeypatch.setenv('JAX_PROCESS_INDEX', '0')
    monkeypatch.setenv('FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS', '7000')
    records = [dict(event='run_config', backend='tpu', devices=16,
                    processes=4, mlm_loss_backend='pallas')]
    records += [dict(row, event='train_step', step=i+1, tokens=64*512, masked_tokens=300)
                for i, row in enumerate(rows(.2)[:100])]
    records += [dict(event='checkpoint', step=100, seconds=1.)]
    downloaded = []
    executed = []
    prefix = 'gs://test-only/endurance'
    output = tmp_path / 'run'

    def run(command, **kwargs):
        if 'scripts.train_encoder' in command:
            kwargs['stdout'].write('launcher zero is not the JAX primary\n')
        elif command[:3] == ['gcloud', 'storage', 'cp']:
            source, dest = command[3:]
            if source.startswith(prefix + '/calibration-evidence/'):
                worker = int(Path(source).stem.split('-')[1])
                downloaded.append(worker)
                Path(dest).write_text('\n'.join(map(json.dumps, records)) if worker == 2 else 'peer log\n')
            elif dest == prefix + '/plan.json':
                assert downloaded == [0, 1, 2, 3]
                plan_result = json.loads(Path(source).read_text())
                assert plan_result['calibration_primary_worker_log'] == 'worker-2.log'
                assert plan_result['steps'] >= 11250
        elif 'scripts.validate_encoder_scale' in command:
            executed.append(command)
            assert command[command.index('--minimum-seconds')+1] == '1800'
            assert command[command.index('--expected-processes')+1] == '4'
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, 'run', run)
    assert runner.main(['--prefix', prefix, '--devices', '16', '--processes', '4',
                        '--output', str(output)]) == 0
    assert len(executed) == 1
