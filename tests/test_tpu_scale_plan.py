import pytest

from scripts.tpu_scale_plan import build_plan


def plan(**changes):
    args = dict(devices=16, hosts=4, fsdp=4, profiles=['correctness', 'context', 'gpt2'],
                hourly_usd=20., budget_usd=10., seconds_per_profile=300,
                token_manifest='/data/manifest.json', checkpoint_prefix='gs://bucket/new-run')
    return build_plan(**(args | changes))


@pytest.mark.parametrize('devices', [1, 4, 8, 16, 32, 64, 128, 256])
def test_scale_plan_preserves_global_batch_and_bounds(devices):
    result = plan(devices=devices, hosts=max(1, devices // 4), fsdp=min(devices, 4))
    assert result['status'] == 'plan_only'
    for profile in result['profiles']:
        assert profile['status'] == 'not_run'
        command = profile['argv']
        assert command[command.index('--global-batch-size') + 1] == str(devices)
        assert profile['train_tokens_required'] == int(command[command.index('--tokens') + 1]) + 1


@pytest.mark.parametrize('changes', [dict(hosts=3), dict(fsdp=3), dict(budget_usd=.01),
    dict(checkpoint_prefix='/local'), dict(seconds_per_profile=3600), dict(profiles=[]),
    dict(profiles=['gpt2', 'gpt2'])])
def test_scale_plan_rejects_invalid_or_over_budget(changes):
    with pytest.raises(ValueError):
        plan(**changes)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_scale_plan_rejects_nonfinite_prices(value):
    with pytest.raises(ValueError):
        plan(hourly_usd=value)
    with pytest.raises(ValueError):
        plan(budget_usd=value)


def test_sustained_profiles_require_quality_gate_and_bounded_logit_memory():
    result = plan(profiles=['sustained-1k', 'sustained-4k'])
    for profile in result['profiles']:
        command = profile['argv']
        assert command[command.index('--warmup-steps') + 1] == '10'
        assert '--remat' in command and '--loss-chunk-size' in command
        assert '--checkpoint-interval-seconds' in command
        assert 'scripts.validate_training_quality' in profile['quality_gate_argv']
