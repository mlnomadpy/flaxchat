import pytest
from scripts.validate_yat_mmbert import choose_steps


def rows():
    return [
        dict(
            includes_compilation=False,
            seconds=1.0,
            nonpadding_tokens=3000,
            updated=True,
            projection_dense_fallback=False,
            loss=5.0,
        )
        for _ in range(31)
    ]


def test_plan_reserves_evaluation_and_caps_tokens():
    plan = choose_steps(rows(), 1800)
    assert plan["steps"] == 400
    assert plan["steps"] * plan["p90_seconds"] * 1.5 + 1200 <= 1800
    assert choose_steps(rows(), 100000)["steps"] == 16667


def test_plan_blocks_unvalidated_or_unaffordable_training():
    with pytest.raises(ValueError):
        choose_steps(rows()[:20], 5000)
    with pytest.raises(ValueError):
        choose_steps(rows(), 1400)
    bad = rows()
    bad[0]["updated"] = False
    with pytest.raises(ValueError):
        choose_steps(bad, 5000)


def test_runtime_rank_permutation_and_log_ownership():
    from scripts.validate_yat_mmbert import validate_rank_mapping, primary_writer
    assert validate_rank_mapping([2, 3, 0, 1], 4) == 2
    assert validate_rank_mapping([2, 0, 1, 3], 4) == 1  # observed physical v5e-16 order
    assert primary_writer([False, False, True, False]) == 2
    # Runtime log ownership is discovered for every stage, not inferred from SSH rank.
    assert primary_writer([False, True, False, False]) == 1
    for ranks in ([0, 0, 2, 3], [0, 1, 2], [0, 1, 2, 4], [False, 1, 2, 3]):
        with pytest.raises(ValueError):
            validate_rank_mapping(ranks, 4)
    for claims in ([False]*4, [True, True, False, False], [1, False]):
        with pytest.raises(ValueError):
            primary_writer(claims)


def test_source_exposure_checks_accounting():
    from scripts.validate_yat_mmbert import summarize_source_exposure
    rows = [dict(nonpadding_tokens=100, source_nonpadding_tokens=dict(hq=60, english=40))]
    result = summarize_source_exposure(rows, dict(hq=.6, english=.4))
    assert result['realized_token_shares'] == dict(hq=.6, english=.4)
    rows[0]['nonpadding_tokens'] = 101
    with pytest.raises(ValueError, match='accounting'):
        summarize_source_exposure(rows, dict(hq=.6, english=.4))


def test_budget_ignores_profiled_steps():
    normal = rows()
    profiled = [dict(normal[0], includes_profiling=True, seconds=1000.) for _ in range(10)]
    assert choose_steps(normal + profiled, 1800) == choose_steps(normal, 1800)
    with pytest.raises(ValueError, match='21 steady-state'):
        choose_steps(normal[:20] + profiled, 1800)


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_validation_rejects_invalid_attention_alpha_before_cloud_work(monkeypatch, tmp_path, value):
    import sys
    from scripts.validate_yat_mmbert import main
    output = tmp_path / 'untouched'
    monkeypatch.setattr(sys, 'argv', ['validate_yat_mmbert', '--prefix', 'gs://example/run',
        '--output', str(output), '--data', 'missing', '--validation', 'missing',
        '--yat-attention-alpha=' + value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()
