from types import SimpleNamespace as NS
import hashlib
import pytest
from scripts.summarize_jax_trace import compile_operation_groups, main, overlap_ns, summarize, union_ns


def event(name, start, end):
    return NS(name=name, start_ns=start * 1_000_000, end_ns=end * 1_000_000)


def test_nested_events_do_not_inflate_elapsed_time():
    profile = NS(
        planes=[
            NS(name="/host:CPU", lines=[]),
            NS(
                name="/device:TPU:0",
                lines=[
                    NS(name="XLA Modules", events=[event("train", 0, 10)]),
                    NS(
                        name="XLA Ops",
                        events=[
                            event("while", 0, 9),
                            event("tpu_custom_call", 1, 5),
                            event("tpu_custom_call", 3, 7),
                            event("all-reduce", 7, 10),
                        ],
                    ),
                ],
            ),
        ]
    )
    d = summarize(profile)["devices"][0]
    assert d["interval_union_ms"] == dict(
        modules=10, all_ops=10, collectives=3, pallas_named_ops=6
    )
    custom = next(x for x in d["top_operations"] if x["name"] == "tpu_custom_call")
    assert (
        custom["union_ms"] == 6
        and custom["summed_event_ms"] == 8
        and custom["events"] == 2
    )
    assert len(summarize(profile, top=1)["devices"][0]["top_operations"]) == 1


def test_large_hlo_names_are_bounded_without_merging_operations():
    names = ["shared_prefix_" + suffix * 1000 for suffix in ("a", "b")]
    profile = NS(planes=[NS(name="/device:TPU:0", lines=[
        NS(name="XLA Ops", events=[event(names[0], 0, 2), event(names[1], 2, 5),
                                    event("short", 5, 6)])])])
    result = summarize(profile, max_name_chars=14,
                       operation_groups={"late_match": "b{100}$"})
    device = result["devices"][0]
    rows = device["top_operations"]
    assert [r["union_ms"] for r in rows] == [3, 2, 1]
    assert rows[0]["name"] == rows[1]["name"] == "shared_prefix_"
    assert [r["name_sha256"] for r in rows[:2]] == [
        hashlib.sha256(name.encode()).hexdigest() for name in reversed(names)]
    assert all(r["name_truncated"] and r["name_length"] == 1014 for r in rows[:2])
    assert "name_truncated" not in rows[2]
    assert device["interval_union_ms"]["late_match"] == 3
    limited = summarize(profile, top=1)["devices"][0]
    assert limited["operation_names"] == 3 and len(limited["top_operations"]) == 1
    with pytest.raises(ValueError, match="max_name_chars"):
        summarize(profile, max_name_chars=0)


def test_union_handles_disjoint_adjacent_empty_and_invalid_intervals():
    assert union_ns([]) == 0
    assert union_ns([(5, 8), (0, 2), (2, 4), (6, 7)]) == 7
    with pytest.raises(ValueError):
        union_ns([(2, 1)])
    with pytest.raises(ValueError):
        summarize(NS(planes=[]), top=0)


def test_custom_groups_union_nested_matches_without_claiming_exclusive_time():
    profile = NS(planes=[NS(name="/device:TPU:0", lines=[
        NS(name="XLA Modules", events=[event("repair", 0, 20)]),
        NS(name="XLA Ops", events=[event("repair_outer", 1, 10),
            event("repair_inner", 2, 5), event("all-reduce(vocab)", 8, 12)])])])
    selectors = {"repair": "repair_", "vocabulary": r"all-reduce\(vocab\)", "missing": "never"}
    result = summarize(profile, operation_groups=selectors)
    times = result['devices'][0]['interval_union_ms']
    assert times['repair'] == 9  # not 12 from double-counting the nested event
    assert times['vocabulary'] == 4 and times['collectives'] == 4
    assert times['missing'] == 0 and times['modules'] == 20
    assert result['operation_group_patterns'] == selectors
    overlaps = result['devices'][0]['category_overlap_ms']
    assert overlaps['collectives']['repair'] == 2
    assert overlaps['collectives']['vocabulary'] == 4
    assert overlaps['repair']['missing'] == 0


@pytest.mark.parametrize('left,right,expected', [
    ([], [(0, 10)], 0),
    ([(0, 10), (2, 5)], [(3, 7), (4, 6)], 4),
    ([(0, 2), (4, 6)], [(1, 5)], 2),
    ([(0, 2)], [(2, 4)], 0),
])
def test_overlap_counts_nested_intervals_once(left, right, expected):
    assert overlap_ns(left, right) == expected
    assert overlap_ns(right, left) == expected


@pytest.mark.parametrize('groups', [{'modules': 'a'}, {'bad-name': 'a'}, {'empty': ''}])
def test_invalid_custom_groups(groups):
    with pytest.raises(ValueError):
        compile_operation_groups(groups)


@pytest.mark.parametrize('specs', [['missing_equals'], ['x=['], ['x=a', 'x=b']])
def test_cli_rejects_bad_selectors_before_loading_trace(specs, tmp_path):
    args = ['nonexistent.pb', '--output', str(tmp_path/'result.json')]
    for spec in specs:
        args.extend(['--op-group', spec])
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_module_filter_excludes_checkpoint_and_clips_crossing_events():
    profile = NS(planes=[NS(name='/device:TPU:0', lines=[
        NS(name='XLA Modules', events=[event('jit_train_step', 2, 6),
            event('jit_train_step_nested', 3, 5), event('checkpoint', 6, 12),
            event('jit_train_step', 12, 16)]),
        NS(name='XLA Ops', events=[event('spanning', 0, 20),
            event('checkpoint_hash', 6, 12), event('all-reduce', 4, 8)])])])
    result = summarize(profile, module_pattern='^jit_train_step',
                       operation_groups={'spanning': '^spanning$'})
    d = result['devices'][0]
    assert d['matched_module_events'] == 3
    assert d['interval_union_ms'] == dict(modules=8, all_ops=8,
        collectives=2, pallas_named_ops=0, spanning=8)
    rows = {row['name']: row for row in d['top_operations']}
    assert 'checkpoint_hash' not in rows
    assert rows['spanning']['events'] == 1
    assert rows['spanning']['summed_event_ms'] == 8
    assert d['category_overlap_ms']['collectives']['spanning'] == 2
    with pytest.raises(ValueError, match='No matching TPU modules'):
        summarize(profile, module_pattern='absent')


def test_cli_rejects_invalid_module_regex_before_loading_trace(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(['missing.pb', '--output', str(tmp_path/'result.json'),
              '--module-pattern', '['])
    assert error.value.code == 2
