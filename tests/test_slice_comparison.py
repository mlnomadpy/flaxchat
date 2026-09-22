from copy import deepcopy

import pytest

from scripts.compare_tpu_slices import compare


def summary():
    return dict(model_config={'vocab_size': 100}, initial_validation_loss=5., final_validation_loss=3.,
                completed_steps=10, checkpoint_step=10, resolved_config={'steps': 10},
                start_step=0, data_manifest_identity={'hash': 'same'}, tokenizer_identity={'hash': 'same'},
                tokens=1000, checkpoint_policy={'save_every': 10}, invocation_seconds=10.,
                new_committed_tokens=1000, devices=16, process_count=4)


def test_whole_slice_cost_includes_invocation_overhead():
    a = summary()
    b = deepcopy(a) | {'invocation_seconds': 20., 'devices': 4, 'process_count': 1}
    r = compare(a, b, reference_hourly=16, candidate_hourly=4)
    assert r['compute_cost_ratio'] == .5
    assert r['throughput_ratio'] == .5


@pytest.mark.parametrize('change', [dict(tokens=999), dict(data_manifest_identity={}),
    dict(resolved_config={'steps': 10, 'batch': 4}), dict(start_step=2),
    dict(final_validation_loss=6.), dict(checkpoint_policy={}), dict(invocation_seconds=float('nan'))])
def test_unmatched_or_invalid_runs_are_rejected(change):
    with pytest.raises(ValueError):
        compare(summary(), summary() | change, reference_hourly=16, candidate_hourly=4)
