"""Finite unseen-row coverage and cursor replay only; no model/backend calls."""

import json
import numpy as np
import pytest
from flaxchat.encoder_data import (
    CoverageMixtureRows,
    RemainingCoverageMixtureRows,
    file_hash,
)


def corpus(root, lengths=(3, 11, 23), shares=(0.6, 0.25, 0.15)):
    rows, documents, cursor = [], [], 0
    for pool, length in enumerate(lengths):
        rows.extend([[pool + 2, 3, 0, 0] if pool % 2 else [pool + 2, 3, 4, 5]] * length)
        documents.append(
            dict(
                id=str(pool),
                language_script=f"lang{pool}",
                source_group=f"source{pool}",
                first_row=cursor,
                end_row=cursor + length,
            )
        )
        cursor += length
    tokens = np.asarray(rows, np.int32)
    path = root / "documents.jsonl"
    path.write_text("".join(json.dumps(doc) + "\n" for doc in documents))
    manifest = dict(
        split="train",
        documents=len(documents),
        documents_sha256=file_hash(path),
        pad_token_id=0,
        language_counts={
            f"lang{pool}": dict(
                rows=length, nonpadding_tokens=length * (2 if pool % 2 else 4)
            )
            for pool, length in enumerate(lengths)
        },
        source_token_shares={
            f"source{pool}": share for pool, share in enumerate(shares)
        },
    )
    return tokens, manifest


@pytest.mark.parametrize(
    "seed,transition,batch", [(1, 0, 4), (7, 3, 4), (17, 10, 2), (12, 100, 3)]
)
def test_unseen_only_no_repeat_and_full_exact_completion(
    tmp_path, seed, transition, batch
):
    tokens, manifest = corpus(tmp_path)
    legacy = CoverageMixtureRows(tmp_path, tokens, manifest, seed, 0.5)
    legacy.seek(transition, batch)
    historical = legacy.row_exposures.copy()
    sampler = RemainingCoverageMixtureRows(
        tmp_path, tokens, manifest, seed, 0.5, transition, batch
    )
    assert sampler.total_unseen_rows == np.count_nonzero(historical == 0)
    assert sampler.total_unseen_tokens == int(sampler.nonpadding[historical == 0].sum())
    assert (
        sampler.completion_step
        == transition + (sampler.total_unseen_rows + batch - 1) // batch
    )
    collected = []
    for step in range(transition, sampler.completion_step):
        indices = sampler.batch(step, batch)
        selected = indices[indices >= 0]
        assert np.all(historical[selected] == 0)
        collected.extend(selected.tolist())
        assert sum(sampler.token_counts(indices).values()) == int(
            sampler.nonpadding[selected].sum()
        )
        assert sum(sampler.source_token_counts(indices).values()) == int(
            sampler.nonpadding[selected].sum()
        )
        if np.any(indices == -1):
            assert step == sampler.completion_step - 1
            assert np.all(indices[len(selected) :] == -1)
    assert len(collected) == len(set(collected)) == sampler.total_unseen_rows
    assert set(collected) == set(np.flatnonzero(historical == 0))
    assert sampler.remaining_unique_rows == sampler.remaining_unique_tokens == 0
    np.testing.assert_array_equal(
        sampler.row_exposures[historical > 0], historical[historical > 0]
    )
    with pytest.raises(StopIteration):
        sampler.batch(sampler.completion_step, batch)


def test_seek_resumed_suffix_matches_sequential_even_when_weights_exhaust_pools(
    tmp_path,
):
    tokens, manifest = corpus(tmp_path, (3, 17, 53), (0.85, 0.1, 0.05))
    transition, batch = 2, 4
    direct = RemainingCoverageMixtureRows(
        tmp_path, tokens, manifest, 18, 0.5, transition, batch
    )
    sequence = {
        step: direct.batch(step, batch)
        for step in range(transition, direct.completion_step)
    }
    for resume in range(transition, direct.completion_step + 1):
        resumed = RemainingCoverageMixtureRows(
            tmp_path, tokens, manifest, 18, 0.5, transition, batch
        )
        resumed.seek(resume, batch)
        for step in range(resume, resumed.completion_step):
            np.testing.assert_array_equal(resumed.batch(step, batch), sequence[step])
        np.testing.assert_array_equal(resumed.row_exposures, direct.row_exposures)
        assert resumed.exposure_summary() == direct.exposure_summary()


def test_final_padding_and_token_accounting_never_reads_last_row_for_minus_one(
    tmp_path,
):
    tokens, manifest = corpus(tmp_path, (2, 3), (0.9, 0.1))
    sampler = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 1, 0.5, 0, 4)
    sampler.batch(0, 4)
    last = sampler.batch(1, 4)
    assert np.count_nonzero(last == -1) == 3
    assert sum(sampler.token_counts(last).values()) == int(sampler.nonpadding[last[0]])
    assert sum(sampler.token_counts(np.array([-1, -1])).values()) == 0
    with pytest.raises(ValueError):
        sampler.source_token_counts(np.array([-2]))
    with pytest.raises(ValueError):
        sampler.token_counts(np.array([len(tokens)]))


def test_small_pools_repeated_by_legacy_are_excluded_and_statistics_preserved(tmp_path):
    tokens, manifest = corpus(tmp_path, (1, 100), (0.99, 0.01))
    legacy = CoverageMixtureRows(tmp_path, tokens, manifest, 7, 0.5)
    legacy.seek(20, 4)
    assert legacy.row_exposures[0] > 1
    sampler = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 7, 0.5, 20, 4)
    assert sampler.row_exposures[0] == legacy.row_exposures[0]
    assert len(sampler._remaining_orders[0]) == 0
    first = sampler.batch(20, 4)
    assert np.all(first > 0)
    report = sampler.exposure_summary()
    assert report["source0:lang0"]["remaining_unique_rows"] == 0
    assert report["source0:lang0"]["legacy_exposures"] == int(legacy._positions[0])
    assert sampler.policy_identity == dict(
        coverage_sampling="remaining_pool_epoch_v1",
        coverage_transition_step=20,
        coverage_transition_batch_size=4,
    )


def test_constructor_metadata_only_never_reads_token_rows_during_legacy_replay(
    tmp_path,
):
    tokens, manifest = corpus(tmp_path, (2, 3), (0.5, 0.5))
    sampler = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 17, 0.5, 1000, 8)
    assert sampler.total_unseen_rows == 0 and sampler.completion_step == 1000
    assert sampler.remaining_unique_tokens == 0
    sampler.seek(1000, 8)
    with pytest.raises(StopIteration):
        sampler.batch(1000, 8)


@pytest.mark.parametrize(
    "transition,batch", [(True, 4), (-1, 4), (1.0, 4), (0, True), (0, 0), (0, 4.0)]
)
def test_strict_transition_types(tmp_path, transition, batch):
    tokens, manifest = corpus(tmp_path)
    with pytest.raises(ValueError):
        RemainingCoverageMixtureRows(
            tmp_path, tokens, manifest, 1, 0.5, transition, batch
        )


def test_strict_cursor_and_fixed_batch(tmp_path):
    tokens, manifest = corpus(tmp_path)
    sampler = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 4, 0.5, 1, 4)
    for step, size in [
        (0, 4),
        (True, 4),
        (1.0, 4),
        (1, True),
        (1, 2),
        (sampler.completion_step + 1, 4),
    ]:
        with pytest.raises(ValueError):
            sampler.batch(step, size)


def test_chunked_pool_replay_is_invariant_to_chunk_boundaries(tmp_path):
    tokens, manifest = corpus(tmp_path, (3, 17, 53), (0.8, 0.15, 0.05))
    left = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 3, 0.5, 0, 4)
    right = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 3, 0.5, 0, 4)
    whole = left._draw_remaining(50, materialize=True)
    pieces = np.concatenate(
        [right._draw_remaining(n, materialize=True) for n in [1, 2, 3, 4, 5, 6, 7, 22]]
    )
    np.testing.assert_array_equal(whole, pieces)
    np.testing.assert_array_equal(left._remaining_positions, right._remaining_positions)


def test_weighted_nonexhausted_draws_match_independent_scalar_reference(tmp_path):
    tokens, manifest = corpus(tmp_path, (1, 2, 47), (0.75, 0.24, 0.01))
    sampler = RemainingCoverageMixtureRows(tmp_path, tokens, manifest, 26, 0.5, 0, 4)
    rng = np.random.default_rng(np.random.SeedSequence([26, 0x52454D42, 0, 4]))
    cursors = np.zeros(3, dtype=np.int64)
    expected = []
    for uniform in rng.random(len(tokens)):
        active = np.flatnonzero(cursors < sampler._remaining_lengths)
        weights = sampler.row_probabilities[active]
        cdf = np.cumsum(weights / weights.sum())
        cdf[-1] = 1.0
        pool = active[np.searchsorted(cdf, uniform, side="right")]
        expected.append(sampler._remaining_orders[pool][cursors[pool]])
        cursors[pool] += 1
    actual = np.concatenate(
        [sampler.batch(step, 4) for step in range(sampler.completion_step)]
    )
    np.testing.assert_array_equal(actual[actual >= 0], expected)


@pytest.mark.parametrize(
    "seed,exponent", [(True, 0.5), (-1, 0.5), (1, True), (1, float("nan")), (1, 1.1)]
)
def test_strict_seed_and_exponent(tmp_path, seed, exponent):
    tokens, manifest = corpus(tmp_path)
    with pytest.raises(ValueError):
        RemainingCoverageMixtureRows(tmp_path, tokens, manifest, seed, exponent, 0, 4)
