import numpy as np
import pytest

from flaxchat.contamination import clean_block_starts, overlapping_spans


def test_chunk_boundary_and_duplicate_candidate_matches():
    train = np.arange(30)
    candidate = np.array([90, 6, 7, 8, 9, 91, 6, 7, 8, 9])
    assert overlapping_spans(train, candidate, width=4, chunk_tokens=8) == [
        {'candidate_start': 1, 'train_start': 6, 'length': 4},
        {'candidate_start': 6, 'train_start': 6, 'length': 4}]


def test_hash_collision_is_not_evidence(monkeypatch):
    monkeypatch.setattr('flaxchat.contamination.window_hashes',
                        lambda tokens, width: np.zeros(max(0, len(tokens) - width + 1), dtype=np.uint64))
    assert overlapping_spans(np.array([1, 2, 3, 4]), np.array([5, 6, 3, 4]), width=2) == [
        {'candidate_start': 2, 'train_start': 2, 'length': 2}]


def test_clean_blocks_exclude_every_intersected_block_without_splicing():
    assert clean_block_starts(20, [{'candidate_start': 4, 'length': 3}], block_tokens=5) == [10, 15]


def test_empty_short_and_invalid_inputs():
    assert overlapping_spans(np.arange(3), np.arange(2), width=4) == []
    with pytest.raises(ValueError):
        overlapping_spans([], [], width=0)
    with pytest.raises(ValueError):
        overlapping_spans([], [], width=4, chunk_tokens=3)


def test_matches_bruteforce_on_random_tokens():
    rng = np.random.default_rng(7)
    train, candidate = rng.integers(0, 4, 100), rng.integers(0, 4, 40)
    expected = [i for i in range(38) if any(np.array_equal(candidate[i:i+3], train[j:j+3]) for j in range(98))]
    actual = overlapping_spans(train, candidate, width=3, chunk_tokens=7)
    assert [m['candidate_start'] for m in actual] == expected
