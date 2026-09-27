import numpy as np
import pytest

from flaxchat.contamination import clean_block_starts, overlapping_rows, overlapping_spans


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


@pytest.mark.parametrize('chunk_rows', [1, 2, 9])
def test_rows_report_all_training_matches_without_padding_or_boundary_matches(chunk_rows):
    train = np.array([[0, 2, 3, 4], [2, 3, 4, 0], [7, 8, 9, 10], [11, 12, 13, 14]])
    candidate = np.array([[0, 2, 3, 4, 0], [9, 10, 11, 12, 0], [0, 0, 0, 0, 0]])
    matches = overlapping_rows(train, candidate, special_token_ids=[0], width=3, chunk_rows=chunk_rows)
    assert [m['train_row'] for m in matches] == [0, 1]
    assert [m['train_column'] for m in matches] == [1, 0]


def test_rows_verify_collisions_and_candidate_boundaries(monkeypatch):
    monkeypatch.setattr('flaxchat.contamination.window_hashes',
                        lambda tokens, width: np.zeros(max(0, len(tokens) - width + 1), dtype=np.uint64))
    train = np.array([[3, 4, 5], [7, 8, 9]])
    candidate = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matches = overlapping_rows(train, candidate, special_token_ids=[], width=3)
    assert [m['train_row'] for m in matches] == [1]


def test_rows_match_bruteforce():
    rng = np.random.default_rng(21)
    train, candidate = rng.integers(0, 7, (30, 11)), rng.integers(0, 7, (10, 9))
    expected = []
    for row, tokens in enumerate(train):
        if any(0 not in tokens[i:i+3] and np.array_equal(tokens[i:i+3], other[j:j+3])
               for i in range(9) for other in candidate for j in range(7)):
            expected.append(row)
    assert [m['train_row'] for m in overlapping_rows(
        train, candidate, special_token_ids=[0], width=3, chunk_rows=4)] == expected


def test_rows_short_empty_and_invalid():
    rows = np.ones((2, 3), dtype=np.int32)
    assert overlapping_rows(rows, rows[:0], special_token_ids=[], width=2) == []
    assert overlapping_rows(rows, rows, special_token_ids=[], width=4) == []
    for invalid in [np.ones(3, dtype=np.int32), np.ones((2, 3))]:
        with pytest.raises(ValueError):
            overlapping_rows(rows, invalid, special_token_ids=[])
    with pytest.raises(ValueError):
        overlapping_rows(rows, rows, special_token_ids=[], chunk_rows=0)


def test_documents_cross_windows_but_never_document_or_special_boundaries():
    from flaxchat.contamination import overlapping_documents
    candidate=[('heldout',np.array([3,4,5,6]))]
    train=[('long',np.arange(10)),('part1',np.array([3,4])),('part2',np.array([5,6])),('special',np.array([3,4,0,5,6]))]
    assert overlapping_documents(train,candidate,special_token_ids=[0],width=4)==[
        dict(train_document_id='long',train_start=3,candidate_document_id='heldout',candidate_start=0,length=4)]


def test_document_hash_collisions_are_verified(monkeypatch):
    from flaxchat.contamination import overlapping_documents
    monkeypatch.setattr('flaxchat.contamination.window_hashes',lambda tokens,width:np.zeros(max(0,len(tokens)-width+1),dtype=np.uint64))
    assert overlapping_documents([('a',np.array([1,2,3]))],[('b',np.array([4,5,6]))],special_token_ids=[],width=3)==[]
    with pytest.raises(ValueError):
        overlapping_documents([('a',np.array([1])),('a',np.array([2]))],[],special_token_ids=[],width=2)


def test_document_prefix_filter_requires_full_width_match():
    from flaxchat.contamination import overlapping_documents
    prefix = np.arange(10, 18, dtype=np.int32)
    candidate = np.concatenate((prefix, np.arange(30, 72, dtype=np.int32)))
    false_match = np.concatenate((prefix, np.arange(100, 142, dtype=np.int32)))
    true_match = np.concatenate(([7], candidate, [8]))
    matches = overlapping_documents(
        [('same_prefix', false_match), ('exact', true_match)],
        [('heldout', candidate)], special_token_ids=[], width=50)
    assert matches == [dict(train_document_id='exact', train_start=1,
                            candidate_document_id='heldout', candidate_start=0, length=50)]


@pytest.mark.parametrize('width', [1, 4, 8, 9, 50])
def test_document_prefix_filter_matches_bruteforce(width):
    from flaxchat.contamination import overlapping_documents
    rng = np.random.default_rng(1337 + width)
    candidate = [('heldout', rng.integers(1, 50, 110, dtype=np.int32))]
    train = [(f'train-{i}', rng.integers(1, 50, 120, dtype=np.int32)) for i in range(4)]
    train[1][1][20:20 + width] = candidate[0][1][10:10 + width]
    train[2][1][20:20 + width] = candidate[0][1][10:10 + width]
    train[2][1][20 + width // 2] = 0  # A special token invalidates the span.
    matches = overlapping_documents(train, candidate, special_token_ids=[0], width=width)
    expected = set()
    for identity, tokens in train:
        for start in range(len(tokens) - width + 1):
            span = tokens[start:start + width]
            if 0 in span:
                continue
            if any(np.array_equal(span, heldout[i:i + width])
                   for _, heldout in candidate for i in range(len(heldout) - width + 1)):
                expected.add(identity)
                break
    assert {match['train_document_id'] for match in matches} == expected
    for match in matches:
        source = dict(train)[match['train_document_id']]
        heldout = dict(candidate)[match['candidate_document_id']]
        start, other = match['train_start'], match['candidate_start']
        assert np.array_equal(source[start:start + width], heldout[other:other + width])
        assert 0 not in source[start:start + width]
