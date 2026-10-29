"""Data-only corruption checks for the final, partially populated batch."""

import numpy as np
import pytest

from flaxchat.mlm import mask_prepared_rows, mask_tokens


POLICY = dict(
    seed=3,
    step=19,
    vocab_size=32,
    mask_token_id=4,
    special_token_ids=[0, 4],
    probability=1.0,
)


def test_real_rows_keep_identical_corruption_and_example_ids():
    tokens = np.arange(8, 24, dtype=np.int32).reshape(4, 4)
    ids = np.array([3, 1, 0])
    actual = mask_prepared_rows(tokens, ids, pad_token_id=0, **POLICY)
    expected = mask_tokens(tokens[ids], example_ids=ids, **POLICY)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_final_padding_is_not_last_row_or_a_training_target():
    tokens = np.arange(8, 24, dtype=np.int32).reshape(4, 4)
    before = tokens.copy()
    x, y = mask_prepared_rows(
        tokens, np.array([1, -1, -1]), pad_token_id=0, allow_padding=True, **POLICY
    )
    np.testing.assert_array_equal(y[0], tokens[1])
    assert np.all(x[1:] == 0) and np.all(y[1:] == -1)
    np.testing.assert_array_equal(tokens, before)


@pytest.mark.parametrize(
    "ids,padding", [([-1], False), ([-2], True), ([4], True), ([1.5], True)]
)
def test_invalid_row_ids_fail_before_indexing(ids, padding):
    with pytest.raises(ValueError, match="row IDs"):
        mask_prepared_rows(
            np.ones((4, 4), dtype=np.int32),
            np.asarray(ids),
            pad_token_id=0,
            allow_padding=padding,
            **POLICY,
        )


def test_empty_device_slice_has_zero_targets():
    x, y = mask_prepared_rows(
        np.ones((4, 4), dtype=np.int32),
        np.array([-1, -1]),
        pad_token_id=0,
        allow_padding=True,
        **POLICY,
    )
    assert not np.any(x) and np.all(y == -1)
