import numpy as np
import pytest

from scripts.evaluate_prepared_checkpoint import prepare_choices


class Tokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, texts, prepend):
        return [[prepend] + [ord(c) for c in text] for text in texts]


def test_answer_boundary_padding_and_lengths():
    padded, start, lengths, original = prepare_choices(
        Tokenizer(), {'query': 'Q', 'choices': ['ab', 'acde']}, 10)
    assert start == 4  # BOS, Q, newline, common answer prefix a
    np.testing.assert_array_equal(lengths, [5, 7])
    for i, row in enumerate(original):
        np.testing.assert_array_equal(padded[i, :len(row)], row)
        assert not np.any(padded[i, len(row):])


@pytest.mark.parametrize('choices, limit', [(['a', 'abcd'], 4), (['same', 'same'], 20), (['', 'a'], 20)])
def test_rejects_truncation_and_empty_scored_continuations(choices, limit):
    with pytest.raises(ValueError):
        prepare_choices(Tokenizer(), {'query': 'Q', 'choices': choices}, limit)
