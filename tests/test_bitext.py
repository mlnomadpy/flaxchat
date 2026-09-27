import numpy as np
import pytest
from flaxchat.bitext import bitext_metrics


def test_weighted_f1_is_not_accuracy_for_collapsed_predictions():
    result = bitext_metrics([0, 0, 0], [0, 1, 2], corpus_size=3)
    assert result == pytest.approx(dict(precision=1/9, recall=1/3, f1=1/6, accuracy=1/3))


def test_repeated_gold_targets_are_weighted_by_support():
    result = bitext_metrics([0, 1, 1, 1], [0, 0, 1, 2], corpus_size=3)
    assert result == pytest.approx(dict(precision=7/12, recall=1/2, f1=11/24, accuracy=1/2))


def test_perfect_and_incorrect_pairs_in_large_corpus():
    assert all(v == 1 for v in bitext_metrics([2, 999999], [2, 999999], corpus_size=1000000).values())
    assert all(v == 0 for v in bitext_metrics([2, 1], [1, 2], corpus_size=3).values())


@pytest.mark.parametrize('predictions,labels,size', [([], [], 3), ([0], [0, 1], 3),
    ([1.0], [0], 3), ([True], [0], 3), ([3], [0], 3), ([-1], [0], 3),
    ([0], [3], 3), ([0], [0], 0), ([[0]], [[0]], 3), ([np.nan], [0], 3)])
def test_invalid_evidence_is_rejected(predictions, labels, size):
    with pytest.raises(ValueError):
        bitext_metrics(predictions, labels, corpus_size=size)
