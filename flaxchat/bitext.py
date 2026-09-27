"""Paired bitext metrics; similarity search and benchmark protocol are separate."""

import numpy as np


def bitext_metrics(predictions, labels, *, corpus_size):
    """Support-weighted multiclass precision/recall/F1 and pair accuracy.

    A corpus row is a class, including repeated target labels when supplied.
    Zero-division classes contribute zero, matching the MTEB bitext metric
    convention. This alone does not establish official benchmark parity.
    """
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if type(corpus_size) is not int or corpus_size < 1:
        raise ValueError('Positive corpus size required')
    for values in (predictions, labels):
        if (values.ndim != 1 or not len(values)
                or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < 0) or np.any(values >= corpus_size)):
            raise ValueError('Nonempty in-range integer corpus indices required')
    if predictions.shape != labels.shape:
        raise ValueError('One prediction per gold pair required')
    # Compact observed classes to avoid allocating a corpus_size squared matrix
    # (or even a corpus_size vector for a very large retrieval corpus).
    _, inverse = np.unique(np.concatenate((labels, predictions)), return_inverse=True)
    truth, predicted = np.split(inverse, 2)
    support = np.bincount(truth, minlength=int(inverse.max()) + 1)
    selected = np.bincount(predicted, minlength=len(support))
    correct = truth == predicted
    true_positive = np.bincount(truth[correct], minlength=len(support))
    precision = np.divide(true_positive, selected, out=np.zeros(len(support)), where=selected != 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(support)), where=support != 0)
    f1 = np.divide(2 * true_positive, support + selected, out=np.zeros(len(support)), where=(support + selected) != 0)
    return dict(precision=float(np.average(precision, weights=support)),
                recall=float(np.average(recall, weights=support)),
                f1=float(np.average(f1, weights=support)),
                accuracy=float(np.mean(correct)))
