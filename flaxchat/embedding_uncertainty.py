"""Reporting-only query bootstrap for already measured retrieval scores.

This module neither evaluates embeddings nor selects checkpoints. Callers must
bind query/qrels/corpus identity and scoring protocol separately from model
identity. Query bootstrap assumes independent queries; related query groups need
cluster resampling instead. Native metric units are preserved.
"""
from collections.abc import Mapping
import hashlib
import json
import math
import platform
import random
import re

MAX_DRAWS = 50_000_000
MAX_RESAMPLES = 100_000
POLICY = 'query-percentile-bootstrap-linear-quantiles-v1'


def _identity(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit SHA256 query/protocol identity required')
    return value


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _scores(scores):
    if not isinstance(scores, Mapping) or len(scores) < 2:
        raise ValueError('At least two independently identified query scores required')
    if any(not isinstance(key, str) or not key for key in scores):
        raise ValueError('Nonempty string query IDs required')
    ordered = sorted(scores)
    values = []
    for key in ordered:
        value = scores[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Finite numeric query scores required')
        try:
            value = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError('Finite numeric query scores required') from error
        if not math.isfinite(value):
            raise ValueError('Finite numeric query scores required')
        values.append(value)
    return ordered, values


def _mean(values):
    # Divide before summation to avoid avoidable overflow of a finite mean.
    result = math.fsum(value / len(values) for value in values)
    if not math.isfinite(result):
        raise ValueError('Finite mean required')
    return result


def _quantile(sorted_values, probability):
    index = (len(sorted_values) - 1) * probability
    first = int(index)
    last = min(first + 1, len(sorted_values) - 1)
    fraction = index - first
    return sorted_values[first] * (1 - fraction) + sorted_values[last] * fraction


def _interval(values, *, metric, confidence, resamples, seed, max_draws):
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError('Measured metric name required')
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError('Confidence must be finite and strictly between zero and one')
    if (type(resamples) is not int or not 2 <= resamples <= MAX_RESAMPLES
            or type(seed) is not int or not 0 <= seed < 2 ** 64
            or type(max_draws) is not int or not 1 <= max_draws <= MAX_DRAWS
            or len(values) * resamples > max_draws):
        raise ValueError('Bounded integer resamples, seed and total bootstrap draws required')
    rng = random.Random(seed)
    means = sorted(_mean([values[rng.randrange(len(values))]
                          for _ in range(len(values))]) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return {
        'format': 'flaxchat-query-uncertainty-v1', 'policy': POLICY,
        'reporting_only': True, 'selection_policy_changed': False,
        'metric': metric, 'aggregation': 'query_arithmetic_mean',
        'estimate': _mean(values),
        'interval': [_quantile(means, tail), _quantile(means, 1 - tail)],
        'confidence': float(confidence), 'query_count': len(values),
        'resamples': resamples, 'seed': seed, 'bootstrap_draws': len(values) * resamples,
        'max_draws': max_draws, 'score_scale': 'unchanged_native_units',
        'resampling_unit': 'query', 'quantile': 'linear-order-statistic-interpolation',
        'rng': 'python-random.Random-randrange', 'python_version': platform.python_version(),
        'assumption': 'queries independent; fixed corpus/judgments/protocol',
        'limitations': 'conditional query uncertainty only; no training-seed, corpus, judgment or contamination uncertainty',
    }


def bootstrap_query_mean(scores, *, metric, query_identity_sha256,
                         protocol_identity_sha256, confidence=0.95,
                         resamples=2000, seed=0, max_draws=MAX_DRAWS):
    """Percentile interval for one existing query-mean metric, without rescaling."""
    query_identity = _identity(query_identity_sha256)
    protocol_identity = _identity(protocol_identity_sha256)
    ids, values = _scores(scores)
    receipt = _interval(values, metric=metric, confidence=confidence,
                        resamples=resamples, seed=seed, max_draws=max_draws)
    receipt.update(query_identity_sha256=query_identity,
                   protocol_identity_sha256=protocol_identity,
                   query_ids_sha256=_digest(ids),
                   measured_scores_sha256=_digest(list(zip(ids, values, strict=True))))
    return receipt


def bootstrap_paired_delta(baseline_scores, candidate_scores, *, metric,
                           baseline_query_identity_sha256, candidate_query_identity_sha256,
                           baseline_protocol_identity_sha256, candidate_protocol_identity_sha256,
                           confidence=0.95, resamples=2000, seed=0, max_draws=MAX_DRAWS):
    """Resample matched query deltas: candidate minus baseline, not two intervals.

    Both identity digests must bind the same query corpus/qrels/selection and the
    same metric definition/units. Model-specific prompts can differ only under
    an explicitly shared matched-comparison protocol. Digests are caller evidence
    references; this helper does not itself authenticate the source artifacts.
    """
    query_identity = _identity(baseline_query_identity_sha256)
    protocol_identity = _identity(baseline_protocol_identity_sha256)
    if (query_identity != _identity(candidate_query_identity_sha256)
            or protocol_identity != _identity(candidate_protocol_identity_sha256)):
        raise ValueError('Paired comparison requires identical query and scoring protocol identity')
    ids, baseline = _scores(baseline_scores)
    candidate_ids, candidate = _scores(candidate_scores)
    if ids != candidate_ids:
        raise ValueError('Paired comparison requires identical complete query IDs')
    deltas = [right - left for left, right in zip(baseline, candidate, strict=True)]
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError('Finite paired query deltas required')
    receipt = _interval(deltas, metric=metric, confidence=confidence,
                        resamples=resamples, seed=seed, max_draws=max_draws)
    receipt.update(aggregation='paired_query_mean_delta', direction='candidate_minus_baseline',
                   query_identity_sha256=query_identity,
                   protocol_identity_sha256=protocol_identity,
                   query_ids_sha256=_digest(ids),
                   baseline_scores_sha256=_digest(list(zip(ids, baseline, strict=True))),
                   candidate_scores_sha256=_digest(list(zip(ids, candidate, strict=True))),
                   baseline_estimate=_mean(baseline), candidate_estimate=_mean(candidate))
    return receipt
