"""Model-free reporting checks using literal measured-score fixtures."""
import math
import unittest

from flaxchat.embedding_uncertainty import bootstrap_paired_delta, bootstrap_query_mean

IDENTITY = 'a' * 64
PROTOCOL = 'b' * 64


def single(scores, **kwargs):
    return bootstrap_query_mean(scores, metric='ndcg@10', query_identity_sha256=IDENTITY,
                                protocol_identity_sha256=PROTOCOL, resamples=500, **kwargs)


def paired(left, right, **kwargs):
    return bootstrap_paired_delta(
        left, right, metric='ndcg@10', baseline_query_identity_sha256=IDENTITY,
        candidate_query_identity_sha256=kwargs.pop('candidate_query_identity_sha256', IDENTITY),
        baseline_protocol_identity_sha256=PROTOCOL,
        candidate_protocol_identity_sha256=kwargs.pop('candidate_protocol_identity_sha256', PROTOCOL),
        resamples=500, **kwargs)


class QueryUncertaintyTests(unittest.TestCase):
    def test_constant_scores_preserve_native_units(self):
        result = single({'q2': 75., 'q1': 75.})
        self.assertEqual(result['estimate'], 75.)
        self.assertEqual(result['interval'], [75., 75.])
        self.assertTrue(result['reporting_only'])
        self.assertFalse(result['selection_policy_changed'])
        self.assertEqual(result['score_scale'], 'unchanged_native_units')

    def test_two_query_interval_and_exact_mean(self):
        result = single({'q1': 0., 'q2': 1.})
        self.assertEqual(result['estimate'], .5)
        self.assertEqual(result['interval'], [0., 1.])
        self.assertEqual(result['query_count'], 2)
        self.assertEqual(result['bootstrap_draws'], 1000)

    def test_input_order_invariance(self):
        scores = {'c': .9, 'b': .2, 'a': .4}
        self.assertEqual(single(scores, seed=24), single(dict(reversed(list(scores.items()))), seed=24))

    def test_same_seed_replays_and_different_seed_changes_draws(self):
        scores = {str(i): i / 13 for i in range(14)}
        self.assertEqual(single(scores, seed=21), single(scores, seed=21))
        self.assertNotEqual(single(scores, seed=21)['interval'], single(scores, seed=22)['interval'])

    def test_paired_shift_removes_shared_query_variation(self):
        result = paired({'a': .1, 'b': .6, 'c': .3}, {'c': .5, 'a': .3, 'b': .8})
        self.assertAlmostEqual(result['estimate'], .2)
        for endpoint in result['interval']:
            self.assertAlmostEqual(endpoint, .2)
        self.assertEqual(result['direction'], 'candidate_minus_baseline')
        self.assertEqual(result['aggregation'], 'paired_query_mean_delta')

    def test_identical_pair_has_zero_uncertainty(self):
        scores = {'a': 0., 'b': 1., 'c': .3}
        result = paired(scores, scores)
        self.assertEqual(result['interval'], [0., 0.])
        self.assertEqual(result['estimate'], 0.)

    def test_query_ids_and_identity_mismatch_fail(self):
        scores = {'a': .1, 'b': .2}
        with self.assertRaisesRegex(ValueError, 'query IDs'):
            paired(scores, {'a': .1, 'c': .2})
        for name in ('candidate_query_identity_sha256', 'candidate_protocol_identity_sha256'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'identity'):
                paired(scores, scores, **{name: 'c' * 64})

    def test_invalid_scores_fail(self):
        fixtures = ({}, {'a': 1.}, {'a': 1., '': 2.}, {'a': 1., 2: 2.},
                    {'a': 1., 'b': True}, {'a': 1., 'b': '1'}, {'a': 1., 'b': math.nan},
                    {'a': 1., 'b': math.inf}, {'a': 1., 'b': 10 ** 1000}, [])
        for scores in fixtures:
            with self.subTest(scores=str(scores)[:50]), self.assertRaises(ValueError):
                single(scores)

    def test_finite_large_mean_and_nonfinite_delta(self):
        result = single({'a': 1e308, 'b': 1e308})
        self.assertEqual(result['estimate'], 1e308)
        with self.assertRaisesRegex(ValueError, 'deltas'):
            paired({'a': -1e308, 'b': 0.}, {'a': 1e308, 'b': 0.})

    def test_bounds_reject_before_sampling(self):
        scores = {'a': 0., 'b': 1.}
        for options in ({'confidence': 0}, {'confidence': 1}, {'confidence': math.nan},
                        {'confidence': True}, {'seed': -1}, {'seed': True},
                        {'seed': 2 ** 64}, {'max_draws': 999}, {'max_draws': 50_000_001}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                single(scores, **options)
        for resamples in (0, 1, True, 100001):
            with self.subTest(resamples=resamples), self.assertRaises(ValueError):
                bootstrap_query_mean(scores, metric='mrr@10', query_identity_sha256=IDENTITY,
                                     protocol_identity_sha256=PROTOCOL, resamples=resamples)

    def test_identity_and_metric_required(self):
        for identity in ('', 'a' * 63, 'G' * 64):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                bootstrap_query_mean({'a': 0., 'b': 1.}, metric='ndcg@10',
                                     query_identity_sha256=identity, protocol_identity_sha256=PROTOCOL)
        with self.assertRaisesRegex(ValueError, 'metric'):
            bootstrap_query_mean({'a': 0., 'b': 1.}, metric=' ',
                                 query_identity_sha256=IDENTITY, protocol_identity_sha256=PROTOCOL)

    def test_score_receipt_changes_when_actual_measurements_change(self):
        first = single({'a': .1, 'b': .2})
        second = single({'a': .1, 'b': .3})
        self.assertEqual(first['query_ids_sha256'], second['query_ids_sha256'])
        self.assertNotEqual(first['measured_scores_sha256'], second['measured_scores_sha256'])


if __name__ == '__main__':
    unittest.main()
