"""Host diagnostic schema only; inputs are literal arrays, no model/backend."""
import importlib.util
import json
from pathlib import Path
import numpy as np


def helper():
    path = Path(__file__).with_name('test_embedding_gradient_cache_physical_tpu.py')
    spec = importlib.util.spec_from_file_location('cache_diagnostic_definition', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_near_zero_violation_reports_original_bound_and_named_location():
    result = helper()._leaf_diagnostics(np.array([4.283338967e-6], np.float32),
                      np.array([7.080473096e-6], np.float32), rtol=.02, atol=2e-6)
    assert result['mismatch_count'] == 1 and result['first_mismatches'][0]['index'] == [0]
    assert result['rtol'] == .02 and result['atol'] == 2e-6
    assert result['max_abs_delta'] > 2e-6 and result['relative_l2_delta'] > .3
    json.dumps(result, allow_nan=False)


def test_nonfinite_and_zero_reference_are_explicit_without_invalid_json():
    diagnostic = helper()._leaf_diagnostics
    result = diagnostic(np.array([np.nan]), np.array([0.]), rtol=.02, atol=2e-6)
    assert not result['finite'] and result['mismatch_count'] == 1
    assert result['first_mismatches'][0]['actual'] is None
    json.dumps(result, allow_nan=False)
    result = diagnostic(np.array([0.]), np.array([0.]), rtol=.02, atol=2e-6)
    assert result['relative_l2_delta'] is None and result['mismatch_count'] == 0


def test_nonfinite_loss_retains_reason_in_json_safe_evidence():
    diagnostic = helper()._loss_diagnostic
    for value, reason in [(np.nan, 'nan'), (np.inf, 'positive_infinity'),
                          (-np.inf, 'negative_infinity')]:
        result = diagnostic(np.asarray(value))
        assert result == {'value': None, 'finite': False, 'nonfinite_kind': reason}
        json.dumps(result, allow_nan=False)
    assert diagnostic(np.float32(.5)) == {'value': .5, 'finite': True, 'nonfinite_kind': None}


def test_summary_preserves_named_failed_and_nonfinite_leaves():
    module = helper()
    passed = {'path': 'safe', **module._leaf_diagnostics(np.array([0.]), np.array([0.]), rtol=.02, atol=2e-6)}
    failed = {'path': 'adam.mu.kernel', **module._leaf_diagnostics(np.array([7.1e-6]), np.array([4.3e-6]), rtol=.02, atol=2e-6)}
    nonfinite = {'path': 'adam.nu.alpha', **module._leaf_diagnostics(np.array([np.nan]), np.array([0.]), rtol=.02, atol=2e-6)}
    result = module._comparison_summary({'optimizer': [passed, failed, nonfinite]})
    assert result == {'optimizer': {'leaf_count': 3, 'failing_leaves': ['adam.mu.kernel', 'adam.nu.alpha'],
                                    'mismatch_count': 2, 'nonfinite_leaf_count': 1}}
    json.dumps(result, allow_nan=False)


def test_gradient_within_original_bound_can_exceed_adam_first_moment_bound():
    diagnostic = helper()._leaf_diagnostics
    cache_mu = np.array([4.283338967e-6], np.float32)
    full_mu = np.array([7.080473096e-6], np.float32)
    factor = 1 - .9
    cache_gradient, full_gradient = cache_mu / factor, full_mu / factor
    broad = diagnostic(cache_gradient, full_gradient, rtol=.02, atol=.002)
    implied = diagnostic(cache_gradient, full_gradient, rtol=.02, atol=2e-6 / factor)
    moments = diagnostic(cache_mu, full_mu, rtol=.02, atol=2e-6)
    assert broad['mismatch_count'] == 0
    assert implied['mismatch_count'] == moments['mismatch_count'] == 1
    observed_delta = cache_mu - full_mu
    gradient_delta = factor * (cache_gradient - full_gradient)
    relation = diagnostic(observed_delta, gradient_delta, rtol=2e-6, atol=2e-8)
    assert relation['mismatch_count'] == 0
    json.dumps({'broad': broad, 'implied': implied, 'moments': moments, 'relation': relation}, allow_nan=False)


def test_embedding_boundary_reports_exact_drift_and_absent_shape_matched_scope():
    module = helper()
    equal = module._leaf_diagnostics(np.array([1.], np.float32), np.array([1.], np.float32), rtol=0., atol=0.)
    different = module._leaf_diagnostics(np.array([1.000001], np.float32), np.array([1.], np.float32), rtol=0., atol=0.)
    result = module._embedding_boundary_summary({'cache_full_embeddings': [equal, different],
                                               'cache_full_embedding_cotangents': [equal]})
    assert result['cache_full_embeddings'] == {'all_finite': True, 'exactly_equal': False, 'mismatch_count': 1}
    assert result['cache_full_embedding_cotangents']['exactly_equal'] is True
    assert result['cache_chunked_embeddings'] is None
    json.dumps(result, allow_nan=False)
