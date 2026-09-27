"""Strict FFN comparison allowing sub-BF16 differences in FP32 weight gradients.

Cross-implementation identity differs from exact checkpoint-resume identity.
All fields except dk remain exact. For dk, require identical BF16 values, zero
locations and signs, plus relative L2 error no larger than FP32 machine epsilon.
This is a numerical admission policy, not a model-quality guarantee.
"""
import ml_dtypes
import numpy as np


FIELDS = ('loss', 'output', 'dx', 'dk', 'dalpha')


def validate_ffn_numerical_equivalence(reference, candidate):
    if set(reference) != set(FIELDS) or set(candidate) != set(FIELDS):
        raise ValueError('Expected exactly loss, output, dx, dk and dalpha')
    failures = []
    measurements = {}
    for name in FIELDS:
        a, b = np.asarray(candidate[name]), np.asarray(reference[name])
        if a.shape != b.shape:
            failures.append(f'{name}: shape mismatch')
            continue
        if a.dtype != b.dtype or a.dtype != np.float32:
            failures.append(f'{name}: expected saved float32 arrays')
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            failures.append(f'{name}: nonfinite')
            continue
        exact = np.array_equal(a, b)
        if name != 'dk':
            if not exact:
                failures.append(f'{name}: not exact')
            measurements[name] = {'exact': bool(exact)}
            continue
        error = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
        scale = float(np.linalg.norm(b.astype(np.float64)))
        epsilon = float(np.finfo(np.float32).eps)
        same_bf16 = np.array_equal(a.astype(ml_dtypes.bfloat16), b.astype(ml_dtypes.bfloat16))
        same_zeros = np.array_equal(a == 0, b == 0)
        same_signs = np.array_equal(np.signbit(a), np.signbit(b))
        if not (same_bf16 and same_zeros and same_signs and error <= epsilon * scale):
            failures.append('dk: outside FP32 epsilon/BF16/zero/sign bounds')
        measurements[name] = dict(exact=bool(exact), same_bf16=bool(same_bf16),
            same_zeros=bool(same_zeros), same_signs=bool(same_signs), error_l2=error,
            reference_l2=scale, relative_l2=error/scale if scale else None,
            relative_l2_limit=epsilon)
    return dict(passed=not failures, failures=failures, fields=measurements,
                scope='Numerical FFN comparison only; exact resume remains a separate requirement')
