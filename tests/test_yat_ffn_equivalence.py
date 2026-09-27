import numpy as np
import pytest
from scripts.yat_ffn_equivalence import validate_ffn_numerical_equivalence


def fixture():
    return {name: np.ones((4,), np.float32) for name in ('loss','output','dx','dk','dalpha')}


def test_accepts_fp32_epsilon_while_reporting_nonidentity():
    ref = fixture()
    actual = {k:v.copy() for k,v in ref.items()}
    actual['dk'][0] = np.nextafter(np.float32(1.), np.float32(2.))
    result = validate_ffn_numerical_equivalence(ref, actual)
    assert result['passed'] and not result['fields']['dk']['exact']


@pytest.mark.parametrize('kind', ['too_large', 'zero', 'sign', 'nan', 'shape', 'output', 'bf16'])
def test_rejects_precision_null_and_shape_regressions(kind):
    ref = fixture()
    actual = {k:v.copy() for k,v in ref.items()}
    if kind == 'too_large':
        actual['dk'][0] += np.float32(1e-5)  # Still rounds to the same BF16.
    elif kind == 'zero':
        ref['dk'][0] = 0
        actual['dk'][0] = np.float32(1e-40)
    elif kind == 'sign':
        ref['dk'][0] = np.float32(1e-40)
        actual['dk'][0] = -ref['dk'][0]
    elif kind == 'nan':
        actual['dk'][0] = np.nan
    elif kind == 'shape':
        actual['dk'] = actual['dk'][None, :]
    elif kind == 'output':
        actual['output'][0] = np.nextafter(np.float32(1.), np.float32(2.))
    elif kind == 'bf16':
        # Small vector-relative error can cross a BF16 rounding boundary.
        ref['dk'][0] = np.float32(1.00390625)
        actual['dk'][0] = np.nextafter(ref['dk'][0], np.float32(2.))
    result = validate_ffn_numerical_equivalence(ref, actual)
    assert not result['passed']
