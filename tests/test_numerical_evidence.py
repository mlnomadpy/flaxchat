import json

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.numerical_evidence import read_numerical_evidence, write_numerical_evidence


def test_device_bf16_values_and_dtype_survive_reload(tmp_path):
    x = jnp.asarray([0, -0., .01, -3.5, 1e-30, jnp.inf], jnp.bfloat16)
    destination = tmp_path / "case"
    write_numerical_evidence(destination, {"q": x, "alpha": jnp.float32(.01),
        "segments": jnp.asarray([[0, -1]], jnp.int32)}, metadata={"backend": "cpu"})
    values, report = read_numerical_evidence(destination)
    np.testing.assert_array_equal(values["q"].view(np.uint32), np.asarray(x, dtype=np.float32).view(np.uint32))
    assert report['arrays']['q']['dtype'] == 'bfloat16'
    assert values['q'].dtype == np.float32
    assert values['alpha'].shape == () and values['segments'].dtype == np.int32
    with pytest.raises(FileExistsError):
        write_numerical_evidence(destination, {'q': x}, metadata={})


def test_corruption_and_manifest_mismatch_are_rejected(tmp_path):
    write_numerical_evidence(tmp_path/'case', {'x': np.ones((2, 3))}, metadata={})
    manifest = tmp_path/'case/manifest.json'
    original = manifest.read_text()
    report = json.loads(original)
    report['arrays']['x']['shape'] = [6]
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='shape/dtype'):
        read_numerical_evidence(tmp_path/'case')
    manifest.write_text(original)
    with (tmp_path/'case/arrays.npz').open('ab') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        read_numerical_evidence(tmp_path/'case')


@pytest.mark.parametrize('arrays,metadata', [({}, {}), ({'../x': [1]}, {}),
    ({'x': np.array([object()])}, {}), ({'x': [1]}, {'loss': float('nan')})])
def test_invalid_payload_leaves_no_case(tmp_path, arrays, metadata):
    with pytest.raises(ValueError):
        write_numerical_evidence(tmp_path/'case', arrays, metadata=metadata)
    assert not (tmp_path/'case').exists()


def test_yat_benchmark_preserves_nonfinite_inputs_and_gradients(tmp_path):
    from scripts.benchmark_yat import measure
    with pytest.raises(ValueError, match='Nonfinite'):
        measure(jnp.sqrt, (jnp.asarray([-1., 4.]),), repeats=1,
                failure_directory=tmp_path/'failure')
    arrays, report = read_numerical_evidence(tmp_path/'failure')
    np.testing.assert_array_equal(arrays['input_0'], [-1., 4.])
    assert np.isnan(arrays['output'][0])
    assert np.isnan(arrays['gradient_0'][0])
    assert report['metadata']['reason'] == 'Nonfinite YAT output or gradient'
