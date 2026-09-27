import json

import numpy as np
import pytest
from flax import nnx
import optax

from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from scripts.compare_checkpoint_numerics import compare_arrays, compare_checkpoints


def test_tiny_drift_is_measured_without_claiming_exactness():
    x = np.ones(262145, np.float32)  # Partial final chunk.
    y = x.copy()
    y[-1] = np.nextafter(y[-1], np.float32(2))
    result = compare_arrays(x, y)
    assert result['changed_elements'] == 1 and not result['exact_bytes']
    assert result['max_abs'] == np.finfo(np.float32).eps
    assert result['relative_l2'] == pytest.approx(np.finfo(np.float32).eps / np.sqrt(x.size))


def test_zero_reference_signed_zero_and_nonfinite():
    result = compare_arrays(np.array([0.], np.float32), np.array([-0.], np.float32))
    assert not result['exact_bytes'] and result['sign_changes'] == 1
    assert result['relative_l2'] is None
    with pytest.raises(ValueError, match='Nonfinite'):
        compare_arrays(np.array([0.]), np.array([np.nan]))
    with pytest.raises(ValueError, match='shape or dtype'):
        compare_arrays(np.ones(2), np.ones((1, 2)))


def test_real_checkpoint_drift_and_manifest_integrity(tmp_path):
    model = nnx.Linear(3, 4, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    for name in ('reference', 'candidate'):
        if name == 'candidate':
            model.kernel[...] = model.kernel[...] + np.float32(1e-5)
        with create_checkpoint_manager(str(tmp_path/name), async_checkpointing=False) as manager:
            save_checkpoint(manager, 1, model, optimizer, {}, training_state={'step': np.array(1)})
    a, b = tmp_path/'reference/1', tmp_path/'candidate/1'
    result = compare_checkpoints(a, b)
    assert result['complete'] and result['identity_equal']
    assert result['comparisons']['model_state']["['kernel']"]['changed_elements'] == 12
    assert all(v['exact_bytes'] for v in result['comparisons']['optimizer_state'].values())
    manifest_path = b/'manifest/metadata'
    manifest = json.loads(manifest_path.read_text())
    manifest['model_state']["['kernel']"]['sha256'] = 'corrupt'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        compare_checkpoints(a, b)
