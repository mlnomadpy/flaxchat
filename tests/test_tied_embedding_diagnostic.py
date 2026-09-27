from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
import pytest

from scripts.diagnose_tied_embedding import variants, diagnose
from tests.test_mlm_accumulation_replay import fixture


def test_instrumented_paths_preserve_loss_and_partition_embedding_derivative():
    _, model, _, _ = fixture()
    before = [np.asarray(v).copy() for v in jax.tree.leaves(nnx.state(model, nnx.Param))]
    x = jnp.tile(jnp.array([[[5, 6, 0]], [[6, 7, 0]]]), (1, jax.device_count(), 1))
    y = jnp.where(x != 0, x, -1)
    arrays, report = diagnose(model, x, y, Mesh(np.asarray(jax.devices()), ('data',)))
    baseline = report['modes']['full']
    for row in report['modes'].values():
        assert row['finite']
        assert row['loss'] == pytest.approx(baseline['loss'], abs=1e-6)
        assert row['reference_loss'] == pytest.approx(baseline['reference_loss'], abs=1e-6)
        assert row['absolute_max'] < 2e-6
    for mode in ('batched', 'per_example'):
        assert report['closure'][mode]['finite']
        np.testing.assert_allclose(arrays['closure_' + mode], 0, atol=2e-6)
    for a, b in zip(before, jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unsupported_loss_cannot_silently_bypass_decoder_instrumentation():
    _, model, _, _ = fixture()
    model.config = replace(model.config, mlm_loss_backend='xla_full')
    with pytest.raises(ValueError, match='native xla'):
        list(variants(model))


def test_nonfinite_error_is_reported_without_invalid_json():
    import json
    from scripts.diagnose_tied_embedding import error_summary
    report = error_summary(np.array([np.nan]))
    assert report == {'finite': False, 'absolute_max': None}
    json.dumps(report, allow_nan=False)
