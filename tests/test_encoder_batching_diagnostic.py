import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scripts import diagnose_encoder_batching as diagnostic
from tests.test_mlm_accumulation_replay import fixture


def inputs():
    x = jnp.array([[[5, 6, 0], [6, 7, 0]], [[7, 5, 0], [5, 7, 0]]])
    return x, jnp.where(x != 0, x, -1)


def test_stage_order_output_controls_and_source_model_preserved():
    config, model, _, _ = fixture()
    before = [np.asarray(v).copy() for v in jax.tree.leaves(nnx.state(model, nnx.Param))]
    arrays, report = diagnostic.diagnose(model, *inputs(), device_count=1)
    assert report['stage_order'] == ['embedding', 'embedding_norm', 'layer_000',
                                    'encoder_output', 'head_features', 'logits']
    for value in report['instrumentation_controls'].values():
        assert value['finite'] and value['max_abs'] < 2e-6
    np.testing.assert_allclose(arrays['batched_logits'], arrays['plain_batched'], atol=2e-6)
    assert arrays['loss_controls'].shape == (2, 3)
    assert report['loss_control_columns'] == ['forward_sum', 'autodiff_sum', 'per_example_forward_sum']
    assert report['loss_controls']['forward_vs_autodiff']['max_abs'] < 2e-6
    assert model.config == config and model.layers[0].config == config
    for a, b in zip(before, jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_instrumentation_change_remains_visible(monkeypatch):
    _, model, _, _ = fixture()
    original = diagnostic.stages
    def perturbed(m, ids):
        result = original(m, ids)
        result['logits'] = result['logits'] + .25
        return result
    monkeypatch.setattr(diagnostic, 'stages', perturbed)
    _, report = diagnostic.diagnose(model, *inputs(), device_count=1)
    assert all(v['max_abs'] > .24 for v in report['instrumentation_controls'].values())


def test_empty_and_oversized_fixtures_are_rejected_before_execution():
    _, model, _, _ = fixture()
    with pytest.raises(ValueError, match='nonempty'):
        diagnostic.diagnose(model, np.empty((0, 2, 3)), np.empty((0, 2, 3)), device_count=1)
    large = np.ones((1, 128, 1024), np.int32)
    with pytest.raises(ValueError, match='million logits'):
        diagnostic.diagnose(model, large, large, device_count=1)


def test_nonfinite_and_inactive_rows_are_distinguished():
    active = np.array([False, True])
    a, b = np.zeros((2, 1)), np.array([[1.0], [0.0]])
    result = diagnostic.comparison(a, b, active)
    assert result['max_abs'] == 1 and result['active_rows_max_abs'] == 0
    result = diagnostic.comparison(a, np.full_like(a, np.nan), active)
    assert not result['finite'] and result['max_abs'] is None


def test_host_loss_sum_differences_are_not_rounded_away():
    result = diagnostic.comparison(np.array([1.0]), np.array([1.0 + 1e-8]), np.array([True]))
    assert not result['exact'] and result['max_abs'] > 0
