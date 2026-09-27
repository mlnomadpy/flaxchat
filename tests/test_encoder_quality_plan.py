import copy
import pytest
from scripts.plan_encoder_quality import plan


def evidence():
    identity = dict(resolved_config=dict(steps=1972, batch_size=64), tokenizer='tokenizer',
                    data_manifest='real-corpus', source_python_sha256='source',
                    initial_weights_sha256={'model.safetensors': 'weights'})
    records = [dict(event='run_config', backend='tpu', devices=4, processes=1,
                    mlm_loss_backend='pallas', input_identity=copy.deepcopy(identity))]
    records += [dict(event='train_step', step=i + 1, updated=True,
                     projection_dense_fallback=False, tokens=32768, nonpadding_tokens=25000,
                     includes_compilation=i == 0, masked_tokens=3600, loss=2.,
                     seconds=100. if i == 0 else 1.) for i in range(100)]
    records.append(dict(event='checkpoint', step=100))
    kwargs = dict(expected_identity=identity, expected_nonpadding=[25000] * 100,
                  devices=4, processes=1, steps=1972, batch_size=64,
                  sequence_length=512, remaining_seconds=5000, hourly_usd=10.8,
                  budget_usd=20, reserve_seconds=900)
    return records, kwargs


def test_full_recipe_is_priced_without_crediting_calibration():
    records, kwargs = evidence()
    result = plan(records, **kwargs)
    assert result['estimated_seconds'] == 1972 * 1.25 + 900
    assert result['estimated_compute_usd'] == pytest.approx(10.095)
    assert result['actual_billing_usd'] is None
    assert result['quality_qualified'] is False
    assert result['resumes_calibration'] is False


@pytest.mark.parametrize('field,value', [('backend', 'cpu'), ('devices', 8),
                                        ('processes', 2), ('mlm_loss_backend', 'xla')])
def test_rejects_wrong_physical_configuration(field, value):
    records, kwargs = evidence()
    records[0][field] = value
    with pytest.raises(ValueError, match='Physical'):
        plan(records, **kwargs)


@pytest.mark.parametrize('key', ['data_manifest', 'source_python_sha256', 'tokenizer',
                                 'initial_weights_sha256', 'resolved_config'])
def test_rejects_changed_calibration_identity(key):
    records, kwargs = evidence()
    records[0]['input_identity'][key] = 'different'
    with pytest.raises(ValueError, match='identity'):
        plan(records, **kwargs)


@pytest.mark.parametrize('field,value', [('nonpadding_tokens', 2000), ('updated', False),
                                        ('projection_dense_fallback', True), ('seconds', float('nan')),
                                        ('masked_tokens', 30000), ('includes_compilation', True)])
def test_rejects_sparse_fixture_or_invalid_updates(field, value):
    records, kwargs = evidence()
    records[2][field] = value
    with pytest.raises(ValueError, match='density'):
        plan(records, **kwargs)


@pytest.mark.parametrize('field,value,match', [('remaining_seconds', 1000, 'lease'),
                                             ('budget_usd', 1, 'budget'),
                                             ('steps', 1000, 'horizon')])
def test_never_truncates_recipe_to_fit(field, value, match):
    records, kwargs = evidence()
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        plan(records, **kwargs)


def test_missing_final_checkpoint_rejected():
    records, kwargs = evidence()
    with pytest.raises(ValueError, match='checkpoint'):
        plan(records[:-1], **kwargs)


def test_missing_update_rejected():
    records, kwargs = evidence()
    del records[2]
    with pytest.raises(ValueError, match='consecutive'):
        plan(records, **kwargs)
