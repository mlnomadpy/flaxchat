import pytest
from scripts.compare_encoder_mlm import compare, MATCHED_FIELDS


def reports():
    common = dict(checkpoint_context='candidate', checkpoint_step=1972, initial_weights_sha256={'model.safetensors':'abc'},
                  runtime={'jax':'pinned'}, model_source_sha256={'encoder.py':'pinned'}, evaluation_source_sha256='source', evaluation_config_sha256='config',
                  evaluation_batch_size=8, backend='tpu', devices=['TPU v5 lite']*4,
                  selected_rows_sha256='rows', masked_tokens=100, evaluated_rows=4,
                  excluded_exact_or_duplicate_rows=0, seed=2026, mask_probability=.15,
                  tokenizer_sha256='tokenizer', validation_manifest_sha256='manifest',
                  overlap_check='exact rows')
    return (dict(common,parameter_source='initial_pretrained',masked_token_loss=2.),
            dict(common,parameter_source='checkpoint',masked_token_loss=1.5))


def test_matched_mlm_improvement_is_not_production_qualification():
    result = compare(*reports())
    assert result['relative_loss_reduction'] == .25
    assert result['candidate_minus_baseline'] == -.5
    assert result['lower_development_mlm_loss']
    assert not result['production_quality_qualified']


@pytest.mark.parametrize('key', MATCHED_FIELDS)
def test_rejects_each_missing_protocol_field(key):
    baseline, candidate = reports()
    del candidate[key]
    with pytest.raises(ValueError, match='Unmatched'):
        compare(baseline,candidate)


@pytest.mark.parametrize('key', MATCHED_FIELDS)
def test_rejects_each_changed_protocol_field(key):
    baseline, candidate = reports()
    candidate[key] = 'changed'
    with pytest.raises(ValueError, match='Unmatched'):
        compare(baseline,candidate)


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-1,'1.0',None])
def test_rejects_invalid_loss(value):
    baseline,candidate = reports()
    candidate['masked_token_loss'] = value
    with pytest.raises(ValueError,match='finite'):
        compare(baseline,candidate)


def test_reports_regression_without_relabeling_it_as_success():
    baseline,candidate=reports()
    candidate['masked_token_loss']=3.
    result=compare(baseline,candidate)
    assert result['candidate_minus_baseline']==1.
    assert not result['lower_development_mlm_loss']
    with pytest.raises(ValueError,match='starting-weight'):
        compare(candidate,baseline)
