import pytest
from scripts.summarize_encoder_gradient_health import summarize


def rows():
    return [dict(event='train_step', step=i+1, loss=2., updated=True,
                 gradient_norm_before_clip=n, gradient_clip_scale=min(1., 1/max(n, 1.)))
            for i, n in enumerate((0., .5, 2., 1e9))]


def test_norm_statistics_do_not_claim_optimizer_update_size():
    report = summarize(rows(), first=1, last=4)
    assert report['gradient_norm_before_clip'] == dict(minimum=0., median=1.25, maximum=1e9)
    assert report['clipped_steps'] == 2 and report['zero_gradient_steps'] == 1
    assert report['scale_below_1e_minus_6_steps'] == 1
    assert 'alone does not imply small Adam updates' in report['scope']


@pytest.mark.parametrize('field,value', [('gradient_norm_before_clip', float('nan')),
    ('gradient_norm_before_clip', -1.), ('gradient_clip_scale', 0.),
    ('gradient_clip_scale', 2.), ('gradient_clip_scale', .5),
    ('gradient_clip_scale', True), ('gradient_norm_before_clip', None),
    ('updated', False), ('loss', float('inf'))])
def test_bad_or_missing_telemetry_cannot_be_reported_healthy(field, value):
    records = rows()
    records[0][field] = value
    with pytest.raises(ValueError):
        summarize(records, first=1, last=4)


@pytest.mark.parametrize('records', [rows()[:-1], rows()[::-1], rows()+[rows()[0]]])
def test_partial_reordered_or_duplicate_steps_are_rejected(records):
    with pytest.raises(ValueError, match='complete, ordered, unique'):
        summarize(records, first=1, last=4)
