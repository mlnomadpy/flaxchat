import json
from typing import Any
import pytest
from scripts.compare_yat_training import compare


def records(seconds, source):
    return [dict(event='run_config', backend='tpu', devices=16, processes=4,
                 input_identity=dict(source_python_sha256=source, data_manifest='data',
                                     tokenizer='tokenizer', initial_weights_sha256={'model':'weights'},
                                     resolved_config=dict(seed=42, batch_size=128))),
            *[dict(event='train_step', step=i, loss=10-i*.1, updated=True,
                   projection_dense_fallback=False, tokens=128*512,
                   masked_tokens=7000, nonpadding_tokens=50000,
                   learning_rate=.001, seconds=seconds if i>1 else 100,
                   includes_compilation=i==1, includes_profiling=i==3)
              for i in range(1,5)], dict(event='checkpoint', step=4, seconds=20)]


def write(path, rows):
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    return path


def run(tmp_path, a, b):
    return compare(write(tmp_path/'baseline.log', a), write(tmp_path/'candidate.log', b),
                   steps=4, batch=128, length=512, hourly_usd=9.6)


def test_matched_useful_throughput_excludes_compile_profile_and_checkpoints(tmp_path):
    out=run(tmp_path,records(2,'old'),records(1,'new'))
    assert out['matched'] and out['useful_token_speedup']==2
    assert out['baseline']['measured_seconds']==4
    assert out['candidate']['nonpadding_tokens_per_second']==50000
    assert out['max_absolute_loss_difference']==0
    assert out['candidate']['estimated_warm_compute_usd_per_billion_useful_tokens']==pytest.approx(53.333333)


@pytest.mark.parametrize('field,value', [('nonpadding_tokens',49000),('masked_tokens',7100),
    ('learning_rate',.002),('includes_profiling',True),('source_nonpadding_tokens',{'hq':50000})])
def test_unmatched_exposure_or_profile_window_rejected(tmp_path,field,value):
    a,b=records(2,'old'),records(1,'new')
    b[2][field]=value
    with pytest.raises(ValueError,match='Unmatched'):
        run(tmp_path,a,b)


def test_changed_data_identity_rejected(tmp_path):
    a,b=records(2,'old'),records(1,'new')
    b[0]['input_identity']['data_manifest']='different'
    with pytest.raises(ValueError,match='mismatch'):
        run(tmp_path,a,b)


def test_nonfinite_loss_and_missing_checkpoint_are_not_performance_results(tmp_path):
    a,b=records(2,'old'),records(1,'new')
    b[2]['loss']=float('nan')
    with pytest.raises(ValueError,match='Invalid update'):
        run(tmp_path,a,b)
    with pytest.raises(ValueError,match='checkpoint'):
        run(tmp_path,a,records(1,'new')[:-1])


@pytest.mark.parametrize('key',['tokenizer','initial_weights_sha256','resolved_config'])
def test_matching_missing_identity_is_not_evidence(tmp_path,key):
    a,b=records(2,'old'),records(1,'new')
    for log in (a,b):
        del log[0]['input_identity'][key]
    with pytest.raises(ValueError,match='Incomplete'):
        run(tmp_path,a,b)


@pytest.mark.parametrize('other_change', [None, 'seed', 'tokenizer'])
def test_accumulation_comparison_allows_only_requested_mode(tmp_path, other_change):
    a,b=records(2,'old'),records(1,'new')
    b[0]['input_identity']['resolved_config']['local_gradient_accumulation']=True
    if other_change == 'seed':
        b[0]['input_identity']['resolved_config']['seed']=99
    if other_change == 'tokenizer':
        b[0]['input_identity']['tokenizer']='other'
    baseline,candidate=write(tmp_path/'a',a),write(tmp_path/'b',b)
    with pytest.raises(ValueError,match='mismatch'):
        compare(baseline,candidate,steps=4,batch=128,length=512,hourly_usd=9.6)
    kwargs: dict[str, Any] = dict(steps=4,batch=128,length=512,hourly_usd=9.6,local_accumulation=True)
    if other_change:
        with pytest.raises(ValueError,match='mismatch'):
            compare(baseline,candidate,**kwargs)
    else:
        result=compare(baseline,candidate,**kwargs)
        assert result['permitted_config_difference']=={'local_gradient_accumulation':{'baseline':False,'candidate':True}}


@pytest.mark.parametrize('baseline_mode,candidate_mode', [(True,False),(True,True),(False,False),(False,1)])
def test_accumulation_comparison_requires_boolean_transition(tmp_path,baseline_mode,candidate_mode):
    a,b=records(2,'old'),records(1,'new')
    for log,value in [(a,baseline_mode),(b,candidate_mode)]:
        log[0]['input_identity']['resolved_config']['local_gradient_accumulation']=value
    with pytest.raises(ValueError,match='Require baseline'):
        compare(write(tmp_path/'a',a),write(tmp_path/'b',b),steps=4,batch=128,length=512,
                hourly_usd=9.6,local_accumulation=True)
