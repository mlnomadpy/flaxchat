"""Selection/commit bookkeeping contracts only; no model/backend execution."""
import pytest
from flaxchat.embedding_recovery import evaluation_due, selection_record, reconcile_best, require_save_success


@pytest.mark.parametrize('cursor,horizon', [(4, 8), (4, 4)])
def test_failed_quality_resume_is_terminal_including_horizon(cursor, horizon):
    from flaxchat.embedding_recovery import validate_resume_evaluation
    from flaxchat.embedding_quality import quality_gate
    baseline = fixture()[0]['quality_selection']['baseline']
    metrics = {'source': {**baseline['source'], 'mrr': .1}}
    state = {'baseline': baseline, 'last_evaluation': {'step': 4, 'metrics': metrics,
        'gate': quality_gate(metrics, baseline, max_regression=.02)}}
    with pytest.raises(ValueError, match='blocked by failed'):
        validate_resume_evaluation(state, cursor=cursor, horizon=horizon, every=4, max_regression=.02)
    state['last_evaluation']['gate']['passed'] = True
    with pytest.raises(ValueError, match='disagrees'):
        validate_resume_evaluation(state, cursor=cursor, horizon=horizon, every=4, max_regression=.02)


def test_resume_evaluation_requires_latest_scheduled_evidence():
    from flaxchat.embedding_recovery import validate_resume_evaluation
    from flaxchat.embedding_quality import quality_gate
    baseline = fixture()[0]['quality_selection']['baseline']
    record = {'step': 0, 'metrics': baseline, 'gate': quality_gate(baseline, baseline, max_regression=.02)}
    state = {'baseline': baseline, 'last_evaluation': record}
    assert validate_resume_evaluation(state, cursor=2, horizon=8, every=4, max_regression=.02) == record
    with pytest.raises(ValueError, match='cursor differs'):
        validate_resume_evaluation(state, cursor=4, horizon=8, every=4, max_regression=.02)


def fixture(step=4, score=.7):
    baseline = {'source': {'mrr': .5, 'recall_at_1': .4, 'recall_at_10': .8}}
    expected = {'resolved_config': {'steps': 12}, 'tokenizer': 'tokenizer',
                'source_python_sha256': 'source', 'data_manifest': 'data'}
    receipt = {'step': step, 'manifest_sha256': 'a' * 64,
               'model_state': {'parameter': {'sha256': 'b' * 64}}}
    metadata = {'step': step, 'resolved_config': expected['resolved_config'],
                'tokenizer_identity': 'tokenizer', 'source_python_sha256': 'source',
                'data_manifest_identity': 'data', 'committed_receipt': receipt,
                'quality_selection': selection_record(step, score, baseline)}
    return metadata, expected


def test_segment_stop_does_not_create_a_selection_point():
    assert [step for step in range(13) if evaluation_due(step, 4, 12)] == [0, 4, 8, 12]
    assert not evaluation_due(2, 4, 12)
    assert evaluation_due(11, 4, 11)  # True stage horizon is always selected.


def test_ahead_best_is_preserved_and_bound_to_artifact():
    selected, expected = fixture(8, .8)
    previous, _ = fixture(4, .7)
    quality = reconcile_best(None, previous, expected, horizon=12, every=4)
    reconciled = reconcile_best(quality, selected, expected, horizon=12, every=4)
    assert reconciled['best_step'] == 8 and reconciled['best_score'] == .8
    assert reconciled['best_receipt'] == selected['committed_receipt']
    assert reconcile_best(reconciled, selected, expected, horizon=12, every=4) == reconciled


def test_same_step_wrong_artifact_and_incompatible_baseline_rejected():
    metadata, expected = fixture()
    quality = reconcile_best(None, metadata, expected, horizon=12, every=4)
    altered = {**metadata, 'committed_receipt': {**metadata['committed_receipt'], 'manifest_sha256': 'c' * 64}}
    with pytest.raises(ValueError, match='artifact identity'):
        reconcile_best(quality, altered, expected, horizon=12, every=4)
    with pytest.raises(ValueError, match='baseline'):
        reconcile_best({**quality, 'baseline': {'other': {}}}, metadata, expected, horizon=12, every=4)


def test_older_worse_wrong_stage_and_off_cadence_best_rejected():
    best, expected = fixture(8, .8)
    quality = reconcile_best(None, best, expected, horizon=12, every=4)
    for step, score in [(4, .7), (12, .6), (12, .8), (10, .9)]:
        candidate, _ = fixture(step, score)
        with pytest.raises(ValueError):
            reconcile_best(quality, candidate, expected, horizon=12, every=4)
    with pytest.raises(ValueError, match='stage differs'):
        reconcile_best(None, best, {**expected, 'tokenizer': 'wrong'}, horizon=12, every=4)


def test_initial_best_bootstrap_is_explicit_and_only_step_zero():
    metadata, expected = fixture(0, .5)
    pending = {'baseline': metadata['quality_selection']['baseline'], 'best_step': 0, 'best_score': .5, 'best_receipt': None}
    with pytest.raises(ValueError):
        reconcile_best(pending, metadata, expected, horizon=12, every=4)
    assert reconcile_best(pending, metadata, expected, horizon=12, every=4, bootstrap=True)['best_step'] == 0
    later, _ = fixture()
    with pytest.raises(ValueError):
        reconcile_best({**pending, 'best_step': 4, 'best_score': .7}, later, expected, horizon=12, every=4, bootstrap=True)


def test_save_refusal_is_never_success():
    require_save_success(True, 4)
    for rejected in (False, None, 1):
        with pytest.raises(RuntimeError, match='rejected'):
            require_save_success(rejected, 4)
