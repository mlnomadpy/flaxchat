"""Model-free stage-global selection and dual-checkpoint reconciliation."""
import math
from flaxchat.embedding_contract import canonical_hash


def evaluation_due(step, every, horizon):
    if any(type(value) is not int for value in (step, every, horizon)) or not 0 <= step <= horizon or every < 1 or horizon < 1:
        raise ValueError('Invalid evaluation cursor/cadence')
    return step == 0 or step % every == 0 or step == horizon


def selection_record(step, score, baseline):
    if type(step) is not int or step < 0 or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError('Invalid selected checkpoint score/cursor')
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError('Selected checkpoint requires development baseline')
    return {'step': step, 'score': float(score), 'baseline': baseline}


def reconcile_best(quality, metadata, expected, *, horizon, every, bootstrap=False):
    """Bind recent recovery to the durable authoritative best (possibly ahead).

    Does not rewind optimizer/sample cursor. A best ahead of recent is preserved
    while earlier steps replay; it must belong to this exact stage and baseline.
    Missing/older/incompatible best fails rather than silently losing selection.
    """
    selection = metadata.get('quality_selection')
    if not isinstance(selection, dict):
        raise ValueError('Best checkpoint lacks authenticated selection metadata')
    record = selection_record(selection.get('step'), selection.get('score'), selection.get('baseline'))
    step = metadata.get('step')
    if step != record['step'] or not evaluation_due(step, every, horizon):
        raise ValueError('Best checkpoint is outside stage evaluation cadence')
    actual = {'resolved_config': metadata.get('resolved_config'),
              'tokenizer': metadata.get('tokenizer_identity'),
              'source_python_sha256': metadata.get('source_python_sha256'),
              'data_manifest': metadata.get('data_manifest_identity')}
    if actual != expected:
        raise ValueError('Best checkpoint immutable stage differs')
    receipt = metadata.get('committed_receipt')
    if not isinstance(receipt, dict) or receipt.get('step') != step or not receipt.get('manifest_sha256') or not receipt.get('model_state'):
        raise ValueError('Best checkpoint requires committed artifact receipt')
    if quality is not None:
        previous = selection_record(quality.get('best_step'), quality.get('best_score'), quality.get('baseline'))
        if canonical_hash(previous['baseline']) != canonical_hash(record['baseline']):
            raise ValueError('Best/recovery development baseline differs')
        if step < previous['step'] or record['score'] < previous['score']:
            raise ValueError('Durable best is older/worse than committed recovery selection')
        if step == previous['step']:
            if record['score'] != previous['score'] or (quality.get('best_receipt') != receipt
                    and not (bootstrap and step == 0 and quality.get('best_receipt') is None)):
                raise ValueError('Best/recovery selected artifact identity differs')
        elif record['score'] <= previous['score']:
            raise ValueError('Ahead best must be a strict selection improvement')
    return {'baseline': record['baseline'], 'best_score': record['score'],
            'best_step': step, 'best_receipt': receipt}


def require_save_success(saved, step):
    if saved is not True:
        raise RuntimeError(f'Checkpoint save request was rejected at step {step}')


def validate_resume_evaluation(quality, *, cursor, horizon, every, max_regression):
    """A committed failed quality decision is terminal for this exact stage."""
    from flaxchat.embedding_quality import quality_gate
    record = quality.get('last_evaluation')
    if not isinstance(record, dict):
        raise ValueError('Resume lacks authenticated last development evaluation')
    latest_due = horizon if cursor == horizon else (cursor // every) * every
    if record.get('step') != latest_due or not evaluation_due(latest_due, every, horizon):
        raise ValueError('Resume development evaluation cursor differs from stage cadence')
    actual = quality_gate(record.get('metrics'), quality.get('baseline'), max_regression=max_regression)
    if record.get('gate') != actual:
        raise ValueError('Resume development gate disagrees with authenticated metrics')
    if not actual['passed']:
        raise ValueError('Resume blocked by failed development gate; use an explicit reviewed new stage')
    return record
