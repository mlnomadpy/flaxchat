"""Model-free stage-global selection and dual-checkpoint reconciliation."""
import math
import copy
import json
import re
from pathlib import Path
from flaxchat.embedding_contract import canonical_hash


def checkpoint_stage_identity(metadata):
    return {'resolved_config': metadata.get('resolved_config'),
            'tokenizer': metadata.get('tokenizer_identity'),
            'source_python_sha256': metadata.get('source_python_sha256'),
            'data_manifest': metadata.get('data_manifest_identity')}


def admit_quality_migration(path, expected, output):
    """Only a pinned report-only policy/source migration with preserved schedule."""
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Migration specification exceeds bounded size')
    spec = json.loads(path.read_text())
    if (spec.get('format') != 'flaxchat-quality-policy-migration-v1'
            or spec.get('checkpoint_prefix') != output
            or spec.get('target_source_python_sha256') != expected['source_python_sha256']):
        raise ValueError('Migration target/source identity mismatch')
    old = spec.get('previous_identity')
    if not isinstance(old, dict) or set(old) != set(expected):
        raise ValueError('Migration requires complete previous stage identity')
    for key in ('checkpoint', 'best'):
        pin = spec.get(key, {})
        if (type(pin.get('step')) is not int or pin['step'] < (1 if key == 'checkpoint' else 0)
                or not re.fullmatch('[0-9a-f]{64}', str(pin.get('manifest_sha256', '')))):
            raise ValueError('Migration requires committed checkpoint and best pins')
    before, after = copy.deepcopy(old), copy.deepcopy(expected)
    for identity in (before, after):
        identity.pop('source_python_sha256')
        identity['resolved_config'].pop('source_receipt', None)
    previous_recipe, next_recipe = before['resolved_config'], after['resolved_config']
    old_policy = previous_recipe['development_policy'].get('regression_action', 'stop')
    new_policy = next_recipe['development_policy'].pop('regression_action', 'stop')
    previous_recipe['development_policy'].pop('regression_action', None)
    if old_policy != 'stop' or new_policy != 'report-only':
        raise ValueError('Migration only changes stop to report-only')
    old_steps = previous_recipe['steps']
    schedule = next_recipe.pop('schedule_steps', next_recipe['steps'])
    old_schedule = previous_recipe.pop('schedule_steps', old_steps)
    if schedule != old_schedule or next_recipe['steps'] < old_steps:
        raise ValueError('Migration must preserve the exact learning-rate schedule')
    next_recipe['steps'] = old_steps
    if before != after:
        raise ValueError('Migration changes data, optimizer, model or other immutable recipe fields')
    return spec, {'format': spec['format'], 'specification_sha256': canonical_hash(spec),
                  'previous_identity_sha256': canonical_hash(old),
                  'source_checkpoint': spec['checkpoint'], 'source_best': spec['best'],
                  'preserved_schedule_steps': schedule}


def migration_restore_identity(metadata, expected, migration, *, best=False):
    actual = checkpoint_stage_identity(metadata)
    if actual == expected:
        return expected
    if migration is None or actual != migration['previous_identity']:
        raise ValueError('Checkpoint immutable stage differs from admitted migration')
    pin = migration['best' if best else 'checkpoint']
    receipt = metadata.get('committed_receipt', {})
    if metadata.get('step') != pin['step'] or receipt.get('manifest_sha256') != pin['manifest_sha256']:
        raise ValueError('Checkpoint differs from independently pinned migration artifact')
    return migration['previous_identity']


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


def validate_resume_evaluation(quality, *, cursor, horizon, every, max_regression, regression_action='stop'):
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
    if regression_action not in ('stop', 'report-only'):
        raise ValueError('Unknown development regression action')
    if not actual['passed'] and regression_action == 'stop':
        raise ValueError('Resume blocked by failed development gate; use an explicit reviewed new stage')
    return record
