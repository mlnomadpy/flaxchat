"""Pure metadata checks; no JAX, model execution, TPU mocks, or cloud calls."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('goat_recovery_external', ROOT / 'recovery.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture():
    original = r.read(ROOT / 'original-qualification.json')
    baseline = r.read(ROOT / 'original-random.json')
    reference = r.read(ROOT / 'original-evaluate2000.json')
    identity = {k: original[k] for k in ('schema_version', 'passed', 'backend', 'devices',
        'processes', 'source_tree_sha256', 'training_arguments_sha256', 'manifest_sha256',
        'runtime_lock_sha256', 'random_initialization')}
    identity['manifest_sha256'] = 'f' * 64
    required = [x['id'] for x in original['tests']]
    return identity, original, baseline, reference, required, r.sha(ROOT / 'original-tests.xml')


def test_original_real_receipt_and_baseline_are_accepted():
    r.validate_original(*fixture())
    assert 'jax' not in sys.modules


@pytest.mark.parametrize('key', ['source_tree_sha256', 'runtime_lock_sha256', 'training_arguments_sha256'])
def test_changed_source_runtime_or_training_identity_rejected(key):
    args = list(fixture())
    args[0][key] = '0' * 64
    with pytest.raises(ValueError, match='identity differs'):
        r.validate_original(*args)


@pytest.mark.parametrize('mutation', ['nan', 'loss', 'rows', 'tokens', 'backend', 'devices', 'step'])
def test_recovery_evaluation_gate(mutation):
    reference = fixture()[3]
    report = copy.deepcopy(reference)
    report['checkpoint_step'] = 35500
    report['masked_token_loss'] = reference['masked_token_loss'] * .99
    r.validate_report(report, reference, step=35500)
    changes = {'nan': ('masked_token_loss', float('nan')), 'loss': ('masked_token_loss', 4.1),
        'rows': ('selected_rows_sha256', 'wrong'), 'tokens': ('masked_tokens', 0),
        'backend': ('backend', 'cpu'), 'devices': ('devices', []), 'step': ('checkpoint_step', 0)}
    key, value = changes[mutation]
    report[key] = value
    with pytest.raises(ValueError, match='gate failed'):
        r.validate_report(report, reference, step=35500)


def test_exact_resume_update_admission():
    events = [dict(step=i, updated=True, loss=3., masked_tokens=20,
                   gradient_norm_before_clip=.5) for i in range(35501, 35505)]
    r.validate_updates(events, 35500, 35504)
    for bad in (events[1:], events[:-1], events[::-1], events[:1] + events[2:]):
        with pytest.raises(ValueError):
            r.validate_updates(bad, 35500, 35504)
    events[1]['updated'] = False
    with pytest.raises(ValueError):
        r.validate_updates(events, 35500, 35504)


def test_original_stage_flags_are_fixed():
    args = r.fixed_arguments(r.ORIGINAL_ROOT, r.OUTPUT)
    assert (args.steps, args.batch_size, args.accumulation_steps, args.save_every) == (100000, 64, 4, 250)
    assert args.random_init and not args.migrate_to_goat and args.goat_score_source == 'input'
    assert args.learning_rate is None
    assert not r.EVIDENCE.startswith(r.OUTPUT)


def test_original_evidence_hash_failure():
    args = list(fixture())
    args[-1] = '0' * 64
    with pytest.raises(ValueError, match='XML hash'):
        r.validate_original(*args)


def test_full_model_batch16_protocol_required():
    reference = fixture()[3]
    report = copy.deepcopy(reference)
    report.update(checkpoint_step=35500, evaluation_batch_size=16, evaluated_rows=512)
    r.validate_batch16_report(report, reference, step=35500)
    for field, value in [('evaluation_batch_size', 8), ('evaluation_batch_size', 64), ('evaluated_rows', 8)]:
        wrong = report | {field: value}
        with pytest.raises(ValueError, match='batch16'):
            r.validate_batch16_report(wrong, reference, step=35500)
    with pytest.raises(ValueError, match='gate failed'):
        r.validate_batch16_report(report | {'masked_token_loss': float('nan')}, reference, step=35500)


def test_new_manifest_qualification_and_budget_binding():
    import json
    manifest = json.loads((ROOT / 'run.json').read_text())
    assert 'recovery-checkpoint35500' in manifest['qualification']['required_tests']
    assert 'recovery-checkpoint3000' not in manifest['qualification']['required_tests']
    assert 'range(36000, 100001, 2000)' in (ROOT / 'recovery.py').read_text()
    assert manifest['workload'][manifest['workload'].index('--max-seconds') + 1] == '32400'
    assert manifest['deployment']['attempt_seconds'] == 36000
    assert r.EVIDENCE.endswith('1009c/training-evidence')
