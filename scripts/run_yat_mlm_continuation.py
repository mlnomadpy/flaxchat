"""Bounded physical-TPU MLM continuation from an explicitly authenticated base."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def parent_config(args):
    parent = args.root / 'parent'
    metadata = json.loads((parent / 'checkpoint-metadata.json').read_text())
    manifest = json.loads((parent / 'checkpoint-manifest.json').read_text())
    if (canonical_hash(metadata) != args.parent_metadata_sha256
            or canonical_hash(manifest) != args.parent_manifest_sha256
            or metadata.get('model_family') != 'modernbert'
            or manifest.get('step') != 48236):
        raise ValueError('Pinned step48236 MLM parent identity differs')
    config = metadata['resolved_config']['encoder']
    if getattr(args, 'random_init', False):
        if (json.loads((parent / 'config.json').read_text()) != config
                or hashlib.sha256((parent / 'tokenizer.json').read_bytes()).hexdigest()
                != metadata.get('tokenizer_identity')):
            raise ValueError('Pinned random-initialization architecture/tokenizer reference differs')
    return config


def training_arguments(args):
    goat_attention_score(args)
    if getattr(args, 'random_init', False) and getattr(args, 'migrate_to_goat', False):
        raise ValueError('Random initialization cannot migrate parent weights')
    config = parent_config(args)
    rate = getattr(args, 'learning_rate', None)
    rate = rate if rate is not None else (3e-4 if getattr(args, 'random_init', False) else 1e-5)
    argv = ['--config', str(args.root / 'parent/config.json'),
            '--data', str(args.root / 'corpus/train'), '--output', args.output,
            '--steps', str(args.steps), '--batch-size', str(args.batch_size),
            '--accumulation-steps', str(args.accumulation_steps), '--learning-rate', str(rate),
            '--lr-schedule', 'cosine', '--warmup-steps', '500', '--final-lr-ratio', '.05',
            '--save-every', str(args.save_every), '--keep-checkpoints', '3',
            '--seed', '1006', '--language-exponent', '.5', '--coverage-sampling',
            '--mask-probability', '.15', '--compile-diagnostics']
    if not getattr(args, 'random_init', False):
        argv += ['--initialize-from-public', str(args.root / 'parent'),
                 '--initialize-public-metadata-sha256', args.parent_metadata_sha256,
                 '--initialize-public-manifest-sha256', args.parent_manifest_sha256]
    mapping = {'compute_dtype': 'dtype', 'residual_dtype': 'residual-dtype',
               'attention_backend': 'attention-backend', 'loss_chunk_size': 'loss-chunk-size',
               'mlm_projection': 'mlm-projection', 'mlm_loss_backend': 'mlm-loss-backend',
               'mlm_vocab_tile': 'mlm-vocab-tile', 'yat_attention_block_size': 'yat-attention-block-size',
               'yat_global_attention_block_size': 'yat-global-attention-block-size'}
    for name in ('ffn_type', 'yat_epsilon', 'yat_alpha', 'yat_attention_alpha',
                 'attention_score', 'yat_compute_mode', 'yat_ffn_compute_mode',
                 'yat_softmax_backward', 'yat_attention_implementation',
                 'yat_ffn_backward_block', 'mlm_projection_capacity'):
        mapping[name] = name.replace('_', '-')
    for name, flag in mapping.items():
        if config.get(name) is not None:
            argv += ['--' + flag, str(config[name])]
    if config.get('yat_local_shards'):
        argv += ['--yat-local-shards']
    if config.get('use_remat') is False:
        argv += ['--no-remat']
    if getattr(args, 'migrate_to_goat', False):
        argv += ['--migrate-to-goat', '--attention-score', 'goat']
    if getattr(args, 'random_init', False):
        argv += ['--attention-score', goat_attention_score(args)]
    return argv


def goat_attention_score(args):
    source = getattr(args, 'goat_score_source', 'projected_values')
    if source not in ('projected_values', 'input'):
        raise ValueError('Unknown GOAT score source')
    if source == 'input' and not getattr(args, 'random_init', False):
        raise ValueError('Input-scored GOAT requires an explicit new random stage')
    return 'goat_input' if source == 'input' else 'goat'


def preflight(args):
    """Hash actual token arrays/provenance without constructing an encoder."""
    from scripts import train_encoder
    from flaxchat.encoder import EncoderConfig
    from flaxchat.encoder_data import load_prepared_rows, file_hash
    corpus = args.root / 'corpus'
    complete = json.loads((corpus / 'COMPLETE.json').read_text())
    for key, name in (('recipe_sha256', 'recipe.json'),
                      ('train_manifest_sha256', 'train/manifest.json'),
                      ('validation_manifest_sha256', 'validation/manifest.json')):
        if complete.get(key) != file_hash(corpus / name):
            raise ValueError('Completed corpus identity differs: ' + name)
    config = EncoderConfig(**parent_config(args))
    for split in ('train', 'validation'):
        rows, manifest = load_prepared_rows(args.root / 'corpus' / split, config,
            minimum_rows=args.batch_size if split == 'train' else 1)
        if rows.shape[1] != 512:
            raise ValueError('This MLM continuation requires512-token document rows')
        if manifest.get('tokenizer_sha256') != file_hash(args.root / 'parent/tokenizer.json'):
            raise ValueError('Corpus tokenizer differs from pinned parent')
        if manifest.get('split') != split:
            raise ValueError('Corpus split identity differs')
    train_encoder.run(train_encoder.parser().parse_args(training_arguments(args) + ['--preflight-only']))




def stage_options(args):
    """Forward stage selection identically to every metadata/physical child."""
    flags = []
    if getattr(args, 'migrate_to_goat', False):
        flags += ['--migrate-to-goat']
    if getattr(args, 'random_init', False):
        flags += ['--random-init']
    if getattr(args, 'goat_score_source', 'projected_values') != 'projected_values':
        flags += ['--goat-score-source', args.goat_score_source]
    if getattr(args, 'learning_rate', None) is not None:
        flags += ['--learning-rate', str(args.learning_rate)]
    return flags


def goat_stage(args):
    return getattr(args, 'migrate_to_goat', False) or getattr(args, 'random_init', False)


def baseline_action(args):
    return 'evaluate-random' if getattr(args, 'random_init', False) else 'evaluate-parent'


def baseline_evidence_key(args):
    return 'random_initializer_evaluation_sha256' if getattr(args, 'random_init', False) else 'parent_evaluation_sha256'


def qualification_status_key(args):
    return ('physical_random_update_resume_qualified' if getattr(args, 'random_init', False)
            else 'physical_import_update_resume_qualified')


def install_random_baseline(evaluator):
    """Same seed/config as fresh trainer; evaluation places it on the same data mesh."""
    from flax import nnx
    constructor = evaluator.ModernBert
    def seeded_encoder(config, **_kwargs):
        if config.attention_score not in ('goat', 'goat_input'):
            raise ValueError('Random GOAT baseline requires GOAT checkpoint configuration')
        return constructor(config, rngs=nnx.Rngs(1006))
    evaluator.ModernBert = seeded_encoder
    # evaluate() otherwise restores the step4 weights into its fresh seed0 model.
    # It still authenticates the committed checkpoint metadata and data policy.
    evaluator.restore_model_from_checkpoint = lambda *_args, **_kwargs: None


def validate_quality_report(args, report, baseline):
    factor = 1.05 if getattr(args, 'random_init', False) else 1.15
    if (report['selected_rows_sha256'] != baseline['selected_rows_sha256']
            or report['masked_tokens'] != baseline['masked_tokens']
            or not math.isfinite(baseline['masked_token_loss'])
            or not math.isfinite(report['masked_token_loss'])
            or report['masked_token_loss'] > factor * baseline['masked_token_loss']):
        raise RuntimeError(f'Heldout MLM loss exceeded {(factor - 1) * 100:.0f}% baseline regression or row identity changed')


def validate_goat_test_report(path, *, input_scores=False):
    """Require executed physical tests for the selected attention architecture."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    suites = list(root.iter('testsuite'))
    minimum = 30 if input_scores else 14
    if (not suites or sum(int(x.get('tests', '0')) for x in suites) < minimum
            or any(int(x.get(key, '0')) for x in suites
                   for key in ('skipped', 'failures', 'errors'))):
        raise RuntimeError('GOAT physical tests did not all execute and pass')
    if input_scores:
        names = {case.get('name') for case in root.iter('testcase')}
        required = {'test_input_goat_geometry_is_independent_of_value_projection[0]',
                    'test_input_goat_geometry_is_independent_of_value_projection[1]',
                    'test_migrated_bf16_mlm_update_checkpoint_exact_resume[goat_input]'}
        if not required <= names:
            raise RuntimeError('Input GOAT score/value separation or resume cases are missing')


def isolated_preflight(args):
    """Run metadata admission in an exited CPU-only process, never lease the TPU.

    Importing train_encoder can initialize JAX's backend even for metadata-only
    admission. The supervisor must remain model/JAX-free so physical children
    have exclusive access to libtpu. This child constructs no model.
    """
    command = [sys.executable, '-m', 'scripts.run_yat_mlm_continuation',
               '--root', str(args.root), '--output', args.output,
               '--parent-metadata-sha256', args.parent_metadata_sha256,
               '--parent-manifest-sha256', args.parent_manifest_sha256,
               '--steps', str(args.steps), '--batch-size', str(args.batch_size),
               '--accumulation-steps', str(args.accumulation_steps),
               '--save-every', str(args.save_every), '--preflight-only']
    command += stage_options(args)
    process = subprocess.Popen(command, env=os.environ | {'JAX_PLATFORMS': 'cpu'},
                               start_new_session=True)
    try:
        code = process.wait(timeout=min(args.max_seconds, 900))
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        # Reap even on timeout/interruption, and kill the group before any
        # physical test or training process can be started by this supervisor.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def physical_child(args):
    import jax
    if jax.default_backend() != 'tpu' or jax.process_count() != 1 or jax.device_count() != 8:
        raise RuntimeError('This continuation requires one physical eight-device TPU host')
    if args.action == 'train':
        from scripts import train_encoder
        argv = training_arguments(args) + ['--stop-after', str(args.boundary)]
        if args.resume:
            argv += ['--resume']
        train_encoder.run(train_encoder.parser().parse_args(argv))
        return
    from scripts import evaluate_encoder, train_encoder
    public_receipt = None
    if args.action == 'evaluate-random':
        if not getattr(args, 'random_init', False):
            raise ValueError('Random baseline requires random initialization stage')
        install_random_baseline(evaluate_encoder)
    if args.action == 'evaluate-parent' and getattr(args, 'random_init', False):
        raise ValueError('Random stage cannot import trained parent weights')
    if args.action == 'evaluate-parent':
        from flaxchat.encoder import EncoderConfig
        from flaxchat.encoder_data import file_hash
        parent = args.root / 'parent'
        config = EncoderConfig(**parent_config(args))
        committed, public_receipt = train_encoder.public_initialization_metadata(
            parent, config, file_hash(parent / 'tokenizer.json'),
            args.parent_metadata_sha256, args.parent_manifest_sha256)
        # Reuse the exact child checkpoint's row/masking evaluation policy while
        # substituting authenticated original MLM weights, never HF dot/GeGLU.
        def restore_parent(model, *_args, **_kwargs):
            train_encoder.restore_public_mlm(model, parent, committed)
        evaluate_encoder.restore_model_from_checkpoint = restore_parent
        if getattr(args, 'migrate_to_goat', False):
            # Preserve child data/masking selection, but build the original parent
            # architecture for the paired pre-migration reference.
            load_metadata = evaluate_encoder.load_checkpoint_metadata
            def parent_evaluation_metadata(*pos, **kwargs):
                metadata = load_metadata(*pos, **kwargs)
                return metadata | {'resolved_config': metadata['resolved_config'] | {
                    'encoder': parent_config(args)}}
            evaluate_encoder.load_checkpoint_metadata = parent_evaluation_metadata
    report = evaluate_encoder.evaluate(args.output, str(args.root / 'corpus/validation'),
        train_data=str(args.root / 'corpus/train'), batch_size=8, max_rows=512,
        seed=2026, checkpoint_step=args.boundary, balanced_languages=True)
    if public_receipt:
        report.update(parameter_source='authenticated_public_mlm_parent',
                      parent_initialization=public_receipt, checkpoint_step=48236,
                      evaluation_policy_checkpoint_step=args.boundary)
    if args.action == 'evaluate-random':
        report.update(parameter_source='random_initializer', initialization_seed=1006,
                      initialization_policy='random_goat_fresh_optimizer_schedule_cursor_no_parent_weights',
                      attention_score=goat_attention_score(args),
                      checkpoint_step=0, evaluation_policy_checkpoint_step=args.boundary)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def qualification_identity(args):
    from scripts.representation_run import source_snapshot, source_tree_digest
    values = {name: os.environ.get(variable) for name, variable in (
        ('manifest_sha256', 'FLAXCHAT_RUN_MANIFEST_SHA256'),
        ('runtime_lock_sha256', 'FLAXCHAT_RUNTIME_LOCK_SHA256'))}
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in '0123456789abcdef' for c in value) for value in values.values()):
        raise ValueError('Bound cloud manifest and runtime identities are required')
    tree = source_snapshot(args.root / 'source')
    if not tree:
        raise ValueError('Frozen source tree is missing')
    result = dict(schema_version=1, passed=True, backend='tpu', devices=8, processes=1,
                  source_tree_sha256=source_tree_digest(tree),
                  training_arguments_sha256=canonical_hash(training_arguments(args)), **values)
    if getattr(args, 'random_init', False):
        result['random_initialization'] = dict(seed=1006, attention_score=goat_attention_score(args),
            reference_metadata_sha256=args.parent_metadata_sha256,
            reference_manifest_sha256=args.parent_manifest_sha256,
            weights_loaded=False, quality_baseline='identical_random_initializer',
            maximum_loss_ratio=1.05)
    return result


def required_qualification_tests(args):
    if getattr(args, 'random_init', False):
        names = ['mlm-random-initialization-update', 'mlm-random-checkpoint-resume',
                 'mlm-heldout-random-initializer-gate', 'goat-physical-forward-backward-migration']
        if goat_attention_score(args) == 'goat_input':
            names.append('goat-input-score-value-separation')
        return names
    names = ['mlm-parent-import-update', 'mlm-checkpoint-resume', 'mlm-heldout-parent-gate']
    if getattr(args, 'migrate_to_goat', False):
        names.append('goat-physical-forward-backward-migration')
    return names


def verify_continuation(args, logs):
    qualified = json.loads((args.root / 'mlm-qualification.json').read_text())
    expected = qualification_identity(args)
    if any(qualified.get(key) != value for key, value in expected.items()):
        raise ValueError('MLM qualification no longer matches deployed source/runtime/stage')
    required = set(required_qualification_tests(args))
    tests = qualified.get('tests', [])
    if len(tests) != len(required) or {item.get('id') for item in tests if item.get('status') == 'passed'} != required:
        raise ValueError('MLM physical qualification tests are incomplete')
    if goat_stage(args):
        xml = logs / 'goat-physical-tests.xml'
        if (not xml.is_file() or hashlib.sha256(xml.read_bytes()).hexdigest()
                != qualified.get('goat_physical_tests_sha256')):
            raise ValueError('GOAT physical qualification evidence changed')
    receipt = json.loads((logs / 'status.json').read_text())
    if (receipt.get('completed_step') != 8 or receipt.get(qualification_status_key(args)) is not True
            or receipt.get('status') != 'qualified'):
        raise ValueError('Expected completed physical MLM qualification at step8')
    baseline = json.loads((logs / (baseline_action(args) + '-4.json')).read_text())
    evaluation = json.loads((logs / 'evaluate-8.json').read_text())
    if (canonical_hash(baseline) != qualified.get(baseline_evidence_key(args))
            or canonical_hash(evaluation) != qualified.get('step8_evaluation_sha256')):
        raise ValueError('Qualified heldout evidence changed')
    return receipt, baseline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--parent-metadata-sha256', required=True)
    p.add_argument('--parent-manifest-sha256', required=True)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--migrate-to-goat', action='store_true')
    initialization.add_argument('--random-init', action='store_true')
    p.add_argument('--goat-score-source', choices=['projected_values', 'input'],
                   default='projected_values', help='Input scores use x directly; values still use W_V')
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--accumulation-steps', type=int, default=4)
    p.add_argument('--max-seconds', type=int, default=39600)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--preflight-only', action='store_true')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--qualify-only', action='store_true')
    modes.add_argument('--continue-qualified', action='store_true')
    p.add_argument('--action', choices=['supervise', 'train', 'evaluate', 'evaluate-parent', 'evaluate-random'], default='supervise')
    p.add_argument('--boundary', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report', type=Path)
    args = p.parse_args(argv)
    goat_attention_score(args)
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Learning rate must be finite and positive')
    parent_config(args)
    if args.preflight_only:
        preflight(args)
        return
    if args.action != 'supervise':
        physical_child(args)
        return
    if (not args.output.startswith('gs://') or args.steps <= 500 or args.eval_every < 8
            or min(args.save_every, args.max_seconds, args.batch_size, args.accumulation_steps) < 1):
        raise ValueError('Invalid bounded MLM continuation configuration')
    isolated_preflight(args)
    logs = args.root / 'mlm-run-evidence'
    logs.mkdir(parents=True, exist_ok=True)
    saved = verify_continuation(args, logs) if args.continue_qualified else None
    bound_identity = qualification_identity(args) if args.qualify_only else None
    claim = logs / ('continuation-claim.json' if args.continue_qualified else 'claim.json')
    with claim.open('x') as stream:
        json.dump({'started_unix': time.time(), 'objective': 'masked_language_modeling',
                   'arguments': vars(args)}, stream, default=str)
    def stop_for_deadline(_signum, _frame):
        raise TimeoutError('MLM supervisor received termination signal')
    signal.signal(signal.SIGTERM, stop_for_deadline)
    deadline = time.monotonic() + args.max_seconds
    receipt = {'objective': 'masked_language_modeling', 'status': 'running', 'completed_step': 0,
               'parent_step': None if args.random_init else 48236, 'evaluations': []}
    if args.random_init:
        receipt.update(initialization='random', initialization_seed=1006,
                       initialization_policy='random_goat_fresh_optimizer_schedule_cursor_no_parent_weights',
                       attention_score=goat_attention_score(args), architecture_reference_step=48236)
    if saved:
        receipt = saved[0] | {'status': 'running'}
    base = [sys.executable, '-m', 'scripts.run_yat_mlm_continuation', '--root', str(args.root),
            '--output', args.output, '--parent-metadata-sha256', args.parent_metadata_sha256,
            '--parent-manifest-sha256', args.parent_manifest_sha256, '--steps', str(args.steps),
            '--batch-size', str(args.batch_size), '--accumulation-steps', str(args.accumulation_steps),
            '--save-every', str(args.save_every)]
    base += stage_options(args)

    def publish():
        (logs / 'status.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        subprocess.run(['gcloud', 'storage', 'rsync', str(logs), args.output.rstrip('/') + '-evidence',
                        '--recursive'], timeout=90, check=True)

    def run(action, boundary, *, resume=False):
        remaining = deadline - time.monotonic() - 120
        if remaining <= 0:
            raise TimeoutError('Bounded MLM worker deadline reached')
        command = base + ['--action', action, '--boundary', str(boundary)]
        if resume:
            command += ['--resume']
        report = logs / f'{action}-{boundary}.json'
        if action != 'train':
            command += ['--report', str(report)]
        log = logs / f'{action}-{boundary}.log'
        with log.open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('MLM child reached bounded worker deadline') from error
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if code:
            raise RuntimeError(f'{action} boundary{boundary} failed; see {log.name}')
        if action == 'train':
            events = []
            for line in log.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get('event') == 'train_step':
                    events.append(event)
            if (not events or events[-1]['step'] != boundary
                    or any(not item['updated'] or item['masked_tokens'] <= 0
                           or not math.isfinite(item['loss'])
                           or not math.isfinite(item['gradient_norm_before_clip']) for item in events)):
                raise RuntimeError('MLM update acceptance failed')
            receipt['completed_step'] = boundary
            publish()
            return None
        return json.loads(report.read_text())

    def check_quality(report, baseline):
        validate_quality_report(args, report, baseline)
        receipt['evaluations'].append({'step': report['checkpoint_step'], 'loss': report['masked_token_loss']})
        publish()

    try:
        if args.continue_qualified:
            baseline = saved[1]
        else:
            if goat_stage(args):
                # Gate the actual shared-V derivatives and diagonal masking on
                # this physical TPU before restoring or updating production weights.
                env = os.environ | {'FLAXCHAT_PHYSICAL_TPU': '1'}
                with (logs / 'goat-physical-tests.log').open('wb') as stream:
                    subprocess.run([sys.executable, '-m', 'pytest', '-q',
                        'tests/test_goat_physical_tpu.py',
                        '--junitxml=' + str(logs / 'goat-physical-tests.xml')],
                        cwd=args.root / 'source', env=env, stdout=stream,
                        stderr=subprocess.STDOUT, check=True,
                        timeout=max(1, deadline - time.monotonic() - 120))
                validate_goat_test_report(logs / 'goat-physical-tests.xml',
                    input_scores=goat_attention_score(args) == 'goat_input')
                receipt['goat_physical_tests_passed'] = True
                publish()
            run('train', 4)
            baseline = run(baseline_action(args), 4)
            receipt['random_initializer_masked_token_loss' if args.random_init else 'parent_masked_token_loss'] = baseline['masked_token_loss']
            check_quality(run('evaluate', 4), baseline)
            run('train', 8, resume=True)
            evaluated = run('evaluate', 8)
            check_quality(evaluated, baseline)
            receipt[qualification_status_key(args)] = True
            if args.qualify_only:
                if evaluated.get('backend') != 'tpu' or len(evaluated.get('devices', [])) != 8:
                    raise ValueError('Physical evaluation topology evidence differs')
                if qualification_identity(args) != bound_identity:
                    raise ValueError('Deployed stage changed during qualification')
                qualification = dict(bound_identity,
                    step8_evaluation_sha256=canonical_hash(evaluated),
                    tests=[{'id': name, 'status': 'passed'} for name in required_qualification_tests(args)])
                qualification[baseline_evidence_key(args)] = canonical_hash(baseline)
                if goat_stage(args):
                    qualification['goat_physical_tests_sha256'] = hashlib.sha256(
                        (logs / 'goat-physical-tests.xml').read_bytes()).hexdigest()
                text = json.dumps(qualification, indent=2, allow_nan=False) + '\n'
                (args.root / 'mlm-qualification.json').write_text(text)
                (logs / 'mlm-qualification.json').write_text(text)
                receipt['status'] = 'qualified'
                return
            publish()
        boundary = min(args.eval_every, args.steps)
        while receipt['completed_step'] < args.steps:
            run('train', boundary, resume=True)
            check_quality(run('evaluate', boundary), baseline)
            boundary = min(boundary + args.eval_every, args.steps)
        receipt['status'] = 'completed'
    except TimeoutError:
        receipt['status'] = 'lease_finished_checkpoint_recovery_available'
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        publish()


if __name__ == '__main__':
    main()
