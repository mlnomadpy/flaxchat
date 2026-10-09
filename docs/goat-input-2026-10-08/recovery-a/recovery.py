"""External, JAX-free supervisor for exact GOAT-input checkpoint27750 recovery."""
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

ORIGINAL_MANIFEST = '3e87cbe501a2b568f7028dc69661f1e7c013d4e50f4ef144df79abdae4b3dcfc'
ORIGINAL_ROOT = '/tmp/flaxchat-representation-yat-embed-torch-goat-input-1007b-input-goat-scratch-base-mlm'
OUTPUT = 'gs://azettaai-yat-eval-0929/goat-input-1007b/yat-embed-torch-goat-input-1007b/input-goat-scratch-base-mlm/checkpoints'
EVIDENCE = 'gs://azettaai-yat-eval-0929/goat-input-1008a/training-evidence'
PARENT_METADATA = '35cff96b97c0a9d5ba401c39545dfeff62e1c73e527d5f5661ba0960042bf4f4'
PARENT_MANIFEST = 'bf891c59974ec83f69349e08e704a99388fd8bdad6fd356ee2be9dfa1fb1b4d3'
REFERENCE_LOSS = 3.869532731642492


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fixed_arguments(root, output):
    return argparse.Namespace(root=Path(root), output=output, steps=100000,
        batch_size=64, accumulation_steps=4, save_every=250, eval_every=2000,
        parent_metadata_sha256=PARENT_METADATA, parent_manifest_sha256=PARENT_MANIFEST,
        random_init=True, migrate_to_goat=False, goat_score_source='input', learning_rate=None)


def validate_original(identity, original, baseline, evaluation2000, required, original_xml_sha):
    expected = identity | {'manifest_sha256': ORIGINAL_MANIFEST}
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError('Original qualification source/runtime/training arguments identity differs')
    tests = original.get('tests', [])
    if len(tests) != len(required) or {x.get('id') for x in tests if x.get('status') == 'passed'} != set(required):
        raise ValueError('Original qualification tests are incomplete')
    if original.get('goat_physical_tests_sha256') != original_xml_sha:
        raise ValueError('Original physical XML hash differs')
    if original.get('random_initializer_evaluation_sha256') != digest(baseline):
        raise ValueError('Original random baseline hash differs')
    if (evaluation2000.get('checkpoint_step') != 2000
            or evaluation2000.get('masked_token_loss') != REFERENCE_LOSS):
        raise ValueError('Original step2000 reference differs')
    validate_report(evaluation2000, baseline, maximum_ratio=1.05)


def validate_report(report, reference, *, step=None, maximum_ratio=1.05):
    loss, base = report.get('masked_token_loss'), reference.get('masked_token_loss')
    if (not isinstance(loss, (float, int)) or not math.isfinite(loss)
            or not isinstance(base, (float, int)) or not math.isfinite(base)
            or loss > maximum_ratio * base
            or report.get('selected_rows_sha256') != reference.get('selected_rows_sha256')
            or not isinstance(report.get('selected_rows_sha256'), str)
            or not report.get('masked_tokens')
            or report.get('masked_tokens') != reference.get('masked_tokens')
            or report.get('backend') != 'tpu' or len(report.get('devices', [])) != 8
            or (step is not None and report.get('checkpoint_step') != step)):
        raise ValueError('Recovery heldout loss/topology/row identity gate failed')


def validate_batch16_report(report, reference, *, step):
    validate_report(report, reference, step=step)
    if report.get('evaluation_batch_size') != 16 or report.get('evaluated_rows') != 512:
        raise ValueError('Recovery requires explicitly qualified batch16 / 512-row evaluation')


def validate_updates(events, start, boundary):
    if (not events or events[0].get('step') != start + 1 or events[-1].get('step') != boundary
            or [x.get('step') for x in events] != list(range(start + 1, boundary + 1))
            or any(x.get('updated') is not True or x.get('masked_tokens', 0) <= 0
                   or not math.isfinite(x.get('loss', float('nan')))
                   or not math.isfinite(x.get('gradient_norm_before_clip', float('nan'))) for x in events)):
        raise ValueError('Recovery updates did not resume exactly or contain nonfinite/skipped updates')


class DeadlineReached(TimeoutError):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evidence-prefix', required=True)
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=32400)
    args = parser.parse_args(argv)
    if (str(args.root) != ORIGINAL_ROOT or args.output != OUTPUT
            or args.evidence_prefix != EVIDENCE or not 120 < args.max_seconds <= 32400):
        raise ValueError('Recovery root/output/evidence namespace or deadline differs from approved stage')
    sys.path.insert(0, str(args.root / 'source'))
    from scripts.run_yat_mlm_continuation import (qualification_identity, training_arguments,
        required_qualification_tests, validate_goat_test_report)
    stage = fixed_arguments(args.root, args.output)
    identity = qualification_identity(stage)
    # This call also authenticates the pinned architecture/tokenizer reference.
    training_arguments(stage)
    original = read(args.root / 'original-qualification.json')
    baseline = read(args.root / 'original-evaluate-random-4.json')
    reference = read(args.root / 'original-evaluate2000.json')
    original_xml = args.root / 'original-goat-physical-tests.xml'
    required = required_qualification_tests(stage)
    validate_original(identity, original, baseline, reference, required, sha(original_xml))
    validate_goat_test_report(original_xml, input_scores=True)
    original8 = read(args.root / 'original-evaluate-8.json')
    if digest(original8) != original.get('step8_evaluation_sha256'):
        raise ValueError('Original step8 evaluation hash differs')
    validate_report(original8, baseline, step=8)
    bound = identity | {'recovery_driver_sha256': sha(__file__),
        'original_qualification_sha256': sha(args.root / 'original-qualification.json'),
        'original_evaluate2000_sha256': sha(args.root / 'original-evaluate2000.json'),
        'checkpoint_namespace': args.output, 'recovery_step': 27750,
        'evaluation_protocol': {'evaluation_batch_size': 16, 'evaluated_rows': 512,
            'seed': 2026, 'balanced_languages': True, 'batch8_full_model_status': 'known_nonfinite_unresolved'}}
    logs = args.root / 'recovery-a-evidence'
    logs.mkdir(exist_ok=True)
    with (logs / ('qualification-claim.json' if args.qualify_only else 'continuation-claim.json')).open('x') as stream:
        json.dump({'started_unix': time.time(), 'identity': bound}, stream)
    deadline = time.monotonic() + args.max_seconds
    status = {'objective': 'masked_language_modeling', 'attention_score': 'goat_input',
              'status': 'running', 'completed_step': 27750, 'evaluations': [], 'identity': bound}
    def stopped(signum, frame):
        raise DeadlineReached('Recovery supervisor termination signal')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    def publish():
        write(logs / 'status.json', status)
        subprocess.run(['gcloud', 'storage', 'rsync', str(logs), args.evidence_prefix, '--recursive'],
                       timeout=90, check=True)
    def child(command, logname, env=None):
        remaining = deadline - time.monotonic() - 120
        if remaining <= 0:
            raise DeadlineReached('Recovery lease deadline')
        path = logs / logname
        with path.open('wb') as stream:
            process = subprocess.Popen(command, cwd=args.root / 'source', env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise DeadlineReached('Recovery child deadline') from error
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if code:
            raise RuntimeError(f'Recovery child exited {code}: {path.name}')
        return path
    base = [sys.executable, '-m', 'scripts.run_yat_mlm_continuation',
        '--root', str(args.root), '--output', args.output,
        '--parent-metadata-sha256', PARENT_METADATA, '--parent-manifest-sha256', PARENT_MANIFEST,
        '--random-init', '--goat-score-source', 'input', '--steps', '100000',
        '--batch-size', '64', '--accumulation-steps', '4', '--save-every', '250']
    def evaluate(step):
        report = logs / f'evaluate-{step}.json'
        # Qualified protocol override lives outside frozen training source.
        # The full-size batch8 forward defect is unresolved, not silently fixed.
        program = ("import json,sys; from pathlib import Path; import jax; "
                   "assert jax.default_backend() == 'tpu' and jax.device_count() == 8 and jax.process_count() == 1; "
                   "from scripts.evaluate_encoder import evaluate; "
                   "r=evaluate(sys.argv[1], sys.argv[2], train_data=sys.argv[3], batch_size=16, "
                   "max_rows=512, seed=2026, checkpoint_step=int(sys.argv[4]), balanced_languages=True); "
                   "Path(sys.argv[5]).write_text(json.dumps(r, indent=2, allow_nan=False))")
        child([sys.executable, '-c', program, args.output, str(args.root / 'corpus/validation'),
               str(args.root / 'corpus/train'), str(step), str(report)], f'evaluate-{step}.log')
        return read(report)
    def verify_latest():
        report = logs / 'checkpoint-admission.json'
        program = ('import json,sys; from pathlib import Path; '
                   'from flaxchat.checkpoint import load_checkpoint_metadata; '
                   'm=load_checkpoint_metadata(sys.argv[1]); '
                   'Path(sys.argv[2]).write_text(json.dumps(m))')
        child([sys.executable, '-c', program, args.output, str(report)], 'checkpoint-admission.log')
        metadata = read(report)
        if (metadata.get('step') != 27750 or
                metadata.get('resolved_config', {}).get('encoder', {}).get('attention_score') != 'goat_input'):
            raise ValueError('Latest committed checkpoint is not corrected input-GOAT step27750')
    try:
        verify_latest()
        xml = logs / 'goat-physical-tests.xml'
        if args.qualify_only:
            child([sys.executable, '-m', 'pytest', '-q', 'tests/test_goat_physical_tpu.py',
                '--junitxml=' + str(xml)], 'goat-physical-tests.log',
                os.environ | {'FLAXCHAT_PHYSICAL_TPU': '1'})
            validate_goat_test_report(xml, input_scores=True)
            report = evaluate(27750)
            validate_batch16_report(report, reference, step=27750)
            if qualification_identity(stage) != identity:
                raise ValueError('Source/runtime changed during recovery qualification')
            qualified = bound | {'tests': [{'id': name, 'status': 'passed'} for name in
                required + ['recovery-checkpoint27750', 'recovery-heldout-batch16']], 'goat_physical_tests_sha256': sha(xml),
                'step27750_evaluation_sha256': digest(report)}
            write(args.root / 'mlm-qualification.json', qualified)
            write(logs / 'mlm-qualification.json', qualified)
            status.update(status='qualified', evaluations=[{'step': 27750, 'loss': report['masked_token_loss']}])
            return
        qualified = read(args.root / 'mlm-qualification.json')
        expected_tests = set(required + ['recovery-checkpoint27750', 'recovery-heldout-batch16'])
        if (any(qualified.get(k) != v for k, v in bound.items())
                or len(qualified.get('tests', [])) != len(expected_tests)
                or {t.get('id') for t in qualified.get('tests', []) if t.get('status') == 'passed'} != expected_tests
                or qualified.get('goat_physical_tests_sha256') != sha(xml)
                or qualified.get('step27750_evaluation_sha256') != digest(read(logs / 'evaluate-27750.json'))):
            raise ValueError('Fresh recovery qualification is missing or differs')
        validate_goat_test_report(xml, input_scores=True)
        validate_batch16_report(read(logs / 'evaluate-27750.json'), reference, step=27750)
        for boundary in range(28000, 100001, 2000):
            start = status['completed_step']
            log = child(base + ['--action', 'train', '--resume', '--boundary', str(boundary)],
                        f'train-{boundary}.log')
            events = []
            for line in log.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get('event') == 'train_step':
                    events.append(event)
            validate_updates(events, start, boundary)
            status['completed_step'] = boundary
            publish()
            report = evaluate(boundary)
            validate_batch16_report(report, read(logs / 'evaluate-27750.json'), step=boundary)
            status['evaluations'].append({'step': boundary, 'loss': report['masked_token_loss']})
            publish()
        status['status'] = 'completed'
    except DeadlineReached as error:
        status.update(status='lease_finished_checkpoint_recovery_available', reason=str(error))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        publish()


if __name__ == '__main__':
    main()
