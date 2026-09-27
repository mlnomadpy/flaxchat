"""Aggregate every physical TPU worker's encoder recovery evidence, failing closed."""
import argparse
import hashlib
import json
import math
from pathlib import Path


CONFIG_KEYS = ('compute_dtype', 'residual_dtype', 'mlm_projection', 'mlm_loss_backend',
               'mlm_vocab_tile', 'batch_size', 'lengths')
REQUIRED_STAGES = ('hardware', 'train-512', 'interrupt', 'resume')


def summarize(root, *, expected_processes, expected_devices, controller=None):
    if expected_processes < 2 or expected_devices < expected_processes or expected_devices % expected_processes:
        raise ValueError('Require a valid multi-process device topology')
    failures, workers, ranks = [], [], []
    configs = []
    manifests = []
    try:
        control = json.loads(Path(controller or Path(root) / 'controller.json').read_text())
        rows = control['workers']
        if (control.get('passed') is not True or len(rows) != expected_processes
                or sorted(r['rank'] for r in rows) != list(range(expected_processes))
                or any(type(r.get('returncode')) is not int or r['returncode'] != 0 for r in rows)):
            failures.append('Controller did not complete on every physical worker')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(f'Missing or invalid controller evidence: {exc}')
    updates = {'train-512': [], 'resume': []}
    for path in sorted(Path(root).rglob('summary.json')):
        try:
            report = json.loads(path.read_text())
            hardware = json.loads(path.with_name('hardware.json').read_text())
            if hardware != report['hardware']:
                raise ValueError('Hardware probe and summary disagree')
            rank = hardware['process_index']
            if type(rank) is not int:
                raise ValueError('Process index must be an integer')
            ranks.append(rank)
            if (hardware['backend'], hardware['process_count'], hardware['device_count']) != ('tpu', expected_processes, expected_devices):
                failures.append(f'{path}: topology mismatch')
            config = {key: report[key] for key in CONFIG_KEYS}
            configs.append(config)
            if (config['lengths'] != [512] or type(config['batch_size']) is not int
                    or config['batch_size'] <= 0 or config['batch_size'] % expected_devices):
                failures.append(f'{path}: invalid recovery configuration')
            stages = report['stages']
            for name in REQUIRED_STAGES:
                matching = [stage for stage in stages if stage['name'] == name]
                code = -9 if name == 'interrupt' else 0
                if len(matching) != 1 or matching[0].get('passed') is not True or matching[0].get('code') != code:
                    failures.append(f'{path}: missing or failed {name}')
            if report['required_passed'] is not True or report['failures']:
                failures.append(f'{path}: required validation failed')
            if 'FAULT_INJECTION: encoder SIGKILL after committed step 3' not in path.with_name('interrupt.log').read_text():
                raise ValueError('Missing committed fault evidence')
            baseline = json.loads(path.with_name('baseline-manifest.json').read_text())
            resumed = json.loads(path.with_name('recovery-manifest.json').read_text())
            if any(type(m.get('step')) is not int or m['step'] != 6 for m in (baseline, resumed)):
                raise ValueError('Wrong checkpoint manifest step')
            for key in ('model_state', 'optimizer_state', 'training_state'):
                if not baseline[key] or baseline[key] != resumed[key]:
                    raise ValueError('Raw recovery state mismatch')
            manifests.append(baseline)
            recovery_path = path.with_name('recovery.json')
            recovery = json.loads(recovery_path.read_text())
            if any(recovery.get(key) is not True for key in ('model_state', 'optimizer_state', 'training_state')):
                failures.append(f'{path}: recovery mismatch')
            for stage_name in updates:
                for line in path.with_name(f'{stage_name}.log').read_text().splitlines():
                    if not line.startswith('{'):
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError('Log record must be an object')
                    if event.get('event') == 'train_step':
                        if rank != 0:
                            failures.append(f'{path}: {stage_name} update logged by nonzero process')
                        else:
                            updates[stage_name].append(event)
            workers.append(dict(process_index=rank, summary=str(path),
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                recovery_sha256=hashlib.sha256(recovery_path.read_bytes()).hexdigest(),
                update_log_sha256={name: hashlib.sha256(path.with_name(f'{name}.log').read_bytes()).hexdigest()
                                   for name in updates}))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            failures.append(f'{path}: incomplete evidence ({exc})')
    if sorted(ranks) != list(range(expected_processes)) or len(workers) != expected_processes:
        failures.append('Require exactly one complete report per physical process')
    if configs and any(config != configs[0] for config in configs):
        failures.append('Workers disagree on validation configuration')
    if any(m != manifests[0] for m in manifests[1:]):
        failures.append('Workers disagree on raw checkpoint manifests')
    for stage_name, expected_steps in (('train-512', list(range(1, 7))), ('resume', [4, 5, 6])):
        events = updates[stage_name]
        try:
            if ([event['step'] for event in events] != expected_steps
                    or any(type(event['step']) is not int for event in events)):
                failures.append(f'{stage_name}: missing, duplicate, or out-of-order update records')
            if any(event['updated'] is not True or not math.isfinite(event['masked_tokens']) or event['masked_tokens'] <= 0 or not math.isfinite(event['loss']) for event in events):
                failures.append(f'{stage_name}: empty or rejected updates')
            if configs and any(
                type(event['tokens']) is not int
                or event['tokens'] != configs[0]['batch_size'] * 512
                or event['masked_tokens'] > event['tokens']
                or not math.isfinite(event['seconds']) or event['seconds'] <= 0
                for event in events
            ):
                failures.append(f'{stage_name}: invalid token counts or timing')
            if configs and configs[0]['mlm_projection'] == 'masked' and any(event['projection_dense_fallback'] is not False for event in events):
                failures.append(f'{stage_name}: masked projection fell back to dense')
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f'{stage_name}: invalid update records ({exc})')
    return dict(scope='physical_multi_host_encoder_recovery', passed=not failures,
                quality_qualified=False, expected_processes=expected_processes,
                expected_devices=expected_devices, configuration=configs[0] if configs else None,
                failures=failures, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, help='Successful gcp_tpu_run controller summary; defaults to input-dir/controller.json')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--expected-processes', type=int, required=True)
    parser.add_argument('--expected-devices', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input_dir, expected_processes=args.expected_processes, expected_devices=args.expected_devices, controller=args.controller)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
