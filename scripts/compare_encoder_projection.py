"""Compare only matched successful MLM training cases; retain missing/failure evidence."""
import argparse
import hashlib
import re
import json
from pathlib import Path


def verify_training_evidence(root, case):
    from scripts.benchmark_encoder_projection import summarize_steps
    name = case['name']
    if (not re.fullmatch(r'train-(dense|masked)-(xla|xla_full|xla_local|pallas)-b[0-9]+-s[0-9]+', name)
            or name != f"train-{case['projection']}-{case['backend']}-b{case['batch']}-s{case['length']}"):
        raise ValueError('Invalid training evidence identity')
    log, manifest_path = Path(root) / f'{name}.log', Path(root) / f'{name}-manifest.json'
    if (hashlib.sha256(log.read_bytes()).hexdigest() != case['training_log_sha256']
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != case['manifest_sha256']):
        raise ValueError('Raw benchmark evidence checksum mismatch')
    records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
    measured = summarize_steps(records, expected_tokens=case['batch'] * case['length'])
    for key in ('steady_tokens_per_second', 'measured_steps', 'warmup_steps', 'first_loss', 'final_loss'):
        if measured[key] != case[key]:
            raise ValueError('Benchmark summary disagrees with raw updates')
    configs = [r for r in records if r.get('event') == 'run_config']
    if len(configs) != 1:
        raise ValueError('Missing unique physical runtime record')
    config = configs[0]
    if (config['backend'] != 'tpu' or type(config['devices']) is not int or config['devices'] <= 0
            or type(config['processes']) is not int or config['processes'] <= 0
            or config['devices'] % config['processes']
            or config['mlm_loss_backend'] != case['backend'] or config['mlm_projection'] != case['projection']):
        raise ValueError('Runtime backend/topology disagrees with benchmark')
    manifest = json.loads(manifest_path.read_text())
    identity = manifest['identity']
    recipe = identity['resolved_config']
    if (type(manifest.get('step')) is not int or manifest['step'] != 55
            or any(not manifest.get(key) for key in ('model_state', 'optimizer_state', 'training_state'))
            or recipe['steps'] != 55 or recipe['batch_size'] != case['batch']
            or recipe['encoder']['mlm_loss_backend'] != case['backend']
            or recipe['encoder']['mlm_projection'] != case['projection']
            or not re.fullmatch('[0-9a-f]{64}', identity['source_python_sha256'])):
        raise ValueError('Checkpoint/source identity disagrees with benchmark')
    return True


def compare(report, evidence_root=None):
    results = report['results']
    training = [r for r in results if r['name'].startswith('train-') and r.get('passed')]
    evidence_failures = []
    if evidence_root is not None:
        verified = []
        for case in training:
            try:
                verify_training_evidence(evidence_root, case)
                verified.append(case)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                evidence_failures.append(f"{case['name']}: {exc}")
        training = verified
    comparisons = []
    for r in training:
        baseline = next((b for b in training if b['batch'] == r['batch'] and b['length'] == r['length']
                         and b['projection'] == 'dense' and b['backend'] == 'xla'), None)
        if baseline is None:
            continue
        memory = [d['statistics'].get('peak_bytes_in_use') for item in r.get('memory', [])
                  for d in item['devices'] if d.get('statistics')]
        comparisons.append(dict(name=r['name'], batch=r['batch'], length=r['length'],
            backend=r['backend'], projection=r['projection'],
            tokens_per_second=r['steady_tokens_per_second'],
            speedup=r['steady_tokens_per_second']/baseline['steady_tokens_per_second'],
            peak_bytes_per_device=max(memory) if memory else None,
            first_loss_absolute_delta=abs(r['first_loss']-baseline['first_loss']),
            final_loss_absolute_delta=abs(r['final_loss']-baseline['final_loss']),
            steady_usd_per_billion_input_tokens=r['steady_usd_per_billion_input_tokens'],
            invocation_usd_per_billion_input_tokens=r['invocation_usd_per_billion_input_tokens'],
            overflow_steps=r['overflow_steps']))
    fused_qualified = any(r.get('tile') and r.get('passed') and r.get('kind') == 'projection_forward_backward' for r in results)
    variants = [('dense', 'xla'), ('masked', 'xla')] + ([('masked', 'pallas')] if fused_qualified else [])
    cases = report.get('planned_cases', ((4,512),(16,512),(32,512),(4,2048)))
    if 'kernel_rows' in report:
        requested_rows = set(report['kernel_rows'])
        tiles = {r.get('tile') for r in results if r.get('tile') and r.get('kind') == 'projection_forward_backward'}
        fused_qualified = any({r.get('projected_rows') for r in results if r.get('tile') == tile and r.get('passed')
                               and r.get('kind') == 'projection_forward_backward'} == requested_rows for tile in tiles)
        fused_qualified &= any(r['name'] == 'model-check-pallas' and r.get('passed') for r in results)
        variants = [('dense', 'xla'), ('masked', 'xla'), ('masked', 'xla_full'), ('masked', 'pallas')]
    expected = {f'train-{p}-{b}-b{batch}-s{length}' for batch,length in cases for p,b in variants}
    passed = {r['name'] for r in training}
    return dict(comparisons=comparisons, missing_or_failed_training=sorted(expected-passed),
                failures=[dict(name=r['name'], returncode=r['returncode']) for r in results if not r.get('passed')],
                fused_kernel_qualified=fused_qualified, quality_qualified=False,
                raw_evidence_verified=evidence_root is not None and not evidence_failures,
                evidence_failures=evidence_failures,
                complete=evidence_root is not None and not evidence_failures and expected <= passed and fused_qualified)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    result=compare(json.loads(args.report.read_text()), evidence_root=args.report.parent)
    text=json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text)
