"""Compare only matched successful MLM training cases; retain missing/failure evidence."""
import argparse
import json
from pathlib import Path


def compare(report):
    results = report['results']
    training = [r for r in results if r['name'].startswith('train-') and r.get('passed')]
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
                complete=expected <= passed and fused_qualified)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    result=compare(json.loads(args.report.read_text()))
    text=json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text)
