"""Compare matched warm training logs without confusing kernel and model speed."""
import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.validate_encoder_scale import validate_updates


def compare(baseline, candidate, *, steps, batch, length, hourly_usd, local_accumulation=False) -> dict[str, Any]:
    if not math.isfinite(hourly_usd) or hourly_usd <= 0:
        raise ValueError('Hourly rate must be finite and positive')
    logs = []
    summaries = []
    for index, path in enumerate((baseline, candidate)):
        records = [json.loads(line) for line in Path(path).read_text().splitlines()
                   if line.startswith('{')]
        configs = [r for r in records if r.get('event') == 'run_config']
        if len(configs) != 1:
            raise ValueError('Require exactly one run configuration per log')
        config = configs[0].copy()
        identity = config['input_identity'].copy()
        if (not isinstance(identity.get('resolved_config'), dict)
                or not {'seed', 'batch_size'} <= identity['resolved_config'].keys()
                or not identity.get('tokenizer') or not identity.get('data_manifest')
                or not identity.get('initial_weights_sha256')):
            raise ValueError('Incomplete input, tokenizer, configuration, or weight identity')
        if local_accumulation:
            resolved = identity['resolved_config'].copy()
            mode = resolved.pop('local_gradient_accumulation', False)
            if mode is not (index == 1):
                raise ValueError('Require baseline local accumulation disabled and candidate enabled')
            identity['resolved_config'] = resolved
        source_hash = identity.pop('source_python_sha256')
        if not isinstance(source_hash, str) or not source_hash:
            raise ValueError('Missing source identity')
        config['input_identity'] = identity
        rows = [r for r in records if r.get('event') == 'train_step']
        logs.append((config, rows, source_hash))
        summary = validate_updates(path, first=1, last=steps, batch=batch, length=length)
        rate = summary.get('nonpadding_tokens_per_second')
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Useful-token throughput is required')
        summary['estimated_warm_compute_usd_per_billion_useful_tokens'] = hourly_usd*1e9/rate/3600
        summaries.append(summary)
    if logs[0][0] != logs[1][0]:
        raise ValueError('Configuration, data, initialization, or topology mismatch')
    fields = ('tokens', 'nonpadding_tokens', 'masked_tokens', 'learning_rate',
              'language_nonpadding_tokens', 'source_nonpadding_tokens',
              'includes_compilation', 'includes_profiling')
    differences = []
    for a, b in zip(logs[0][1], logs[1][1], strict=True):
        if any(key not in row for row in (a, b)
               for key in ('learning_rate', 'includes_profiling')):
            raise ValueError('Incomplete training or profiling evidence')
        if any(a.get(key) != b.get(key) for key in fields):
            raise ValueError(f'Unmatched token exposure or timing window at step {a["step"]}')
        differences.append(dict(step=a['step'], baseline_loss=a['loss'],
                                candidate_loss=b['loss'], absolute_loss_difference=abs(a['loss']-b['loss'])))
    return dict(
        matched=True, permitted_config_difference=(
            {'local_gradient_accumulation': {'baseline': False, 'candidate': True}}
            if local_accumulation else {}),
        baseline=summaries[0], candidate=summaries[1],
        useful_token_speedup=summaries[1]['nonpadding_tokens_per_second']/summaries[0]['nonpadding_tokens_per_second'],
        max_absolute_loss_difference=max(row['absolute_loss_difference'] for row in differences),
        loss_trajectory=differences, source_hashes=[row[2] for row in logs],
        hourly_usd=hourly_usd,
        scope='Warm training only; compilation, profiling and checkpoint time excluded. Cost is an estimate at the supplied whole-slice rate, not billed spend. Matching logs cannot prove same physical allocation or that only the intended source changed; verify launcher receipts and frozen source manifests separately. This does not qualify production model quality.',
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-local-accumulation', action='store_true',
                        help='Permit only the explicit disabled-to-enabled accumulation mode change')
    args = parser.parse_args(argv)
    result = compare(args.baseline, args.candidate, steps=args.steps, batch=args.batch,
                     length=args.length, hourly_usd=args.hourly_usd,
                     local_accumulation=args.compare_local_accumulation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
