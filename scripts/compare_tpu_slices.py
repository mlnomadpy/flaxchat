"""Compare matched training summaries using explicit whole-slice hourly rates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from flaxchat.training_quality import evaluate


def compare(reference, candidate, *, reference_hourly, candidate_hourly):
    if not all(math.isfinite(x) and x > 0 for x in (reference_hourly, candidate_hourly)):
        raise ValueError('Positive finite whole-slice hourly rates required')
    for summary in (reference, candidate):
        if not evaluate(summary)['passed']:
            raise ValueError('Both runs must pass the training quality gate')
        if summary.get('start_step') != 0:
            raise ValueError('Matched benchmarks must start from initialization')
    for key in ('model_config', 'data_manifest_identity', 'tokenizer_identity', 'tokens', 'checkpoint_policy'):
        if key not in reference or key not in candidate or reference[key] != candidate[key]:
            raise ValueError(f'Unmatched {key}')
    # Runtime is separate evidence: different backends/versions are explicit
    # comparison dimensions, not silently part of the training recipe.
    left, right = [dict(s['resolved_config']) for s in (reference, candidate)]
    left.pop('runtime', None)
    right.pop('runtime', None)
    if left != right:
        raise ValueError('Unmatched resolved training recipe')
    def metrics(summary, hourly):
        seconds, tokens = summary['invocation_seconds'], summary['new_committed_tokens']
        if not math.isfinite(seconds) or seconds <= 0 or tokens <= 0 or tokens != summary['tokens']:
            raise ValueError('Invalid invocation accounting')
        return {'devices': summary['devices'], 'hosts': summary['process_count'],
                'invocation_seconds': seconds, 'tokens': tokens,
                'tokens_per_second': tokens / seconds,
                'compute_usd': hourly * seconds / 3600,
                'usd_per_million_tokens': hourly * seconds / 3600 / tokens * 1e6,
                'validation_loss': summary['final_validation_loss'],
                'hourly_rate_input': hourly, 'source_python_sha256': summary.get('source_python_sha256')}
    a, b = metrics(reference, reference_hourly), metrics(candidate, candidate_hourly)
    return {'status': 'matched_recipe', 'reference': a, 'candidate': b,
            'throughput_ratio': b['tokens_per_second'] / a['tokens_per_second'],
            'compute_cost_ratio': b['compute_usd'] / a['compute_usd'],
            'validation_loss_delta': b['validation_loss'] - a['validation_loss'],
            'runtime_equal': reference['resolved_config'].get('runtime') == candidate['resolved_config'].get('runtime'),
            'limitations': ['Invocation cost excludes provisioning, setup, recovery experiments and cleanup.',
                           'Hourly prices are explicit estimates, not billed Spot charges.',
                           'Topology and reduction ordering differ; loss is reported, not assumed identical.',
                           'Source hashes are reported; review source differences before interpreting performance.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-hourly', type=float, required=True)
    parser.add_argument('--candidate-hourly', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()),
                     reference_hourly=args.reference_hourly, candidate_hourly=args.candidate_hourly)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
