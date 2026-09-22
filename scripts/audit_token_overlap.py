"""Audit exact train/validation token overlap and identify untouched eval blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flaxchat.contamination import clean_block_starts, overlapping_spans
from flaxchat.token_pool import TokenPool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--span-tokens', type=int, default=50)
    parser.add_argument('--sequence-length', type=int, default=1024)
    args = parser.parse_args(argv)
    pool = TokenPool(args.token_manifest)
    validation = pool.arrays['validation'][:]
    matches = overlapping_spans(pool.arrays['train'], validation, width=args.span_tokens)
    starts = clean_block_starts(len(validation), matches, block_tokens=args.sequence_length + 1)
    report = {'data_identity': pool.identity, 'span_tokens': args.span_tokens,
              'train_tokens': len(pool.arrays['train']), 'validation_tokens': len(validation),
              'matching_spans': matches, 'clean_block_starts': starts,
              'block_tokens': args.sequence_length + 1,
              'status': 'exact_span_audit_complete',
              'limitations': ['Exact token spans only; edited, semantic and shorter overlap may remain.',
                              'Clean blocks are relative only to this exact training pool.',
                              'Existing benchmark data is unchanged for matched comparisons.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'matching_spans'}))
    print(f'Matching spans: {len(matches)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
