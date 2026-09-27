"""Compare matched development MLM reports without claiming downstream quality."""
import argparse
import json
import math
from pathlib import Path


MATCHED_FIELDS = (
    'checkpoint_context', 'checkpoint_step', 'initial_weights_sha256', 'evaluation_source_sha256',
    'runtime', 'model_source_sha256', 'evaluation_config_sha256', 'evaluation_batch_size', 'backend', 'devices',
    'selected_rows_sha256', 'masked_tokens', 'evaluated_rows',
    'excluded_exact_or_duplicate_rows', 'seed', 'mask_probability',
    'tokenizer_sha256', 'validation_manifest_sha256', 'overlap_check',
)


def compare(baseline, candidate):
    if baseline.get('parameter_source') != 'initial_pretrained' or candidate.get('parameter_source') != 'checkpoint':
        raise ValueError('Expected starting-weight baseline and trained-checkpoint candidate')
    for key in MATCHED_FIELDS:
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            raise ValueError(f'Unmatched evaluation protocol: {key}')
    if not baseline['initial_weights_sha256']:
        raise ValueError('Missing initial weight provenance')
    for report in (baseline, candidate):
        if not isinstance(report.get('masked_token_loss'), (int, float)) or not math.isfinite(report['masked_token_loss']) or report['masked_token_loss'] < 0:
            raise ValueError('Expected finite nonnegative MLM loss')
        if report['masked_tokens'] <= 0 or report['evaluated_rows'] <= 0:
            raise ValueError('Empty evaluation')
    before, after = baseline['masked_token_loss'], candidate['masked_token_loss']
    return dict(protocol_matched=True, baseline_loss=before, candidate_loss=after,
                candidate_minus_baseline=after-before,
                relative_loss_reduction=(before-after)/before if before else None,
                lower_development_mlm_loss=after < before,
                protocol={key:baseline[key] for key in MATCHED_FIELDS},
                production_quality_qualified=False,
                scope='Matched development MLM only; not a downstream, multilingual per-language, or leaderboard qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
