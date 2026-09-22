"""Packaged, fail-closed loss sanity gate for prepared-training summaries.

This is a prerequisite for a larger experiment, not proof of benchmark quality
or decontamination. Choose a stricter task-specific threshold before spending.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluate(summary, *, max_validation_loss=None):
    vocab = summary['model_config']['vocab_size']
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab < 2:
        raise ValueError('A concrete vocabulary size of at least two is required')
    uniform = math.log(vocab)
    maximum = uniform if max_validation_loss is None else max_validation_loss
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('Maximum validation loss must be finite and positive')
    initial, final = summary['initial_validation_loss'], summary['final_validation_loss']
    completed, checkpoint = summary['completed_steps'], summary['checkpoint_step']
    requested = summary['resolved_config']['steps']
    checks = {
        'finite_nonnegative_loss': all(isinstance(x, (float, int)) and not isinstance(x, bool)
            and math.isfinite(x) and x >= 0 for x in (initial, final)),
        'committed_requested_horizon': all(isinstance(x, int) and not isinstance(x, bool)
            and x > 0 for x in (completed, checkpoint, requested)) and completed == checkpoint == requested,
    }
    checks['heldout_loss_improved'] = checks['finite_nonnegative_loss'] and final < initial
    checks['below_uniform_baseline'] = checks['finite_nonnegative_loss'] and final < uniform
    checks['within_declared_threshold'] = checks['finite_nonnegative_loss'] and final <= maximum
    return {'passed': all(checks.values()), 'checks': checks,
            'initial_validation_loss': initial if checks['finite_nonnegative_loss'] else None,
            'final_validation_loss': final if checks['finite_nonnegative_loss'] else None,
            'uniform_loss': uniform, 'max_validation_loss': maximum,
            'source_python_sha256': summary.get('source_python_sha256'),
            'limitations': ['Loss sanity only; no contamination, statistical-significance or downstream quality claim.',
                            'The threshold and held-out corpus must be chosen before comparing experiments.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-validation-loss', type=float)
    args = parser.parse_args(argv)
    result = evaluate(json.loads(args.summary.read_text()), max_validation_loss=args.max_validation_loss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
