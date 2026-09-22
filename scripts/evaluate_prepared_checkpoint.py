"""Bounded zero-shot ARC likelihood evaluation of a prepared-training checkpoint.

Uses a pinned split, deterministic sample, fixed padding, per-example records,
exact training-span exclusion, and a Wilson interval. This is not full CORE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def prepare_choices(tokenizer, item, max_length):
    from flaxchat.eval import find_common_length
    tokens = tokenizer([item['query'] + '\n' + c for c in item['choices']],
                       prepend=tokenizer.get_bos_token_id())
    start = find_common_length(tokens)
    if not 1 <= start < min(map(len, tokens)):
        raise ValueError('Empty answer continuation')
    if max(map(len, tokens)) > max_length:
        raise ValueError('Example exceeds declared context limit; truncation is forbidden')
    padded = np.full((len(tokens), max_length), tokenizer.get_bos_token_id(), dtype=np.int32)
    for i, row in enumerate(tokens):
        padded[i, :len(row)] = row
    return padded, start, np.asarray([len(t) for t in tokens]), tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--token-manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-examples', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--overlap-tokens', type=int, default=13)
    parser.add_argument('--holdout-blocks', type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.max_examples, args.max_length, args.overlap_tokens, args.holdout_blocks) < 1:
        parser.error('Bounds must be positive')
    from scripts.validate_tpu import source_digest
    from flaxchat.token_pool import TokenPool
    from flaxchat.tokenizer import HuggingFaceTokenizer
    from flaxchat.contamination import overlapping_spans, clean_block_starts
    from flaxchat.eval import CORE_TASKS, normalize_core_item, _wilson_interval
    from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint, validate_checkpoint_tokenizer
    from flaxchat.common import compute_init
    from flaxchat.gpt import GPT, GPTConfig
    from datasets import load_dataset
    from flax import nnx
    import jax
    import jax.numpy as jnp
    pool = TokenPool(args.token_manifest)
    tokenizer = HuggingFaceTokenizer.from_directory(str(pool.path.parent / 'tokenizer'))
    spec = CORE_TASKS['arc_easy']
    data = load_dataset('allenai/ai2_arc', 'ARC-Easy', revision=spec['revision'], split=spec['split'])
    indices = sorted(random.Random(1234).sample(range(len(data)), min(args.max_examples, len(data))))
    items = [normalize_core_item(data[i]) for i in indices]
    # One bounded scan of training tokens for all candidate strings. Matches
    # crossing candidate boundaries are discarded, never counted as overlap.
    encoded, boundaries, combined = [], [], []
    for item in items:
        row = []
        for text in [item['query'], *item['choices']]:
            tokens = tokenizer.encode(text)
            row.append((len(combined), len(combined) + len(tokens)))
            combined.extend(tokens)
        boundaries.append(row)
        encoded.append(item)
    matches = overlapping_spans(pool.arrays['train'], np.asarray(combined), width=args.overlap_tokens)
    contaminated = {i for i, spans in enumerate(boundaries) if any(
        left <= m['candidate_start'] and m['candidate_start'] + m['length'] <= right
        for left, right in spans for m in matches)}
    compute_init()
    metadata = load_checkpoint_metadata(args.checkpoint)
    if metadata.get('data_manifest_identity') != pool.identity:
        raise ValueError('Overlap audit pool must match the checkpoint training data')
    validate_checkpoint_tokenizer(metadata, tokenizer, tokenizer_path=str(pool.path.parent / 'tokenizer'))
    config = GPTConfig(**metadata['model_config'])
    if args.max_length > config.sequence_len:
        raise ValueError('Requested evaluation length exceeds trained context')
    model = GPT(config, rngs=nnx.Rngs(0))
    restore_model_from_checkpoint(model, args.checkpoint, step=metadata['step'])
    @nnx.jit
    def score(model, tokens, start, lengths):
        logits = model(tokens)
        logp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
        nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
        pos = jnp.arange(tokens.shape[1] - 1)[None, :]
        mask = (pos >= start - 1) & (pos < lengths[:, None] - 1)
        return jnp.sum(jnp.where(mask, nll, 0), axis=1) / jnp.sum(mask, axis=1)
    validation = pool.arrays['validation'][:]
    heldout_matches = overlapping_spans(pool.arrays['train'], validation, width=50)
    block_size = min(config.sequence_len, 1024)
    starts = clean_block_starts(len(validation), heldout_matches, block_tokens=block_size + 1)[:args.holdout_blocks]
    if not starts:
        raise ValueError('No exact-span-clean heldout blocks')
    @nnx.jit
    def heldout_loss(model, x, y):
        return model(x, y)
    losses = []
    for start in starts:
        block = jnp.asarray(validation[start:start + block_size + 1], dtype=jnp.int32)
        losses.append(float(heldout_loss(model, block[:-1][None, :], block[1:][None, :])))
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite heldout loss')
    records = []
    for i, (index, item) in enumerate(zip(indices, encoded, strict=True)):
        row = {'index': index, 'gold': item['gold']}
        if i in contaminated:
            row.update(status='excluded_training_overlap')
        else:
            try:
                padded, start, lengths, _ = prepare_choices(tokenizer, item, args.max_length)
            except ValueError as exc:
                row.update(status='excluded_context', reason=str(exc))
            else:
                scores = np.asarray(score(model, jnp.asarray(padded), start, jnp.asarray(lengths)))
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite candidate scores')
                prediction = int(np.argmin(scores))
                row.update(status='scored', scores=scores.tolist(), prediction=prediction,
                           correct=prediction == item['gold'], choice_count=len(scores))
        records.append(row)
        print(json.dumps(row), flush=True)
    scored = [r for r in records if r['status'] == 'scored']
    correct = sum(r['correct'] for r in scored)
    report = {'status': 'complete' if scored else 'no_eligible_examples', 'task': 'arc_easy',
              'dataset': spec['dataset'], 'revision': spec['revision'], 'split': spec['split'],
              'seed': 1234, 'fewshot': 0, 'scoring': 'mean_continuation_nll_common_token_prefix',
              'max_length': args.max_length, 'overlap_tokens': args.overlap_tokens,
              'candidate_content_sha256': hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
              'data_identity': pool.identity, 'checkpoint': args.checkpoint, 'checkpoint_step': metadata['step'],
              'training_source_python_sha256': metadata.get('source_python_sha256'),
              'evaluation_source_python_sha256': source_digest(Path(__file__).resolve().parents[1]),
              'clean_holdout': {'loss': float(np.mean(losses)), 'block_losses': losses, 'block_starts': starts, 'tokens': len(starts) * block_size, 'exclusion_span_tokens': 50},
              'sample_count': len(scored), 'sampled_count': len(indices), 'correct': correct,
              'accuracy': correct / len(scored) if scored else None,
              'random_choice_baseline': float(np.mean([1 / r['choice_count'] for r in scored])) if scored else None,
              'confidence_interval_95': _wilson_interval(correct, len(scored)), 'records': records,
              'limitations': ['Small zero-shot diagnostic; not full ARC or CORE replication.',
                              'Exact-span exclusion only; near-duplicates may remain.',
                              'Context and overlap exclusions change the evaluated sample.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if scored else 1


if __name__ == '__main__':
    raise SystemExit(main())
