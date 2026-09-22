"""Prepare bounded, pinned FineWeb tokens on CPU before allocating a TPU."""
from __future__ import annotations

import argparse
from pathlib import Path

from flaxchat.token_pool import write_streaming_pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    parser.add_argument('--subset', default='sample-10BT')
    parser.add_argument('--revision', required=True, help='Immutable dataset commit SHA')
    parser.add_argument('--tokenizer', default='gpt2')
    parser.add_argument('--tokenizer-revision', required=True, help='Immutable tokenizer commit SHA')
    parser.add_argument('--shard-tokens', type=int, default=16_000_000)
    parser.add_argument('--train-tokens', type=int, default=2_000_000)
    parser.add_argument('--validation-tokens', type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.shuffle_buffer < 1:
        parser.error("--shuffle-buffer must be positive")
    import re
    if not all(re.fullmatch('[0-9a-f]{40}', value) for value in (args.revision, args.tokenizer_revision)):
        parser.error('Both revisions must be full commit SHAs')
    if min(args.train_tokens, args.validation_tokens) < 2:
        parser.error('Token budgets must be at least two')
    import logging
    logging.getLogger('httpx').setLevel(logging.WARNING)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define an EOS token')
    dataset = iter(load_dataset(args.dataset, args.subset, revision=args.revision, split='train', streaming=True).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer))
    def token_chunks(budget):
        remaining = budget
        while remaining:
            try:
                document = next(dataset)
            except StopIteration as exc:
                raise ValueError('Dataset exhausted before requested token budget') from exc
            text = document.get('text', '')
            if text:
                chunk = (tokenizer.encode(text[:100000], add_special_tokens=False) + [tokenizer.eos_token_id])[:remaining]
                remaining -= len(chunk)
                yield chunk
    splits = {'train': token_chunks(args.train_tokens), 'validation': token_chunks(args.validation_tokens)}
    identity = {'dataset': args.dataset, 'subset': args.subset, 'revision': args.revision,
                'tokenizer': args.tokenizer, 'tokenizer_revision': args.tokenizer_revision,
                'seed': args.seed, 'shuffle_buffer': args.shuffle_buffer,
                'split_policy': 'shuffled-train-first-disjoint-documents-v2', 'document_char_limit': 100000}
    from flaxchat.tokenizer import HuggingFaceTokenizer
    from flaxchat.dataloader import _tokenizer_identity
    identity['tokenizer_identity'] = _tokenizer_identity(HuggingFaceTokenizer(tokenizer.backend_tokenizer))
    path = write_streaming_pool(args.output, splits, identity, len(tokenizer), shard_tokens=args.shard_tokens)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
