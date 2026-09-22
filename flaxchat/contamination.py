"""Bounded exact token-span overlap auditing with collision verification.

This detects copied spans, not semantic or edited near-duplicates. Train shards
are scanned in chunks with overlap so matches crossing boundaries are retained.
"""
from __future__ import annotations

import numpy as np


def window_hashes(tokens, width):
    values = np.asarray(tokens, dtype=np.uint64)
    if width < 1:
        raise ValueError('Span length must be positive')
    size = len(values) - width + 1
    hashes = np.zeros(max(0, size), dtype=np.uint64)
    for offset in range(width if size > 0 else 0):
        hashes *= np.uint64(1000003)
        hashes += values[offset:offset + size] + np.uint64(1)
    return hashes


def overlapping_spans(train, candidate, *, width=50, chunk_tokens=262144):
    """Return every candidate start with an exact training match and a witness."""
    if width < 1 or chunk_tokens < width:
        raise ValueError('Positive span length and a chunk at least that long required')
    candidate = np.asarray(candidate)
    hashes = window_hashes(candidate, width)
    if not len(hashes):
        return []
    order = np.argsort(hashes)
    sorted_hashes = hashes[order]
    matches = {}
    for start in range(0, max(0, len(train) - width + 1), chunk_tokens):
        block = np.asarray(train[start:min(len(train), start + chunk_tokens + width - 1)])
        train_hashes = window_hashes(block, width)
        left = np.searchsorted(sorted_hashes, train_hashes, side='left')
        right = np.searchsorted(sorted_hashes, train_hashes, side='right')
        for index in np.flatnonzero(left != right):
            for position in order[left[index]:right[index]]:
                position = int(position)
                if position not in matches and np.array_equal(block[index:index + width], candidate[position:position + width]):
                    matches[position] = start + int(index)
        if len(matches) == len(hashes):
            break
    return [{'candidate_start': position, 'train_start': matches[position], 'length': width}
            for position in sorted(matches)]


def clean_block_starts(length, matches, *, block_tokens=1025):
    """Select original contiguous nonoverlapping blocks, never splice fragments."""
    if block_tokens < 2:
        raise ValueError('Evaluation blocks need input and target tokens')
    return [start for start in range(0, length - block_tokens + 1, block_tokens)
            if not any(m['candidate_start'] < start + block_tokens and
                       m['candidate_start'] + m['length'] > start for m in matches)]
