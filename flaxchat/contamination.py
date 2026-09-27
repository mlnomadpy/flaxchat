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


def overlapping_rows(train, candidate, *, special_token_ids, width=50, chunk_rows=256):
    """Return one exact witness for EVERY matching training row.

    Spans never cross row boundaries or contain special tokens. Candidate hashes
    are indexed in memory; training rows are scanned in bounded chunks. This is
    an exact-span audit, not a semantic/near-duplicate detector.
    """
    if width < 1 or chunk_rows < 1:
        raise ValueError('Positive span width and chunk rows required')
    for rows in (train, candidate):
        if rows.ndim != 2 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError('Expected two-dimensional integer token rows')

    def valid_hashes(rows):
        flat = np.asarray(rows).reshape(-1)
        if rows.shape[1] < width:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
        hashes = window_hashes(flat, width)
        starts = np.arange(len(hashes), dtype=np.int64)
        bad = np.concatenate(([0], np.cumsum(np.isin(flat, special_token_ids))))
        valid = ((starts % rows.shape[1] <= rows.shape[1] - width)
                 & (bad[starts + width] == bad[starts]))
        return hashes[valid], starts[valid]

    candidate_hashes, candidate_starts = valid_hashes(candidate)
    order = np.argsort(candidate_hashes)
    sorted_hashes = candidate_hashes[order]
    candidate_flat = np.asarray(candidate).reshape(-1)
    matches = []
    for start in range(0, len(train), chunk_rows):
        block = train[start:start + chunk_rows]
        hashes, starts = valid_hashes(block)
        left = np.searchsorted(sorted_hashes, hashes, side='left')
        right = np.searchsorted(sorted_hashes, hashes, side='right')
        found = set()
        for index in np.flatnonzero(left != right):
            row, column = divmod(int(starts[index]), train.shape[1])
            if row in found:
                continue
            for candidate_index in order[left[index]:right[index]]:
                position = int(candidate_starts[candidate_index])
                if np.array_equal(block[row, column:column + width],
                                  candidate_flat[position:position + width]):
                    candidate_row, candidate_column = divmod(position, candidate.shape[1])
                    matches.append(dict(train_row=start + row, train_column=column,
                                        candidate_row=candidate_row, candidate_column=candidate_column,
                                        length=width))
                    found.add(row)
                    break
    return matches


def overlapping_documents(train, candidate, *, special_token_ids, width=50):
    """One verified witness per matching training document, across row windows.

    Inputs yield (unique document ID, complete 1D integer token array). Candidate
    content and short-prefix hashes are indexed in memory; training documents are
    streamed. A prefix match is only a filter: the full span is always compared.
    Never concatenate documents or remove interior special tokens.
    """
    if type(width) is not int or width < 1:
        raise ValueError('Positive integer span width required')
    prefix_width = min(width, 8)

    def checked(documents):
        seen = set()
        for identity, tokens in documents:
            tokens = np.asarray(tokens)
            if (not isinstance(identity, str) or not identity or identity in seen
                    or tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer)):
                raise ValueError('Unique document IDs and one-dimensional integer tokens required')
            seen.add(identity)
            yield identity, tokens

    def eligible(tokens):
        count = max(0, len(tokens) - width + 1)
        if not count:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
        hashes = window_hashes(tokens, prefix_width)[:count]
        starts = np.arange(count, dtype=np.int64)
        bad = np.concatenate(([0], np.cumsum(np.isin(tokens, special_token_ids))))
        valid = bad[starts + width] == bad[starts]
        return hashes[valid], starts[valid]

    candidates = list(checked(candidate))
    hash_parts, start_parts, doc_parts = [], [], []
    for index, (_, tokens) in enumerate(candidates):
        hashes, starts = eligible(tokens)
        hash_parts.append(hashes)
        start_parts.append(starts)
        doc_parts.append(np.full(len(starts), index, dtype=np.int64))
    hashes = np.concatenate(hash_parts) if hash_parts else np.array([], dtype=np.uint64)
    starts = np.concatenate(start_parts) if start_parts else np.array([], dtype=np.int64)
    docs = np.concatenate(doc_parts) if doc_parts else np.array([], dtype=np.int64)
    order = np.argsort(hashes)
    sorted_hashes = hashes[order]
    matches = []
    for identity, tokens in checked(train):
        hashes, positions = eligible(tokens)
        left = np.searchsorted(sorted_hashes, hashes, side='left')
        right = np.searchsorted(sorted_hashes, hashes, side='right')
        found = False
        for index in np.flatnonzero(left != right):
            position = int(positions[index])
            for other in order[left[index]:right[index]]:
                document, candidate_tokens = candidates[int(docs[other])]
                start = int(starts[other])
                if np.array_equal(tokens[position:position + width], candidate_tokens[start:start + width]):
                    matches.append(dict(train_document_id=identity, train_start=position,
                                        candidate_document_id=document, candidate_start=start, length=width))
                    found = True
                    break
            if found:
                break
    return matches
