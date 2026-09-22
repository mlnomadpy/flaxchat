"""Immutable local token pools; no Arrow, network, padding, or hidden epochs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_pool(output: Path, splits: dict, identity: dict, vocab_size: int) -> Path:
    """Write a bounded preparation result into a new directory."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'version': 1, 'dtype': '<u4', 'vocab_size': vocab_size,
                'identity': identity, 'splits': {}}
    for split, tokens in splits.items():
        values = np.asarray(tokens)
        if not np.issubdtype(values.dtype, np.integer) or values.ndim != 1 or values.size < 2 or np.any(values < 0) or np.any(values >= vocab_size):
            raise ValueError('Each split needs at least two valid token IDs')
        path = output / f'{split}.bin'
        values.astype('<u4').tofile(path)
        manifest['splits'][split] = {'file': path.name, 'tokens': int(values.size), 'sha256': sha256(path)}
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return path


class TokenPool:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest['version'] not in (1, 2) or self.manifest['dtype'] != '<u4':
            raise ValueError('Unsupported token pool format')
        self.identity = {'manifest_sha256': sha256(self.path)}
        self.vocab_size = int(self.manifest['vocab_size'])
        self.arrays = {}
        for split in ('train', 'validation'):
            entry = self.manifest['splits'][split]
            if self.manifest['version'] == 2:
                self.arrays[split] = _ShardedTokens(self.path.parent, entry, self.vocab_size)
                continue
            shard = (self.path.parent / entry['file']).resolve()
            if shard.parent != self.path.parent:
                raise ValueError('Token shard must be inside the pool directory')
            if shard.stat().st_size != entry['tokens'] * 4 or sha256(shard) != entry['sha256']:
                raise ValueError(f'Corrupt {split} token shard')
            values = np.memmap(shard, dtype='<u4', mode='r')
            if values.size < 2 or np.max(values) >= self.vocab_size:
                raise ValueError('Invalid token IDs')
            self.arrays[split] = values

    def batch(self, split: str, step: int, batch_size: int, sequence_length: int):
        return self.batch_rows(split, step, batch_size, sequence_length, 0, batch_size)

    def batch_rows(self, split: str, step: int, batch_size: int, sequence_length: int,
                   row_start: int, row_stop: int):
        """Read a global row interval supplied by the actual device sharding.

        Mesh topology can interleave host device IDs. Rank order is not a
        reliable substitute for the global row indices assigned by JAX.
        """
        if step < 0 or batch_size <= 0 or sequence_length <= 0:
            raise ValueError('Invalid batch coordinates')
        if not 0 <= row_start < row_stop <= batch_size:
            raise ValueError('Invalid global row interval')
        count = batch_size * sequence_length
        start = step * count
        values = self.arrays[split]
        if start + count + 1 > len(values):
            raise ValueError(f'{split} token pool exhausted at batch {step}; prepare more data')
        start += row_start * sequence_length
        count = (row_stop - row_start) * sequence_length
        shape = (row_stop - row_start, sequence_length)
        x = np.asarray(values[start:start + count], dtype=np.int32).reshape(shape)
        y = np.asarray(values[start + 1:start + count + 1], dtype=np.int32).reshape(shape)
        return x, y

    def local_batch(self, split: str, step: int, global_batch: int,
                    sequence_length: int, process_index: int, process_count: int):
        """Read only this host's contiguous part of the canonical global batch."""
        if process_count < 1 or not 0 <= process_index < process_count:
            raise ValueError('Invalid process coordinates')
        if global_batch <= 0 or global_batch % process_count:
            raise ValueError('Global batch must be divisible by process count')
        local_batch = global_batch // process_count
        return self.batch_rows(split, step, global_batch, sequence_length,
                               process_index * local_batch, (process_index + 1) * local_batch)


def write_streaming_pool(output, splits, identity, vocab_size, *, shard_tokens=16_000_000):
    """Atomically publish a v2 pool from bounded iterables of token chunks."""
    import os
    import shutil
    import tempfile
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if shard_tokens < 2 or vocab_size < 1 or vocab_size > 2**31:
        raise ValueError('Invalid shard size or vocabulary')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    manifest = {'version': 2, 'dtype': '<u4', 'vocab_size': vocab_size,
                'identity': identity, 'splits': {}}
    try:
        for split in ('train', 'validation'):
            entries, total, current, handle = [], 0, 0, None
            digest = hashlib.sha256()
            try:
                for chunk in splits[split]:
                    values = np.asarray(chunk)
                    if values.size == 0:
                        continue
                    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= vocab_size):
                        raise ValueError('Invalid token chunk')
                    offset = 0
                    while offset < len(values):
                        if handle is None:
                            name = f'{split}-{len(entries):06d}.bin'
                            handle = (staging / name).open('wb')
                            digest = hashlib.sha256()
                            current = 0
                        count = min(shard_tokens - current, len(values) - offset)
                        payload = values[offset:offset + count].astype('<u4').tobytes()
                        handle.write(payload)
                        digest.update(payload)
                        current += count
                        total += count
                        offset += count
                        if current == shard_tokens:
                            handle.flush()
                            os.fsync(handle.fileno())
                            handle.close()
                            handle = None
                            entries.append({'file': name, 'tokens': current, 'sha256': digest.hexdigest()})
                if handle is not None:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    handle = None
                    entries.append({'file': name, 'tokens': current, 'sha256': digest.hexdigest()})
            finally:
                if handle is not None:
                    handle.close()
            if total < 2:
                raise ValueError('Each split needs at least two valid token IDs')
            manifest['splits'][split] = {'tokens': total, 'shards': entries}
        (staging / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        # Never merge into or overwrite an existing published pool.
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
        return output / 'manifest.json'
    except BaseException:
        shutil.rmtree(staging)
        raise


class _ShardedTokens:
    """Verify each immutable shard on first access; retain at most one mmap."""
    def __init__(self, root, entry, vocab_size):
        self.root, self.entries, self.vocab_size = root, entry['shards'], vocab_size
        self.ends = np.cumsum([item['tokens'] for item in self.entries], dtype=np.int64)
        if not len(self.ends) or int(self.ends[-1]) != entry['tokens']:
            raise ValueError('Invalid shard token counts')
        self.verified = set()
        for item in self.entries:
            path = (root / item['file']).resolve()
            if path.parent != root or item['tokens'] < 1 or path.stat().st_size != item['tokens'] * 4:
                raise ValueError('Invalid token shard')

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise ValueError('Only contiguous token ranges are supported')
        start, stop, _ = index.indices(len(self))
        result = np.empty(max(0, stop - start), dtype=np.int32)
        cursor = start
        while cursor < stop:
            shard_index = int(np.searchsorted(self.ends, cursor, side='right'))
            entry = self.entries[shard_index]
            path = self.root / entry['file']
            values = np.memmap(path, mode='r', dtype='<u4')
            try:
                if shard_index not in self.verified:
                    if sha256(path) != entry['sha256']:
                        raise ValueError('Corrupt token shard')
                    for offset in range(0, len(values), 262144):
                        if np.max(values[offset:offset + 262144]) >= self.vocab_size:
                            raise ValueError('Invalid token IDs')
                    self.verified.add(shard_index)
                base = 0 if shard_index == 0 else int(self.ends[shard_index - 1])
                end = min(stop, int(self.ends[shard_index]))
                result[cursor - start:end - start] = values[cursor - base:end - base]
                cursor = end
            finally:
                cast(Any, values)._mmap.close()
        return result
