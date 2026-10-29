"""Shared CPU validation of prepared encoder rows before model allocation."""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_prepared_rows(directory, config, *, minimum_rows=1):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'flaxchat-encoder-rows-v1' or not manifest.get('tokenizer_sha256'):
        raise ValueError('A prepared encoder manifest with tokenizer identity is required')
    if manifest.get('vocab_size') != config.vocab_size or manifest.get('pad_token_id') != config.pad_token_id:
        raise ValueError('Prepared tokenizer vocabulary/padding does not match model')
    if manifest.get('mask_token_id', config.mask_token_id) != config.mask_token_id:
        raise ValueError('Prepared mask token does not match model')
    special = manifest.get('special_token_ids')
    if (not isinstance(special, list) or any(type(t) is not int or not 0 <= t < config.vocab_size for t in special)
            or len(set(special)) != len(special)
            or not {config.pad_token_id, config.mask_token_id} <= set(special)):
        raise ValueError('Invalid special token policy: padding and mask tokens must be recorded')
    if manifest.get('tokens_sha256') != file_hash(directory / 'tokens.npy'):
        raise ValueError('Prepared data checksum mismatch')
    tokens = np.load(directory / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    if (tokens.ndim != 2 or tokens.shape[0] < minimum_rows
            or not 0 < tokens.shape[1] <= config.max_position_embeddings):
        raise ValueError('Invalid prepared row shape or insufficient rows for global batch')
    if tokens.dtype != np.int32:
        raise ValueError('Prepared tokens must have int32 dtype')
    for key, actual in [('rows', tokens.shape[0]), ('sequence_length', tokens.shape[1])]:
        if key in manifest and (type(manifest[key]) is not int or manifest[key] != actual):
            raise ValueError('Prepared manifest dimensions do not match tokens')
    for start in range(0, len(tokens), 1024):
        chunk = tokens[start:start + 1024]
        if np.any((chunk < 0) | (chunk >= config.vocab_size)):
            raise ValueError('Prepared token outside vocabulary')
    return tokens, manifest


class EpochRows:
    """Replayable epoch permutations; retains at most one row-index vector."""
    def __init__(self, rows, seed, shuffle=False):
        if rows <= 0 or seed < 0:
            raise ValueError('Positive row count and nonnegative seed required')
        self.rows, self.seed, self.shuffle = rows, seed, shuffle
        self.epoch = None
        self.order = None

    def batch(self, step, size):
        if step < 0 or size <= 0:
            raise ValueError('Invalid batch cursor')
        positions = np.arange(step * size, (step + 1) * size, dtype=np.int64)
        if not self.shuffle:
            return positions % self.rows
        result = np.empty(size, dtype=np.int64)
        epochs = positions // self.rows
        for epoch in np.unique(epochs):
            if self.epoch != int(epoch):
                self.order = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)])).permutation(self.rows)
                self.epoch = int(epoch)
            take = epochs == epoch
            assert self.order is not None
            result[take] = self.order[positions[take] % self.rows]
        return result


class LanguageRows:
    """Replayable temperature sampling with expected NONPADDING token shares.

    Sampling is with replacement. Languages are weighted by token_count**exponent,
    then corrected for mean row occupancy so padding does not alter token shares.
    Each step is independently seeded; workers slice the same global row IDs.
    """
    def __init__(self, directory, tokens, manifest, seed, exponent):
        if seed < 0 or not np.isfinite(exponent) or not 0 <= exponent <= 1:
            raise ValueError('Require nonnegative seed and language exponent in [0, 1]')
        if manifest.get('split') != 'train':
            raise ValueError('Language sampling requires an explicit training split')
        path = Path(directory) / 'documents.jsonl'
        if not manifest.get('documents_sha256') or file_hash(path) != manifest['documents_sha256']:
            raise ValueError('Language sampling requires verified document provenance')
        groups, seen, cursor = {}, set(), 0
        with path.open() as stream:
            for line in stream:
                doc = json.loads(line)
                language = doc.get('language_script')
                end = doc.get('end_row')
                if (not isinstance(language, str) or not language or not isinstance(doc.get('id'), str)
                        or not doc['id'] or doc['id'] in seen or doc.get('first_row') != cursor
                        or type(end) is not int or not cursor < end <= len(tokens)):
                    raise ValueError('Invalid language document coverage or identity')
                groups.setdefault(language, []).extend(range(cursor, end))
                seen.add(doc['id'])
                cursor = end
        if cursor != len(tokens) or len(seen) != manifest.get('documents'):
            raise ValueError('Incomplete language document provenance')
        self.languages = sorted(groups)
        if set(manifest.get('language_counts', {})) != set(self.languages):
            raise ValueError('Language inventory does not match document provenance')
        self.indices = [np.asarray(groups[name], dtype=np.int64) for name in self.languages]
        self.row_language = np.empty(len(tokens), dtype=np.int32)
        self.nonpadding = np.empty(len(tokens), dtype=np.int64)
        for start in range(0, len(tokens), 1024):
            self.nonpadding[start:start + 1024] = np.sum(tokens[start:start + 1024] != manifest['pad_token_id'], axis=1)
        counts = []
        for i, indices in enumerate(self.indices):
            self.row_language[indices] = i
            counts.append(int(self.nonpadding[indices].sum()))
            recorded = manifest.get('language_counts', {}).get(self.languages[i], {})
            if recorded.get('rows') != len(indices) or recorded.get('nonpadding_tokens') != counts[-1]:
                raise ValueError('Language token accounting does not match prepared rows')
        counts = np.asarray(counts, dtype=np.float64)
        if not len(counts) or np.any(counts <= 0):
            raise ValueError('Language groups must contain nonpadding tokens')
        self.target_token_shares = counts ** exponent
        self.target_token_shares /= self.target_token_shares.sum()
        self.row_probabilities = self.target_token_shares / (counts / np.array([len(i) for i in self.indices]))
        self.row_probabilities /= self.row_probabilities.sum()
        self.seed = seed

    def batch(self, step, size):
        if step < 0 or size <= 0:
            raise ValueError('Invalid batch cursor')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        languages = rng.choice(len(self.indices), size=size, p=self.row_probabilities)
        result = np.empty(size, dtype=np.int64)
        for i in np.unique(languages):
            selected = languages == i
            result[selected] = rng.choice(self.indices[i], size=int(selected.sum()))
        return result

    def token_counts(self, indices):
        return {name: int(self.nonpadding[indices][self.row_language[indices] == i].sum())
                for i, name in enumerate(self.languages)}


class MixtureRows(LanguageRows):
    """Fixed source token shares, with temperature sampling inside each source.

    Provenance and policy live in the checksummed manifest, so resume rejects
    changing the mixture. Row occupancy correction preserves expected token
    shares when languages and sources have different amounts of padding.
    """
    def __init__(self, directory, tokens, manifest, seed, exponent):
        super().__init__(directory, tokens, manifest, seed, exponent)
        shares = manifest.get('source_token_shares')
        if (not isinstance(shares, dict) or not shares
                or any(not isinstance(k, str) or not k or type(v) not in (int, float)
                       or not np.isfinite(v) or v <= 0 for k, v in shares.items())
                or not np.isclose(sum(shares.values()), 1., rtol=0, atol=1e-8)):
            raise ValueError('Require positive source token shares summing to one')
        groups = {}
        self.sources = sorted(shares)
        self.row_source = np.empty(len(tokens), dtype=np.int32)
        with (Path(directory) / 'documents.jsonl').open() as stream:
            for line in stream:
                doc = json.loads(line)
                source = doc.get('source_group')
                if source not in shares:
                    raise ValueError('Document source missing from mixture recipe')
                indices = range(doc['first_row'], doc['end_row'])
                groups.setdefault((source, doc['language_script']), []).extend(indices)
                self.row_source[doc['first_row']:doc['end_row']] = self.sources.index(source)
        if {s for s, _ in groups} != set(shares):
            raise ValueError('Missing mixture source data')
        self.pools = sorted(groups)
        self.indices = [np.asarray(groups[key], dtype=np.int64) for key in self.pools]
        counts = np.asarray([self.nonpadding[i].sum() for i in self.indices], dtype=np.float64)
        weights = counts ** exponent
        self.target_token_shares = np.zeros(len(self.pools))
        for source, share in shares.items():
            selected = np.asarray([s == source for s, _ in self.pools], dtype=np.bool_)
            self.target_token_shares[selected] = share * weights[selected] / weights[selected].sum()
        self.row_probabilities = self.target_token_shares / (counts / np.asarray([len(i) for i in self.indices]))
        self.row_probabilities /= self.row_probabilities.sum()

    def source_token_counts(self, indices):
        return {name: int(self.nonpadding[indices][self.row_source[indices] == i].sum())
                for i, name in enumerate(self.sources)}


class CoverageMixtureRows(MixtureRows):
    """Deterministic mixture with one pass per pool before any row is reused.

    The cursor is the global training step and batch size. Pool choices use a
    sequential RNG stream; seeking replays only pool choices, then reconstructs
    per-row exposure from complete pool epochs and the current permutation.
    No sampler state needs to be added to model checkpoints.
    """

    def __init__(self, directory, tokens, manifest, seed, exponent):
        super().__init__(directory, tokens, manifest, seed, exponent)
        self._rng = None
        self._next_step = 0
        self._positions = np.zeros(len(self.pools), dtype=np.int64)
        self.row_exposures = np.zeros(len(tokens), dtype=np.int64)
        self._order_cache = {}
        self.seek(0, 1)

    def _order(self, pool, epoch):
        cached = self._order_cache.get(pool)
        if cached is None or cached[0] != epoch:
            self._order_cache[pool] = (epoch, np.random.default_rng(
                np.random.SeedSequence([self.seed, 0xC0FEBABE, pool, epoch])
            ).permutation(self.indices[pool]))
        return self._order_cache[pool][1]

    def seek(self, step, size):
        """Restore an exact cursor using the checkpoint's completed step."""
        if step < 0 or size <= 0:
            raise ValueError('Invalid batch cursor')
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0xC0FEBABE]))
        self._positions[:] = 0
        remaining = step * size
        boundaries = np.cumsum(self.row_probabilities)
        while remaining:
            take = min(remaining, 1_000_000)
            pools = np.searchsorted(boundaries, self._rng.random(take), side='right')
            self._positions += np.bincount(pools, minlength=len(self.pools))
            remaining -= take
        self.row_exposures[:] = 0
        for pool, indices in enumerate(self.indices):
            epoch, offset = divmod(int(self._positions[pool]), len(indices))
            self.row_exposures[indices] = epoch
            if offset:
                self.row_exposures[self._order(pool, epoch)[:offset]] += 1
        self._next_step = step
        self._batch_size = size

    def batch(self, step, size):
        if step < 0 or size <= 0:
            raise ValueError('Invalid batch cursor')
        if step != self._next_step or size != self._batch_size:
            self.seek(step, size)
        if self._rng is None:
            raise RuntimeError('Coverage sampler RNG is not initialized')
        pools = np.searchsorted(np.cumsum(self.row_probabilities), self._rng.random(size), side='right')
        result = np.empty(size, dtype=np.int64)
        for pool in np.unique(pools):
            positions = np.flatnonzero(pools == pool)
            length = len(self.indices[pool])
            cursor = int(self._positions[pool])
            for position in positions:
                epoch, offset = divmod(cursor, length)
                result[position] = self._order(int(pool), epoch)[offset]
                cursor += 1
            self._positions[pool] = cursor
        np.add.at(self.row_exposures, result, 1)
        self._next_step += 1
        return result

    def exposure_summary(self):
        return {':'.join(name): {'rows': len(indices),
                                 'seen_rows': int(np.count_nonzero(self.row_exposures[indices])),
                                 'total_exposures': int(self._positions[i]),
                                 'max_row_exposures': int(self.row_exposures[indices].max())}
                for i, (name, indices) in enumerate(zip(self.pools, self.indices, strict=True))}


class RemainingCoverageMixtureRows(CoverageMixtureRows):
    """One finite pass over rows never seen before a frozen legacy transition.

    Legacy pool-epoch exposure is reconstructed exactly. The new weighted stream
    removes exhausted pools and renormalizes its original weights. Per-pool
    permutations and RNG are pinned by seed, transition step and batch size.
    Final padding is -1, never an index into the corpus. No model state changes.
    """
    def __init__(self, directory, tokens, manifest, seed, exponent, transition_step, transition_batch_size):
        if (type(seed) is not int or seed < 0 or type(transition_step) is not int or transition_step < 0
                or type(transition_batch_size) is not int or transition_batch_size < 1
                or type(exponent) not in (int, float) or not np.isfinite(exponent) or not 0 <= exponent <= 1):
            raise ValueError('Finite integer seed/transition step and positive transition batch required')
        self._remaining_initialized = False
        super().__init__(directory, tokens, manifest, seed, exponent)
        if np.any(~np.isfinite(self.row_probabilities)) or np.any(self.row_probabilities <= 0):
            raise ValueError('Finite positive weights required for every remaining coverage pool')
        CoverageMixtureRows.seek(self, transition_step, transition_batch_size)
        self.transition_step, self.transition_batch_size = transition_step, transition_batch_size
        self._historical_seen = self.row_exposures > 0
        self.historical_seen_rows = int(np.count_nonzero(self._historical_seen))
        self._legacy_positions = self._positions.copy()
        self._legacy_summary = CoverageMixtureRows.exposure_summary(self)
        self._remaining_orders = []
        for pool, indices in enumerate(self.indices):
            unseen = indices[~self._historical_seen[indices]]
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, 0x52454D41, transition_step, transition_batch_size, pool]))
            self._remaining_orders.append(rng.permutation(unseen))
        self._order_cache.clear()
        self._remaining_lengths = np.asarray([len(order) for order in self._remaining_orders], dtype=np.int64)
        self.total_unseen_rows = int(self._remaining_lengths.sum())
        self._remaining_positions = np.zeros(len(self.pools), dtype=np.int64)
        self._remaining_tokens = np.asarray([self._token_sum(order) for order in self._remaining_orders], dtype=np.int64)
        self.total_unseen_tokens = int(self._remaining_tokens.sum())
        self.completion_step = transition_step + (self.total_unseen_rows + transition_batch_size - 1) // transition_batch_size
        self.policy_identity = {'coverage_sampling': 'remaining_pool_epoch_v1',
            'coverage_transition_step': transition_step, 'coverage_transition_batch_size': transition_batch_size}
        self._remaining_initialized = True
        self.seek(transition_step, transition_batch_size)

    def _token_sum(self, order):
        return sum(int(self.nonpadding[order[first:first + 65536]].sum()) for first in range(0, len(order), 65536))

    @property
    def remaining_unique_rows(self):
        return self.total_unseen_rows - int(self._remaining_positions.sum())

    @property
    def remaining_unique_tokens(self):
        return int(self._remaining_tokens.sum())

    def _validate_cursor(self, step, size):
        if (type(step) is not int or type(size) is not int or size != self.transition_batch_size
                or not self.transition_step <= step <= self.completion_step):
            raise ValueError('Remaining coverage cursor must use frozen transition batch within finite horizon')

    def _draw_remaining(self, count, *, materialize):
        # Reuse each uniform when a pool exhausts. Vector boundaries therefore
        # cannot change the stream when replay uses bounded million-row chunks.
        uniform = self._rng.random(count)
        result = np.full(count, -1, dtype=np.int64) if materialize else None
        offset = 0
        while offset < count:
            available = self._remaining_lengths - self._remaining_positions
            active = np.flatnonzero(available > 0)
            if not len(active):
                break
            weights = self.row_probabilities[active]
            boundaries = np.cumsum(weights / weights.sum())
            boundaries[-1] = 1.
            choices = active[np.searchsorted(boundaries, uniform[offset:], side='right')]
            counts = np.bincount(choices, minlength=len(self.pools))
            exhausted = np.flatnonzero((available > 0) & (counts >= available))
            take = len(choices)
            for pool in exhausted:
                take = min(take, int(np.flatnonzero(choices == pool)[int(available[pool]) - 1]) + 1)
            if take < len(choices):
                counts = np.bincount(choices[:take], minlength=len(self.pools))
            if not take:
                raise RuntimeError('Nonexhausted pool scheduling made no progress')
            if materialize:
                for pool in np.flatnonzero(counts):
                    positions = np.flatnonzero(choices[:take] == pool)
                    cursor = int(self._remaining_positions[pool])
                    ids = self._remaining_orders[pool][cursor:cursor + len(positions)]
                    result[offset + positions] = ids
                    self.row_exposures[ids] = 1
                    self._remaining_tokens[pool] -= int(self.nonpadding[ids].sum())
            self._remaining_positions += counts
            offset += take
        self._positions[:] = self._legacy_positions + self._remaining_positions
        return result

    def seek(self, step, size):
        if not self._remaining_initialized:
            return CoverageMixtureRows.seek(self, step, size)
        self._validate_cursor(step, size)
        self._rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, 0x52454D42, self.transition_step, self.transition_batch_size]))
        self._remaining_positions[:] = 0
        self.row_exposures[~self._historical_seen] = 0
        remaining = min((step - self.transition_step) * size, self.total_unseen_rows)
        while remaining:
            take = min(remaining, 1_000_000)
            self._draw_remaining(take, materialize=False)
            remaining -= take
        for pool, order in enumerate(self._remaining_orders):
            cursor = int(self._remaining_positions[pool])
            self.row_exposures[order[:cursor]] = 1
            self._remaining_tokens[pool] = self._token_sum(order[cursor:])
        self._positions[:] = self._legacy_positions + self._remaining_positions
        self._next_step, self._batch_size = step, size

    def batch(self, step, size):
        self._validate_cursor(step, size)
        if step != self._next_step:
            self.seek(step, size)
        if step == self.completion_step:
            raise StopIteration('Every previously unseen corpus row has been consumed')
        result = self._draw_remaining(size, materialize=True)
        self._next_step += 1
        return result

    def _valid_indices(self, indices):
        values = np.asarray(indices)
        if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < -1) or np.any(values >= len(self.nonpadding))):
            raise ValueError('Remaining coverage row IDs must be valid or final -1 padding')
        return values[values >= 0]

    def token_counts(self, indices):
        return LanguageRows.token_counts(self, self._valid_indices(indices))

    def source_token_counts(self, indices):
        return MixtureRows.source_token_counts(self, self._valid_indices(indices))

    def exposure_summary(self):
        summary = CoverageMixtureRows.exposure_summary(self)
        for pool, name in enumerate(self.pools):
            key = ':'.join(name)
            summary[key].update(legacy_seen_rows=self._legacy_summary[key]['seen_rows'],
                legacy_exposures=self._legacy_summary[key]['total_exposures'],
                remaining_unique_rows=int(self._remaining_lengths[pool] - self._remaining_positions[pool]),
                remaining_unique_tokens=int(self._remaining_tokens[pool]),
                new_unique_exposures=int(self._remaining_positions[pool]))
        return summary
