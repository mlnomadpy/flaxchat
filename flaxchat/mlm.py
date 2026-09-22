"""Reproducible masked-token batches, independent of host partitioning."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def _replacement_vocabulary(vocab_size, special_ids):
    allowed = np.arange(vocab_size, dtype=np.int32)
    allowed = allowed[~np.isin(allowed, special_ids)]
    allowed.flags.writeable = False
    return allowed


def mask_tokens(tokens, *, seed, step, example_ids, vocab_size, mask_token_id,
                special_token_ids, probability=0.15, segment_ids=None):
    """BERT 80/10/10 corruption, with unshifted labels (-1 means ignored).

    Each row has a PRNG derived from seed, optimizer step and stable example ID.
    Persist those inputs, rather than a process-local RNG, for exact resumption.
    Special tokens are neither prediction targets nor random replacements.
    """
    tokens = np.asarray(tokens)
    example_ids = np.asarray(example_ids)
    if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError('tokens must be a two-dimensional integer array')
    if example_ids.shape != (len(tokens),) or not np.issubdtype(example_ids.dtype, np.integer):
        raise ValueError('example_ids must contain one integer per row')
    if not 0 <= probability <= 1 or seed < 0 or step < 0 or np.any(example_ids < 0):
        raise ValueError('Invalid probability, seed, step, or example ID')
    if vocab_size <= 0 or not 0 <= mask_token_id < vocab_size or np.any((tokens < 0) | (tokens >= vocab_size)):
        raise ValueError('Token IDs must be inside vocabulary')
    specials = set(special_token_ids) | {mask_token_id}
    if any(t < 0 or t >= vocab_size for t in specials):
        raise ValueError('Special token outside vocabulary')
    allowed = _replacement_vocabulary(vocab_size, tuple(sorted(specials)))
    if not len(allowed):
        raise ValueError('Vocabulary must contain non-special tokens')
    eligible = ~np.isin(tokens, list(specials))
    if segment_ids is not None:
        if np.shape(segment_ids) != tokens.shape:
            raise ValueError('segment_ids must match tokens')
        eligible &= np.asarray(segment_ids) >= 0
    inputs = tokens.astype(np.int32, copy=True)
    targets = np.full(tokens.shape, -1, dtype=np.int32)
    for i, example_id in enumerate(example_ids):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, int(example_id)]))
        selected = (rng.random(tokens.shape[1]) < probability) & eligible[i]
        choice = rng.random(tokens.shape[1])
        replacement = rng.choice(allowed, size=tokens.shape[1])
        targets[i, selected] = tokens[i, selected]
        inputs[i, selected & (choice < .8)] = mask_token_id
        randomize = selected & (choice >= .8) & (choice < .9)
        inputs[i, randomize] = replacement[randomize]
    return inputs, targets
