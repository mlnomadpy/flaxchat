"""Explicit weights-only migration to V-only GOAT; never an optimizer resume."""
from dataclasses import replace

from flax import nnx


def migrate_yat_to_goat(source, target):
    """Copy an authenticated, already-loaded YAT encoder into a GOAT encoder.

    Keep the V third of each combined projection and all other parameters.
    Caller must create a fresh optimizer and record the source checkpoint identity.
    All compatibility checks finish before any target parameter is changed.
    """
    a, b = source.config, target.config
    if a.attention_score != 'yat_softmax' or b.attention_score not in ('goat', 'goat_input'):
        raise ValueError('Migration requires yat_softmax source and goat/goat_input target')
    if replace(a, attention_score=b.attention_score, yat_local_shards=False) != b:
        raise ValueError('GOAT migration may only change attention architecture and local sharding')
    old = nnx.state(source, nnx.Param).flat_state()
    new = nnx.state(target, nnx.Param).flat_state()
    old_map = dict(old)
    updates = []
    consumed = set()
    for path, variable in new:
        source_path = path
        is_v = len(path) == 4 and path[0] == 'layers' and path[2:] == ('v', 'kernel')
        if is_v:
            source_path = (*path[:2], 'qkv', 'kernel')
        if source_path not in old_map:
            raise ValueError(f'Missing source parameter: {source_path}')
        value = old_map[source_path][...]
        if is_v:
            if value.shape != (a.hidden_size, 3 * a.hidden_size):
                raise ValueError('Unexpected QKV layout')
            value = value[:, 2 * a.hidden_size:]
        if value.shape != variable.shape or value.dtype != variable.dtype:
            raise ValueError(f'Incompatible target parameter: {path}')
        consumed.add(source_path)
        updates.append((variable, value))
    if consumed != set(dict(old)):
        raise ValueError('Unconsumed source parameters')
    for variable, value in updates:
        variable[...] = value
    nnx.update(target, new.to_nested_state())
