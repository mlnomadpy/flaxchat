"""Deterministic, masked-token-weighted held-out evaluation of an encoder checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint
from flaxchat.common import replicate_on_mesh
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.mlm import mask_tokens
from flaxchat.training import place_host_batch
from scripts.train_encoder import file_hash


def evaluate(checkpoint, data, *, train_data, batch_size=8, max_rows=1024, seed=2026):
    if batch_size <= 0 or max_rows <= 0 or seed < 0:
        raise ValueError('Invalid evaluation bounds or seed')
    metadata = load_checkpoint_metadata(checkpoint)
    if metadata.get('model_family') != 'modernbert':
        raise ValueError('Expected an encoder checkpoint')
    identity = metadata['resolved_config']
    config = EncoderConfig(**identity['encoder'])
    if config.attention_backend != 'xla':
        raise ValueError('Portable evaluation currently requires an XLA encoder checkpoint')
    arrays = []
    for directory in (train_data, data):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['tokenizer_sha256'] != metadata['tokenizer_identity']:
            raise ValueError('Evaluation tokenizer mismatch')
        if file_hash(directory / 'tokens.npy') != manifest['tokens_sha256']:
            raise ValueError('Prepared data checksum mismatch')
        if manifest['special_token_ids'] != identity['special_token_ids']:
            raise ValueError('Special token policy mismatch')
        arrays.append(np.load(directory / 'tokens.npy', mmap_mode='r', allow_pickle=False))
    if file_hash(Path(train_data) / 'manifest.json') != metadata['data_manifest_identity']:
        raise ValueError('Training pool identity mismatch')
    train, validation = arrays
    for array in arrays:
        if array.ndim != 2 or array.dtype != np.int32 or array.shape[1] > config.max_position_embeddings:
            raise ValueError('Invalid evaluation row layout')
    def row_hash(row):
        return hashlib.sha256(np.asarray(row[row != config.pad_token_id], dtype='<i4').tobytes()).digest()
    training_rows = {row_hash(row) for row in train}
    chosen = []
    excluded = 0
    seen = set()
    for i, row in enumerate(validation):
        digest = row_hash(row)
        if digest in training_rows or digest in seen:
            excluded += 1
            continue
        seen.add(digest)
        chosen.append(i)
        if len(chosen) == max_rows:
            break
    if not chosen:
        raise ValueError('No held-out rows after exact-row overlap exclusion')
    model = ModernBert(config, rngs=nnx.Rngs(0))
    restore_model_from_checkpoint(model, checkpoint, expected_identity={'resolved_config': identity})
    loss_fn = nnx.jit(lambda m, x, y: m(x, y))
    execution_batch = batch_size
    mesh = None
    if config.mlm_loss_backend in ('pallas', 'xla_full'):
        devices = jax.device_count()
        execution_batch = ((batch_size + devices - 1) // devices) * devices
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('data',))
        nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    numerator, denominator = 0., 0
    for start in range(0, len(chosen), batch_size):
        indices = np.array(chosen[start:start + batch_size])
        x, y = mask_tokens(validation[indices], seed=seed, step=0, example_ids=indices,
            vocab_size=config.vocab_size, mask_token_id=config.mask_token_id,
            special_token_ids=identity['special_token_ids'], probability=identity['mask_probability'])
        count = int((y >= 0).sum())
        missing = execution_batch - len(x)
        x = np.pad(x, ((0, missing), (0, 0)), constant_values=config.pad_token_id)
        y = np.pad(y, ((0, missing), (0, 0)), constant_values=-1)
        x_device = place_host_batch(x, mesh) if mesh is not None else jnp.asarray(x)
        y_device = place_host_batch(y, mesh) if mesh is not None else jnp.asarray(y)
        loss = float(loss_fn(model, x_device, y_device))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite evaluation loss')
        numerator += loss * count
        denominator += count
    if not denominator:
        raise ValueError('Evaluation contains no selected masked tokens')
    return dict(masked_token_loss=numerator / denominator, masked_tokens=denominator,
                evaluated_rows=len(chosen), excluded_exact_or_duplicate_rows=excluded, seed=seed,
                mask_probability=identity['mask_probability'],
                overlap_check='exact unpadded token rows only; not substring or semantic decontamination',
                tokenizer_sha256=metadata['tokenizer_identity'],
                validation_manifest_sha256=file_hash(Path(data) / 'manifest.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-rows', type=int, default=1024)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = evaluate(a.checkpoint, a.data, train_data=a.train_data,
                      batch_size=a.batch_size, max_rows=a.max_rows, seed=a.seed)
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
