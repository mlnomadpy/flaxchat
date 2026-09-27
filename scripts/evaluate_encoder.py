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
from flaxchat.runtime import runtime_identity
from flaxchat.training import place_host_batch
from flaxchat.encoder_data import file_hash, load_prepared_rows


def evaluation_order(data, rows, *, seed, balanced_languages=False):
    """Round-robin shuffled language rows; validate complete document provenance."""
    if not balanced_languages:
        return np.arange(rows), None
    groups = {}
    labels = []
    cursor = 0
    for line in (Path(data) / 'documents.jsonl').read_text().splitlines():
        doc = json.loads(line)
        name, start, end = doc.get('language_script'), doc.get('first_row'), doc.get('end_row')
        if (not isinstance(name, str) or not name or type(start) is not int
                or type(end) is not int or start != cursor or not start < end <= rows):
            raise ValueError('Invalid evaluation language document coverage')
        groups.setdefault(name, []).extend(range(start, end))
        labels.extend([name] * (end - start))
        cursor = end
    if cursor != rows:
        raise ValueError('Incomplete evaluation language document coverage')
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(groups[name]).tolist() for name in sorted(groups)]
    order = [pool[i] for i in range(max(map(len, pools))) for pool in pools if i < len(pool)]
    return np.asarray(order), labels


def host_evaluation_rows(x, y, *, rank, processes):
    """Partition an already padded global batch without duplicating examples."""
    if processes < 1 or not 0 <= rank < processes or len(x) != len(y) or len(x) % processes:
        raise ValueError('Invalid evaluation host partition')
    size = len(x) // processes
    return x[rank * size:(rank + 1) * size], y[rank * size:(rank + 1) * size]


def initial_weight_identity(metadata, config, directory):
    """Require the exact original pretrained tensors for a paired baseline."""
    from scripts.train_encoder import pretrained_inventory
    if config.ffn_type != 'geglu' or config.attention_score != 'dot_product':
        raise ValueError('Initial-weight baseline requires the released GeGLU/dot-product architecture')
    expected = metadata.get('initial_weights_sha256')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Checkpoint does not record initial pretrained weight hashes')
    actual = pretrained_inventory(directory, config)
    if actual != expected:
        raise ValueError('Initial pretrained weight identity mismatch')
    return actual


def evaluate(checkpoint, data, *, train_data, batch_size=8, max_rows=1024, seed=2026,
             initial_pretrained=None, checkpoint_step=None, balanced_languages=False):
    if batch_size <= 0 or max_rows <= 0 or seed < 0:
        raise ValueError('Invalid evaluation bounds or seed')
    if checkpoint_step is not None and checkpoint_step < 0:
        raise ValueError('Checkpoint step must be nonnegative')
    metadata = load_checkpoint_metadata(checkpoint, step=checkpoint_step)
    selected_step = metadata['step']
    if metadata.get('model_family') != 'modernbert':
        raise ValueError('Expected an encoder checkpoint')
    identity = metadata['resolved_config']
    config = EncoderConfig(**identity['encoder'])
    if config.attention_backend != 'xla':
        raise ValueError('Portable evaluation currently requires an XLA encoder checkpoint')
    arrays = []
    for directory in (train_data, data):
        directory = Path(directory)
        tokens, manifest = load_prepared_rows(directory, config)
        if manifest['tokenizer_sha256'] != metadata['tokenizer_identity']:
            raise ValueError('Evaluation tokenizer mismatch')
        if manifest['special_token_ids'] != identity['special_token_ids']:
            raise ValueError('Special token policy mismatch')
        arrays.append(tokens)
    if file_hash(Path(train_data) / 'manifest.json') != metadata['data_manifest_identity']:
        raise ValueError('Training pool identity mismatch')
    train, validation = arrays
    def row_hash(row):
        return hashlib.sha256(np.asarray(row[row != config.pad_token_id], dtype='<i4').tobytes()).digest()
    training_rows = {row_hash(row) for row in train}
    chosen = []
    excluded = 0
    seen = set()
    order, languages = evaluation_order(data, len(validation), seed=seed,
                                        balanced_languages=balanced_languages)
    for i in order:
        row = validation[i]
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
    initial_hashes = (initial_weight_identity(metadata, config, initial_pretrained)
                      if initial_pretrained is not None else None)
    model = ModernBert(config, rngs=nnx.Rngs(0))
    if initial_pretrained is None:
        restore_model_from_checkpoint(model, checkpoint, step=selected_step, expected_identity={'resolved_config': identity})
    else:
        from scripts.train_encoder import load_pretrained
        if load_pretrained(model, initial_pretrained) != initial_hashes:
            raise ValueError('Initial pretrained weights changed while loading')
    loss_fn = nnx.jit(lambda m, x, y: m(x, y))
    execution_batch = batch_size
    mesh = None
    if jax.process_count() > 1 or config.yat_local_shards or config.mlm_loss_backend in ('pallas', 'xla_full', 'xla_local'):
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
        if mesh is not None:
            x, y = host_evaluation_rows(x, y, rank=jax.process_index(), processes=jax.process_count())
        x_device = place_host_batch(x, mesh) if mesh is not None else jnp.asarray(x)
        y_device = place_host_batch(y, mesh) if mesh is not None else jnp.asarray(y)
        loss = float(loss_fn(model, x_device, y_device))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite evaluation loss')
        numerator += loss * count
        denominator += count
    if not denominator:
        raise ValueError('Evaluation contains no selected masked tokens')
    return dict(parameter_source='initial_pretrained' if initial_pretrained is not None else 'checkpoint',
                checkpoint_context=str(checkpoint), checkpoint_step=selected_step, initial_weights_sha256=metadata.get('initial_weights_sha256'),
                evaluation_source_sha256=file_hash(Path(__file__)),
                runtime=runtime_identity(),
                model_source_sha256={str(path.relative_to(Path(__file__).parents[1])):file_hash(path)
                                     for path in sorted((Path(__file__).parents[1]/'flaxchat').glob('*.py'))},
                evaluation_config_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                evaluation_batch_size=batch_size, backend=jax.default_backend(),
                devices=[d.device_kind for d in jax.devices()],
                selected_rows_sha256=hashlib.sha256(np.asarray(chosen, dtype='<i8').tobytes()).hexdigest(),
                masked_token_loss=numerator / denominator, masked_tokens=denominator,
                evaluated_rows=len(chosen), excluded_exact_or_duplicate_rows=excluded, seed=seed,
                row_selection='balanced_languages' if balanced_languages else 'prefix',
                language_rows=({name: sum(languages[i] == name for i in chosen)
                                for name in sorted(set(languages))} if languages is not None else None),
                mask_probability=identity['mask_probability'],
                overlap_check='exact unpadded token rows only; not substring or semantic decontamination',
                tokenizer_sha256=metadata['tokenizer_identity'],
                validation_manifest_sha256=file_hash(Path(data) / 'manifest.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-step', type=int, help='Pin the checkpoint step; otherwise select latest once')
    p.add_argument('--initial-pretrained', help='Evaluate the exact starting weights using checkpoint evaluation policy')
    p.add_argument('--data', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-rows', type=int, default=1024)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--balanced-languages', action='store_true', help='Round-robin shuffled language rows using document provenance')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = evaluate(a.checkpoint, a.data, train_data=a.train_data,
                      batch_size=a.batch_size, max_rows=a.max_rows, seed=a.seed, initial_pretrained=a.initial_pretrained,
                      checkpoint_step=a.checkpoint_step, balanced_languages=a.balanced_languages)
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
