"""Train a ModernBERT MLM on prepared, unshifted document rows.

Run with python -m scripts.train_encoder --help. Data parallel training uses
one global JAX mesh; parameters are replicated. FSDP is not yet implemented for
this encoder. Use the existing external TPU budget/deadline supervisor on GCP.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
from jax.sharding import Mesh

from flaxchat.checkpoint import create_checkpoint_manager, restore_model_from_checkpoint, save_checkpoint
from flaxchat.common import replicate_on_mesh
from flaxchat.encoder import EncoderConfig, ModernBert, import_hf_weights
from flaxchat.mlm import mask_tokens
from flaxchat.training import apply_gradients_if_finite, place_host_batch, gather_process_metadata


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained(model, directory):
    """Load a local HF safetensors snapshot; never execute repository code."""
    from safetensors.numpy import load_file
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    if not index.exists() and not (directory / 'model.safetensors').exists():
        raise ValueError('Convert pytorch_model.bin first with python -m scripts.convert_encoder_checkpoint SNAPSHOT')
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    tensors = {}
    provenance = {}
    for name in names:
        path = (directory / name).resolve()
        if path.parent != directory.resolve():
            raise ValueError('Checkpoint shard must be inside snapshot directory')
        provenance[name] = file_hash(path)
        shard = load_file(str(path))
        if set(tensors) & set(shard):
            raise ValueError('Duplicate checkpoint tensors')
        tensors.update(shard)
    import_hf_weights(model, tensors)
    return provenance


def run(args):
    if args.distributed and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if args.preflight_only and args.resume:
        raise ValueError('Preflight validates local inputs, not checkpoint resume')
    if args.steps <= 0 or args.batch_size <= 0 or args.save_every <= 0 or args.seed < 0:
        raise ValueError('steps, batch-size, save-every must be positive; seed nonnegative')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 < args.mask_probability <= 1:
        raise ValueError('Invalid learning rate or mask probability')
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        raise ValueError('stop-after must be within training horizon')
    if args.batch_size % jax.device_count():
        raise ValueError('Global batch must be divisible by global device count')
    if jax.process_count() > 1 and not args.output.startswith('gs://') and not args.shared_local_checkpoints:
        raise ValueError('Multi-host checkpoints require gs:// or an explicitly shared filesystem')
    raw = json.loads(Path(args.config).read_text())
    overrides = dict(compute_dtype=args.dtype, residual_dtype=args.residual_dtype, attention_backend=args.attention_backend,
                     loss_chunk_size=args.loss_chunk_size, use_remat=not args.no_remat,
                     mlm_projection=args.mlm_projection, mlm_loss_backend=args.mlm_loss_backend,
                     mlm_vocab_tile=args.mlm_vocab_tile)
    config = EncoderConfig.from_hf(raw, **overrides) if raw.get('model_type') else EncoderConfig(**(raw | overrides))
    if config.attention_backend == 'splash' and jax.device_count() > 1:
        raise ValueError('Multi-device encoder training currently requires --attention-backend xla')
    data_dir = Path(args.data)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    if manifest.get('format') != 'flaxchat-encoder-rows-v1' or not manifest.get('tokenizer_sha256'):
        raise ValueError('A prepared encoder manifest with tokenizer identity is required')
    if manifest['vocab_size'] != config.vocab_size or manifest['pad_token_id'] != config.pad_token_id:
        raise ValueError('Prepared tokenizer vocabulary/padding does not match model')
    if manifest.get('mask_token_id', config.mask_token_id) != config.mask_token_id:
        raise ValueError('Prepared mask token does not match model')
    if not {config.pad_token_id, config.mask_token_id} <= set(manifest['special_token_ids']):
        raise ValueError('Padding and mask tokens must be recorded among tokenizer special tokens')
    data_hash = file_hash(data_dir / 'tokens.npy')
    if manifest['tokens_sha256'] != data_hash:
        raise ValueError('Prepared data checksum mismatch')
    tokens = np.load(data_dir / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    if tokens.ndim != 2 or tokens.shape[0] < args.batch_size or tokens.shape[1] > config.max_position_embeddings:
        raise ValueError('Invalid prepared row shape or insufficient rows for global batch')
    if tokens.dtype != np.int32:
        raise ValueError('Prepared tokens must have int32 dtype')
    # Validate in bounded chunks before accelerator compilation/allocation.
    for start in range(0, len(tokens), 1024):
        chunk = tokens[start:start + 1024]
        if np.any((chunk < 0) | (chunk >= config.vocab_size)):
            raise ValueError('Prepared token outside vocabulary')
    if args.pretrained:
        tokenizer_path = Path(args.pretrained) / 'tokenizer.json'
        if not tokenizer_path.exists() or file_hash(tokenizer_path) != manifest['tokenizer_sha256']:
            raise ValueError('Prepared tokenizer must match pretrained snapshot tokenizer.json')
        pretrained_config = EncoderConfig.from_hf(json.loads((Path(args.pretrained) / 'config.json').read_text()), **overrides)
        if pretrained_config != config:
            raise ValueError('Pretrained architecture does not match requested configuration')
    identity = dict(encoder=asdict(config), steps=args.steps, batch_size=args.batch_size,
                    seed=args.seed, learning_rate=args.learning_rate, mask_probability=args.mask_probability,
                    special_token_ids=manifest['special_token_ids'])
    source_digest = hashlib.sha256(''.join(file_hash(path) for path in (
        Path(__file__), Path(__file__).parents[1] / 'flaxchat/encoder.py',
        Path(__file__).parents[1] / 'flaxchat/mlm.py',
        Path(__file__).parents[1] / 'flaxchat/training.py',
        Path(__file__).parents[1] / 'flaxchat/common.py',
        Path(__file__).parents[1] / 'flaxchat/checkpoint.py',
        Path(__file__).parents[1] / 'flaxchat/fused_cross_entropy.py')).encode()).hexdigest()
    metadata = dict(source_python_sha256=source_digest, resolved_config=identity, tokenizer_identity=manifest['tokenizer_sha256'],
                    data_manifest_identity=file_hash(data_dir / 'manifest.json'), model_family='modernbert')
    expected = dict(resolved_config=identity, tokenizer=metadata['tokenizer_identity'],
                    data_manifest=metadata['data_manifest_identity'], source_python_sha256=source_digest)
    identities = gather_process_metadata(expected)
    if any(item != identities[0] for item in identities):
        raise ValueError('Workers disagree on data, tokenizer, source, or training configuration')
    if args.preflight_only:
        print(json.dumps(dict(event='input_preflight', passed=True,
                              config=identity, rows=len(tokens), sequence_length=tokens.shape[1])), flush=True)
        return
    model = ModernBert(config, rngs=nnx.Rngs(args.seed))
    if args.pretrained and not args.resume:
        metadata['initial_weights_sha256'] = load_pretrained(model, args.pretrained)
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.),
                              optax.adamw(args.learning_rate, weight_decay=.01)), wrt=nnx.Param)
    start_step = 0
    if args.resume:
        metadata, state = restore_model_from_checkpoint(model, args.output, optimizer=optimizer,
                                                expected_identity=expected, load_training_state=True)
        if state is None:
            raise ValueError('Checkpoint has no training cursor')
        start_step = int(state['completed_steps'][0])
        optimizer.step[...] = jnp.asarray(state['optimizer_updates'][0], dtype=optimizer.step.dtype)
    nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    manager = create_checkpoint_manager(args.output, async_checkpointing=False)
    if not args.resume and manager.latest_step() is not None:
        manager.close()
        raise ValueError('Output already contains checkpoints; use --resume or a new directory')

    @nnx.jit
    def train_step(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(lambda m: m(x, y))(model)
        selected = (y >= 0).sum()
        capacity = min(y.shape[1], ((y.shape[1] + 4 * config.loss_chunk_size - 1) //
                                    (4 * config.loss_chunk_size)) * config.loss_chunk_size)
        overflow = (config.mlm_projection == 'masked') & jnp.any((y >= 0).sum(axis=1) > capacity)
        # Empty-mask batches must not cause an AdamW weight-decay-only update.
        updated = nnx.cond(selected > 0,
                           lambda m, o, g, loss_value: apply_gradients_if_finite(m, o, g, loss_value),
                           lambda m, o, g, loss_value: jnp.array(False), model, optimizer, grads, loss)
        return loss, selected, updated, overflow

    if jax.process_index() == 0:
        print(json.dumps(dict(event='run_config', backend=jax.default_backend(),
            devices=jax.device_count(), processes=jax.process_count(),
            compute_dtype=config.compute_dtype, residual_dtype=config.residual_dtype,
            mlm_projection=config.mlm_projection,
            mlm_loss_backend=config.mlm_loss_backend, mlm_vocab_tile=config.mlm_vocab_tile,
            parameter_dtype='float32', optimizer_dtype='float32',
            matmul_precision=str(jax.default_matmul_precision.value))), flush=True)
    end = args.stop_after or args.steps
    local_batch = args.batch_size // jax.process_count()
    try:
        for step in range(start_step, end):
            started = time.monotonic()
            global_ids = (np.arange(args.batch_size, dtype=np.int64) + step * args.batch_size) % len(tokens)
            local_ids = global_ids[jax.process_index() * local_batch:(jax.process_index() + 1) * local_batch]
            x, y = mask_tokens(tokens[local_ids], seed=args.seed, step=step, example_ids=local_ids,
                vocab_size=config.vocab_size, mask_token_id=config.mask_token_id,
                special_token_ids=manifest['special_token_ids'], probability=args.mask_probability)
            loss, selected, updated, overflow = train_step(model, optimizer, place_host_batch(x, mesh), place_host_batch(y, mesh))
            loss_value, selected_value, updated_value = float(loss), int(selected), bool(updated)
            if selected_value and not updated_value:
                raise FloatingPointError('Rejected nonfinite encoder optimizer update')
            elapsed = time.monotonic() - started
            if jax.process_index() == 0:
                print(json.dumps(dict(event='train_step', step=step + 1, loss=loss_value, masked_tokens=selected_value,
                                      updated=updated_value, tokens=args.batch_size * tokens.shape[1],
                                      projection_dense_fallback=bool(overflow),
                                      seconds=elapsed, includes_compilation=step == start_step,
                                      tokens_per_second=args.batch_size * tokens.shape[1] / elapsed)), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == end:
                checkpoint_started = time.monotonic()
                save_checkpoint(manager, step + 1, model, optimizer, metadata,
                                training_state={'completed_steps': np.array([step + 1], dtype=np.int32),
                                                'optimizer_updates': np.array([int(optimizer.step[...])], dtype=np.int32)})
                manager.wait_until_finished()
                if jax.process_index() == 0:
                    print(json.dumps(dict(event='checkpoint', step=step + 1,
                        seconds=time.monotonic() - checkpoint_started)), flush=True)
    finally:
        manager.close()
        if jax.process_index() == 0:
            print(json.dumps(dict(event='device_memory', devices=[
                dict(device=str(d), statistics=d.memory_stats()) for d in jax.local_devices()])), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Encoder config or HF ModernBERT config.json')
    p.add_argument('--data', required=True, help='Prepared tokens.npy and manifest.json directory')
    p.add_argument('--output', required=True)
    p.add_argument('--pretrained', help='Local HF snapshot with config, tokenizer, and safetensors')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--stop-after', type=int, help='Stop early while preserving resume horizon')
    p.add_argument('--batch-size', type=int, default=8, help='Global batch across all hosts')
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--mask-probability', type=float, default=.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--residual-dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16',
                   help='Dense/attention compute; master parameters and Adam state remain FP32')
    p.add_argument('--attention-backend', choices=['xla', 'splash'], default='xla')
    p.add_argument('--loss-chunk-size', type=int, default=128)
    p.add_argument('--mlm-projection', choices=['dense', 'masked'], default='dense',
                   help='Experimental masked-position projection with loss-preserving overflow fallback')
    p.add_argument('--mlm-loss-backend', choices=['xla', 'xla_full', 'pallas'], default='xla')
    p.add_argument('--mlm-vocab-tile', type=int, default=1024)
    p.add_argument('--no-remat', action='store_true')
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--preflight-only', action='store_true', help='Validate local config/tokenizer/data without training or creating checkpoints')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--shared-local-checkpoints', action='store_true',
                   help='Confirm output is a filesystem shared by every worker')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
