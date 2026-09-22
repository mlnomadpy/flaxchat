"""Train a GPT on immutable local token pools with exact topology-independent data cursors."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import math


def parse_tokens(value):
    value = value.upper().strip()
    scale = {'B': 1e9, 'M': 1e6, 'K': 1e3}
    return int(float(value[:-1]) * scale[value[-1]]) if value[-1] in scale else int(value)


def checkpoint_due(step, stop, every_steps, elapsed_seconds, interval_seconds):
    return step == stop or (step % every_steps == 0 if every_steps is not None else elapsed_seconds >= interval_seconds)


def main(argv=None):
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-manifest', required=True, help='Prepared local manifest.json; no live streaming')
    parser.add_argument('--compute-dtype', choices=('auto', 'float32', 'bfloat16'), default='auto')
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--batch-per-device', type=int, default=1)
    parser.add_argument('--global-batch-size', type=int, help='Keep batch and schedule fixed across topologies')
    parser.add_argument('--fsdp', type=int, default=1, help='Parameter sharding axis size')
    parser.add_argument('--loss-chunk-size', type=int, default=0)
    parser.add_argument('--remat', action='store_true')
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--tokens', default='1B')
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-batches', type=int, default=1)
    cadence = parser.add_mutually_exclusive_group()
    cadence.add_argument('--save-every', type=int, help='Explicit update-count checkpoint cadence')
    cadence.add_argument('--checkpoint-interval-seconds', type=float, default=300., help='Wall-clock cadence, default 300s; always save at final/clean stop')
    parser.add_argument('--max-vocab-size', type=int, default=60000)
    parser.add_argument('--run-name', default='gpt2-base')
    parser.add_argument('--ckpt-dir', default='artifacts/fineweb-gpt/checkpoints')
    parser.add_argument('--artifact-dir', default='artifacts/fineweb-gpt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Stop cleanly at an absolute update count without changing the LR schedule')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tie-embeddings', dest='tie_embeddings', action='store_true', default=True,
                       help='Share input/output weights (default); standard GPT initializes the shared table at stddev 0.02')
    group.add_argument('--untie-embeddings', dest='tie_embeddings', action='store_false')
    args = parser.parse_args(argv)
    if min(args.depth, args.batch_per_device, args.seq_len, args.eval_every, args.eval_batches) <= 0:
        parser.error('Depth, batch size, sequence length, and evaluation/save intervals must be positive')
    if (args.save_every is not None and args.save_every <= 0) or not math.isfinite(args.checkpoint_interval_seconds) or args.checkpoint_interval_seconds <= 0:
        parser.error('Checkpoint cadence must be finite and positive')
    if args.lr <= 0 or args.warmup_steps < 0:
        parser.error('LR must be positive and warmup nonnegative')
    if args.global_batch_size is not None and args.global_batch_size <= 0:
        parser.error('--global-batch-size must be positive')

    import jax
    import numpy as np
    import hashlib
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flaxchat.common import COMPUTE_DTYPE, compute_init, replicate_on_mesh, setup_mesh, place_array
    from flaxchat.gpt import GPT, GPTConfig, training_attention_backend
    from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint
    from flaxchat.token_pool import TokenPool
    from flaxchat.training import apply_gradients_if_finite, initialize_sharded, gradients_for_microbatches, pretraining_optimizer
    from scripts.validate_tpu import source_digest
    from jax.experimental import multihost_utils

    mesh = compute_init()
    if args.fsdp < 1 or jax.device_count() % args.fsdp:
        parser.error('--fsdp must divide the global device count')
    if args.fsdp > 1:
        mesh = setup_mesh(fsdp=args.fsdp)
    process_count, process_index = jax.process_count(), jax.process_index()
    if process_count > 1 and not args.ckpt_dir.startswith('gs://'):
        parser.error('Multi-host checkpoints require a shared gs:// directory')
    pool = TokenPool(args.token_manifest)
    if pool.vocab_size > args.max_vocab_size:
        parser.error('Token pool vocabulary exceeds --max-vocab-size')
    batch_size = args.global_batch_size or args.batch_per_device * jax.device_count()
    if batch_size < 1 or batch_size % jax.device_count():
        parser.error('Global batch must be positive and divisible by global device count')
    if args.accumulation_steps < 1 or batch_size % (args.accumulation_steps * jax.device_count()):
        parser.error("Global batch must divide accumulation steps times device count")
    tokens_per_step = batch_size * args.seq_len
    steps = parse_tokens(args.tokens) // tokens_per_step
    if steps < 1:
        parser.error(f'Token budget must cover at least {tokens_per_step} tokens')
    # Preflight every required range before compiling or updating weights.
    pool.local_batch('train', steps - 1, batch_size, args.seq_len, process_index, process_count)
    pool.local_batch('validation', args.eval_batches - 1, batch_size, args.seq_len, process_index, process_count)
    stop = steps if args.stop_after is None else min(steps, args.stop_after)
    if stop < 1:
        parser.error('--stop-after must be positive')
    warmup = min(args.warmup_steps, steps - 1)
    attention_backend, attention_reason = training_attention_backend('auto', jax.default_backend(), jax.device_count())
    config = GPTConfig(compute_dtype=(np.dtype(COMPUTE_DTYPE).name if args.compute_dtype == 'auto' else args.compute_dtype), sequence_len=args.seq_len, vocab_size=pool.vocab_size,
                       n_layer=args.depth, n_head=args.depth, n_kv_head=args.depth,
                       n_embd=args.depth * 64, window_pattern='L',
                       tie_embeddings=args.tie_embeddings, standard_gpt=True, attention_backend=attention_backend,
                       loss_chunk_size=args.loss_chunk_size, use_remat=args.remat)
    def initialize():
        model = GPT(config, rngs=nnx.Rngs(42))
        optimizer, _ = pretraining_optimizer(model, kind=args.optimizer, learning_rate=args.lr,
                                            warmup_steps=warmup, steps=steps)
        return model, optimizer
    model, optimizer = initialize_sharded(initialize, mesh, fsdp=args.fsdp)
    # The schedule is a pure function; no second model or optimizer allocation.
    import optax
    schedule = (optax.warmup_cosine_decay_schedule(0., args.lr, warmup, steps, end_value=args.lr * .05)
                if warmup else optax.cosine_decay_schedule(args.lr, steps, alpha=.05))
    resolved = {'model': asdict(config), 'global_batch_size': batch_size, 'steps': steps,
                'lr': args.lr, 'warmup_steps': warmup, 'seed': 42, 'optimizer': ('adamw-clip1-wd0.01' if args.optimizer == 'adamw' else 'normuon-adamw-default-groups-matrix-wd0.01'),
                'accumulation_steps': args.accumulation_steps}
    from flaxchat.runtime import runtime_identity
    resolved['runtime'] = runtime_identity()
    identity = {'resolved_config': resolved, 'data_manifest': pool.identity,
                'tokenizer': pool.manifest['identity'].get('tokenizer_identity', pool.manifest['identity'])}
    source_hash = source_digest(Path(__file__).resolve().parents[1])
    identity['source_python_sha256'] = source_hash
    agreement = hashlib.sha256(json.dumps({**identity, 'source': source_hash}, sort_keys=True).encode()).digest()
    multihost_utils.assert_equal(np.frombuffer(agreement, dtype=np.uint8),
                                'Workers must use identical code, data, tokenizer and schedule')
    output = Path(args.artifact_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from etils import epath
    checkpoint_path = epath.Path(args.ckpt_dir) if args.ckpt_dir.startswith('gs://') else epath.Path(args.ckpt_dir).resolve()
    def place_state(state):
        if args.fsdp == 1:
            return replicate_on_mesh(state, mesh)
        def place(leaf):
            spec = P('fsdp') if np.ndim(leaf) >= 2 and leaf.shape[0] % args.fsdp == 0 else P()
            return place_array(leaf, NamedSharding(mesh, spec))
        return jax.tree.map(place, state)
    nnx.update(model, place_state(nnx.state(model)))
    nnx.update(optimizer, place_state(nnx.state(optimizer)))
    start = 0
    if args.resume:
        _, state = restore_model_from_checkpoint(model, str(checkpoint_path), optimizer=optimizer,
                                                  expected_identity=identity, load_training_state=True)
        start = int(state['next_batch'])
        if int(state['update_step']) != start or start > stop:
            raise ValueError('Invalid checkpoint cursor or stop precedes checkpoint')
        optimizer.step[...] = start
    elif checkpoint_path.exists() and any(checkpoint_path.iterdir()):
        raise ValueError('Checkpoint directory is not empty; use --resume or a new directory')
    nnx.update(model, place_state(nnx.state(model)))
    nnx.update(optimizer, place_state(nnx.state(optimizer)))
    sharding = NamedSharding(mesh, P(('data', 'fsdp')))

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(current):
            return current(x, y)
        if args.accumulation_steps > 1:
            shape = (args.accumulation_steps, batch_size // args.accumulation_steps, args.seq_len)
            layout = NamedSharding(mesh, P(None, ('data', 'fsdp')))
            x = jax.lax.with_sharding_constraint(x.reshape(shape), layout)
            y = jax.lax.with_sharding_constraint(y.reshape(shape), layout)
            loss, grads = gradients_for_microbatches(model, x, y)
        else:
            loss, grads = nnx.value_and_grad(loss_fn)(model)
        return loss, apply_gradients_if_finite(model, optimizer, grads, loss)

    @nnx.jit
    def eval_step(model, x, y):
        return model(x, y)

    def batch(split, step, prefetched=None):
        def read(index, side):
            rows = index[0]
            start = 0 if rows.start is None else rows.start
            stop = batch_size if rows.stop is None else rows.stop
            pair = prefetched[(start, stop)] if prefetched is not None else pool.batch_rows(split, step, batch_size, args.seq_len, start, stop)
            return pair[side]
        return tuple(jax.make_array_from_callback((batch_size, args.seq_len), sharding,
                     lambda index, side=side: read(index, side)) for side in (0, 1))

    def evaluate():
        loss = float(np.mean([float(eval_step(model, *batch('validation', i))) for i in range(args.eval_batches)]))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        return loss

    # Verify the actual global array mapping, not just arithmetic host offsets.
    # Bounded to one batch and skipped on a single host.
    if process_count > 1:
        actual = multihost_utils.process_allgather(batch('train', min(start, steps - 1))[0], tiled=True)
        expected = pool.batch('train', min(start, steps - 1), batch_size, args.seq_len)[0]
        np.testing.assert_array_equal(np.asarray(actual), expected)

    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    metadata = {'resolved_config': resolved, 'model_config': asdict(config), 'tokenizer_identity': identity['tokenizer'],
                'data_manifest_identity': pool.identity, 'source_revision': revision,
                'source_python_sha256': source_hash}
    manager = create_checkpoint_manager(str(checkpoint_path), async_checkpointing=False)
    from flaxchat.prefetch import BackgroundPrefetcher
    local_ranges = sorted({(index[0].start or 0, index[0].stop or batch_size)
                          for index in sharding.addressable_devices_indices_map((batch_size, args.seq_len)).values()})
    def host_batches():
        for step in range(start, stop):
            yield step, {bounds: pool.batch_rows('train', step, batch_size, args.seq_len, *bounds)
                         for bounds in local_ranges}
    host_source = host_batches()
    prefetcher = BackgroundPrefetcher(lambda: next(host_source), None, None, place=False)
    metrics = []
    started = time.monotonic()
    last_saved = start if args.resume else None
    completed = start
    last_checkpoint_time = time.monotonic()
    try:
        initial_validation = evaluate()
        for step in range(start, stop):
            tick = time.monotonic()
            batch_step, host_rows = next(prefetcher)
            if batch_step != step:
                raise ValueError('Prefetch cursor differs from consumed update')
            loss, updated = train_step(model, optimizer, *batch('train', step, host_rows))
            loss = float(loss)
            if not bool(updated):
                raise FloatingPointError(f'Nonfinite loss/gradients at update {step}; state not advanced')
            completed = step + 1
            row = {'step': completed, 'loss': loss, 'tokens': completed * tokens_per_step,
                   'step_seconds': time.monotonic() - tick, 'learning_rate': float(schedule(step))}
            if completed % args.eval_every == 0 or completed == stop:
                row['validation_loss'] = evaluate()
            metrics.append(row)
            with (output / f'metrics-process-{process_index}.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            due = checkpoint_due(completed, stop, args.save_every,
                                 time.monotonic() - last_checkpoint_time, args.checkpoint_interval_seconds)
            if process_count > 1 and args.save_every is None:
                due = bool(multihost_utils.broadcast_one_to_all(np.asarray(due), is_source=process_index == 0))
            if due:
                save_started = time.monotonic()
                save_checkpoint(manager, completed, model, optimizer, {**metadata, 'step': completed},
                                training_state=replicate_on_mesh({
                                    'next_batch': np.asarray(completed, dtype=np.int32),
                                    'update_step': np.asarray(completed, dtype=np.int32)}, mesh))
                manager.wait_until_finished()
                last_saved = completed
                row['checkpoint_seconds'] = time.monotonic() - save_started
                last_checkpoint_time = time.monotonic()
                event = {'event': 'checkpoint_committed', 'step': completed,
                         'checkpoint_seconds': row['checkpoint_seconds']}
                with (output / f'metrics-process-{process_index}.jsonl').open('a') as handle:
                    handle.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
        summary = {**metadata, 'backend': jax.default_backend(), 'devices': jax.device_count(),
                   'process_count': jax.process_count(), 'process_index': process_index,
                   'local_device_count': jax.local_device_count(), 'mesh': dict(mesh.shape),
                   'attention_backend': attention_backend,
                   'attention_fallback_reason': attention_reason, 'start_step': start, 'completed_steps': completed,
                   'checkpoint_step': last_saved, 'tokens': completed * tokens_per_step,
                   'initial_validation_loss': initial_validation, 'final_validation_loss': evaluate(),
                   'elapsed_seconds': time.monotonic() - started, 'metrics': metrics}
        summary['invocation_seconds'] = time.monotonic() - invocation_started
        summary['checkpoint_policy'] = {'save_every': args.save_every, 'interval_seconds': args.checkpoint_interval_seconds}
        summary['new_committed_tokens'] = (completed - start) * tokens_per_step
        summary['end_to_end_tokens_per_second'] = summary['new_committed_tokens'] / summary['invocation_seconds']
        summary['device_memory_stats'] = [device.memory_stats() for device in jax.local_devices()]
        summary['model_shardings'] = sorted({str(leaf.sharding) for leaf in jax.tree.leaves(nnx.state(model))})
        summary['local_batch_row_ranges'] = sorted({
            (index[0].start or 0, index[0].stop or batch_size)
            for index in sharding.addressable_devices_indices_map((batch_size, args.seq_len)).values()})
        summary['first_step_compile_and_execute_seconds'] = metrics[0]['step_seconds'] if metrics else None
        summary['steady_tokens_per_second'] = (
            tokens_per_step * (len(metrics) - 1) / sum(row['step_seconds'] for row in metrics[1:])
            if len(metrics) > 1 else None)
        (output / f'training_summary-process-{process_index}.json').write_text(json.dumps(summary, indent=2) + '\n')
        if process_index == 0:
            (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        prefetcher.stop()
        manager.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
