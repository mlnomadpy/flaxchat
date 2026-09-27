"""Train a ModernBERT MLM on prepared, unshifted document rows.

Run with python -m scripts.train_encoder --help. Data parallel training uses
one global JAX mesh; parameters are replicated by default. Experimental FSDP
shards divisible matrix parameters and optimizer state along their first axis. Use the existing external TPU budget/deadline supervisor on GCP.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from flaxchat.checkpoint import create_checkpoint_manager, load_checkpoint_metadata, restore_model_from_checkpoint, save_checkpoint
from flaxchat.common import replicate_on_mesh
from flaxchat.encoder import EncoderConfig, ModernBert, import_hf_weights
from flaxchat.mlm import mask_tokens
from flaxchat.profiling import HostGCTiming, TrainingTrace
from flaxchat.encoder_data import MixtureRows, CoverageMixtureRows, file_hash, load_prepared_rows, EpochRows, LanguageRows
from flaxchat.training import initialize_sharded, apply_gradients_if_finite, place_host_batch, gather_process_metadata, gradients_for_mlm_microbatches, gradients_for_local_mlm_microbatches


def pretrained_shapes(config):
    """HF tensor header shapes, without constructing a model or device arrays."""
    h, i = config.hidden_size, config.intermediate_size
    shapes = {'model.embeddings.tok_embeddings.weight': (config.vocab_size, h),
              'model.embeddings.norm.weight': (h,), 'model.final_norm.weight': (h,),
              'head.dense.weight': (h, h), 'head.norm.weight': (h,),
              'decoder.bias': (config.vocab_size,)}
    for layer in range(config.num_hidden_layers):
        prefix = f'model.layers.{layer}.'
        shapes.update({prefix + key: shape for key, shape in {
            'attn.Wqkv.weight': (3 * h, h), 'attn.Wo.weight': (h, h),
            'mlp.Wi.weight': (2 * i, h), 'mlp.Wo.weight': (h, i),
            'mlp_norm.weight': (h,)}.items()})
        if layer:
            shapes[prefix + 'attn_norm.weight'] = (h,)
    return shapes


def pretrained_inventory(directory, config=None):
    """Inspect and hash local weight shards without allocating a JAX model."""
    from safetensors import safe_open
    directory = Path(directory).resolve()
    index = directory / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map'] if index.exists() else None
    if weight_map is not None and (not isinstance(weight_map, dict) or not weight_map):
        raise ValueError('Checkpoint index must contain a nonempty weight map')
    names = sorted(set(weight_map.values())) if weight_map is not None else ['model.safetensors']
    hashes, tensors, shapes = {}, {}, {}
    for name in names:
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError('Checkpoint shard must exist inside snapshot directory')
        hashes[name] = file_hash(path)
        with safe_open(str(path), framework='numpy') as shard:
            for key in shard.keys():
                if key in tensors:
                    raise ValueError('Duplicate checkpoint tensors')
                tensors[key] = name
                view = shard.get_slice(key)
                shapes[key] = tuple(view.get_shape())
    if not tensors or (weight_map is not None and tensors != weight_map):
        raise ValueError('Checkpoint tensors do not match shard index')
    if config is not None:
        expected = pretrained_shapes(config)
        extra = set(shapes) - set(expected) - {'decoder.weight'}
        if extra or set(expected) - set(shapes):
            raise ValueError('Pretrained tensor names do not match architecture')
        for key, shape in expected.items():
            if shapes[key] != shape:
                raise ValueError(f'Pretrained tensor shape mismatch: {key}')
        if 'decoder.weight' in shapes and shapes['decoder.weight'] != expected['model.embeddings.tok_embeddings.weight']:
            raise ValueError('Tied decoder shape mismatch')
        # Validate all headers before materializing any tensor content. A bad
        # dimension must not turn a bounded row scan into a huge allocation.
        for name in names:
            with safe_open(str(directory / name), framework='numpy') as shard:
                for key in shard.keys():
                    view = shard.get_slice(key)
                    if view.get_dtype() not in ('F16', 'BF16', 'F32', 'F64'):
                        raise ValueError(f'Nonfloating pretrained tensor: {key}')
                    for start in range(0, shapes[key][0], 1024):
                        values = view[start:min(start + 1024, shapes[key][0])]
                        if not np.isfinite(values).all():
                            raise ValueError(f'Nonfinite pretrained tensor: {key}')
        if 'decoder.weight' in tensors:
            embedding_key = 'model.embeddings.tok_embeddings.weight'
            with safe_open(str(directory / tensors[embedding_key]), framework='numpy') as embed, safe_open(
                    str(directory / tensors['decoder.weight']), framework='numpy') as decoder:
                left, right = embed.get_slice(embedding_key), decoder.get_slice('decoder.weight')
                for start in range(0, config.vocab_size, 1024):
                    if not np.array_equal(left[start:min(start + 1024, config.vocab_size)], right[start:min(start + 1024, config.vocab_size)]):
                        raise ValueError('Decoder weights must be tied to embeddings')
    return hashes


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


def validate_resume_cursor(state, *, horizon, stop):
    values = []
    for key in ('completed_steps', 'optimizer_updates'):
        value = np.asarray(state.get(key))
        if value.shape != (1,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Invalid encoder checkpoint cursor')
        values.append(int(value[0]))
    completed, updates = values
    if not 0 <= updates <= completed <= horizon:
        raise ValueError('Checkpoint cursor or optimizer updates outside training horizon')
    if stop < completed:
        raise ValueError('Requested stop precedes restored checkpoint cursor')
    return completed, updates


def learning_rate_schedule(args):
    if not 0 <= args.warmup_steps < args.steps or not 0 <= args.final_lr_ratio <= 1:
        raise ValueError('Invalid warmup or final learning-rate ratio')
    if args.lr_schedule == 'constant':
        if args.warmup_steps:
            raise ValueError('Warmup requires cosine schedule')
        return lambda step: jnp.asarray(args.learning_rate)
    if args.warmup_steps:
        return optax.warmup_cosine_decay_schedule(0., args.learning_rate,
            args.warmup_steps, args.steps, end_value=args.learning_rate * args.final_lr_ratio)
    return optax.cosine_decay_schedule(args.learning_rate, args.steps, alpha=args.final_lr_ratio)


def initialization_metadata(directory, step, config, tokenizer_identity):
    """Pin a new stage's parent; exact-run resume remains a separate operation."""
    parent = load_checkpoint_metadata(directory, step)
    parent_encoder = parent.get('resolved_config', {}).get('encoder')
    if (parent.get('model_family') != 'modernbert'
            or not isinstance(parent_encoder, dict) or not parent_encoder):
        raise ValueError('Initialization checkpoint encoder configuration differs')
    parent_config = EncoderConfig(**parent_encoder)
    execution_fields = ('yat_local_shards', 'yat_attention_block_size', 'yat_global_attention_block_size')
    overrides = {name: getattr(config, name) for name in execution_fields}
    if replace(parent_config, **overrides) != config:
        raise ValueError('Initialization checkpoint encoder configuration differs')
    if parent.get('tokenizer_identity') != tokenizer_identity:
        raise ValueError('Initialization checkpoint tokenizer differs')
    receipt = dict(checkpoint=str(directory), step=parent['step'],
                   metadata_sha256=hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest(),
                   policy='model_weights_only; fresh_optimizer_schedule_and_data_cursor',
                   execution_overrides={name: dict(parent=getattr(parent_config, name), current=value)
                                        for name, value in overrides.items() if getattr(parent_config, name) != value})
    return parent, receipt


def run(args):
    jit_partial_mode = getattr(args, 'nnx_jit_partial', False)
    fsdp = getattr(args, 'fsdp', 1)
    manifest_chunk_mib = getattr(args, 'checkpoint_manifest_chunk_mib', None)
    if manifest_chunk_mib is not None and (type(manifest_chunk_mib) is not int or not 1 <= manifest_chunk_mib <= 64):
        raise ValueError('Checkpoint manifest chunk size must be 1–64 MiB')
    trace = TrainingTrace(args.profile_dir, skip=args.profile_skip_steps, steps=args.profile_steps)
    if args.distributed and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if manifest_chunk_mib is None:
        # Full-model save/resume qualified on one physical TPU device. Keep
        # the prior transfer bound elsewhere until those topologies qualify.
        manifest_chunk_mib = 16 if jax.default_backend() == 'tpu' and jax.device_count() == 1 else 1
    if type(fsdp) is not int or fsdp < 1 or jax.device_count() % fsdp:
        raise ValueError('fsdp must be a positive divisor of the global device count')
    if fsdp > 1 and (args.local_gradient_accumulation or args.yat_local_shards or jit_partial_mode):
        raise ValueError('Experimental FSDP requires global accumulation, global YAT kernels, and ordinary NNX JIT')
    if args.preflight_only and args.resume:
        raise ValueError('Preflight validates local inputs, not checkpoint resume')
    if args.initialize_step is not None and (not args.initialize_from_checkpoint or args.initialize_step < 1):
        raise ValueError('initialize-step requires an initialization checkpoint and a positive step')
    if args.initialize_from_checkpoint and args.resume:
        raise ValueError('Choose checkpoint initialization for a new stage or exact resume')
    if args.initialize_from_checkpoint and (
            args.initialize_from_checkpoint.rstrip('/') == args.output.rstrip('/')
            or (not args.output.startswith('gs://') and not args.initialize_from_checkpoint.startswith('gs://')
                and Path(args.initialize_from_checkpoint).resolve() == Path(args.output).resolve())):
        raise ValueError('A new stage requires a different output directory from its parent')
    if args.steps <= 0 or args.batch_size <= 0 or args.save_every <= 0 or args.seed < 0:
        raise ValueError('steps, batch-size, save-every must be positive; seed nonnegative')
    if args.keep_checkpoints < 0:
        raise ValueError('keep-checkpoints must be nonnegative; zero retains all')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 < args.mask_probability <= 1:
        raise ValueError('Invalid learning rate or mask probability')
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        raise ValueError('stop-after must be within training horizon')
    schedule = learning_rate_schedule(args)
    if args.accumulation_steps < 1 or args.batch_size % (args.accumulation_steps * jax.device_count()):
        raise ValueError('Global batch must be divisible by accumulation steps times global device count')
    if jax.process_count() > 1 and not args.output.startswith('gs://') and not args.shared_local_checkpoints:
        raise ValueError('Multi-host checkpoints require gs:// or an explicitly shared filesystem')
    raw = json.loads(Path(args.config).read_text())
    overrides = dict(compute_dtype=args.dtype, residual_dtype=args.residual_dtype, attention_backend=args.attention_backend,
                     yat_local_shards=args.yat_local_shards, yat_attention_block_size=args.yat_attention_block_size, yat_global_attention_block_size=args.yat_global_attention_block_size,
                     loss_chunk_size=args.loss_chunk_size, use_remat=not args.no_remat,
                     mlm_projection=args.mlm_projection, mlm_loss_backend=args.mlm_loss_backend,
                     mlm_vocab_tile=args.mlm_vocab_tile)
    overrides.update({name: getattr(args, name) for name in ('ffn_type', 'yat_epsilon', 'yat_alpha', 'yat_attention_alpha', 'attention_score', 'yat_compute_mode', 'yat_ffn_compute_mode', 'yat_softmax_backward', 'yat_attention_implementation', 'yat_ffn_backward_block', 'mlm_projection_capacity')
                      if getattr(args, name) is not None})
    config = EncoderConfig.from_hf(raw, **overrides) if raw.get('model_type') else EncoderConfig(**(raw | overrides))
    if config.attention_backend == 'splash' and jax.device_count() > 1:
        raise ValueError('Multi-device encoder training currently requires --attention-backend xla')
    data_dir = Path(args.data)
    tokens, manifest = load_prepared_rows(data_dir, config, minimum_rows=args.batch_size)
    if args.language_exponent is not None and args.shuffle:
        raise ValueError('Choose language sampling or epoch shuffle, not both')
    if args.coverage_sampling and (manifest.get('source_token_shares') is None or args.language_exponent is None):
        raise ValueError('Coverage sampling requires a source mixture and language exponent')
    if manifest.get('source_token_shares') is not None and args.language_exponent is None:
        raise ValueError('A source mixture requires explicit within-source language sampling')
    sampler = (CoverageMixtureRows if args.coverage_sampling else MixtureRows) if manifest.get('source_token_shares') is not None else LanguageRows
    rows = (sampler(data_dir, tokens, manifest, args.seed, args.language_exponent)
            if args.language_exponent is not None else EpochRows(len(tokens), args.seed, shuffle=args.shuffle))
    if args.pretrained:
        tokenizer_path = Path(args.pretrained) / 'tokenizer.json'
        if not tokenizer_path.exists() or file_hash(tokenizer_path) != manifest['tokenizer_sha256']:
            raise ValueError('Prepared tokenizer must match pretrained snapshot tokenizer.json')
        pretrained_config = EncoderConfig.from_hf(json.loads((Path(args.pretrained) / 'config.json').read_text()), **overrides)
        if pretrained_config != config:
            raise ValueError('Pretrained architecture does not match requested configuration')
    parent, initialization = (initialization_metadata(args.initialize_from_checkpoint, args.initialize_step,
                                config, manifest['tokenizer_sha256'])
                              if args.initialize_from_checkpoint else (None, None))
    initial_weights = pretrained_inventory(args.pretrained, config) if args.pretrained and not args.resume and parent is None else None
    from flaxchat.runtime import runtime_identity
    identity = dict(runtime=runtime_identity(), encoder=asdict(config), steps=args.steps, batch_size=args.batch_size,
                    seed=args.seed, learning_rate=args.learning_rate, mask_probability=args.mask_probability,
                    lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps, final_lr_ratio=args.final_lr_ratio,
                    accumulation_steps=args.accumulation_steps, shuffle=args.shuffle,
                    language_exponent=args.language_exponent,
                    special_token_ids=manifest['special_token_ids'])
    if args.coverage_sampling:
        identity['coverage_sampling'] = 'pool_epoch_v1'
    if fsdp > 1:
        identity['fsdp'] = fsdp
    if args.local_gradient_accumulation:
        identity['local_gradient_accumulation'] = True
    if jit_partial_mode:
        identity['nnx_jit_partial'] = True
    source_digest = hashlib.sha256(''.join(file_hash(path) for path in (
        Path(__file__), Path(__file__).parents[1] / 'flaxchat/encoder.py',
        Path(__file__).parents[1] / 'flaxchat/mlm.py',
        Path(__file__).parents[1] / 'flaxchat/yat.py',
        Path(__file__).parents[1] / 'flaxchat/yat_attention.py',
        Path(__file__).parents[1] / 'flaxchat/yat_attention_centered.py',
        Path(__file__).parents[1] / 'flaxchat/profiling.py',
        Path(__file__).parents[1] / 'flaxchat/encoder_data.py',
        Path(__file__).parents[1] / 'flaxchat/training.py',
        Path(__file__).parents[1] / 'flaxchat/common.py',
        Path(__file__).parents[1] / 'flaxchat/checkpoint.py',
        Path(__file__).parents[1] / 'flaxchat/fused_cross_entropy.py')).encode()).hexdigest()
    metadata = dict(source_python_sha256=source_digest, resolved_config=identity, tokenizer_identity=manifest['tokenizer_sha256'],
                    data_manifest_identity=file_hash(data_dir / 'manifest.json'), model_family='modernbert')
    expected = dict(resolved_config=identity, tokenizer=metadata['tokenizer_identity'],
                    data_manifest=metadata['data_manifest_identity'], source_python_sha256=source_digest)
    identities = gather_process_metadata(expected | {'initial_weights_sha256': initial_weights,
                                                     'initialization': initialization})
    if any(item != identities[0] for item in identities):
        raise ValueError('Workers disagree on data, tokenizer, source, weights, runtime, or training configuration')
    if args.preflight_only:
        print(json.dumps(dict(event='input_preflight', passed=True,
                              config=identity, input_identity=identities[0],
                              rows=len(tokens), sequence_length=tokens.shape[1])), flush=True)
        return
    mesh = (Mesh(np.asarray(jax.devices()).reshape(-1, fsdp), ('data', 'fsdp'))
            if fsdp > 1 else Mesh(np.asarray(jax.devices()), ('data',)))
    data_axes = ('data', 'fsdp') if fsdp > 1 else 'data'

    def make_optimizer(current):
        return nnx.Optimizer(current, optax.chain(optax.clip_by_global_norm(1.),
            optax.adamw(schedule if args.lr_schedule != 'constant' else args.learning_rate,
                        weight_decay=.01)), wrt=nnx.Param)

    def initialize():
        current = ModernBert(config, rngs=nnx.Rngs(args.seed))
        return current, make_optimizer(current)

    if fsdp > 1:
        model, optimizer = initialize_sharded(initialize, mesh, fsdp=fsdp)
    else:
        model = ModernBert(config, rngs=nnx.Rngs(args.seed))
    if args.pretrained and not args.resume and parent is None:
        metadata['initial_weights_sha256'] = load_pretrained(model, args.pretrained)
        if metadata['initial_weights_sha256'] != initial_weights:
            raise ValueError('Pretrained weights changed after preflight')
    if fsdp == 1:
        nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    if parent is not None:
        if initialization is None:
            raise ValueError('Missing initialization receipt')
        # Pin the selected step even when a writer adds a newer parent checkpoint.
        _, checked = initialization_metadata(args.initialize_from_checkpoint, parent['step'],
                                              config, manifest['tokenizer_sha256'])
        if checked != initialization:
            raise ValueError('Initialization checkpoint metadata changed after preflight')
        restored_parent = restore_model_from_checkpoint(model, args.initialize_from_checkpoint,
            step=parent['step'], expected_identity=dict(resolved_config=parent['resolved_config'],
                                                       tokenizer=parent['tokenizer_identity']))
        if not isinstance(restored_parent, dict) or dict(restored_parent, step=parent['step']) != parent:
            raise ValueError('Initialization checkpoint metadata changed during restore')
        metadata['initialization'] = initialization
    if fsdp == 1:
        optimizer = make_optimizer(model)
    start_step = 0
    if args.resume:
        metadata, state = restore_model_from_checkpoint(model, args.output, optimizer=optimizer,
                                                expected_identity=expected, load_training_state=True)
        if state is None:
            raise ValueError('Checkpoint has no training cursor')
        start_step, updates = validate_resume_cursor(state, horizon=args.steps, stop=args.stop_after or args.steps)
        optimizer.step[...] = jnp.asarray(updates, dtype=optimizer.step.dtype)
        if start_step == (args.stop_after or args.steps):
            if jax.process_index() == 0:
                print(json.dumps(dict(event='already_completed', step=start_step, optimizer_updates=updates)), flush=True)
            return
    if isinstance(rows, CoverageMixtureRows):
        rows.seek(start_step, args.batch_size)
    if fsdp == 1:
        nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    def verify_fsdp_layout(phase):
        if fsdp == 1:
            return
        leaves = jax.tree.leaves(nnx.state((model, optimizer)))
        sharded = 0
        for leaf in leaves:
            spec = P('fsdp') if leaf.ndim >= 2 and leaf.shape[0] % fsdp == 0 else P()
            if not leaf.sharding.is_equivalent_to(NamedSharding(mesh, spec), leaf.ndim):
                raise RuntimeError(f'FSDP state layout changed during {phase}: {leaf.shape}')
            sharded += spec == P('fsdp')
        if jax.process_index() == 0:
            print(json.dumps(dict(event='fsdp_layout', phase=phase, leaves=len(leaves),
                                  sharded_leaves=sharded, factor=fsdp)), flush=True)

    verify_fsdp_layout('initialization_or_restore')
    manager = create_checkpoint_manager(args.output, max_to_keep=args.keep_checkpoints or None,
                                        async_checkpointing=False)
    if not args.resume and manager.latest_step() is not None:
        manager.close()
        raise ValueError('Output already contains checkpoints; use --resume or a new directory')

    def train_step(model, optimizer, x, y):
        if args.accumulation_steps > 1 or args.local_gradient_accumulation:
            shape = (args.accumulation_steps, args.batch_size // args.accumulation_steps, x.shape[1])
            layout = NamedSharding(mesh, P(None, data_axes))
            inputs = jax.lax.with_sharding_constraint(x.reshape(shape), layout)
            targets = jax.lax.with_sharding_constraint(y.reshape(shape), layout)
            if args.local_gradient_accumulation:
                loss, grads = gradients_for_local_mlm_microbatches(model, inputs, targets, mesh)
            else:
                loss, grads = gradients_for_mlm_microbatches(model, inputs, targets)
        else:
            loss, grads = nnx.value_and_grad(lambda m: m(x, y))(model)
        # Observe the same global norm used by the optimizer's unit clipping.
        # A finite loss alone can hide exploding gradients and tiny updates.
        gradient_norm = optax.tree.norm(grads)
        gradient_clip_scale = jnp.minimum(1., 1. / jnp.maximum(gradient_norm, 1.))
        selected = (y >= 0).sum()
        capacity = config.projection_capacity(y.shape[1])
        overflow = (config.mlm_projection == 'masked') & jnp.any((y >= 0).sum(axis=1) > capacity)
        # Empty-mask batches must not cause an AdamW weight-decay-only update.
        updated = nnx.cond(selected > 0,
                           lambda m, o, g, loss_value: apply_gradients_if_finite(m, o, g, loss_value),
                           lambda m, o, g, loss_value: jnp.array(False), model, optimizer, grads, loss)
        return (loss, selected, updated, overflow, (x != config.pad_token_id).sum(),
                gradient_norm, gradient_clip_scale)

    if jit_partial_mode:
        # Bind after restore/placement. The partial wrapper's argument zero is
        # the single flattened model+optimizer list, not the original model.
        train_step = nnx.jit_partial(train_step, model, optimizer,
            graph=True, graph_updates=False,
            donate_argnums=(0,) if args.donate_state else ())
    else:
        if fsdp > 1:
            # NNX also applies input StateSharding to mutated state outputs.
            # Preserve parameter and optimizer layouts across every update.
            def state_layout(node):
                return nnx.StateSharding(
                    jax.tree.map(lambda leaf: leaf.sharding, nnx.state(node)))

            batch_layout = NamedSharding(mesh, P(data_axes))
            train_step = nnx.jit(
                train_step,
                in_shardings=(state_layout(model), state_layout(optimizer), batch_layout, batch_layout),
                out_shardings=NamedSharding(mesh, P()),
                donate_argnums=(0, 1) if args.donate_state else (),
            )
        else:
            train_step = nnx.jit(train_step,
                donate_argnums=(0, 1) if args.donate_state else ())

    if jax.process_index() == 0:
        print(json.dumps(dict(event='run_config', backend=jax.default_backend(),
            devices=jax.device_count(), processes=jax.process_count(),
            input_identity=identities[0],
            compute_dtype=config.compute_dtype, residual_dtype=config.residual_dtype,
            mlm_projection=config.mlm_projection,
            mlm_loss_backend=config.mlm_loss_backend, mlm_vocab_tile=config.mlm_vocab_tile,
            parameter_dtype='float32', optimizer_dtype='float32', donate_state=args.donate_state,
            nnx_jit_partial=jit_partial_mode, fsdp=fsdp, mesh_shape=dict(mesh.shape),
            checkpoint_manifest_chunk_mib=manifest_chunk_mib,
            matmul_precision=str(jax.default_matmul_precision.value))), flush=True)
    end = args.stop_after or args.steps
    local_batch = args.batch_size // jax.process_count()
    if isinstance(rows, LanguageRows) and jax.process_index() == 0:
        pool_names = [":".join(pool) for pool in rows.pools] if isinstance(rows, MixtureRows) else rows.languages
        print(json.dumps(dict(event="mixture_sampling" if isinstance(rows, MixtureRows) else "language_sampling", policy="pool_epoch_without_replacement" if isinstance(rows, CoverageMixtureRows) else "with_replacement",
            target_token_shares=dict(zip(pool_names, rows.target_token_shares.tolist(), strict=True)),
            expected_row_exposures={name: args.steps * args.batch_size * float(p) / len(ids)
                for name, p, ids in zip(pool_names, rows.row_probabilities, rows.indices, strict=True)})), flush=True)
    step_executable = train_step
    gc_timing = HostGCTiming(getattr(args, 'host_gc_diagnostics', False))
    try:
        for step in range(start_step, end):
            gc_before = gc_timing.snapshot() if gc_timing.enabled else None
            started = time.monotonic()
            if step == start_step and args.compile_diagnostics:
                print(json.dumps(dict(event='training_phase', phase='preparing_batch', process=jax.process_index())), flush=True)
            global_ids = rows.batch(step, args.batch_size)
            local_ids = global_ids[jax.process_index() * local_batch:(jax.process_index() + 1) * local_batch]
            x, y = mask_tokens(tokens[local_ids], seed=args.seed, step=step, example_ids=local_ids,
                vocab_size=config.vocab_size, mask_token_id=config.mask_token_id,
                special_token_ids=manifest['special_token_ids'], probability=args.mask_probability)
            learning_rate = float(np.asarray(schedule(int(optimizer.step[...]))))
            if step == start_step and args.compile_diagnostics:
                print(json.dumps(dict(event='training_phase', phase='placing_batch', process=jax.process_index())), flush=True)
            x_device, y_device = place_host_batch(x, mesh, data_axes=data_axes), place_host_batch(y, mesh, data_axes=data_axes)
            step_args = (x_device, y_device) if jit_partial_mode else (model, optimizer, x_device, y_device)
            batch_prepared_at = time.monotonic()
            if step == start_step and args.compile_diagnostics:
                tick = time.monotonic()
                print(json.dumps(dict(event='training_phase', phase='lowering_started', process=jax.process_index())), flush=True)
                lowered = train_step.lower(*step_args)
                print(json.dumps(dict(event='training_phase', phase='lowering_finished', seconds=time.monotonic()-tick, process=jax.process_index())), flush=True)
                tick = time.monotonic()
                step_executable = lowered.compile()
                print(json.dumps(dict(event='training_phase', phase='compilation_finished', seconds=time.monotonic()-tick, process=jax.process_index())), flush=True)
                try:
                    memory_stats = step_executable.memory_analysis()
                    memory = str(memory_stats)
                    memory_bytes = {name: getattr(memory_stats, name, None) for name in (
                        'argument_size_in_bytes', 'output_size_in_bytes',
                        'temp_size_in_bytes', 'alias_size_in_bytes',
                        'generated_code_size_in_bytes', 'host_argument_size_in_bytes',
                        'host_output_size_in_bytes', 'host_temp_size_in_bytes',
                        'host_alias_size_in_bytes', 'host_generated_code_size_in_bytes')}
                except NotImplementedError:
                    memory = 'unavailable'
                    memory_bytes = None
                print(json.dumps(dict(event='compiled_memory', process=jax.process_index(), statistics=memory,
                                      bytes=memory_bytes,
                                      scope='Compiler buffer statistics, not measured device peak or total VM memory')), flush=True)
                print(json.dumps(dict(event='training_phase', phase='execution_started', epoch=time.time(), process=jax.process_index())), flush=True)
            execution_started_at = time.monotonic()
            with trace.step(step - start_step):
                (loss, selected, updated, overflow, nonpadding,
                 gradient_norm, gradient_clip_scale) = step_executable(*step_args)
                # Synchronize inside the trace so device execution is captured.
                loss_value, selected_value, updated_value = float(loss), int(selected), bool(updated)
            if selected_value and not updated_value:
                raise FloatingPointError('Rejected nonfinite encoder optimizer update')
            execution_finished_at = time.monotonic()
            gc_after = gc_timing.snapshot() if gc_timing.enabled else None
            elapsed = execution_finished_at - started
            if jax.process_index() == 0:
                print(json.dumps(dict(event='train_step', step=step + 1, loss=loss_value, masked_tokens=selected_value,
                                      updated=updated_value, tokens=args.batch_size * tokens.shape[1],
                                      nonpadding_tokens=int(nonpadding), nonpadding_tokens_per_second=int(nonpadding) / elapsed,
                                      learning_rate=learning_rate,
                                      gradient_norm_before_clip=float(gradient_norm),
                                      gradient_clip_scale=float(gradient_clip_scale),
                                      includes_profiling=bool(args.profile_dir and args.profile_skip_steps <= step - start_step < args.profile_skip_steps + args.profile_steps),
                                      language_nonpadding_tokens=rows.token_counts(global_ids) if isinstance(rows, LanguageRows) else None,
                                      source_nonpadding_tokens=rows.source_token_counts(global_ids) if isinstance(rows, MixtureRows) else None,
                                      projection_dense_fallback=bool(overflow),
                                      batch_preparation_seconds=batch_prepared_at - started,
                                      compilation_diagnostics_seconds=execution_started_at - batch_prepared_at,
                                      synchronized_execution_and_profile_seconds=execution_finished_at - execution_started_at,
                                      timing_scope='Host wall time; batch placement may be asynchronous and complete during synchronized execution; profiling start/export is included when enabled',
                                      host_gc_seconds=(gc_after[0] - gc_before[0]) if gc_after is not None and gc_before is not None else None,
                                      host_gc_collections=(gc_after[1] - gc_before[1]) if gc_after is not None and gc_before is not None else None,
                                      seconds=elapsed, includes_compilation=step == start_step,
                                      tokens_per_second=args.batch_size * tokens.shape[1] / elapsed)), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == end:
                if isinstance(rows, CoverageMixtureRows) and jax.process_index() == 0:
                    print(json.dumps(dict(event='coverage_sampling', step=step + 1,
                                          pools=rows.exposure_summary())), flush=True)
                verify_fsdp_layout(f'before_checkpoint_{step + 1}')
                checkpoint_started = time.monotonic()
                save_checkpoint(manager, step + 1, model, optimizer, metadata,
                                manifest_chunk_bytes=manifest_chunk_mib * 1024 * 1024,
                                training_state={'completed_steps': np.array([step + 1], dtype=np.int32),
                                                'optimizer_updates': np.array([int(optimizer.step[...])], dtype=np.int32)})
                manager.wait_until_finished()
                if jax.process_index() == 0:
                    print(json.dumps(dict(event='checkpoint', step=step + 1,
                        manifest_chunk_mib=manifest_chunk_mib,
                        seconds=time.monotonic() - checkpoint_started)), flush=True)
    finally:
        gc_timing.close()
        try:
            trace.close()
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
    p.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--final-lr-ratio', type=float, default=.05)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--local-gradient-accumulation', action='store_true',
                   help='Experimental: accumulate on each device before reducing gradients once per update')
    p.add_argument('--shuffle', action='store_true', help='Deterministic epoch permutations, replayable from cursor')
    p.add_argument('--language-exponent', type=float, help='Opt-in language token-temperature sampling in [0,1]; requires corpus document provenance')
    p.add_argument('--coverage-sampling', action='store_true',
                   help='With a source mixture, sample each source/language pool without replacement until it is exhausted')
    p.add_argument('--ffn-type', choices=['geglu', 'yat_glu'],
                   help='Opt-in YAT feature branch with linear gate; changes pretrained model behavior')
    p.add_argument('--yat-ffn-backward-block', type=int, help='Direct BF16 FFN backward coordinate tile; forward distance order is unchanged')
    p.add_argument('--yat-compute-mode', choices=['mixed', 'bf16', 'bf16_adaptive'], help='BF16 direct reference or adaptive dot-reuse with selective direct fallback; benchmark before scaling')
    p.add_argument('--yat-ffn-compute-mode', choices=['mixed', 'bf16', 'bf16_adaptive'],
                   help='Optional FFN-only override; attention retains yat-compute-mode. Experimental; requires separate validation.')
    p.add_argument('--yat-softmax-backward', choices=['factored', 'max_centered'],
                   help='Opt-in BF16 gradient centering; recorded in checkpoint identity; full-model TPU qualification pending')
    p.add_argument('--yat-attention-implementation', choices=['standard', 'centered_fp32_scores'],
                   help='Experimental centered backward with BF16 geometry and FP32 scores; requires direct BF16 mode and both attention tile sizes')
    p.add_argument('--yat-global-attention-block-size', type=int, default=0, help='Experimental global YAT query tiling; zero retains dense reference')
    p.add_argument('--yat-attention-block-size', type=int, default=0, help='Opt-in local YAT query tiling; 64 is TPU-benchmarked; zero retains dense reference')
    p.add_argument('--yat-local-shards', action='store_true', help='Experimental chip-local YAT geometry and attention using explicit data shard_map')
    p.add_argument('--attention-score', choices=['dot_product', 'yat_softmax'],
                   help='Exact YAT scores followed by softmax; experimental dense reference')
    p.add_argument('--yat-epsilon', type=float, choices=[.01], help='Fixed YAT distance regularizer: 0.01')
    p.add_argument('--yat-alpha', type=float, help='Initial value of the trainable per-block YAT alpha')
    p.add_argument('--yat-attention-alpha', type=float,
                   help='Optional separate initial attention alpha; stays trainable and is recorded in checkpoint identity')
    p.add_argument('--mask-probability', type=float, default=.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--keep-checkpoints', type=int, default=3, help='Retained checkpoints; 0 keeps every checkpoint')
    p.add_argument('--residual-dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16',
                   help='Dense/attention compute; master parameters and Adam state remain FP32')
    p.add_argument('--attention-backend', choices=['xla', 'splash'], default='xla')
    p.add_argument('--loss-chunk-size', type=int, default=128)
    p.add_argument('--mlm-projection-capacity', type=int,
                   help='Optional per-row masked-target capacity, independent of loss chunk size; overflow remains loss-preserving')
    p.add_argument('--mlm-projection', choices=['dense', 'masked'], default='dense',
                   help='Experimental masked-position projection with loss-preserving overflow fallback')
    p.add_argument('--mlm-loss-backend', choices=['xla', 'xla_full', 'xla_local', 'pallas'], default='xla')
    p.add_argument('--mlm-vocab-tile', type=int, default=1024)
    p.add_argument('--no-remat', action='store_true')
    p.add_argument('--donate-state', action='store_true', help='Opt-in model/optimizer buffer reuse; benchmark memory and throughput')
    p.add_argument('--profile-dir', help='Local trace directory; one subdirectory per JAX process')
    p.add_argument('--profile-skip-steps', type=int, default=2, help='Warmup updates before bounded trace, relative to this invocation')
    p.add_argument('--profile-steps', type=int, default=1, help='Number of steady-state updates to trace')
    p.add_argument('--compile-diagnostics', action='store_true', help='Report separate first-step placement, lowering, compilation and execution phases')
    p.add_argument('--host-gc-diagnostics', action='store_true', help='Observe Python GC duration overlapping training steps without changing collector behavior')
    p.add_argument('--fsdp', type=int, default=1, help='Experimental first-axis parameter/optimizer sharding factor; full trainer TPU qualification pending')
    p.add_argument('--nnx-jit-partial', action='store_true', help='Fixed-graph NNX binding to reduce per-step Python traversal; exact save/resume qualified on single-chip v5e, other topologies require qualification')
    p.add_argument('--checkpoint-manifest-chunk-mib', type=int, help='Manifest hash transfer size (1–64 MiB); default 16 on a single TPU device, 1 elsewhere; checkpoint format unchanged')
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--preflight-only', action='store_true', help='Validate local config/tokenizer/data without training or creating checkpoints')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--initialize-from-checkpoint', help='New stage from encoder model weights; fresh optimizer, schedule and cursor')
    p.add_argument('--initialize-step', type=int, help='Pin parent checkpoint step; otherwise latest is resolved once before preflight')
    p.add_argument('--shared-local-checkpoints', action='store_true',
                   help='Confirm output is a filesystem shared by every worker')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
