"""Pure, testable helpers for numerically safe optimizer updates."""

import jax
import jax.numpy as jnp
from flax import nnx


def accumulation_dtype(name: str):
    """Resolve the configured gradient accumulation dtype."""
    aliases = {
        "float32": jnp.float32,
        "fp32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported gradient accumulation dtype: {name!r}") from exc


def tree_all_finite(tree):
    return jax.tree.reduce(
        lambda finite, leaf: finite & jnp.all(jnp.isfinite(leaf)),
        tree,
        initializer=jnp.array(True),
    )


def apply_gradients_if_finite(model, optimizer, grads, loss):
    """Atomically update model and optimizer, or preserve both on nonfinite input."""
    finite = jnp.isfinite(loss) & tree_all_finite(grads)

    def update(model, optimizer, grads):
        before_model = jax.tree.map(jnp.copy, nnx.state(model))
        before_optimizer = jax.tree.map(jnp.copy, nnx.state(optimizer))
        optimizer.update(model, grads)
        candidate_model, candidate_optimizer = nnx.state(model), nnx.state(optimizer)
        accepted = tree_all_finite(candidate_model) & tree_all_finite(candidate_optimizer)
        def choose(new, old):
            return jnp.where(accepted, new, old)
        nnx.update(model, jax.tree.map(choose, candidate_model, before_model))
        nnx.update(optimizer, jax.tree.map(choose, candidate_optimizer, before_optimizer))
        return accepted

    def skip(model, optimizer, grads):
        del model, optimizer, grads
        return jnp.array(False)

    updated = nnx.cond(finite, update, skip, model, optimizer, grads)
    return updated


def gradients_for_microbatches(model, all_inputs, all_targets, dtype=jnp.float32):
    """Average microbatch gradients in an explicit accumulation dtype."""
    params = nnx.state(model, nnx.Param)
    zeros = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=dtype), params)

    @nnx.scan(
        in_axes=(nnx.Carry, None, 0, 0),
        out_axes=(nnx.Carry, 0),
    )
    def micro_step(accumulated, current_model, inputs, targets):
        def loss_fn(current_model):
            return current_model(inputs, targets)

        loss, grads = nnx.value_and_grad(loss_fn)(current_model)
        grads = jax.tree.map(lambda grad: grad.astype(dtype), grads)
        accumulated = jax.tree.map(jnp.add, accumulated, grads)
        return accumulated, loss

    accumulated, losses = micro_step(zeros, model, all_inputs, all_targets)
    count = all_inputs.shape[0]
    averaged = jax.tree.map(lambda grad: grad / count, accumulated)
    return jnp.mean(losses), averaged


def place_host_batch(array, mesh, *, batch_axis=0):
    """Place process-local rows into one global batch, including accumulation.

    Every process contributes local_device_count * per_device_batch rows.
    Accumulation uses axis 1; the microstep axis is replicated.
    """
    from jax.sharding import NamedSharding, PartitionSpec
    axes = [None] * array.ndim
    axes[batch_axis] = 'data'
    sharding = NamedSharding(mesh, PartitionSpec(*axes))
    return jax.make_array_from_process_local_data(sharding, array)


def gather_process_metadata(value):
    """Collect small JSON resume metadata in rank order on every worker."""
    import json
    import numpy as np
    from jax.experimental import multihost_utils
    if jax.process_count() == 1:
        return [value]
    encoded = json.dumps(value, sort_keys=True).encode()
    sizes = np.asarray(multihost_utils.process_allgather(np.asarray(len(encoded), np.int32))).reshape(-1)
    payload = np.zeros(int(sizes.max()), np.uint8)
    payload[:len(encoded)] = np.frombuffer(encoded, np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(payload)).reshape(len(sizes), -1)
    return [json.loads(bytes(row[:size])) for row, size in zip(gathered, sizes, strict=True)]


def initialize_sharded(factory, mesh, *, fsdp=1):
    """Compile initialization directly into target shardings, without full leaves.

    The factory may return a model or (model, optimizer). Abstract evaluation
    determines layout before any parameter or optimizer buffer is allocated.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    if fsdp < 1 or mesh.size % fsdp:
        raise ValueError('fsdp must divide the mesh size')
    abstract = nnx.eval_shape(factory)
    graph, state = nnx.split(abstract)
    def layout(leaf):
        shape = getattr(leaf, 'shape', ())
        spec = P('fsdp') if len(shape) >= 2 and fsdp > 1 and shape[0] % fsdp == 0 else P()
        return NamedSharding(mesh, spec)
    shardings = jax.tree.map(layout, state)
    @jax.jit(out_shardings=shardings)
    def initialize():
        return nnx.split(factory())[1]
    return nnx.merge(graph, initialize())


def pretraining_optimizer(model, *, kind, learning_rate, warmup_steps, steps):
    """One optimizer recipe shared by training and portable checkpoint restore."""
    import optax
    schedule = (optax.warmup_cosine_decay_schedule(0., learning_rate, warmup_steps, steps,
                                                  end_value=learning_rate * .05)
                if warmup_steps else optax.cosine_decay_schedule(learning_rate, steps, alpha=.05))
    if kind == 'muon':
        from flaxchat.config import FlaxChatConfig
        from flaxchat.optim import setup_optimizer
        config = FlaxChatConfig(model=model.config)
        config.training.matrix_lr = learning_rate
        optimizer = setup_optimizer(model, config, weight_decay_scaled=.01,
                                    lr_schedule_fn=lambda step: schedule(step) / learning_rate)
    elif kind == 'adamw':
        optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.),
                                  optax.adamw(schedule, weight_decay=.01)), wrt=nnx.Param)
    else:
        raise ValueError(f'Unsupported optimizer recipe: {kind}')
    return optimizer, schedule
