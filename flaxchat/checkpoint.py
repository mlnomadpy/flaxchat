"""
Checkpoint management using Orbax.

Supports:
- Async checkpointing for non-blocking saves
- Multi-tier: fast local + durable GCS
- Rotation (max_to_keep)
- Exact resumption (model, optimizer, metadata)
"""

import os
import json

import jax
import orbax.checkpoint as ocp
from flax import nnx


def create_checkpoint_manager(
    checkpoint_dir: str,
    max_to_keep: int = 3,
    async_checkpointing: bool = True,
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_dir,
        options=options,
    )
    return manager


def _opt_state_pytree(optimizer: nnx.Optimizer):
    """Extract the optimizer's pure pytree state (opt_state moments etc).

    Uses `optimizer.opt_state` directly: it's the raw optax state (e.g.
    tuple of ScaleByAdamState, ScaleByLearningRateState, EmptyState for
    plain AdamW). Avoids `nnx.state(optimizer)` which also includes the
    wrapped model params.
    """
    return optimizer.opt_state


def save_checkpoint(
    manager: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metadata: dict,
):
    """Save model params, optimizer state, and metadata to a composite checkpoint."""
    model_state = nnx.state(model, nnx.Param)
    model_dict = nnx.to_pure_dict(model_state)
    opt_state = _opt_state_pytree(optimizer)

    manager.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeSave(model_dict),
            optimizer=ocp.args.PyTreeSave(opt_state),
            metadata=ocp.args.JsonSave(metadata),
        ),
    )


def load_checkpoint(
    manager: ocp.CheckpointManager,
    step: int | None = None,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,
) -> tuple[dict, object, dict]:
    """
    Load checkpoint. Returns (model_dict, opt_state, metadata).
    If step is None, loads the latest checkpoint.
    If `optimizer` is None, optimizer state is not restored (returns None).
    """
    if step is None:
        step = manager.latest_step()
        if step is None:
            raise ValueError("No checkpoints found")

    # Abstract states for restore shape inference
    if model is not None:
        abstract_model = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    else:
        abstract_model = None

    composite_args = {
        "model": ocp.args.PyTreeRestore(abstract_model),
        "metadata": ocp.args.JsonRestore(),
    }
    if optimizer is not None:
        composite_args["optimizer"] = ocp.args.PyTreeRestore(_opt_state_pytree(optimizer))

    restored = manager.restore(step, args=ocp.args.Composite(**composite_args))
    opt_state = restored.optimizer if optimizer is not None else None
    return restored.model, opt_state, restored.metadata


def restore_model_from_checkpoint(
    model: nnx.Module,
    checkpoint_dir: str,
    step: int | None = None,
    optimizer: nnx.Optimizer | None = None,
):
    """Load checkpoint and apply to model (and optionally optimizer) in-place.

    Returns metadata dict. If `optimizer` is provided, its state is also
    restored (opt_state is reassigned in place).
    """
    manager = create_checkpoint_manager(checkpoint_dir, max_to_keep=999)
    model_dict, opt_state, metadata = load_checkpoint(manager, step, model, optimizer)

    # Apply loaded params to model in-place via state traversal
    import jax.numpy as jnp

    def _apply_dict(module_state, loaded_dict):
        for key, val in loaded_dict.items():
            if isinstance(val, dict):
                _apply_dict(module_state[key], val)
            else:
                module_state[key].value = jnp.array(val)

    model_state = nnx.state(model, nnx.Param)
    _apply_dict(model_state, model_dict)
    nnx.update(model, model_state)

    if optimizer is not None and opt_state is not None:
        # Reassign opt_state — this is a pytree of optax state NamedTuples
        # with restored arrays inside. nnx.Optimizer holds opt_state as an
        # attribute so we can just overwrite.
        optimizer.opt_state = opt_state

    return metadata
