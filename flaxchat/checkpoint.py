"""Versioned, integrity-checked Orbax checkpoints for exact resumption."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


CHECKPOINT_FORMAT_VERSION = 2


class CheckpointCompatibilityError(ValueError):
    """Raised before live state is mutated when a checkpoint is incompatible."""


def create_checkpoint_manager(
    checkpoint_dir: str,
    max_to_keep: int = 3,
    async_checkpointing: bool = True,
) -> ocp.CheckpointManager:
    """Create an atomic Orbax manager honoring the async policy."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        enable_async_checkpointing=async_checkpointing,
        cleanup_tmp_directories=True,
    )
    return ocp.CheckpointManager(directory=checkpoint_dir, options=options)


def _opt_state_pytree(optimizer: nnx.Optimizer):
    return optimizer.opt_state


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_manifest(tree) -> dict[str, dict[str, Any]]:
    """Return a canonical schema and content digest for each array leaf."""
    manifest = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(jax.device_get(leaf))
        name = jax.tree_util.keystr(path)
        manifest[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return manifest


def _checkpoint_manifest(step, model_state, opt_state, metadata, training_state):
    identities = {
        "resolved_config": metadata.get("resolved_config", metadata.get("model_config")),
        "tokenizer": metadata.get("tokenizer_identity", "unavailable"),
        "data_manifest": metadata.get("data_manifest_identity", "unavailable"),
        "source_revision": metadata.get("source_revision", "unavailable"),
    }
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "model_state": _state_manifest(model_state),
        "optimizer_state": _state_manifest(opt_state),
        "training_state": _state_manifest(training_state) if training_state is not None else {},
        "metadata_sha256": _json_hash(metadata),
        "identity": identities,
        "identity_sha256": _json_hash(identities),
    }


def save_checkpoint(
    manager: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metadata: dict,
    *,
    training_state: dict | None = None,
):
    """Save all persistent model variables, optimizer and resumable run state."""
    model_state = nnx.to_pure_dict(nnx.state(model))
    opt_state = _opt_state_pytree(optimizer)
    metadata = dict(metadata)
    manifest = _checkpoint_manifest(
        step, model_state, opt_state, metadata, training_state
    )
    items = {
        "model": ocp.args.PyTreeSave(model_state),
        "optimizer": ocp.args.PyTreeSave(opt_state),
        "metadata": ocp.args.JsonSave(metadata),
        "manifest": ocp.args.JsonSave(manifest),
    }
    if training_state is not None:
        items["training_state"] = ocp.args.PyTreeSave(training_state)
    return manager.save(step, args=ocp.args.Composite(**items))


def _validate_manifest(manifest, model_state, opt_state, metadata, training_state):
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint format {manifest.get('format_version')!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    actual = {
        "model_state": _state_manifest(model_state),
        "optimizer_state": _state_manifest(opt_state) if opt_state is not None else None,
        "training_state": _state_manifest(training_state) if training_state is not None else {},
    }
    if actual["model_state"] != manifest.get("model_state"):
        raise CheckpointCompatibilityError("Model checkpoint state is incomplete or corrupt")
    if opt_state is not None and actual["optimizer_state"] != manifest.get("optimizer_state"):
        raise CheckpointCompatibilityError("Optimizer checkpoint state is incomplete or corrupt")
    if training_state is not None and actual["training_state"] != manifest.get("training_state"):
        raise CheckpointCompatibilityError("Training checkpoint state is incomplete or corrupt")
    if _json_hash(metadata) != manifest.get("metadata_sha256"):
        raise CheckpointCompatibilityError("Checkpoint metadata is corrupt")


def _restore_args_on_current_topology(metadata):
    """Build explicit restore args without trusting checkpoint sharding files.

    Training state has no live target tree, so Orbax otherwise reconstructs the
    sharding saved by the writer.  That is unsafe when a checkpoint moves from
    one accelerator topology to another.  These small bookkeeping arrays are
    deliberately restored onto a current local device instead.
    """
    devices = jax.local_devices()
    if not devices:
        raise CheckpointCompatibilityError("No local JAX device is available for restore")
    current_sharding = jax.sharding.SingleDeviceSharding(devices[0])
    sharding_tree = jax.tree.map(lambda _: current_sharding, metadata)
    return ocp.checkpoint_utils.construct_restore_args(
        metadata, sharding_tree=sharding_tree
    )


def load_checkpoint(
    manager: ocp.CheckpointManager,
    step: int | None = None,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,
    *,
    load_training_state: bool = False,
):
    """Load and validate a checkpoint without mutating live objects.

    For compatibility, returns ``(model_state, metadata)`` unless optimizer or
    training state was requested; then it returns a four-tuple containing both.
    """
    if step is None:
        step = manager.latest_step()
        if step is None:
            raise ValueError("No checkpoints found")

    model_abstract = nnx.to_pure_dict(nnx.state(model)) if model is not None else None
    model_restore_args = (
        ocp.checkpoint_utils.construct_restore_args(model_abstract)
        if model_abstract is not None else None
    )
    items = {
        "model": ocp.args.PyTreeRestore(model_abstract, restore_args=model_restore_args),
        "metadata": ocp.args.JsonRestore(),
        "manifest": ocp.args.JsonRestore(),
    }
    if optimizer is not None:
        optimizer_abstract = _opt_state_pytree(optimizer)
        items["optimizer"] = ocp.args.PyTreeRestore(
            optimizer_abstract,
            restore_args=ocp.checkpoint_utils.construct_restore_args(optimizer_abstract),
        )
    if load_training_state:
        # A freshly opened Composite manager has no item handlers registered,
        # so ``manager.item_metadata`` cannot discover this tree reliably.
        step_directory = manager._get_read_step_directory(step, manager.directory)
        training_metadata = ocp.PyTreeCheckpointHandler().metadata(
            step_directory / "training_state"
        )
        items["training_state"] = ocp.args.PyTreeRestore(
            training_metadata,
            restore_args=_restore_args_on_current_topology(training_metadata)
        )
    try:
        restored = manager.restore(step, args=ocp.args.Composite(**items))
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint {step} is incomplete, corrupt, or incompatible: {exc}"
        ) from exc

    opt_state = restored.optimizer if optimizer is not None else None
    training_state = restored.training_state if load_training_state else None
    _validate_manifest(
        restored.manifest, restored.model, opt_state, restored.metadata, training_state
    )
    if optimizer is None and not load_training_state:
        return restored.model, restored.metadata
    return restored.model, opt_state, restored.metadata, training_state


def restore_model_from_checkpoint(
    model: nnx.Module,
    checkpoint_dir: str,
    step: int | None = None,
    optimizer: nnx.Optimizer | None = None,
    *,
    expected_identity: dict | None = None,
    load_training_state: bool = False,
):
    """Validate first, then atomically apply restored model/optimizer state."""
    manager = create_checkpoint_manager(
        checkpoint_dir, max_to_keep=999, async_checkpointing=False
    )
    try:
        loaded = load_checkpoint(
            manager, step, model, optimizer, load_training_state=load_training_state
        )
        if optimizer is None and not load_training_state:
            model_dict, metadata = loaded
            opt_state = training_state = None
        else:
            model_dict, opt_state, metadata, training_state = loaded

        if expected_identity is not None:
            actual = {
                "resolved_config": metadata.get("resolved_config", metadata.get("model_config")),
                "tokenizer": metadata.get("tokenizer_identity", "unavailable"),
                "data_manifest": metadata.get("data_manifest_identity", "unavailable"),
                "source_revision": metadata.get("source_revision", "unavailable"),
            }
            mismatches = {
                key: (actual.get(key), value)
                for key, value in expected_identity.items()
                if actual.get(key) != value
            }
            if mismatches:
                raise CheckpointCompatibilityError(
                    f"Checkpoint identity mismatch: {mismatches}"
                )

        model_state = nnx.state(model)
        pure_state = nnx.to_pure_dict(model_state)
        if _state_manifest(pure_state).keys() != _state_manifest(model_dict).keys():
            raise CheckpointCompatibilityError("Live model state schema does not match checkpoint")
        nnx.replace_by_pure_dict(model_state, model_dict)
        nnx.update(model, model_state)
        if optimizer is not None and opt_state is not None:
            optimizer.opt_state = opt_state
        if load_training_state:
            return metadata, training_state
        return metadata
    finally:
        manager.close()
