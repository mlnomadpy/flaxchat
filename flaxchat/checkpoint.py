"""Versioned, integrity-checked Orbax checkpoints for exact resumption."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Any, cast

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
    if not checkpoint_dir.startswith("gs://"):
        checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
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


@lru_cache(maxsize=64)
def _chunk_reader(sharding, width):
    """Reuse compilations across same-shaped leaves and subsequent saves."""
    @jax.jit(out_shardings=sharding)
    def read(value, start):
        return jax.lax.dynamic_slice_in_dim(value.reshape(-1), start, width)
    return read


def _state_manifest(tree) -> dict[str, dict[str, Any]]:
    """Return a canonical schema and content digest for each array leaf."""
    manifest = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        digest = hashlib.sha256()
        if not hasattr(leaf, "dtype"):
            leaf = np.asarray(leaf)
        shape, dtype = leaf.shape, leaf.dtype
        # Stream canonical flattened byte chunks. Only a bounded chunk is
        # replicated for hashing; a whole FSDP leaf is never gathered on host.
        chunk_elements = max(1, (1024 * 1024) // np.dtype(dtype).itemsize)
        size = int(np.prod(shape))
        if size and isinstance(leaf, jax.Array):
            if leaf.is_fully_replicated:
                local = leaf.addressable_data(0).reshape(-1)
                for start in range(0, size, chunk_elements):
                    digest.update(np.asarray(jax.device_get(local[start:start + chunk_elements])).tobytes())
            else:
                from jax.sharding import NamedSharding, PartitionSpec
                if not isinstance(leaf.sharding, NamedSharding):
                    raise ValueError('Sharded checkpoint integrity requires NamedSharding')
                # Preserve the input mesh's physical device ordering. TPU
                # topology-aware meshes need not follow jax.devices() order.
                replicated = NamedSharding(leaf.sharding.mesh, PartitionSpec())
                width = min(size, chunk_elements)
                read_chunk = _chunk_reader(replicated, width)
                for start in range(0, size, width):
                    # Dynamic slices clamp at the end, so explicitly select the
                    # trailing partial region from that final bounded window.
                    offset = min(start, size - width)
                    chunk = read_chunk(leaf, np.int32(offset)).addressable_data(0)
                    array = np.asarray(jax.device_get(chunk))
                    digest.update(array[start - offset:].tobytes())
        else:
            flat = np.asarray(jax.device_get(leaf)).reshape(-1)
            for start in range(0, size, chunk_elements):
                digest.update(flat[start:start + chunk_elements].tobytes())
        name = jax.tree_util.keystr(path)
        manifest[name] = {
            "shape": list(shape), "dtype": str(dtype), "sha256": digest.hexdigest(),
        }
    return manifest


def _canonical_manifest_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    """Equate sequence indexes with Orbax's metadata-restored numeric keys."""
    result = {}
    for path, value in manifest.items():
        path = re.sub(r"\[(\d+)\]", r"['\1']", path)
        path = re.sub(r"\.([A-Za-z_]\w*)", r"['\1']", path)
        result[path] = value
    return result


def _checkpoint_manifest(step, model_state, opt_state, metadata, training_state):
    identities = {
        "resolved_config": metadata.get("resolved_config", metadata.get("model_config")),
        "tokenizer": metadata.get("tokenizer_identity", "unavailable"),
        "data_manifest": metadata.get("data_manifest_identity", "unavailable"),
        "source_revision": metadata.get("source_revision", "unavailable"),
        "source_python_sha256": metadata.get("source_python_sha256", "unavailable"),
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
    if _canonical_manifest_paths(actual["model_state"]) != _canonical_manifest_paths(
        manifest.get("model_state", {})
    ):
        raise CheckpointCompatibilityError("Model checkpoint state is incomplete or corrupt")
    if opt_state is not None and _canonical_manifest_paths(
        actual["optimizer_state"]
    ) != _canonical_manifest_paths(manifest.get("optimizer_state", {})):
        raise CheckpointCompatibilityError("Optimizer checkpoint state is incomplete or corrupt")
    if training_state is not None and _canonical_manifest_paths(
        actual["training_state"]
    ) != _canonical_manifest_paths(manifest.get("training_state", {})):
        raise CheckpointCompatibilityError("Training checkpoint state is incomplete or corrupt")
    if _json_hash(metadata) != manifest.get("metadata_sha256"):
        raise CheckpointCompatibilityError("Checkpoint metadata is corrupt")


def _restore_args_on_current_topology(metadata, target=None):
    """Build explicit restore args without trusting checkpoint sharding files.

    Training state has no live target tree, so Orbax otherwise reconstructs the
    sharding saved by the writer.  That is unsafe when a checkpoint moves from
    one accelerator topology to another.  These small bookkeeping arrays are
    deliberately restored onto a current local device instead.
    """
    devices = jax.local_devices()
    if not devices:
        raise CheckpointCompatibilityError("No local JAX device is available for restore")
    if jax.process_count() > 1:
        current_sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('restore',)),
            jax.sharding.PartitionSpec(),
        )
    else:
        current_sharding = jax.sharding.SingleDeviceSharding(devices[0])
    target_leaves = _canonical_manifest_paths({
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(target)[0]
    }) if target is not None else {}
    def sharding_for(path, value):
        key = next(iter(_canonical_manifest_paths({jax.tree_util.keystr(path): None})))
        leaf = target_leaves.get(key)
        if leaf is not None and np.shape(leaf) != tuple(getattr(value, "shape", ())):
            raise CheckpointCompatibilityError(f"Checkpoint shape mismatch at {key}")
        return leaf.sharding if isinstance(leaf, jax.Array) else current_sharding
    sharding_tree = jax.tree_util.tree_map_with_path(sharding_for, metadata)
    return ocp.checkpoint_utils.construct_restore_args(
        metadata, sharding_tree=sharding_tree
    )


def _item_metadata(manager: ocp.CheckpointManager, step: int, item: str):
    """Read an item's stored schema without restoring its writer topology."""
    step_directory = manager._get_read_step_directory(step, manager.directory)
    try:
        return ocp.PyTreeCheckpointHandler().metadata(step_directory / item)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint item {item!r} is incomplete, corrupt, or incompatible: {exc}"
        ) from exc


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

    # Restore against the stored schema so integrity is checked before Orbax can
    # cast TPU values (for example bfloat16 constants) to a CPU target tree.
    model_metadata = _item_metadata(manager, step, "model")
    items = {
        "model": ocp.args.PyTreeRestore(
            model_metadata,
            restore_args=_restore_args_on_current_topology(
                model_metadata, nnx.to_pure_dict(nnx.state(model)) if model is not None else None),
        ),
        "metadata": ocp.args.JsonRestore(),
        "manifest": ocp.args.JsonRestore(),
    }
    if optimizer is not None:
        optimizer_metadata = _item_metadata(manager, step, "optimizer")
        items["optimizer"] = ocp.args.PyTreeRestore(
            optimizer_metadata,
            restore_args=_restore_args_on_current_topology(optimizer_metadata, _opt_state_pytree(optimizer)),
        )
    if load_training_state:
        # A freshly opened Composite manager has no item handlers registered,
        # so ``manager.item_metadata`` cannot discover this tree reliably.
        training_metadata = _item_metadata(manager, step, "training_state")
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

    restored = cast(dict[str, Any], restored)
    opt_state = restored["optimizer"] if optimizer is not None else None
    training_state = restored["training_state"] if load_training_state else None
    _validate_manifest(
        restored["manifest"], restored["model"], opt_state, restored["metadata"], training_state
    )
    # Reconstruct live containers and cast only after stored-byte integrity has
    # passed. This uses the already-restored arrays, with no second storage read.
    def conform(restored_tree, target):
        leaves = _canonical_manifest_paths({
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(restored_tree)[0]
        })
        target_paths = _canonical_manifest_paths({
            jax.tree_util.keystr(path): None
            for path, _ in jax.tree_util.tree_flatten_with_path(target)[0]
        })
        if leaves.keys() != target_paths.keys():
            raise CheckpointCompatibilityError("Live state paths do not match checkpoint")
        def convert(path, reference):
            key = next(iter(_canonical_manifest_paths({jax.tree_util.keystr(path): None})))
            value = leaves[key]
            if np.shape(value) != np.shape(reference):
                raise CheckpointCompatibilityError(f"Live state shape mismatch at {key}")
            if not hasattr(reference, "dtype"):
                return type(reference)(value)
            return value.astype(reference.dtype)
        return jax.tree_util.tree_map_with_path(convert, target)
    model_state = restored["model"]
    if model is not None:
        model_state = conform(model_state, nnx.to_pure_dict(nnx.state(model)))
    if optimizer is not None:
        opt_state = conform(opt_state, _opt_state_pytree(optimizer))
    if optimizer is None and not load_training_state:
        return model_state, restored["metadata"]
    return model_state, opt_state, restored["metadata"], training_state


def load_checkpoint_metadata(checkpoint_dir: str, step: int | None = None) -> dict:
    """Read integrity-checked metadata before constructing a live model."""
    manager = create_checkpoint_manager(
        checkpoint_dir, max_to_keep=999, async_checkpointing=False
    )
    try:
        selected = manager.latest_step() if step is None else step
        if selected is None:
            raise ValueError("No checkpoints found")
        restored = manager.restore(
            selected,
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
                manifest=ocp.args.JsonRestore(),
            ),
        )
        restored = cast(dict[str, Any], restored)
        if _json_hash(restored["metadata"]) != restored["manifest"].get("metadata_sha256"):
            raise CheckpointCompatibilityError("Checkpoint metadata is corrupt")
        metadata = dict(restored['metadata'])
        if metadata.get('step', selected) != selected:
            raise CheckpointCompatibilityError('Checkpoint metadata step differs from selected checkpoint')
        # Older, integrity-checked artifacts omit this redundant field. Bind
        # the returned metadata to the actual selected directory without
        # rewriting the artifact or relaxing its original checksum.
        return {**metadata, 'step': selected}
    except CheckpointCompatibilityError:
        raise
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint metadata is incomplete, corrupt, or incompatible: {exc}"
        ) from exc
    finally:
        manager.close()


def validate_checkpoint_tokenizer(metadata, tokenizer, *, tokenizer_path=None):
    """Reject missing or semantically different tokenizer identities."""
    from flaxchat.dataloader import _tokenizer_identity
    actual = _tokenizer_identity(tokenizer)
    expected = metadata.get("tokenizer_identity")
    if isinstance(expected, str) and tokenizer_path is not None:
        # Released pipeline artifacts use the exact persisted-file hash.
        import hashlib
        from pathlib import Path
        from flaxchat.tokenizer import load_tokenizer, tokenizer_artifact_path
        artifact = Path(tokenizer_artifact_path(tokenizer_path))
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected or _tokenizer_identity(load_tokenizer(tokenizer_path)) != actual:
            raise CheckpointCompatibilityError("Checkpoint tokenizer identity mismatch")
        return actual
    if not isinstance(expected, dict) or not any(key.endswith("sha256") for key in expected):
        raise CheckpointCompatibilityError("Checkpoint lacks a verifiable tokenizer identity; migrate the artifact explicitly")
    if expected != actual:
        raise CheckpointCompatibilityError("Checkpoint tokenizer identity mismatch")
    return actual


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
        loaded: Any = load_checkpoint(
            manager, step, model, optimizer, load_training_state=load_training_state
        )
        if optimizer is None and not load_training_state:
            model_dict, metadata = cast(tuple[Any, dict], loaded)
            opt_state = training_state = None
        else:
            model_dict, opt_state, metadata, training_state = cast(
                tuple[Any, Any, dict, Any], loaded
            )

        if expected_identity is not None:
            actual = {
                "resolved_config": metadata.get("resolved_config", metadata.get("model_config")),
                "tokenizer": metadata.get("tokenizer_identity", "unavailable"),
                "data_manifest": metadata.get("data_manifest_identity", "unavailable"),
                "source_revision": metadata.get("source_revision", "unavailable"),
                "source_python_sha256": metadata.get("source_python_sha256", "unavailable"),
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

        if "model_config" in metadata and hasattr(model, "config"):
            from flaxchat.config import GPTConfig
            if GPTConfig(**metadata["model_config"]) != cast(Any, model).config:
                raise CheckpointCompatibilityError("Live model configuration does not match checkpoint")
        model_state = nnx.state(model)
        pure_state = nnx.to_pure_dict(model_state)
        live_schema = {jax.tree_util.keystr(path): np.shape(leaf)
                       for path, leaf in jax.tree_util.tree_flatten_with_path(pure_state)[0]}
        restored_schema = {jax.tree_util.keystr(path): np.shape(leaf)
                           for path, leaf in jax.tree_util.tree_flatten_with_path(model_dict)[0]}
        if live_schema != restored_schema:
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
