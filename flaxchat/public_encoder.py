"""Portable, weight-only YAT encoder release helpers.

The tensor names are canonical JAX tree paths, so every trainable YAT alpha is
included without a fragile hand-written mapping to upstream ModernBERT names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from safetensors.numpy import load_file, save_file

from flaxchat.checkpoint import _canonical_manifest_paths
from flaxchat.encoder import EncoderConfig, ModernBert


def _named_leaves(tree):
    return _canonical_manifest_paths({
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
    })


def export_state_safetensors(state, output, *, expected_manifest=None):
    """Write a complete state tree; optionally verify training manifest hashes."""
    tensors = {}
    leaves = _named_leaves(state)
    if expected_manifest is not None:
        expected_manifest = _canonical_manifest_paths(expected_manifest)
    if expected_manifest is not None and set(leaves) != set(expected_manifest):
        raise ValueError("Model state paths differ from checkpoint manifest")
    for name, value in leaves.items():
        array = np.asarray(jax.device_get(value)).copy(order="C")
        if expected_manifest is not None:
            record = expected_manifest[name]
            if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
                raise ValueError(f"Model state schema differs at {name}")
            if hashlib.sha256(array.tobytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Model state hash differs at {name}")
        tensors[name] = array
    save_file(tensors, str(output), metadata={"format": "flaxchat-nnx-state-v1"})
    digest = hashlib.sha256()
    with Path(output).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"tensors": len(tensors), "bytes": Path(output).stat().st_size,
            "sha256": digest.hexdigest()}


def load_public_encoder(directory):
    """Construct the exact YAT architecture and load all released parameters."""
    directory = Path(directory)
    config = EncoderConfig(**json.loads((directory / "config.json").read_text()))
    model = ModernBert(config, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    expected = _named_leaves(nnx.to_pure_dict(state))
    tensors = load_file(str(directory / "model.safetensors"))
    if set(tensors) != set(expected):
        raise ValueError("Released tensor names do not match the YAT model")
    for name, reference in expected.items():
        if tensors[name].shape != reference.shape or tensors[name].dtype != np.dtype(reference.dtype):
            raise ValueError(f"Released tensor schema differs at {name}")
    restored = jax.tree_util.tree_map_with_path(
        lambda path, _: jax.device_put(tensors[next(iter(_canonical_manifest_paths({jax.tree_util.keystr(path): None})))]),
        nnx.to_pure_dict(state),
    )
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(model, state)
    return model
