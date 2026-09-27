"""Save or restore a deliberately sharded checkpoint for topology testing."""

from __future__ import annotations

import argparse
import json

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from flaxchat.checkpoint import create_checkpoint_manager, restore_model_from_checkpoint, save_checkpoint


class PortableModel(nnx.Module):
    def __init__(self):
        mesh = Mesh(np.asarray(jax.devices()), ("data",))
        sharding = NamedSharding(mesh, P("data", None))
        values = jnp.arange(32, dtype=jnp.float32).reshape(8, 4)
        self.weight = nnx.Param(jax.device_put(values, sharding))


def make_run():
    model = PortableModel()
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    mesh = model.weight[...].sharding.mesh
    nnx.update(optimizer, jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P("data", None) if x.ndim == 2 else P())),
        nnx.state(optimizer),
    ))
    return model, optimizer


def advance(model, optimizer):
    # Elementwise deterministic updates isolate checkpoint portability from
    # topology-dependent floating-point reduction order.
    gradients = jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param))
    optimizer.update(model, gradients)


def assert_same_state(left, right):
    expected, expected_tree = jax.tree.flatten(nnx.state(left))
    actual, actual_tree = jax.tree.flatten(nnx.state(right))
    assert expected_tree == actual_tree
    for x, y in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        assert x.sharding.is_equivalent_to(y.sharding, x.ndim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "restore"))
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()
    model, optimizer = make_run()
    if args.mode == "save":
        for _ in range(7):
            advance(model, optimizer)
        manager = create_checkpoint_manager(args.checkpoint_dir, async_checkpointing=False)
        try:
            save_checkpoint(
                manager,
                7,
                model,
                optimizer,
                {"writer_device_count": jax.device_count()},
                training_state={"update_step": jnp.asarray(7)},
            )
        finally:
            manager.close()
    else:
        reference_model, reference_optimizer = make_run()
        for _ in range(7):
            advance(reference_model, reference_optimizer)
        model.weight[...] = jnp.zeros_like(model.weight[...])
        metadata, training_state = restore_model_from_checkpoint(
            model,
            args.checkpoint_dir,
            optimizer=optimizer,
            load_training_state=True,
        )
        assert_same_state(reference_model, model)
        assert_same_state(reference_optimizer, optimizer)
        assert int(training_state["update_step"]) == 7
        assert int(optimizer.step[...]) == 7
        assert metadata["writer_device_count"] != jax.device_count()
        advance(model, optimizer)
        advance(reference_model, reference_optimizer)
        assert_same_state(reference_model, model)
        assert_same_state(reference_optimizer, optimizer)
    print(json.dumps({
        "mode": args.mode,
        "device_count": jax.device_count(),
        "shard_count": len(model.weight[...].addressable_shards),
        "optimizer_updates": int(optimizer.step[...]),
    }))


if __name__ == "__main__":
    main()
