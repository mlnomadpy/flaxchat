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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "restore"))
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()
    model = PortableModel()
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    if args.mode == "save":
        manager = create_checkpoint_manager(args.checkpoint_dir, async_checkpointing=False)
        save_checkpoint(
            manager,
            7,
            model,
            optimizer,
            {"writer_device_count": jax.device_count()},
            training_state={"update_step": jnp.asarray(7)},
        )
        manager.close()
    else:
        model.weight[...] = jnp.zeros_like(model.weight[...])
        metadata, training_state = restore_model_from_checkpoint(
            model,
            args.checkpoint_dir,
            optimizer=optimizer,
            load_training_state=True,
        )
        assert training_state is not None
        np.testing.assert_array_equal(
            np.asarray(model.weight[...]), np.arange(32, dtype=np.float32).reshape(8, 4)
        )
        assert int(training_state["update_step"]) == 7
        assert metadata["writer_device_count"] != jax.device_count()
    print(json.dumps({
        "mode": args.mode,
        "device_count": jax.device_count(),
        "shard_count": len(model.weight[...].addressable_shards),
    }))


if __name__ == "__main__":
    main()
