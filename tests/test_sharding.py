import os

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from flaxchat.common import replicate_on_mesh, setup_mesh, shard_batch


def test_default_mesh_consumes_every_device():
    mesh = setup_mesh()
    assert mesh.shape == {"data": jax.device_count(), "fsdp": 1, "tensor": 1}
    assert mesh.devices.size == jax.device_count()


def test_batch_is_sharded_and_state_is_replicated():
    mesh = setup_mesh()
    batch = jnp.arange(jax.device_count() * 4).reshape(jax.device_count(), 4)
    inputs, targets = shard_batch(batch, batch + 1, mesh)
    state = replicate_on_mesh({"weight": jnp.ones((2, 2))}, mesh)
    assert inputs.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), inputs.ndim)
    assert targets.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), targets.ndim)
    assert state["weight"].sharding.is_equivalent_to(NamedSharding(mesh, P()), 2)


def test_virtual_multidevice_job_really_has_eight_devices():
    if "xla_force_host_platform_device_count=8" not in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("dedicated virtual multi-device job only")
    assert jax.device_count() == 8
