# Sharding contract

FlaxChat creates a three-axis JAX mesh named `data`, `fsdp`, and `tensor`.
The default assigns every visible device to `data`; model state is replicated and
batches use `PartitionSpec("data")`. FSDP shards rank-two and larger parameters
along their first dimension only when explicitly requested.

The CPU CI forces eight virtual devices with:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  pytest tests/test_sharding.py -v
```

This job verifies that the mesh consumes all eight devices, arrays carry the
expected named sharding, and the public two-step pretraining command accepts a
global batch divisible by the mesh. Multi-host loaders assign alternating row
groups by process index and reject a checkpoint cursor created under a different
process count or index.
