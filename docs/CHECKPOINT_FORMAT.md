# Checkpoint format

Format version 2 is an Orbax composite checkpoint with these items:

- `model`: every persistent NNX variable, not only trainable parameters;
- `optimizer`: the complete Optax state;
- `training_state`: update/RNG state supplied by the training loop;
- `metadata`: resolved configuration, tokenizer/data/source identities and
  the exact dataloader cursor;
- `manifest`: canonical shapes, dtypes and SHA-256 digests for every state
leaf plus hashes of metadata and compatibility identities.

`scripts.pretrain --resume-from-step=N` restores the model and optimizer before
rebuilding the data iterator. Its saved cursor is then validated against the
ordered dataset manifest, tokenizer, packing policy, and process topology.

Orbax writes through temporary directories and publishes a completed step
atomically. Asynchronous managers must be finalized with
`wait_until_finished()` and `close()` before process exit. Restore validates
the manifest and optional expected identities before mutating the live model
or optimizer. Partial, corrupt, schema-incompatible, or identity-mismatched
checkpoints raise `CheckpointCompatibilityError`.

Every array restore receives an explicit destination sharding. Model and
optimizer leaves use the live object's current sharding; bookkeeping arrays in
`training_state` are restored on a current local device. Restore never falls
back to the writer's serialized sharding, so moving a checkpoint between
device topologies does not silently recreate an incompatible layout.
