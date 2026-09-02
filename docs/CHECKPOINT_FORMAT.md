# Checkpoint format

Format version 2 is an Orbax composite checkpoint with these items:

- `model`: every persistent NNX variable, not only trainable parameters;
- `optimizer`: the complete Optax state;
- `training_state`: update/RNG state supplied by the training loop;
- `metadata`: resolved configuration, tokenizer/data/source identities and
  the exact dataloader cursor;
- `manifest`: canonical shapes, dtypes and SHA-256 digests for every state
  leaf plus hashes of metadata and compatibility identities.

Orbax writes through temporary directories and publishes a completed step
atomically. Asynchronous managers must be finalized with
`wait_until_finished()` and `close()` before process exit. Restore validates
the manifest and optional expected identities before mutating the live model
or optimizer. Partial, corrupt, schema-incompatible, or identity-mismatched
checkpoints raise `CheckpointCompatibilityError`.
