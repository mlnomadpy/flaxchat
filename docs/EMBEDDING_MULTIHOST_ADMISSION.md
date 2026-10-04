# Physical embedding multi-host admission

`python -m scripts.validate_embedding_multihost` is a worker for an **already allocated**, independently guarded TPU slice. It does not request hardware. Launch the same immutable command on every VM worker through the approved supervisor and outer command timeout. A one-host invocation fails; a successful localhost multiprocess simulation cannot satisfy its physical TPU backend gate.

Example for an actual 16-device, two-process slice (verify that inventory first):

```bash
JAX_PLATFORMS=tpu .venv/bin/python -m scripts.validate_embedding_multihost \
  --output gs://YOUR-PRIVATE-BUCKET/FRESH-ADMISSION-PREFIX \
  --expected-devices 16 --expected-processes 2 \
  --batch-size 32 --encoder-chunk-size 16 --timeout-seconds 1200
```

The quoted counts are an example declaration, not evidence that capacity, quota or this topology is available. Keep the outer execution timeout within the independently verified cloud lease, with evidence/cleanup margin. The worker's Python alarm is an additional bound; it cannot reliably interrupt a blocked native collective, so it does not replace the outer timeout or cloud deletion guard. All hosts must install the same frozen source/runtime before launch.

The output must be a new GCS prefix. It receives the actual production trainer's `baseline` and `recovery` checkpoints plus separately owned `receipts/rank-N.json` files and `receipts/summary.json`. This creates synthetic qualification artifacts, not a continuation of the published production model. Preserve this distinction in cost and model reporting.

## Cases and acceptance

- Discover JAX distributed ranks before backend/model operations, require at least two physical processes, and require exact declared global device/process counts. Gather actual local devices and row slices; reject missing/duplicate ranks, overlapping rows/devices, incomplete global ownership, mixed device kinds or source/worker/runtime disagreement.
- Place host-local query/positive/negative embeddings as globally shaped arrays. Compare complete hard-negative loss and all embedding gradients against a fully replicated **TPU** reference. Known alternative positives cross the physical host boundary; another relevant candidate exists only in the mined-negative pool. Assert the expected known-positive mask count.
- Construct a tiny BF16 YAT encoder using real public export and data-preparation APIs, with fixed bias 1, epsilon 0.01 and trainable alpha. Compare actual full encoder parameter gradients against a fully replicated TPU reference. An enabled cache path is compared against the uncached reference. The receipt declares numerical tolerances and complete fixture identities.
- Reuse `train_yat_embedding_finetune.run` and the shared stage parser/admission, rather than maintaining a second training loop. Run four updates uninterrupted, then an identical stage stopped after update two and resumed. Stage evaluation cadence is four, so this is an off-cadence stop. Restore committed GCS manifests and compare model, optimizer and full training-state trees exactly. Compare the selected best artifact and selection metadata too.
- Every physical rank must pass. Rank-zero summary aggregates the rank results; per-rank evidence has unique paths. Missing summaries, failed cases and missing ranks cannot be interpreted as passing.

The default trainer fixture contains explicit negatives. Use `--pair-only` with a separate fresh prefix to qualify the static pair-only encoder path. Use `--encoder-chunk-size 0` for direct execution; a nonzero chunk must divide the global batch and align with the declared global device count. Run each policy under its own recorded identity, not by overwriting previous evidence.

## Limits

The worker qualifies only its actual family/chip/host count, tiny fixture and selected pair/cache policy. Its replicated reference remains on TPUs. It does not run CPU model checks.

Four synthetic updates do not establish sustained multi-host training, production data throughput, production representation quality, v6e acceptance, every scale, or transport-disconnect recovery. The dedicated physical single-host suite includes best/recent commit-fault cases; a separate physical multi-host abrupt-kill/restart scenario remains needed to qualify that fault scope. Always report this limitation instead of treating the new worker as completion of all-scale acceptance.

Model-free tests in `tests/test_embedding_multihost_metadata.py` exercise ownership rejection and complete checkpoint-tree comparison. Those tests can run locally, but are not physical multi-host acceptance. Before allocation, also run the shared complete stage admission and sealed infrastructure manifest checks, then verify current account/project, existing allocations, spending authorization, price basis and independent cleanup.
