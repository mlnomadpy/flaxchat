# Spot TPU validation — 2026-09-21

## Scope and budget

Project: `tpubuilders`. Campaign budget: USD 50 out of the user's USD 900.
The first allocation is a single `v5litepod-8` Spot host in `us-west4-a`.
The unallocated Iowa request was deleted before the Las Vegas request.
An independent controller deletes the dedicated queued resource within two hours;
a VM startup shutdown is a secondary backstop. Resources are explicitly deleted
as soon as evidence is retrieved. Billing estimates are not actual billed charges.

## Changes under validation

- `scripts/prepare_token_pool.py` prepares a bounded corpus on CPU from full
  dataset and tokenizer commit SHAs. Validation and training consume disjoint
  documents. The token pool records its split policy and checksums.
- `flaxchat/token_pool.py` verifies local uint32 token files and provides packed,
  deterministic batches. Exhaustion is an error, not replacement text. Training
  targets contain no padding and EOS remains supervised.
- `scripts/train_gpt2.py` now requires `--token-manifest`. It never imports HF
  datasets or performs remote Parquet I/O. It validates the complete requested
  token budget before training, evaluates held-out loss, refuses nonfinite
  updates, checkpoints completed-update counts, and supports `--resume` and
  `--stop-after` without changing the learning-rate schedule.
- Resume checks model/config, tokenizer/data identity before restoring. A local
  regression compares uninterrupted vs resumed model, optimizer, and cursor
  digests exactly. Existing checkpoint directories require explicit resume.
- FineWeb's Kaggle adapter now requires a mounted token manifest and dataset.
- `scripts/validate_tpu.py` requires a physical TPU backend, runs the acceptance
  stages in bounded child processes, persists partial summaries, and exports
  each pytest result with skip/failure reasons. CPU subprocess tests remain CPU
  tests even when the parent suite runs on TPU.
- The shared training attention selector uses exact XLA attention on a data-
  sharded TPU mesh. The first physical attempt exposed that automatic Splash
  selection could not be partitioned. FineWeb and main pretraining now share
  the existing TinyStories policy.
- Inference inputs and caches now follow the model's device set. A physical
  checkpoint-inference test exposed the previous global-mesh mismatch after
  restoring weights onto one TPU. Four virtual-device regression cases cover
  padded, cached, JIT, and speculative generation.
- `scripts/validate_gpt_checkpoint.py` independently restores the saved model
  and checks deterministic token generation.
- `scripts/validate_spot_recovery.py` forcibly kills its own trainer after a
  committed checkpoint, resumes in a fresh process, and compares final model,
  optimizer, and training-state digests against an uninterrupted baseline.

## Reproduction

Preparation runs on CPU, before reserving TPU capacity:

```bash
python -m scripts.prepare_token_pool \
  --output artifacts/fineweb-pool \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --tokenizer-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --train-tokens 131073 --validation-tokens 8193
```

The preparation environment needs `datasets` and `transformers`; these are not
needed by the prepared-data training process itself. The immutable validation
source archive also contains the prepared pool. Its SHA-256 is recorded in
`artifacts/gcp-validation/source-bundle.json`.

On the eight-device host, install `infra/tpu/validation-environment.txt`, the
project's dev/web/data extras, `httpx`, and CPU PyTorch for the reference tests:

```bash
python -m scripts.validate_tpu \
  --output artifacts/tpu-validation \
  --token-manifest artifacts/gcp-validation/fineweb-pool/manifest.json \
  --timeout-seconds 4500
```

## Gates and limitations

1. TPU backend, at least eight devices, exactly one physical host.
2. Real pinned FineWeb: stop at five updates, reopen and finish fifty updates.
3. GPT-2-sized configuration: ten updates at sequence length 128.
4. Independent checkpoint restore and greedy generation.
5. Main pretraining smoke, including immutable-config backend resolution.
6. Entire pytest suite, including TPU-only attention tests; retain all skips.
7. Offline TinyStories tokenizer → training stages → evaluation → inference.

These are execution/recovery gates, not a convergence or quality claim. The additional 1,024-token context smoke checks the intended model shape, but
a sustained throughput/memory pilot is still required before setting a large
training token budget. Multi-host FineWeb is explicitly rejected rather than
silently duplicating the same data on each worker.

Issue #31 additionally needs a prepared-shard smoke in Kaggle's own runtime.
Issue #12 requires physical multi-host, cross-topology checkpoint, and recovery
acceptance. Release #11, matched comparisons #23, publication #19, and umbrella
#1 must not be marked complete based on this single-host validation.

## Results

All eight final acceptance stages passed. The complete physical TPU suite passed
**406 tests with 2 intentional skips and no failures**. The skips are the non-TPU
fallback test and the dedicated virtual-CPU-device job. A separate eight-virtual-
CPU-device run passed all eight sharding/placement checks. Final CPU suite:
**403 passed, 5 skipped**. Ruff, Pyright, and documentation validation passed.

- Real FineWeb: stop at update 5, resume in a new process, complete update 50.
- GPT-2-sized model: 12 layers, width 768, 12 heads, 50,257-token vocabulary;
  ten updates at sequence length 128, checkpoint restore and greedy generation.
- Main pretraining smoke and TinyStories end-to-end acceptance passed.
- Forced interruption: SIGKILL return code -9; resumed process return code 0;
  final model, optimizer, and cursor digests exactly match the baseline.
- Sequence length 1,024: three updates, 24,576 genuine target tokens;
  validation loss 23.7474 → 17.5644. This short check does not establish sustained
  throughput, long-run stability, or useful model quality.

[Every unit/integration test and skip reason](TPU_TEST_INVENTORY_2026-09-21.md)
is listed separately. [Machine-readable run evidence](../benchmarks/results/gcp-spot-v5e-8-20260921.json)
contains exact commands, environment versions, source hashes, stage timings,
losses, recovery results, and cleanup details. Raw logs and JUnit XML are under
`artifacts/gcp-validation/accepted/` locally and in the dedicated GCS bucket.

## Failures found and fixed

| Finding | Fix and evidence |
|---|---|
| FineWeb TPU process performed live HF/PyArrow I/O | CPU preparation plus local checksummed packed token pool; real FineWeb updates pass on GCP |
| No actual FineWeb resume or held-out evaluation | Exact cursor/config-bound resume, periodic validation, fresh-process and SIGKILL recovery |
| Padded tokens inflated counts; exhaustion generated replacement text | Packed true-token targets, EOS supervision, explicit exhaustion failure |
| Splash auto-selection failed on sharded TPU training | Shared exact-XLA fallback for FineWeb, TinyStories, and main pretraining |
| Checkpoint inference put inputs on a different device set than weights | All generation modes follow the model's device set; four placement regressions and physical checkpoint inference pass |
| Frozen-config mutation introduced during implementation | Immutable replacement; new main-pretraining smoke regression passes on CPU and TPU |
| Incomplete acceptance failure reporting | Timeout-bounded stages, dependency-blocked status, persisted summaries, complete JUnit inventory and retained negative runs |

The first two failed physical attempts remain archived; they are not counted
as successful acceptance. No GitHub issue was closed based only on this run.

## Open GitHub issues and remaining work

| Issue | Status after this validation |
|---|---|
| [#31](https://github.com/mlnomadpy/flaxchat/issues/31) | Data path fixed and verified on GCP. Run one prepared-shard smoke in Kaggle's own runtime before closing. |
| [#12](https://github.com/mlnomadpy/flaxchat/issues/12) | Physical multi-host, cross-topology restore, and multi-host interruption recovery still required. FineWeb explicitly rejects multi-host execution. |
| [#11](https://github.com/mlnomadpy/flaxchat/issues/11) | Release remains outstanding. This campaign tested source plus checksummed changes, not a released package. |
| [#23](https://github.com/mlnomadpy/flaxchat/issues/23) | Matched nanochat/MaxText comparison remains outstanding; not required for single-host training. |
| [#19](https://github.com/mlnomadpy/flaxchat/issues/19) | Public checkpoint/demo remains outstanding; follows useful-model training. |
| [#1](https://github.com/mlnomadpy/flaxchat/issues/1) | Umbrella readiness plan remains open. |

Two dependency deprecation warnings remain (Starlette/httpx and AnyIO
BlockingPortal); neither caused a failure. The preparation utility intentionally
creates a bounded pool. Large training corpora need an appropriately sized CPU
preparation job and token pool, with a sustained pilot before committing the
remaining training budget. No large training run was launched.

## Cost, retention, and teardown

The TPU and its queued resource were explicitly deleted and their absence
verified in both attempted zones. CPU data preparation ran locally. The
conservative TPU estimate is **USD 7.60**, using the full on-demand reference
rate of USD 1.20/chip-hour over the entire node lifetime, including provisioning.
Actual Spot charges should be lower; actual billed cost is **not yet available**.
Storage/network charges are additional. This stays well within the USD 50
validation allocation.

Only the dedicated evidence bucket remains. Remote artifacts expire after
30 days to bound storage costs; local reports and test inventory remain in the
workspace. The immutable source archive hash and tested Python-source digest
are recorded because changes were not yet a published Git commit.
