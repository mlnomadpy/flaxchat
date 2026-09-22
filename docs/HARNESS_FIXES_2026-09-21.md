# Harness fixes and validation — 2026-09-21

This implements the correctness and infrastructure findings in [the harness review](HARNESS_REVIEW_2026-09-21.md). Changes remain in the working tree. No GitHub issue has been closed, release published, or long training run started. Kaggle is excluded.

## Accepted validation

Final Python source digest: `14c688f0d5077d960d7c50620eab05bd08dc264660e0cd78f7cd5f728f4a4148`. Archive SHA-256: `dc53797e6a00e82048d9e66203fbf36b9b898cd944379ec6193f202e9a8445f9`.

- **503 physical TPU unit tests passed, 4 intentional skips** on four chips. All skips have complementary CPU/virtual-device coverage.
- **All 9 physical multi-host stages passed** on two hosts/eight chips in 483.5 seconds: topology, baseline, worker SIGKILL, resume, FSDP baseline/stop/resume, single-host bridge and multi-host restore.
- Model, optimizer and training-state manifests match **exactly** for interrupted DP recovery, FSDP recovery and topology transfer.
- Final-source local checks: 498 passed/9 skipped; 36 BF16 regressions; 25 eight-device checkpoint/accumulation tests; 8 additional eight-device sharding tests; 2 two-process tests. These counts overlap.

The training recipe uses Muon, two-way accumulation, BF16, chunked loss and rematerialization. The single-host topology bridge uses CPU/BF16 and is explicitly distinguished from physical TPU execution. [Complete unit-test inventory](HARNESS_TPU_TEST_RESULTS_2026-09-21.md).

The original four TPU numerical assertions are resolved: dtype-specific scan loss bounds, FP32 initialization rounding allowance, independent analytic accumulation references and BF16 gradient-norm bounds. Measured accumulated/full-batch gradient relative L2 error is 0.556%, under 2%. Near-zero scan leaves retain the original absolute error allowance. Strict FP32 comparisons remain.

A separate **12-layer GPT-2, cross-host FSDP=8 smoke passed**: 8,192 tokens in four updates, global batch 16, sequence length 128, Muon, accumulation, BF16, chunked loss and rematerialization. The final checkpoint committed successfully. Invocation time was 91.4 seconds, including 35.3 seconds for first-step compile/execution. This short run does not establish sustained throughput, convergence or quality-per-dollar improvements; larger pod scales remain unvalidated.

**Cleanup and cost:** both fix-validation TPUs and queues are confirmed deleted; no backup watchdog remains and the final cloud guard completed successfully. Conservative compute estimate across the two fix-validation allocations: **$38.96** ($12.40 initial + $26.56 final), below the $60 reserve. This uses the on-demand planning rate as a conservative estimate; actual Spot billing is unreconciled and ancillary charges/older campaigns are excluded. The $900 total ceiling remains in force.

## Implemented changes

| Review finding | Implementation | Verification / remaining scope |
|---|---|---|
| R1: Muon schedule and silent fallback | Muon evaluates its schedule using its persistent optimizer count. All parameter groups honor zero LR. Unsupported optimizer initialization now fails instead of silently switching to AdamW. Training and portable restore share the optimizer recipe. | Zero-LR all-parameter regression, schedule-count regression, exact Muon checkpoint recovery. |
| R2: incorrect token accounting | Per-device batch size now expands to the correct process-local row count. Both ordinary and accumulated batches become global arrays. Main pretraining asserts actual array size equals the declared update token budget. | Batch-placement tests, eight-device smoke and accumulated prepared-data recovery. |
| R3: divergent production/recovery paths | Shared placement, finite-update, accumulation, initialization and optimizer helpers. Prepared-data trainer supports both AdamW and Muon plus accumulation. Main pretraining honors checkpoint settings, checks resolved identity and source digest, requires shared GCS for multiple hosts and saves every rank's loader state. All ranks participate in evaluation/sampling. | Prepared token pools retain portable global cursors. Legacy parquet resume deliberately requires the original process topology; its unsupported parameter-parallel settings fail clearly. |
| R4: checkpoint memory and duplicate reads | Global sharded leaves are hashed in canonical chunks of at most 1 MiB rather than gathered whole. Restore uses target shardings, verifies stored dtypes, reconstructs containers in memory and performs no second storage read. Prepared trainer initializes model/optimizer directly into target shardings. | Existing corruption/topology tests plus real two-process sharded checkpoint round trip. Large-model peak HBM still needs physical profiling. Hash format remains compatible. |
| R5: Winogrande protocol | Score the suffix conditioned on each filled-in prefix, use summed likelihood, record token boundaries and bump the prompt protocol version. | Golden rendering/scoring regression and existing evaluation tests. Full external harness score parity is not claimed. |
| R6: cost and lifecycle controls | Wait before allocation; finite positive budgets; explicit whole-slice rate; budget timer armed before provisioning; default teardown; one lifecycle for run-once; cleanup failures surface. Added durable Cloud Workflows expiry controller and live execution receipt checks. Physical acceptance now requires an active cloud cleanup receipt. | Launcher timing tests and deployed controller expiry test. Local timers alone are not a guarantee against client loss. Cloud deletion can be delayed or fail; controller verifies absence and fails visibly after bounded retries. |
| R7: actual BF16 compute | Linear/embedding compute dtype is explicit. Master parameters remain FP32. Tied output projection casts weights for compute in both full and cached inference. | Projection JAXPR regression and BF16 value/gradient tests. No TPU speedup claim yet. |
| R8: scan combinations | Standard GPT no longer accesses extension-only scalars. Value embeddings are computed inside the chosen layer branch. Scan/rematerialization combinations work. Removed the incorrect constant compile-size claim. | Standard/extended × loop/scan × remat value-and-gradient comparisons. BF16 comparisons use dtype-appropriate tolerances. |
| R9: data loss and identity | Packing retains document tails and versions the changed packing policy. Legacy data identity hashes file contents. Tokenizer identity includes serialized rules or vocabulary. Prepared pool construction offers deterministic bounded document shuffling with seed recorded. | Data/resume tests. Existing validation pools are immutable and were not regenerated. Dataset curation/mixture quality remains an experiment, not a proven gain. |
| R10: loading/compile overhead | Host prefetch snapshots consumed data/cursors in both trainers. EOF queues can shut down without hanging. GCP workers use a persistent compilation-cache directory. Setup offers a lean training profile without web/dev/Torch dependencies. | Backpressure/error/EOF/cursor tests and exact prepared-data recovery with prefetch enabled. Full image/transitive dependency reproducibility still requires recording the installed environment. |
| R11: full vocabulary loss memory | Opt-in token-chunked loss projects one chunk at a time and rematerializes it during backward. It preserves softcap, masks, padded vocabulary and tied/untied gradients. Inference still returns full logits. | FP32/BF16 loss-and-gradient equivalence, partial final chunk, all-masked targets, exact checkpoint recovery with chunking. Physical memory/throughput benefit is unmeasured. |
| R12: remat and unsupported sharding | Rematerialization is a resolved model option; the prepared trainer exposes it. Unsupported legacy FSDP/tensor settings fail instead of being ignored. Existing explicit XLA fallback remains for unsupported multi-device Splash. | Remat/scan tests and sharded initialization tests. Shard-local Splash and tensor parallelism are not newly implemented or claimed validated. |
| R13: misleading metrics | Correct `TPU v5` runtime alias for v5p. Main training timing includes short runs, and reports newly committed tokens and loop goodput. Prepared trainer reports invocation time and end-to-end committed-token throughput separately from steady step throughput. | Alias test and trainer summaries. Cloud billing and a sustained production benchmark remain separate evidence. |
| R14: SFT/RL safety and recovery | Finite-update rejection preserves both model and optimizer. Load architecture from checkpoint metadata. Persist/check data, tokenizer, schedule and update cursor. Exact single-host resume is implemented. RL decay counts actual optimizer updates; rollout training shapes are fixed; checkpoint steps count completed work. Unsupported multi-host SFT/RL fails explicitly. | Real offline SFT/RL uninterrupted-versus-resumed checkpoint equality. Packed conversation isolation and batched rollout generation remain optional future optimizations. |

Additional decoding fixes cover standard GPT, nonzero smear weights, value-embedding lookup and disabling top-k. Cached generation is compared against the full model after nonzero parameter changes.

## Local evidence

The initial frozen-source checks passed:

- **495 CPU tests**, 8 skips. Four skips require a physical TPU; the other four have complementary virtual-device or two-process coverage.
- **67 tests on eight virtual CPU devices**, including Muon accumulation, FSDP checkpoint transfer, main pretraining and SFT/RL recovery.
- **31 BF16 CPU tests**, including mixed-precision values/gradients and recovery.
- **2 real two-process CPU tests**, covering initialization, collectives, canonical data and sharded checkpoint integrity.
- Ruff, documentation checks and `git diff --check` passed.

Raw logs and all 503 unit-test inventory entries are recorded in `artifacts/harness-fixes/` and [the machine-readable report](../benchmarks/results/harness-fixes-20260921.json). These suites overlap; their counts should not be added as unique tests.

The initial source Python digest is `9960c2ff2671a09c64c7d6d9f38c525dfad3d51aeaef16db2a9c4186688fe3c8`; source archive SHA-256 is `a655eaf38adea0a8fd19ba92c5d1ee627de89b2008e51ddf8b80f70db2dc6a3d`. The base Git commit alone does not identify these uncommitted changes.

## Cloud controller and TPU gate

Deployed `flaxchat-validation-cleanup` in `tpubuilders/us-central1`, revision `000001-b36`. Its dedicated account has only `tpu.nodes.get` and `tpu.nodes.delete`, conditionally limited to `flaxchat-validation-*` nodes/queues in `us-east5-a` and `us-central2-b`. Google's queued-resource deletion API uses the node-delete permission and `force=true` deletes the associated nodes too. [API documentation](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/delete).

Execution `21e7cc0c-447b-4648-8bcc-46729b63e613` waited until its deadline and succeeded with `Resource absent` against a deliberately nonexistent validation queue. This proves deployment, durable waiting and API access; it does not prove deletion of an allocated slice or controller-failure recovery on a real workload.

## Initial physical failures and resolution history

The approved source (`9960c2ff…fe3c8`) ran on Spot v5p-16: two physical hosts and eight chips. Topology, canonical data placement, Muon baseline, deliberate worker interruption and resumed training stages passed. The campaign then failed at the first FSDP checkpoint: its bounded integrity-hash gather reconstructed a mesh in `jax.devices()` order, conflicting with the TPU's topology-aware input mesh. Later stages and final exact-state comparisons did not run; that initial multi-host attempt failed; the final accepted rerun above passed.

The local fix preserves the leaf's `NamedSharding` mesh and also bounds hashing for locally addressable sharded leaves. A regression covers a reversed eight-device mesh, a partial final hash chunk and an empty array. Updated source `9312a2e47caa9cb87be0fcccf8451db863a9123b4b177c561bbd6e088bcec185` passed 495 CPU tests (9 skips), 18 checkpoint tests on eight virtual devices, and 2 real two-process tests. Ruff and diff checks passed. Updated archive SHA-256: `89da723058fa4926df72f00f92f38f42212461349637f23aae811e1774d57056`.

The complete original-source unit suite on four physical TPU chips produced **495 passed, 4 failed, 4 skipped**. All cases, failure messages and skip reasons are in `artifacts/harness-fixes/tpu-unit-inventory-initial.json` and the machine-readable report. Assertions found in the initial run (resolved by the accepted validation above):

- Two extended-model scan/loop loss comparisons differ by approximately 1.17e-5 under BF16; their FP32 diagnostic passes.
- Sharded/eager initialization differs by up to 1.79e-7 in FP32; the current test tolerance is too tight for those observed values, but has not been silently relaxed.
- BF16 microbatch/full-batch gradient comparison exceeds its existing elementwise tolerance. The FP32 diagnostic passes. This motivated the independent analytic reference and recorded numerical-error bound in the final accepted tests.

The initial FP32 diagnostic returned 5 passed and 1 failed (initialization). These results were retained as failure evidence; the accepted final-source suite and recovery campaign above supersede their status.

**Initial allocation closed:** the first TPU and queue were deleted, the backup watchdog stopped and the waiting guard cancelled. After the user explicitly requested all remaining fixes, the corrected archive uploaded successfully and a new guarded Spot validation request was created.

Intermediate source revisions and a bounded scan-only timeout remain in the raw logs. The final accepted suite and multi-host campaign use the same final source digest, verified on both hosts.

The real deletion execution `d901fdfa-3a49-4389-9a8a-d3c358d2e85f` succeeded with `Resource deleted`; independent node/queue listings were empty. Conservative compute estimate for this attempt: **$12.40**, using 22.1 minutes from node creation through confirmed absence at $33.60/hour. This excludes storage/network/Workflow charges and previous attempts. The planned reserve was $60 within the original $900 ceiling. Actual billing has not been reconciled. Spot prices vary; conservative estimates include allocation/setup/deletion overhead and use the whole-slice on-demand planning rate, not an invented Spot discount. [Google TPU pricing](https://cloud.google.com/tpu/pricing).

## Open GitHub work

All six previously open issues remain open. The fixes support [#12 physical acceptance](https://github.com/mlnomadpy/flaxchat/issues/12) and [#1 readiness](https://github.com/mlnomadpy/flaxchat/issues/1), but closure requires reviewed/published evidence. [#23 matched comparisons](https://github.com/mlnomadpy/flaxchat/issues/23), [#11 release](https://github.com/mlnomadpy/flaxchat/issues/11), and [#19 public checkpoint/demo](https://github.com/mlnomadpy/flaxchat/issues/19) require their respective benchmark, release or training/publication work. [#31 Kaggle](https://github.com/mlnomadpy/flaxchat/issues/31) remains excluded.

Research additions such as quantization, new optimizers, curriculum/model averaging, shard-local Splash and packed SFT are not silently enabled. They require quality-matched ablations and measured cost improvement; the correctness fixes do not establish paper-reported gains on this system.
