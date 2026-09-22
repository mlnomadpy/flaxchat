# Harness readiness fixes — September 22, 2026

The loss-sanity failure, repeated checkpoint compilation, checkpoint cadence, cloud timeout handling, SSH command replay and coordinator-rank assumptions are fixed. The final source passed **549 physical TPU unit tests**, **all nine four-host recovery stages**, and **both 1K/4K training quality gates**. All three allocations, including the two failed qualification attempts, are deleted; independent final queue and TPU VM listings are empty.

## Changes

- **Tied-head initialization:** standard GPT initializes its shared input/output table at standard deviation 0.02 instead of the untied recipe's 0.8. Tying remains enabled, retaining approximately 124M parameters for depth 12. Untied/extension initialization and restored checkpoint tensors remain unchanged. Source identity still prevents silently resuming with different code.
- **Checkpoint cost:** bounded chunk readers are cached across same-shaped leaves and repeated saves. Canonical integrity hashes, reversed physical meshes and the 1 MiB temporary chunk bound remain intact. Prepared training defaults to 300 seconds after the previous completed commit; rank zero broadcasts the decision. Final/clean-stop updates always save. Explicit `--save-every` is preserved.
- **Cloud lifecycle:** timed-out reads retry within their deadline; provisioning waits for the existing guarded request rather than creating another. Deletion is asynchronous and both queue and worker absence must be verified. An unknown response never counts as deletion.
- **Remote execution:** a unique one-use identity prevents replay on the same worker. Workload status travels in a completion marker over a successful SSH shell exit; missing/duplicate markers and transport errors fail closed. Deliberate SIGKILL is now reported without gcloud restarting training.
- **Host ordering:** fault injection targets the coordination-service rank, which can differ from TPU runtime process order. Acceptance searches every worker log for the unique event; result collection uses actual runtime filenames.
- **CI and scale plans:** prepared-trainer changes require distributed CI. Eight-device/BF16 jobs include the new quality checks. Sustained plans include BF16, rematerialization, chunked loss, warmup and an explicit quality gate. Generated plans are never represented as completed qualifications.

## Verification

Final Python SHA-256: `effcf924ad1122ebdc730ff7f90888f4b9f7aa6a18058e0bb2e613b0421d868c`. Physical source archive SHA-256: `930e59d763fa191bf8e2bcbeccd1531f75729fa57fe2f9610dea703c66056235`. Base commit: `c311b63a7ff035e7539155ae0955b60d8c9de29e` plus the frozen workspace changes.

- **Local full suite:** 547 passed, seven skipped, four accelerator deselections; coverage **78.35%**, all module floors passed.
- **Physical unit suite:** 549 passed, five skipped, four accelerator deselections in **264.19 seconds**, on one four-chip v5p worker. [Every test and skip reason](HARNESS_READINESS_TPU_TESTS_2026-09-22.md) is recorded. CPU fixtures and mocked operational tests retain those classifications even when run on a TPU VM.
- **Additional checks:** 58 targeted tests on eight virtual CPU devices, 66 BF16 regressions, three real two-process CPU tests, and 51 launcher/cleanup/CI-policy checks passed. These suites overlap; do not sum their counts. The clock-skew test forces a worker to request an early checkpoint and verifies that the coordinator controls the commit without deadlock. Lint, Pyright, documentation validation, wheel build and an installed-wheel model smoke passed.
- **Four-host recovery:** probe, DP baseline, coordinator kill, DP resume, FSDP baseline, FSDP clean stop, FSDP resume, CPU/BF16 bridge and four-host restore passed. Model, optimizer and training-cursor manifest hashes match exactly in all three comparisons.

## Training and measured overhead

Both probes used a Spot **v5p-32: four hosts, 16 chips**, depth 12, global batch 16, FSDP 8, BF16, rematerialization and loss chunks of 128. Each completed 110 updates, including 100 after warmup, on pinned FineWeb/GPT-2-tokenizer data. All four workers passed the quality gate. Uniform-token loss is **10.8249**.

| Profile | Tokens | Held-out loss | Invocation | Checkpoint time |
|---|---:|---:|---:|---:|
| sustained-1k | 1,802,240 | 12.0133 → 7.3879 | 99.62s | 44.07s (44.2%) |
| sustained-4k | 7,208,960 | 12.0080 → 7.1499 | 209.84s | 21.88s (10.4%) |

The 1K probe deliberately saved at steps 55 and 110: **22.77s and 21.31s**. The 4K probe used the new time policy and saved once at its final update: **21.88s**. The earlier campaign took about **40–41s per commit** and spent 63.3% of invocation time checkpointing. The new observations are roughly 46% faster per commit, but topology and data differ, so this is not an isolated causal benchmark.

Median post-warmup steps were **123.6ms at 1K** and **1.293s at 4K**. Steady throughput was about **127,139 / 50,606 tokens/s**; invocation throughput including initialization/evaluation/saves was **18,091 / 34,354 tokens/s**. Allocator peaks across workers were approximately **508MB / 596MB per chip**, not full HBM/compiler traces.

The 16-chip slice qualifies multi-host behavior; it is not a cost-optimal default for a 124M model. The earlier two-host steady result was faster at 1K, though its recipe/data differ. Before production, compare the same batch and quality target on a smaller slice. Distributed attention currently falls back to exact XLA because automatic partitioning of Splash kernels is unsupported; explicitly sharded fused attention is the next substantive 4K performance opportunity.

## Cost and cleanup

| Attempt | Request through verified absence | Conservative compute estimate |
|---|---:|---:|
| cloud | 14.21 min | $15.91 |
| cloud-retry | 14.71 min | $16.47 |
| cloud-final | 26.61 min | $29.81 |

Total: **$62.19**, using **$67.20/hour on-demand** for the whole 16-chip slice, including both failed attempts. **This is not the actual Spot bill**, and ancillary charges are excluded. Each attempt retained a $69.20 reservation under a $70 cap, including cleanup allowance and $2 ancillary reserve: $207.60 total reservations, not actual spending. Both successful training invocations alone correspond to about **$5.78** of this conservative compute estimate; most qualification cost was provisioning, tests, recovery checks and cleanup. [Published TPU pricing](https://cloud.google.com/tpu/pricing).

The first attempt stopped before setup following cloud read/deletion timeouts. The second passed 544 unit tests but exposed command replay and worker-order assumptions during fault injection. Both were deleted before the next request. The third passed all gates and was deleted. Cloud guards remained armed throughout; no allocation was left running.

The latest retrieved project billing snapshot, taken before these three allocations, covers **September 16–22**: **$9.22 usage before savings, -$9.22 savings, $0.00 net**. Its Spot TPU lines were $2.95 v5e and $6.15 v5p. It is posted project usage, not a final campaign invoice. Recent usage and promotional credits still require reconciliation; no spending is inferred from a grant's original face value.

## Scope limits

These are bounded correctness, recovery and loss-sanity passes. They do not establish decontaminated downstream quality, multi-day Spot replacement, arbitrary pod-scale performance or a faithful paper reproduction. The primary cost-saving next step is a matched smaller-slice/batch benchmark; further context-scale speedup requires a validated distributed fused-attention path.

[Machine-readable evidence](../benchmarks/results/harness-readiness-fixes-20260922.json) includes every unit result, all training rows, source identities, recovery comparisons and lifecycle ledgers. Raw evidence is under `artifacts/readiness-fixes/`; frozen source and durable checkpoints are in the existing private validation bucket under `readiness-0922c/`.
