# GOAT base MLM launch — October 6, 2026

Requested: launch training only if the new V-only YAT attention works.

One controller `yat-goat-1006` in `azettaai/us-west4-a` owns this attempt. Its provider-enforced lifetime is 15 hours with DELETE termination; the Spot TPU lease is independently enforced for at most 12 hours. The TPU workflow validates scoped cleanup access before creation. There is no automatic capacity or failure retry.

The frozen `run.json` references the retained, authenticated 1B-token corpus and the public MLM parent at step 48236. The new GOAT stage retains each parent QKV projection's V third and every other compatible weight; it starts a fresh optimizer. No contrastive objective is used.

The gate runs 14 physical TPU cases, then real parent migration, 4 MLM updates, original-parent heldout evaluation, child evaluation, resume to step 8, and evaluation again. Longer training requires all checks to pass, including heldout loss no more than 15% above the original YAT parent. An architecture change is not assumed quality-preserving.

If admitted, continue within the lease using sequence length 512, batch 64 and accumulation 4, with checkpoints every 250 steps (latest 3 retained) and heldout evaluation every 2,000 steps. The 100,000-step schedule is a ceiling; the lease will generally bind first. Repeated exposure to a 1B-token pool is not additional unique data.

Fresh billing observation: $858.63 available under Azetta.ai, expires July 22, 2027. Billing is enabled. Prior conservative campaign reservations total $723.60; this attempt reserves at most $130 including cleanup and ancillary exposure, within the amended $860 campaign cap and original $900 budget. Reservations are not posted spend. The TPU ceiling uses the official us-west4 v5e on-demand price of $1.20 per chip-hour ($9.60 for eight chips); the actual dynamic Spot rate is not verified.

Live durable state:
- `gs://azettaai-yat-eval-0929/goat-mlm-1006/pipeline-status.json`
- Controller evidence: same prefix `/controller/`
- Training/evaluation evidence: checkpoint namespace plus `-evidence/`

Local metadata admission: 25 tests passed, Ruff clean. Physical results must be read from this attempt's current receipts; this document is not a physical pass.

## Corrected attempt B

The first attempt failed before tests or MLM updates: metadata preflight imported JAX in the supervisor and retained the TPU lock. All three first-attempt resources were independently verified absent. The corrected supervisor executes metadata admission in a separate CPU-only process, kills/reaps its process group, and stays JAX-free before physical children. No CPU model computation is performed. Twenty-eight local metadata/orchestration tests pass.

The second source freeze is in `retry-b/run.json`. Controller `yat-goat-1006b`, prefix `gs://azettaai-yat-eval-0929/goat-mlm-1006b`, TPU lease3h, controller provider lifetime6h, workload limit2h after qualification. The schedule remains100,000steps but the lease binds; checkpoints remain250steps/latest3. Reservation41.60USD includingcleanupandancillary, prior853.60 retained, aggregate895.20 under900cap. No physical pass yet.

## Corrected attempt C

Attempt B physically executed14 tests:10passed,4failed. The FP32 comparisons required explicit highest TPU dot precision to match full-precision distance arithmetic; the batch2 MLM fixture was incompatible with an8device projection mesh. Both are corrected without tolerance relaxation. BF16 production arithmetic is unchanged. B resources and controller were independently verified absent.

Failed-run reservations were reconciled only after final cleanup receipts plus later independent empty-zone reads. Original130/41.60 reservations and event histories remain recorded; conservative elapsed-time ceilings retain11.61/9.66 respectively, including full ancillary reserves. These are not posted charges. The latest ledger retains744.87 before a new130reservation, totaling874.87 under unchanged900cap.

Attempt C uses controller `yat-goat-1006c`, prefix `gs://azettaai-yat-eval-0929/goat-mlm-1006c`,12hTPU lease,15hcontrollerproviderdeletion, and10hworkloadceiling after qualification. See `retry-c/run.json` for the exact source/runtime/input freeze. Model execution still requires all physical and quality gates.

## Corrected attempt D

Attempt C passed 13 of 14 physical tests. The remaining exact-resume fixture restored arrays to its single-device target; it now mirrors the trainer's replicated model/optimizer targets and sharded batch. Exact equality requirements remain unchanged. No production MLM updates ran. Queue, TPU, and controller were independently absent after final cleanup. Its elapsed-time reservation ceiling is $11.70, including full ancillary allowance; this is not posted spend.

Attempt D uses controller `yat-goat-1006d` and prefix `gs://azettaai-yat-eval-0929/goat-mlm-1006d`. The unchanged 12-hour TPU lease, 15-hour controller deletion, and 10-hour training ceiling apply. Prior retained reservations $756.57 plus $130 new reservation total $886.57 under the $900 cap. Source and input identities are in `retry-d/run.json`. Training remains conditional on all physical and parent-loss gates.

### Attempt D outcome: physical correctness passed, quality rejected

All 14 physical TPU tests passed (25.11 seconds), including tiny-model exact optimizer/checkpoint resume. The full migrated model completed four MLM updates and saved checkpoint4. On the same 512 balanced-language validation rows and 30,168 masked tokens, original parent loss was **1.2216591398** and GOAT child loss was **8.3892631552** (6.867×). The unchanged 15% regression bound rejected continuation. Full-model step8 resume and sustained training therefore remain unvalidated. This result rejects the direct V-third migration as quality-preserving; it does not establish whether GOAT can learn under a separate adaptation strategy. No long training was launched.

The original public parent remains unchanged. Child checkpoint4 remains at `gs://azettaai-yat-eval-0929/goat-mlm-1006d/yat-embed-torch-goat-1006d/goat-base-mlm/checkpoints`. Further GOAT adaptation requires a distinct bounded recovery plan; do not weaken this gate or describe this child as a production improvement.

Final cleanup: the owned TPU, queue, and controller were independently verified absent. The terminal reservation ceiling is $12.28 including ancillary reserve, not a posted billing charge. All four attempt ceilings are retained with their original reservations and event histories.
