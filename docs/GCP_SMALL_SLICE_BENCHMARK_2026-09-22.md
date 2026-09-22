# Four-chip Spot TPU benchmark and recovery pilot — September 22, 2026

A **Spot v5p-8 (one host, four chips)** is the better measured default for this
approximately 124M-parameter GPT recipe. It reduced estimated invocation compute
cost by **72.5% at 1K** and **82.0% at 4K** relative to the prior four-host,
16-chip v5p-32 run. Both quality gates passed. The full campaign is finished and
independent queue and VM inventories are empty.

## Matched comparison

Both runs used depth 12, tied embeddings, global batch 16, the same immutable
FineWeb token pool, seed 42, AdamW schedule, BF16, rematerialization, chunked loss,
and 110 updates including 10 warmup updates. Checkpoint policies also match:
steps 55/110 at 1K and the 300-second policy at 4K. FSDP changes from 8 to 4 to
fit the smaller topology. This is one observation per context/topology, not a
repeated statistical benchmark.

| Context | Chips | Invocation seconds | Tokens/second including overhead | Final held-out loss | Estimated invocation compute |
|---|---:|---:|---:|---:|---:|
| 1K | 16 | 99.62 | 18,091 | 7.3879 | $1.8596 |
| 1K | 4 | 109.68 | 16,432 | 7.3887 | $0.5118 |
| 4K | 16 | 209.84 | 34,354 | 7.1499 | $3.9170 |
| 4K | 4 | 151.39 | 47,619 | 7.1544 | $0.7065 |

These are conservative estimates using whole-slice **on-demand** inputs of
$67.20/hour and $16.80/hour, based on
[Google's published TPU pricing](https://cloud.google.com/tpu/pricing).
They are **not actual Spot billing**. Invocation costs exclude setup, queueing,
recovery experiments and cleanup. The measured loss deltas are 0.00086 at 1K and
0.00451 at 4K; different reduction ordering can produce different trajectories.

On four chips, median post-warmup steps were 82.16 ms at 1K and 762.27 ms at 4K;
steady throughput was 198,278 and 85,849 tokens/second. The 1K invocation spent
56.42 seconds checkpointing, versus 44.07 seconds on the larger slice, explaining
why faster training steps did not yield a faster whole 1K invocation. The 4K
checkpoint took 25.09 seconds. These intentionally short qualification runs
amortize checkpoint/compilation overhead poorly.

## Longer pilot and recovery

The pilot used the same model and global batch at 1K for **400 updates / 6,553,600
tokens**, with a fixed 400-step learning-rate schedule. The uninterrupted run
reached held-out loss **6.49758**, from 12.01328. A separate run was deliberately
SIGKILLed immediately after committing step 200, then restored in a fresh process
and completed step 400.

The final canonical hashes match **exactly** for all three state groups:
model, optimizer and training cursor. The resumed final loss is also 6.49758.
This validates process interruption after a committed checkpoint; it is not a
real Spot eviction/replacement or a multi-day availability test. Multi-host
fault handling was separately validated in the
[previous campaign](HARNESS_READINESS_FIXES_2026-09-22.md).

## Evaluation and contamination checks

The CPU overlap audit scanned all 7,208,961 training tokens and 65,537 validation
tokens. It found **zero exact shared 50-token spans**. All 63 nonoverlapping
1,025-token validation blocks remain eligible. The scan verifies hash matches
against token IDs and preserves matches crossing training shard/chunk boundaries.
It does not rule out shorter, edited or semantic duplicates.

The downloaded model-only checkpoint passed integrity verification and local
CPU evaluation:

- Eight contiguous audited holdout blocks: **8,192 scored tokens**, mean loss
  **6.05470**. This differs from the training validation loss because it uses a
  different subset and block layout; do not compare the two as a quality gain.
- A deterministic sample of **32 pinned ARC-Easy test questions**, zero-shot
  mean continuation likelihood: **6 correct / 32 = 18.75%**.
- Wilson 95% interval: **8.89%–35.31%**; random-choice baseline is 25%.
- No sampled question was excluded for exact 13-token training overlap or
  context length. Questions and answers are audited individually.

The sample does **not establish above-chance downstream ability**. This is a
small evaluation diagnostic, not the official full ARC protocol, a full CORE
result, or a faithful reproduction of the requested paper. Data revision,
indices, content hash, individual scores and exclusions are recorded in the
[machine-readable report](../benchmarks/results/gcp-small-slice-20260922.json).

## Budget and cleanup

One guarded attempt was made. The supervisor reserved **$21.60** under the
**$25 cap**: a 40-minute attempt plus a 30-minute cleanup allowance at the
conservative whole-slice rate, with a $2 ancillary reserve. The reservation is
not spending.

Actual request-through-verified-absence time was **960.38 seconds (16.01 minutes)**,
corresponding to **$4.48 conservative compute**, including setup, both matched
benchmarks, all pilot/recovery work and cleanup. Ancillary charges are excluded;
actual Spot billing remains pending. Local CPU evaluation added no TPU allocation.
Independent final queued-resource and TPU VM lists both returned `[]`.

The source archive and raw campaign evidence are stored under the existing
project bucket prefix `small-slice-0922/`. That bucket's project membership and
non-public IAM bindings were verified before upload. Its existing 30-day
lifecycle applies. Checkpoints and source data remain private.

## Harness improvements and CI follow-up

- Scale plans accept a fixed global batch independently of device count and
  reject batches that cannot shard across the selected devices.
- A comparison CLI rejects recipe/data/tokenizer/budget/cadence mismatches and
  failed quality gates before reporting per-invocation or per-million-token cost.
- New exact-span audit and bounded prepared-checkpoint evaluation CLIs record
  identities, exclusions, scores and confidence intervals.
- CI scope selection includes these new tools.
- GitHub's previous Linux run exposed a WebSocket race: a background receiver
  could consume a disconnect during the final `done` send, then the outer loop
  attempted another receive. The fix preserves consumed events and next requests;
  deterministic regression cases cover both events. The disconnect test was
  also verified to fail against the original implementation.
- The previous Pages job could not write the revision marker into the
  container-owned build output. The marker now enters `docs/` before Jekyll
  copies it into the published site.

The final local CPU suite passed **571 tests**, with seven skips and four
accelerator deselections. All **52 focused evaluation/comparison/planner tests**
and **21 WebSocket/workflow regressions** passed (overlapping counts). Lint, type
checks, documentation checks and package build passed. Results are recorded
with the campaign evidence. Coverage-instrumented pytest aborted
locally before producing a report; no new local coverage percentage is claimed.
The normal CPU test suite and focused regressions pass.

Training used base commit `4d7b79b8b9cce8a00bfb8676c41e4e440a715974` plus the
fixed-batch planner and its tests, frozen before the later evaluation/CI fixes.
Python source SHA-256:
`1f7229cfaf413a28b526f31a4cffd657543b674b85d85bb1ad2d9d9eb24d76df`.
Archive SHA-256:
`720e66299ba8ae761b4d6243ea35eb538b98220c12ea029bd92283aea081abf5`.
The training code and runtime match the prior benchmark; the source identity
changes because the planner/tests changed. New evaluation and CI code has local
validation, not a claim of inclusion in the frozen TPU source.

## Next work

Use the four-chip slice for this recipe. Keep the default time-based checkpoint
cadence for longer runs; the forced 55/200-step saves here served measurement
and fault injection. Before a larger training budget, reconcile posted Spot
charges, choose a stronger preregistered quality target and expand evaluation
with document/near-duplicate decontamination. Investigate a colocated checkpoint
bucket and fused distributed attention with measured comparisons. Full paper
reproduction still requires a faithful architecture, optimizer, data and
schedule implementation; this campaign does not supply that equivalence.
