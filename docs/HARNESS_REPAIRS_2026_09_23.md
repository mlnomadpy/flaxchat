# Harness repairs — 2026-09-23

This follow-up implements repairs to the audit and scale-campaign findings. It
keeps production multilingual quality separate from infrastructure correctness.
Detailed, locally retained evidence is under `artifacts/encoder-fixes-0923/`.

## Implemented repairs

| Finding | Change | Verification |
|---|---|---|
| Pallas loss differs on released mmBERT weights | Explicit FP32 projection output for BF16 operands in the canonical, local XLA and Pallas implementations | Physical root-cause diagnostic; repaired forward error 0.00001097 versus the earlier 0.00648451 |
| Custom backward rounds FP32 logit cotangents prematurely | Retain FP32 cotangents through transpose matmuls; cast only returned operand gradients | CPU interpreter loss/gradient tests, including multiple token/vocabulary tiles; final TPU result recorded below |
| Accurate Pallas backward exceeds v5e VMEM | Separate backward token tiling from forward tiling; cap backward tiles at 128 rows | Regression covers 128, 256, 512 and 1,024 rows; physical result below |
| Global chunk scan scales poorly | Optional `xla_local` scans within each data device, rematerializes logits and globally normalizes selected-token sums | Four-device CPU comparisons and physical released-weight loss/all-parameter-gradient gate |
| Large full projection exceeds v5e HBM | Memory-bounded local chunked alternative; keep `xla_full` an explicit diagnostic choice | Actual training/recovery and device-memory evidence recorded below; no claim that full projection now fits |
| Missing training recipe controls | Mask-count-weighted accumulation, accepted-update cosine/warmup schedule, deterministic epoch shuffling | Exact-resume tests, unequal/empty microbatch tests and two-process interruption/recovery |
| Document tails discarded | Optional document windows, preserving tokenizer special-token processing on each window | Tail-preservation regression, including Tokenizers 0.23 early-truncation behavior |
| Padded throughput overstates useful work | Nonpadding token counts and throughput in training logs; preparation occupancy statistics | Data-preparation and training tests |
| Pretrained validation occurs too late | CPU shard/header/name/shape/content hashing, bounded nonfinite scans and tied-decoder checks before model construction | Missing/corrupt/wrong-shape/nonfinite/untied snapshot regressions |
| Evaluation accepts invalid prepared data | Shared prepared-data validator for training and evaluation | Checksum-consistent invalid vocabulary, dimensions, special-token policy and format tests |
| Resume silently accepts a stop before its cursor | Cursor/update bounds, contradictory-stop rejection and explicit completed/no-op event | Direct malformed-cursor and real checkpoint recovery tests |
| Older recovery aggregate trusts summary booleans | Require raw model/optimizer/cursor manifests, hardware probe, committed-fault marker and successful controller receipt | Missing/mismatched raw evidence and controller-failure regressions |
| Benchmark comparison trusts passed flags | Verify raw log hash, complete updates, physical runtime record and committed checkpoint/source identity before declaring completion | Missing/tampered logs, stale checkpoint, incorrect backend, missing source and inflated throughput regressions |
| Projection campaign subprocess/upload deadlines | Shared monotonic/wall deadline covering kernel checks, children and evidence uploads | Expiry, system-sleep and hung-upload tests |
| Workload CLI uses aging image Python | Use setup-created Python 3.12 for workload `gcloud`; record/check its interpreter | Setup assertion plus physical checkpoint transfers; initial bootstrap still uses image CLI |

The original loss/global-gradient limits remain 0.001 and 0.03. The released-weight
backend validator additionally enforces a 0.03 **per-parameter** relative-gradient
limit; a small global norm can no longer conceal a failing tensor. The initial
forward-only repair passed the old limits but failed this new stricter check
(worst parameter 0.03852), so that intermediate Pallas build remains unqualified.

## Numerical root cause

The frozen-feature diagnostic compared 131,072 logits from actual mmBERT weights.
All Pallas logits exactly equaled the BF16-rounded XLA logits plus FP32 bias;
none equaled the corresponding original XLA logits. Maximum absolute difference
was 0.06069946. XLA's original cast-down/cast-up expression equaled the explicit
FP32-output projection on this TPU. The implementation now requests FP32 output
explicitly instead of relying on different compiler treatment of that expression.
The custom backward also follows the FP32-output derivative: its cotangent must
not be prematurely rounded to BF16.

The operand and accumulation distinction follows the
[JAX TPU Pallas precision documentation](https://docs.jax.dev/en/latest/pallas/tpu/details.html).
Source identity prevents silent resume across the precision change. Existing
checkpoints are not rewritten or declared compatible without validation.

## Local verification

- Full CPU suite: **833 passed, 19 skipped, 5 accelerator tests deselected**.
- Coverage: **79.69%**; every configured module floor passed. Encoder: 85.19%;
  fused projection: 95.65%; training: 87.96%.
- All four opt-in two-process CPU interruption/recovery variants passed, including
  local XLA with accumulation, warmup/cosine and shuffling.
- Four simulated CPU devices: projection loss/gradient comparisons cover uneven
  mask counts, empty devices, padding, dense/masked execution and overflow.
- Final focused regressions: 67 policy/snapshot/scale tests, 40 benchmark/deadline tests, and 31 projection tests on four simulated devices passed.
- Lint, configured type checking and documentation checks passed.

The window regression caught Tokenizers 0.23 omitting overflow during early right
truncation. Window preparation now fully tokenizes, splits explicitly, and applies
the tokenizer postprocessor per window. This avoids relying on the behavior
reported in [the upstream issue](https://github.com/huggingface/tokenizers/issues/2091).
The CPU suite above includes the correction.

## Physical verification

The four-chip local-XLA released-weight gate passed: loss error 0.000000477,
global relative gradient error 0.00996, maximum per-parameter error 0.02016.
The four-chip local-XLA run completed 1,000 accepted updates in 349.57 measured
seconds (compilation excluded), averaging 93,644.78 input positions/sec. Its
interrupted/resumed final model, optimizer and cursor inventories exactly match
the uninterrupted run. Raw worker evidence passes independent verification.
The composite controller correctly remains failed because its earlier
forward-only Pallas experiment failed the stricter per-parameter gate.

For updates 2–250, local XLA achieved 93,816.39 input positions/sec, versus
59,600.84 in the earlier canonical run: 1.574× throughput, or 36.47% lower steady
compute cost per position at equal chip price. Mask counts match exactly. This
is a before/after **source-version** comparison, not an identical-source A/B test;
the precision contract and logging also changed. Useful nonpadding throughput
was 6,672.99 tokens/sec in that window. Peak live allocation was about 6.95 GiB.

The physical four-chip recipe test also passed six updates with accumulation=2,
cosine scheduling, one warmup update and deterministic shuffling. Killing after
committed update 3 and resuming produced exact final model, optimizer and cursor
inventories. A separate local check verified the raw manifests and fault marker.
This is a tiny-model recipe correctness test, not a multilingual quality result.

The attempted identical-source eight-chip run became unavailable before setup;
no training result was produced. Its deletion was verified. This does not
establish the cause as a Spot eviction or qualify the new backend on eight chips.

The corrected FP32 Pallas backward first exceeded the v5e 16 MiB VMEM limit:
16.43 MiB in the released-model gate and 19.47 MiB in the 1,024-row kernel test.
Backward token tiles now use 128 rows independently of forward tiling. The final
single-chip v5e run passed **all three 128/512/1,024-row kernel checks** at hidden
size 768 and vocabulary tile 1,024, testing both 4,097 and 256,000 vocabulary
entries. Worst relative gradient error across those checks was 0.00010266,
below the unchanged 0.015 kernel limit. Median forward/backward times were
4.14, 16.70 and 32.92 ms respectively; these are isolated projection measurements.

The final released-mmBERT gate also passed on that physical chip: loss error
**0.000012875**, global relative gradient error **0.008070**, and maximum
per-parameter relative error **0.025243**, below the stricter 0.03 gate. The
controller recorded successful workload completion. Evidence is in
`pallas1-evidence/`, with frozen source identity in `fixes-v5.json`.
This qualifies those numerical shapes on one v5e chip; the repaired Pallas
implementation still needs sustained multi-device training/recovery qualification.

## Remaining qualification limits

These changes do not establish production model quality or mmBERT reproduction.
The prior small XNLI continuation regression remains an observed failure; it is
not erased by a numerical or throughput repair. Representative multilingual
pretraining data, development-only recipe selection, untouched downstream tests
and predeclared per-language acceptance criteria are still required.

Encoder FSDP, streaming language-mixture sampling, document packing, days-long
stability, actual Spot eviction/replacement, v6e and all large slices remain
separate work. Gradient accumulation does not allow a global microbatch smaller
than the device count. The observed test subset must not be reused to tune the
recipe. Default training remains conservative; new recipe controls are opt-in.

## Budget and cleanup

This repair campaign has a **$40 conservative reservation cap**, independent of
the completed prior campaign. Reservations are not posted spending. Each resource
has a bounded workload, separate setup deadline and independent cloud cleanup
lease. New allocations wait for verified deletion of the previous node and queue.
The five attempts reserved **$38.60** including cleanup time and ancillary buffers;
this is not their actual cost. Applying conservative hourly ceilings to their
complete request-to-verified-deletion windows gives a **$5.23 compute estimate**,
including provisioning/deletion time, excluding storage/network. It is not posted
billing or measured billable duration; see `cost-estimate.json`.
At 16:31 UTC, the Billing Credits UI displayed
**$859.99** in the marketing credit expiring September 28, 2026. Billing may lag;
the separate program credit is not combined with this balance.
All five controllers verified their nodes and queues absent. A separate inventory
at 16:48 UTC found zero TPU nodes and zero queues in `us-west4-a`, `us-east5-a`
and `us-central1-a`. At 16:49 UTC, storage cleanup verified **57 local evidence
copies** by CRC32C/size before deleting temporary objects; the campaign prefix
has **zero live objects**. The existing seven-day soft-delete policy is unchanged,
so retained deleted objects may still incur storage charges during that period.
Receipts: `final-compute-inventory.json`, `retained-evidence-inventory.json` and
`storage-cleanup-receipt.json`.
