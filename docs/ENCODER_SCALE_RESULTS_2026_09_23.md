# Multilingual encoder qualification — September 23, 2026

Status: bounded campaign complete; all campaign TPUs/queues deleted and the
cloud artifact prefix emptied. **Production quality and all scales remain
unqualified.**

## Protocol

The campaign uses the released `jhu-clsp/mmBERT-base` weights, BF16 compute with
FP32 residuals/master weights/Adam state, masked MLM, global batch 64, sequence
length 512, and learning rate 2e-5. Inputs are 15,360 multilingual premises from
1,024 XNLI training examples in 15 languages. This is a bounded continuation and
infrastructure test, not pretraining from scratch. The fixture averages 36.32
non-padding tokens per row (7.09% occupancy). Input positions/second therefore
must not be used as useful-text throughput or to price production pretraining.

Each scale must complete a baseline, rerun from the same initialization, receive
a deliberate SIGKILL after a committed midpoint checkpoint, resume, and reproduce
the final model, optimizer and data-cursor manifests exactly. The independent
verifier checks raw logs/manifests and every physical worker, rather than trusting
a summary's pass flag. Baseline duration excludes compilation and must exceed
120 measured seconds. This qualifies only a bounded run; it is not evidence of
days-long reliability or recovery from an actual capacity eviction.

## Failures uncovered

- v6e-4: two bounded Spot requests in `us-central1-a` and `us-east5-a` timed out
  waiting for capacity. Both were deleted and absence verified. v6e remains
  unvalidated; these are capacity failures, not a model correctness result.
- External SSH to a ready v5e slice repeatedly timed out. Existing IAP access
  worked. The supervisor now supports IAP, bounds each SSH connection to 20
  seconds, and gives setup a separate 300-second default deadline. No IAM or
  firewall permissions were expanded.
- `xla_full`, four v5e chips, batch 64: compilation required about 23.38 GiB of
  temporary HBM versus 15.75 GiB available. No training updates completed.
  The canonical chunked `xla` path completes this baseline; reported peak live
  allocation was about 6.95 GiB. These allocator/HLO measurements are distinct
  metrics and are not an exact memory-reduction ratio.
- Pallas projection: the real-text/released-weight loss parity check failed.
  Absolute loss difference was 0.00648451 versus the existing 0.001 limit;
  global relative gradient L2 was 0.0125314 versus its 0.03 limit. Across the
  137 retained parameter-gradient diagnostics, the largest per-tensor relative
  error was 0.0412144 (`layers[8].attn_norm.scale`); the global limit is not a
  per-tensor guarantee. The loss mismatch
  persists when prediction features are frozen, confirming a projection/loss
  mismatch independently of encoder execution. The cause is not yet proven. The tolerance was not
  weakened and Pallas was not used for the subsequent endurance runs.

## Scale evidence

| Chips | Hosts | Updates | Measured baseline seconds | Input positions/s | Exact recovery |
|---:|---:|---:|---:|---:|---|
| 4 v5e | 1 | 500 | 274.49 | 59,570.5 | Passed |
| 8 v5e | 1 | 1,000 | 460.27 | 71,121.4 | Passed |
| 8 v5p | 2 | 1,000 | 177.65 | 184,270.1 | Passed |

All tested updates were accepted and finite with no dense fallback. For the
four-chip v5e run, the resumed
250 updates took 137.01 measured seconds. Every completed shape reproduced its
final model, optimizer and cursor manifests exactly.

The v5e-16 request also exhausted its 360-second capacity wait; queue and node
absence were verified. A v5p-16 allocation (eight chips, two physical hosts) is
the bounded multi-host fallback. It passed independent verification on both physical workers, including committed
interruption markers and exact model/optimizer/cursor manifests. Baseline masked
target throughput was 1,856.86 targets/s (four-chip v5e: 598.92). Different
generations and horizons prevent interpreting this as a controlled strong-scaling
ratio. The 32-chip request is deferred to preserve the
campaign cap.

For the same-generation v5e comparison, updates 2–250 use the same logged
recipe, global batch, input order and per-step masked-token counts, before either
run has saved its first checkpoint. Four chips measured 59,600.84 input
positions/s; eight chips measured 71,189.34 (**1.194×**). At equal per-chip-hour
pricing, eight chips therefore require **1.674×** the steady compute cost per
input position. This is one matched window per shape on the padded fixture,
not a production cost estimate; setup/compile/save costs are excluded. Four-chip
v5e is the better measured cost choice for this particular chunked-XLA recipe.

## Quality evidence

The released-model frozen-encoder XNLI probe scores **40.85% macro accuracy** on
512 test pairs per language across 15 languages. It fits one fixed ridge head
using 1,024 English training pairs. Test normalization uses training statistics.
The downloaded dataset revision and NPZ bytes are pinned; exact English pair
overlap between these train/test subsets is zero. A separate NFC/whitespace
normalized audit found no test premise or hypothesis matching the MLM training
texts in any of the 15 languages. This is not semantic decontamination; original
pretraining contamination is unknown.

This is not the published end-to-end fine-tuned XNLI protocol. Its score must not
be compared directly with published mmBERT scores. It does not qualify retrieval,
token classification, long contexts, full language/domain coverage, or production
quality. The actual step-500 checkpoint scores **38.62%**, a **−2.23 percentage
point** macro change. Fourteen of fifteen languages declined; Hindi −5.27 pp,
Russian −5.08 pp and French −4.88 pp were the largest drops. This is descriptive
probe evidence, not a significance claim. Both runs use the same frozen evaluator
bundle and four-chip v5e shape. The small-corpus continuation recipe should not
be promoted as a production candidate. Subsequent tuning must use development
data rather than optimizing against this observed test subset.

## Harness improvements

- Physical endurance driver with bounded subprocesses, per-worker uploads,
  committed midpoint interruption and exact final-state comparison.
- Independent verifier rejects missing workers/fault markers, nonfinite or
  rejected updates, dense fallback, incomplete horizons, incorrect compilation
  flags, backend changes, stale final steps, and mismatched raw checkpoint manifests.
- Fixed multilingual probe and comparison tool reject incompatible datasets,
  missing language scores, inconsistent macro scores and changed sample counts.
  Optional predeclared macro/per-language regression limits make future automated
  comparisons fail when exceeded; no limits were retrofitted to this campaign.
- IAP support, bounded SSH/setup, and early failure-cause recording in the
  controller ledger improve operational diagnosis without weakening cleanup.
- Performance reports distinguish padded input positions from masked targets.

The focused local suite passed 91 tests, and the new validation/evaluation
scripts passed type checking and lint. Physical training evidence is reported
separately from local unit-test coverage.

## Coverage limits

| Requested coverage | Outcome |
|---|---|
| v5e, 4 and 8 chips | Bounded endurance and exact recovery passed |
| v5p, 8 chips / 2 hosts | Bounded physical multi-host endurance and exact recovery passed |
| v6e, 4 chips | Two bounded capacity waits expired; unvalidated |
| v5e, 16 chips / 4 hosts | Bounded capacity wait expired; unvalidated |
| v5e, 1 chip | Fixed batch 64 not memory-qualified; no run |
| v5e, 32/64/128/256 chips | Deferred within this campaign budget; 128/256 also require a different global batch or parallel layout |
| Production multilingual quality | Not qualified; the measured continuation probe regressed |

## Remaining work before production

1. Revise the data/optimization recipe on a separate development split. Use
   representative multilingual training text and evaluate locked full task sets,
   including retrieval, token-level tasks, long contexts and target domains.
   The observed probe regression blocks promotion of this continuation recipe.
2. Resolve Pallas real-text loss parity before qualifying it. Retain per-parameter
   gradient diagnostics as well as the global norm; do not relax the current
   loss gate to hide the discrepancy.
3. Profile the memory-safe XLA projection at larger device counts. Its global
   flatten-and-scan structure is a candidate bottleneck; an explicitly local
   chunked scan inside data sharding is worth testing, but is not validated by
   this campaign. Padding/length bucketing also needs a representative workload.
4. Retry v6e and v5e multi-host capacity with bounded requests, then qualify
   larger shapes and longer wall-clock runs. Encoder FSDP and gradient
   accumulation remain unsupported; a fixed batch of 64 cannot cover 128/256
   devices. Actual eviction recovery and days-long stability remain unproven.

The TPU images also emitted a Google CLI Python 3.10 end-of-support warning.
The training environment uses a separate Python runtime; update the image-bundled
CLI/runtime before depending on its long-term support.

## Budget and evidence

The campaign uses Spot only, cloud-enforced deletion guards, and a conservative
$150 reservation cap. The final reserved amount is $144.40. Reservations use
[on-demand rates](https://cloud.google.com/tpu/pricing) ($1.20/chip-hour for
v5e in us-west4 and $4.20/chip-hour for v5p in us-east5) and include failed
capacity attempts; they are not posted charges or consumed credit. The last
observed marketing credit balance was **$860.14**, down $1.71 from the first
$861.85 observation. This movement is not final campaign spend: delayed usage
and earlier jobs may be included. The
separate program credit has different eligibility and is not added to that
balance for campaign planning.

Local evidence is under `artifacts/encoder-scale-0923/`: immutable source bundle
hashes, launch commands, input identities, capacity/cleanup receipts, numerical
diagnostics, worker logs, and independent qualification reports. See
[the protocol](ENCODER_SCALE_QUALIFICATION.md) for acceptance requirements.

## Verified cleanup

All ten attempts have `resource_absent` receipts. Fresh API inventories found
zero TPU nodes and zero queued resources in all three used zones: us-central1-a,
us-east5-a and us-west4-a. Seventy-six cloud evidence objects were retained with
size and CRC32C verification before deleting the campaign prefix; that prefix
now has zero live objects. A separate bucket-wide live listing was also empty.
Ephemeral checkpoint tensor payloads were removed
after evaluation; raw checkpoint manifests, logs and source/input identities
remain local. The existing seven-day soft-delete policy was preserved, so
retained soft-deleted bytes can still incur storage charges until expiry.

## Subsequent repairs

See [the repair campaign](HARNESS_REPAIRS_2026_09_23.md) for the projection
rounding diagnosis, local chunked XLA backend, training recipe controls and new
validation. The failed Pallas and quality measurements in this report remain
historical evidence for their recorded source versions.
