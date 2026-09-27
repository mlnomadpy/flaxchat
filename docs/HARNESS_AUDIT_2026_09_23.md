# Harness audit — 2026-09-23

## Verdict and scope

The harness supports bounded, supervised experiments and has substantial correctness
and recovery coverage. It is not yet qualified for a production multilingual encoder
pretraining campaign or arbitrary TPU scales. Small execution tests, exact recovery,
released-weight parity, throughput, and downstream quality are distinct claims.

This audit reviews the current working tree based on commit `011125f`, including
uncommitted qualification changes. It covers encoder training/data/evaluation,
causal training and shared checkpoint/runtime code, TPU launch/budget orchestration,
benchmark acceptance, CI, and retained physical evidence. No cloud resources were
allocated and no production code was changed during the audit. Kaggle-specific
work is outside this audit. This is not a new physical TPU certification, current
billing audit, dependency vulnerability scan, or literature review.

## Findings requiring correction

### A1 — P1: pretrained snapshot integrity is checked too late

Evidence: `scripts/train_encoder.py`, lines 124–135. Worker agreement and
`--preflight-only` complete before `load_pretrained` checks or hashes weight files.
The subsequently computed `initial_weights_sha256` is not included in the worker
agreement. The encoder's identity also lacks the runtime fingerprint already used
by `scripts/train_gpt2.py`.

Reproduced locally: a snapshot containing only the real mmBERT `config.json` and
`tokenizer.json`, with no weights, passes input preflight. A missing snapshot can
therefore consume setup/allocation time before failing. Different snapshots or
runtime versions on workers are not rejected by the explicit agreement check;
this is a missing guard, not evidence that prior runs used different weights.

Required fix: CPU validation of shard inventory, safetensors headers, shapes and
hashes before provisioning; pin and compare that inventory and runtime across
workers before model placement. Verify missing/truncated shards and cross-worker
weight/runtime disagreement. Keep resume identity separate from an unused initial
snapshot when restoring a full checkpoint.

### A2 — P1: benchmark acceptance is weaker than qualification acceptance

Evidence: `scripts/benchmark_encoder_projection.py:30` and its campaign success
assignment around line 180. `summarize_steps` requires only some post-warmup records
and truthy `updated` values. It does not enforce the requested 55-step horizon,
ordered unique steps, finite loss/timing, nonempty masks, expected token counts,
or a final checkpoint. The comparison script trusts the resulting `passed` flags.

Reproduced locally: six records with NaN losses, zero masked tokens and no
checkpoint produce a throughput result with one measured step. Current trainer
checks prevent some malformed outputs, but the acceptance layer is not independent
of those assumptions.

Required fix: a common strict record validator parameterized by requested horizon,
warmup, batch and context. Require committed checkpoints and hardware/source
identity, and explicitly distinguish dense fallback from the intended backend.
Add malformed, truncated, stale and duplicate-record tests before publishing more
cost/performance claims. This does not establish that historical measurements are
wrong; it establishes that malformed evidence can pass the reusable gate.

### A3 — P2: recovery aggregation does not independently prove interruption

Evidence: `scripts/summarize_encoder_validation.py:33`. It checks stage success
booleans and three equality booleans in `recovery.json`, but does not consume the
interruption marker/log, raw baseline/resume manifests, hardware log, or controller
completion receipt. Hashing a summary preserves its identity, not the validity of
its assertions.

Reproduced locally: the saved physical `multixla7` campaign still aggregates as
passed after copying only summary, recovery booleans and update logs into a new
directory, omitting interruption logs, hardware logs and manifests entirely.

Required fix: collect and verify raw recovery manifests and committed-fault evidence
for every worker, plus successful controller completion. Fail for missing,
inconsistent or mixed-run evidence. Keep the existing worker and update-log hashes.

### A4 — P2: benchmark cloud operations remain unbounded

Evidence: `scripts/benchmark_encoder_projection.py:118` uploads without a timeout;
`validate_projection_campaign.py` uses unbounded `subprocess.call`. The campaign's
`max_seconds` check occurs only before training cases, uses wall-clock time, and
does not constrain each child to the remaining allowance. Kernel/model checks
also execute before that budget check.

The recently added cloud-call deadlines apply to `validate_encoder_tpu`, not these
entry points. The external supervisor/cleanup lease still bounds the allocation;
this is an inner-workload deadline gap, not a demonstrated unlimited TPU leak.

Required fix: shared monotonic deadline, remaining-time child timeouts, bounded
uploads, and explicit partial-campaign status. Test hung uploads and deadlines
expiring during kernel qualification.

### A5 — P2: CPU isolation does not clear all supported distributed launch metadata

Evidence: `scripts/validate_encoder_tpu.py:15` removes JAX/TPU variables but leaves
`SLURM_NTASKS`, `OMPI_COMM_WORLD_SIZE`, and `PMI_SIZE`. `flaxchat/common.py` uses those
variables to trigger distributed initialization. The frozen-archive preflight
already clears them.

Reproduced locally: all three scheduler variables survive `stage_environment` with
`cpu=True`. On those launchers a supposedly isolated CPU check can initialize a
distributed runtime or wait for peers. The current GCP runner normally supplies
JAX variables, so this is a portability gap rather than a confirmed prior hang.

Required fix: one shared CPU-isolation policy, tested with all recognized launcher
variables and forced CPU device-count settings.

### A6 — P2: encoder resume does not validate the cursor against the requested stop

Evidence: `scripts/train_encoder.py:141` restores `completed_steps`, then uses
`range(start_step, end)` without rejecting a stop before that cursor. In contrast,
`scripts/train_gpt2.py:147` explicitly rejects a stop preceding its restored cursor.

A valid checkpoint at step 6 with `--stop-after 3` can exit successfully without
performing requested work. Checksum validation protects checkpoint contents but
does not make a contradictory invocation valid.

Required fix: validate cursor range and optimizer-update consistency, and define
an explicit already-completed/no-op outcome. Test stop-before-cursor and completed
horizon behavior. This finding is based on control-flow inspection.

### A7 — P2: evaluation input validation is weaker than training validation

Evidence: `scripts/evaluate_encoder.py:30`. It verifies checksums, tokenizer and
special-token policy, and row dtype/layout, but does not reject out-of-vocabulary
IDs as the trainer does. It also does not establish the prepared-data format,
vocabulary/pad/mask fields, or a positive sequence length before execution.
A checksum only proves correspondence with a manifest, not valid model inputs.

Required fix: reuse a common prepared-data validator for train, evaluation and
preflight. Test internally checksum-consistent but invalid evaluation datasets.

## Capability and cost gaps

### B1 — Production encoder training recipe is incomplete

`train_encoder.py` uses constant-LR AdamW with fixed clipping/weight decay,
replicated model/optimizer state, and no gradient accumulation. The causal
`train_gpt2` path already supports schedules, accumulation, prefetch and FSDP;
those capabilities must not be attributed to the encoder.

Add a checkpointed schedule and token budget, configurable optimizer policy,
masked-token-weighted accumulation, deterministic data-order state, then measured
FSDP support. Validate numerical equivalence and exact resume for each addition.
Without sharding, more devices increase data parallelism but do not reduce each
device's parameter/optimizer memory requirement.

### B2 — The encoder data pipeline is a bounded fixture pipeline

`prepare_encoder_data.py:29` truncates each document to one fixed-length row;
`train_encoder.py:180` cycles through rows in deterministic sequential order.
The encoder model supports segment IDs, but the preparation/training path does
not produce or supply packed-document segment/position arrays.

Missing: sharded streaming preparation, deterministic epoch shuffling, resumable
language-mixture sampling, corpus-level deduplication, source/license/language
metadata, padding-efficiency measurement, and document-preserving packing.
Truncation loses document tails; padding and fixed order can waste compute or
bias learning on language-sorted exports. Benchmark these before adding complex
kernels. Do not confuse deterministic replay with representative sampling.

### B3 — Encoder quality is not qualified

`evaluate_encoder.py` reports aggregate masked-token loss and excludes exact
unpadded row duplicates. It builds a Python hash set over the entire training
pool, which is unsuitable for very large corpora. It provides no per-language
metrics, near-duplicate/substring contamination check, initial-versus-final quality
gate, repeated-seed uncertainty, or downstream task evaluation.

Build a pinned multilingual evaluation suite with language-level MLM loss and
chosen downstream tasks, fixed baselines, data provenance and predeclared
acceptance thresholds. Use a reusable external/indexed overlap check at scale.
The causal loss-sanity gate is explicitly not a downstream quality certificate.

### B4 — Physical coverage is narrow

Retained evidence supports single-host four-chip v5p projection tests, two-host
eight-chip v5p encoder recovery at 512 tokens, and single-host 2K/8K execution.
The long-context runs have only two post-compilation timing samples per shape.

Still unqualified: v6e execution, broad v4/v5/v6 scale comparisons, long-context
multi-host recovery, sustained multilingual training, encoder FSDP, and scaling
efficiency across several slices. SFT and RL explicitly reject multi-host runs.
CPU simulated devices and Pallas interpreter checks do not fill these hardware
gaps. Add a small cost-bounded physical matrix only after A1–A4 are addressed.

### B5 — Storage lifecycle and total budget accounting are manual

`RunLedger` persists reservations per output directory and can reconcile posted
billing, but separate runs do not share a project-wide remaining-credit ledger.
The TPU supervisor confirms compute absence; it does not clean checkpoint storage
or track soft-deleted bytes. Ancillary allowances are estimates, not metered caps.

The latest storage cleanup verified zero live validation objects but retained
133.06 GB under the existing soft-delete policy. This is historical audit evidence,
not a new resource/billing check in this turn.

Add per-run object inventory and retention classes, small-report preservation,
explicit checkpoint cleanup receipts, soft-delete expiry accounting, and a shared
campaign budget reconciled with posted gross usage/credits. Storage-policy changes
must be explicitly authorized. Never label successful TPU deletion as zero cost.

### B6 — Checkpoint overhead matters more than tiny kernel wins for short jobs

Encoder saves are synchronous, with a full model/optimizer content digest before
Orbax commit. Recorded checkpoints took about 32–43 seconds; short validation
updates took far less. `checkpoint.py:60` deliberately streams hashes in bounded
chunks, preserving integrity at a nontrivial synchronization/transfer cost.

Profile hashing, serialization, transfer and barriers separately. Add configurable
retention and wall-clock cadence to the encoder, retaining forced final saves.
Evaluate asynchronous/pipelined saving only with fault-injection and corruption
coverage; do not remove integrity checks just to improve benchmark timing.

### B7 — Validation/release coverage has explicit blind spots

Coverage is scoped to `flaxchat`, omits `flaxchat/stages/*`, and does not measure
`scripts/*`; important orchestration scripts therefore do not contribute to the
headline percentage. Per-module floors omit encoder, MLM, fused projection,
operations and training modules. Several legacy modules are excluded from type
checking. These have tests, but not the same enforced coverage policy.

Add risk-based floors or branch coverage for the launchers and acceptance code,
and enforce regression tests for the failures above. Current recent changes are
uncommitted, including new untracked preflight/aggregation scripts. Their local
success is not proof that remote CI or a clean repository checkout contains them.
Publish a reviewed revision with a small, sanitized evidence bundle; then run its
Linux/distributed CI. GCS evidence paths documented before cleanup are historical;
local ignored artifacts are currently the surviving detailed evidence.

## Recommended order

1. Fix A1–A4 before another paid benchmark or a public performance claim.
2. Fix A5–A7 and unify prepared-data/record validation across entry points.
3. Add encoder schedule, deterministic language sampling and a multilingual quality
   baseline; this establishes whether more training actually helps.
4. Measure checkpoint and input overhead, then add weighted accumulation and FSDP
   when memory/throughput measurements justify them.
5. Run a bounded v6e pilot and sustained multi-host campaign with published source,
   environment, evidence, costs and teardown receipts.

## Verification performed

Static lint and full configured type checking passed. Documentation and published
result-provenance checks passed. Direct local probes reproduced A1, A2, A3 and the
retained-environment condition in A5. A4, A6 and A7 are code-inspection findings.
The full non-accelerator CPU suite passed: **696 passed, 13 skipped,
5 deselected**, with **79.43% branch-inclusive coverage** (139.96 seconds).
Accelerator tests were deselected; opt-in distributed/topology tests were not
newly exercised by this single-device invocation. No new TPU, cloud writes, live billing requests, or dependency audit were
performed. Production code was intentionally left unchanged for this audit.

All configured per-module coverage floors also passed. Full test, coverage, lint
and type-check logs are retained locally in `artifacts/harness-audit-2026-09-23/`.
The single warning was a Starlette test-client dependency deprecation.

## Testing upgrade following this audit

The subsequent local changes address the reproduced benchmark-record and CPU
isolation defects (A2 record validation and A5). Benchmark acceptance now requires
the declared horizon, ordered integer steps, finite valid metrics, exact input
token counts, and a final checkpoint. Cloud timeouts and independent physical
provenance remain separate work; this does not fully close all benchmark gaps.

Snapshot preflight now verifies shard availability, safetensors structure, index
membership, uniqueness and content hashes without constructing the model. Weight
hashes and runtime identity participate in worker agreement, and loading rejects
weights that changed after inventory. This closes the missing-weight reproduction
in A1. Architecture-specific tensor-shape/value validation still occurs during
weight import, so a full CPU-only architecture validator remains work to do.
Runtime identity is now part of the encoder recipe: old checkpoints lacking it
will require explicit compatibility/migration handling, not silent resume.

New tests cover corrupt/missing/duplicate shards, inconsistent shard indices,
cross-worker weight/runtime disagreements, rejection before model allocation,
malformed benchmark evidence and all supported distributed launcher variables.
CI routes trainer changes to the snapshot tests. Coverage floors now include
encoder (80%), MLM (75%), fused projection (90%), operations (85%), and training
(80%); policy tests reject both below-floor and absent module reports.

Verification for the testing upgrade: the full CPU run passed 733 tests with
13 skipped and 5 deselected (79.43% coverage). After the final additions, the
focused regression/policy suite passed 138 tests, including ten tests added after
the full run's collection. All three opt-in two-process CPU interruption/recovery
variants passed. New module coverage floors, lint, types and documentation checks
passed. Logs are retained in `artifacts/testing-upgrade-2026-09-23/`. This was local
verification only; no cloud resources were allocated.

## Subsequent physical test-suite campaign

The later Spot TPU campaign attempted every test module on v5p, fixed a test-runner
BF16 Splash precision incompatibility, and passed the targeted retest on v5e.
The combined inventory has 764 distinct passing identities across physical-host
and CPU checks. See [the campaign report](TPU_TEST_RESULTS_2026_09_23.md) for exact
counts, source provenance, unresolved qualification limits and cleanup evidence.
This does not close the production-quality or multi-host evidence gaps above.

## Repair follow-up

The subsequent [repair report](HARNESS_REPAIRS_2026_09_23.md) tracks code changes,
regression tests and fresh TPU evidence against these findings. In particular,
encoder accumulation, scheduling and epoch shuffling are now implemented as
opt-in controls. This audit's capability descriptions above record the state at
the time of review; they are not a claim that later repairs are absent.
