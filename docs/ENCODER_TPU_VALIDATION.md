# Encoder TPU verification — 2026-09-22

Latest bounded endurance and real-data quality evidence:
[September 23 scale campaign](ENCODER_SCALE_RESULTS_2026_09_23.md).
The historical results below do not supersede its real-text Pallas parity failure
or its downstream quality regression.

Project: `tpubuilders`; zone: `us-east5-a`; Spot `v5p-8` (four chips, one VM).
Model: `jhu-clsp/mmBERT-base`, revision
`c5955035435e2bf121cde7f3c8863ef52ff35d82`, 307,786,240 parameters.
Policy: BF16 compute, FP32 residuals/master parameters/Adam state; XLA attention.

## Completed evidence

- Physical topology probe: four TPU devices, one process, JAX 0.11.1, Flax 0.12.9.
- CPU suite on the TPU VM: 596 passed, 8 skipped, 5 accelerator tests deselected;
  coverage 78.92%; all configured per-module coverage floors passed.
- Encoder tests on TPU, including precision variants and exact restart: passed.
- Encoder Splash forward/gradient comparison: passed.
- GPT Splash forward/gradient checks: all four passed after precision isolation.
- Released-weight FP32 output/loss/gradient parity on TPU: passed.
- Released 307.8M-parameter checkpoint: six BF16 training updates at global batch
  four, sequence 512, all finite and accepted; step-6 GCS checkpoint committed.
- Recovery follow-up: physical TPU probe, held-out evaluation, SIGKILL after
  committed step 3, and resume to step 6 all passed. Model, optimizer, and training
  state manifests exactly match the uninterrupted baseline.
- Held-out fixture: 16 rows, 1,177 selected tokens, masked-token loss 0.0573377.
  These synthetic fixtures validate execution only, not production quality.
- BF16-versus-FP32 diagnostic: hidden relative L2 2.02%, logits 0.46%,
  loss 1.83%, first-QKV gradient 3.18%. Strict pointwise gate failed. This is
  numerical drift evidence, not a BF16 parity or model-quality qualification.

## Issues exposed and corrections

| Issue | Correction | Verification |
| --- | --- | --- |
| Single-worker launcher exported distributed coordinator variables; a CPU subprocess test inherited the coordinator and the parent aborted. | Only export distributed variables for multiple workers; isolate CPU validation environments. | Local launcher regressions and subsequent full CPU suite passed. |
| Global highest matmul precision forced invalid FP32 operations inside the BF16 GPT Splash kernel. | Use native precision for BF16 Splash diagnostics; keep highest precision for the FP32 reference gate. | Local regression and four physical TPU checks passed. |
| Released tokenizer labels model mask token 4 as non-special. | Explicit mask ID in data preparation and manifest; protect it from masking targets/random replacement. | Regression, released-data preflight, and six physical TPU updates passed. |
| Controller monotonic time lagged the cloud wall-clock deadline during the third attempt; capacity waiting left insufficient time for recovery tests. | Bound every operation by the earlier monotonic and wall-clock deadline; retain independent cloud cleanup. | Local deadline/launcher suite: 21 passed. Cloud guard deleted the third slice at expiry. |

The first two attempts were stopped by validation and their queues and VMs were
confirmed absent. Neither completed model training. Failure logs are retained.
A fourth bounded attempt reused the committed baseline and passed recovery checks.
Its supervisor exited successfully; both queue and TPU VM absence were independently
confirmed. All four campaign allocations have been cleaned up.
Production quality is not established
by the short synthetic multilingual fixtures used for this verification.

## Throughput interpretation and experimental optimization

The five post-compilation updates in the released-model pilot measured roughly
28.6–30.4K input tokens/s (2,048 tokens per step). First compilation/update took
42.1 seconds; saving the checkpoint took 41.6 seconds. This six-step pilot is a
correctness check, not a sustained throughput qualification.

The prior GPT pilot recorded 197,176 steady-state input tokens/s, but used 12
layers, vocabulary 50,257, and batch 16 × sequence 1,024 (16,384 tokens per step).
The encoder uses 22 layers, vocabulary 256,000, and batch 4 × sequence 512.
The models, vocabulary projections, batch sizes, and parameter sharding differ;
these rates do not isolate a precision-related regression.

An opt-in `--mlm-projection masked` path was added locally after freezing the
recovery workload. It projects static buffers of selected positions, with a
dense overflow fallback that drops no labels. At sequence 512 / chunk 128 it
reduces head rows fourfold when capacity is sufficient; whole-model speedup is
unmeasured. Local encoder/trainer suite: 32 passed, one multiprocess test skipped,
one accelerator test deselected; an additional four-device CPU sharding parity
test passed. The frozen physical TPU results above do not validate this new path.
Physical multi-host encoder validation and a sustained masked-projection TPU
benchmark remain outstanding.

## Artifacts and cost accounting

Evidence is retained under `artifacts/encoder-validation-0922/` and the private
GCS campaign prefix `gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/`.
The original single-host attempts used the independent cloud expiry guard and a
conservative $25 budget. The later two-host attempts below use $35 budgets.
The recorded Spot quote is $1.003363 per chip-hour ($4.013452 per four-chip hour),
checked September 22. Entire resource-lifetime estimates intentionally include
non-billable waiting/deletion periods and are not posted billing or credit usage.

## Later projection follow-up

The preceding recovery campaign predates projection optimization. Subsequent
physical masked/fused measurements, the shared-head rounding fix, and its
released-model gradient regression are recorded in
[projection benchmarks](ENCODER_PROJECTION_BENCHMARK.md). Physical multi-host
encoder recovery was subsequently verified in the bounded campaign below;
downstream model quality remains a separate requirement.

### Backend-specific recovery and topology gates

The qualification runner accepts `--mlm-projection`, `--mlm-loss-backend`, and
`--mlm-vocab-tile` and carries them through baseline training, fault injection,
and resumed training. Nondefault configurations have separate checkpoint and
report paths. Recovery mode resolves the same configuration under its
`--baseline-prefix`; existing default baseline paths remain compatible.

Use `--expected-processes` and `--expected-devices` to assert the intended slice,
and `--batch-size` to choose a global batch divisible by the actual global device
count. Multi mode rejects a one-process runtime; single and recovery modes reject
multiple processes. Hardware evidence and configuration are saved in the summary.
Each workload stage has a 900-second default deadline, adjustable with
`--stage-timeout-seconds`; the external Spot supervisor and durable cleanup guard
are still required to bound the complete allocation.

An expected SIGKILL only passes when the worker also logs the committed step-3
fault marker. Unexplained kills and timeouts prevent resume qualification.
The two-process CPU test exercises an actual committed-checkpoint SIGKILL and
compares final model, optimizer, and training-state checksums against an
uninterrupted baseline for dense XLA, masked chunked XLA, and masked `xla_full`.
Pallas recovery is covered by the single-process CPU interpreter test; physical
Pallas interruption/recovery and physical multi-host encoder acceptance are
reported for the specific two-host configuration below. These checks do not establish downstream model quality.

Download the per-worker result directories into a fresh local directory, then
aggregate physical multi-host recovery evidence with:

```bash
python -m scripts.summarize_encoder_validation \
  --input-dir artifacts/encoder-multi-results \
  --expected-processes 2 --expected-devices 8 \
  --output artifacts/encoder-multi-acceptance.json
```

This rejects missing or duplicate process reports, wrong hardware, mismatched
backend/configuration, missing required stages, and failed recovery checksums.
It also requires six accepted baseline updates and resumed updates 4–6 with
finite losses, selected labels, and no masked-to-dense fallback.
The result records evidence file hashes and always leaves downstream quality
unqualified. Keep the source manifest and cloud ledger with the report.

### Physical multi-host recovery campaign, 2026-09-22

Attempt `multirecovery5` acquired a Spot `v5p-16` in `us-east5-a`. Both hardware
probes confirmed two processes and eight TPU devices. Training failed before
updates because the historical base archive's prepared-data manifest omitted
mask token 4 from `special_token_ids`; the current trainer correctly rejected it.
This is a failed qualification, not a Pallas or multi-host training pass.

The supervisor deleted both node and queued resource and recorded
`resource_absent`. At the recorded eight-chip Spot quote of $8.026904/hour,
observed-READY-to-absence and full-request-to-absence windows estimate
$0.97–$1.46 of compute respectively; these are neither posted billing nor an
exact billing interval, and exclude storage/network charges. Failed evidence is
retained under `artifacts/encoder-validation-0922/multirecovery5/evidence` and
`gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/multirecovery5/`.

For the corrected attempt, the current preparer rebuilt a 32-row, 512-token
synthetic fixture. The exact Pallas training configuration passed CPU input
preflight before provisioning. The fixture and manifest were included directly
in the checksum-verified source overlay, and the runner received its explicit
artifact root, preventing fallback to stale prepared data in the base archive.


Corrected attempt `multirecovery6` **passed physical multi-host encoder recovery**:

- Spot `v5p-16`, two physical hosts/eight TPU devices, JAX 0.11.1 / Flax 0.12.9.
- Released mmBERT-base weights, 307.8M parameters, global batch 16, context 512,
  full 256,000-token vocabulary, masked Pallas projection, vocabulary tile 1024.
- BF16 compute, FP32 residual/master weights/optimizer, data-parallel replicas.
- Six accepted baseline updates; both workers received SIGKILL after committed
  checkpoint 3; restore completed updates 4–6. No dense fallback occurred.
- Final model, optimizer, and training-state checksum manifests exactly matched
  the uninterrupted baseline on both hosts. The all-worker aggregate passed.

Baseline stage: 107.2–107.6 s per worker; fault stage: 96.3 s; resume stage:
130.6–131.1 s. Initial compilation/update took 49.28 s, with five subsequent
steps at 0.0525–0.0554 s (approximately 148k–156k input tokens/s). The final
baseline checkpoint took 35.36 s; resumed final checkpoint 33.83 s. This is a
short recovery smoke, not a sustained throughput/scaling comparison or language
quality result. It does not qualify other contexts, batches, TPU generations,
encoder FSDP, or other backends. The subsequent `xla_full` test is recorded below.

Frozen overlay SHA256:
`3f1825586cb811002cddce02c9408d30e59ff71737e7dfa2503d6f8ef93f16e8`.
The exact source/fixture manifest and worker logs are under
`gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/multirecovery6/`.
Local aggregate: `artifacts/encoder-validation-0922/multirecovery6-acceptance.json`.
The aggregate's `quality_qualified` remains false.

Both attempts finished with verified node and queued-resource absence; the
successful supervisor exited zero. Successful-attempt estimated compute is
$1.45–$2.02; the two attempts together are approximately **$2.42–$3.47**, using
the observed-READY and full-request windows respectively. These estimates include
the wait for teardown, exclude storage/network, and are not posted billing or
credit-balance measurements. The local `cost-estimate.json` files preserve the
rate, windows, and unknown posted-billing fields.

The Spot supervisor now supports `--local-preflight` to run input/archive checks
before any cloud lease or allocation. Failure and timeout tests verify that the
cloud API is never reached in either case. This prevents campaigns from paying
for validation errors that can be detected on the controller first.


### Multi-host `xla_full` recovery, 2026-09-22

Attempt `multixla7` **passed** the same two-host/eight-device Spot `v5p-16`
qualification with `--mlm-loss-backend xla_full`: batch 16, context 512,
masked projection, BF16 compute and FP32 residual/master/optimizer state.
The synthetic fixture was identical to `multirecovery6`. The supervisor ran a
CPU preflight from the checksum-verified extracted source archive before creating
its cloud lease or requesting a TPU. No stale fixture was used.

Both workers completed six accepted baseline updates, were deliberately killed
after checkpoint 3 committed, and resumed updates 4–6. All final model,
optimizer, and training-state checksums exactly matched this backend's
uninterrupted baseline; all-worker acceptance passed with no dense fallback.
This does not assert bitwise equality between different projection backends.

Per-worker stage times: baseline 106.95–107.11 s, interruption 97.08–97.54 s,
resume 131.76–131.96 s. Initial compile/update took 51.46 s; the following five
steps took 0.0518–0.0558 s, approximately 147k–158k input tokens/s. Final
checkpoint times were 32.46 s (baseline) and 33.63 s (resumed). These few updates
establish recovery execution, not steady-state scaling efficiency or language
quality. Initial SSH setup needed transport retries on one worker; the guarded
command identities and allocation were not recreated.

Frozen source overlay SHA256:
`04bdff513f62c8a8273994cb3705bbcc72cb80c256f2be73a7cb699e6caa8298`.
Evidence: `gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/multixla7/`.
Local aggregate: `artifacts/encoder-validation-0922/multixla7-acceptance.json`.

The reusable `scripts.preflight_encoder_archive` now supports this preparation
workflow. It checks the expected digest, rejects unsafe/duplicate archive entries,
requires package roots to prevent importing the installed checkout, and executes
the archived trainer against its archived fixture using CPU-only settings. Nine
focused regressions cover invalid archives, import isolation, and child failures;
static/type checks pass. This prepares future campaigns to reject bad payloads
before paid allocation. Remaining physical work includes long-context sweeps,
v6e execution, broader scale coverage, and realistic multilingual quality checks.

The supervisor exited zero, with independent API checks confirming no TPU VM or
queued resource remained in `us-east5-a`. Estimated compute for `multixla7` is
$1.51–$1.98, using the same recorded Spot quote and observed-READY/full-request
windows. This excludes storage/network and is not posted billing or credit usage.

### Staged long-context execution checks

`validate_encoder_tpu --mode context --lengths 2048 8192` runs an ascending,
single-host context sweep without repeating the already recorded CPU/parity
campaign. Use an explicit `--batch-size`, expected device/process counts, and
backend. A failed stage or evidence check stops the sweep before the larger case.
Each context requires three accepted finite updates, the exact input-token count,
a committed final checkpoint, and peak-memory evidence from every TPU device.
Masked projection must not fall back to dense. Reports use
`scope: context_execution_only` and never claim downstream quality or interruption
recovery at these lengths. Use the frozen-archive preflight for each bundled
context before asking the Spot supervisor to provision hardware.


### Physical 2,048- and 8,192-token sweep, 2026-09-22

Attempt `context8` **passed** both execution checks on a single-host, four-device
Spot `v5p-8`: released mmBERT-base, global batch 4, masked `xla_full`, BF16 compute,
and FP32 residual/master/optimizer state. Frozen-archive preflight passed for both
16-row synthetic fixtures before provisioning. The request became ready within
the five-minute capacity window; no fallback slice was allocated.

| Context | Compile + first update | Following two steps | Approx. input tokens/s | Final checkpoint | Peak allocator usage | Peak reserved memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2,048 | 47.95 s | 0.0821 / 0.0787 s | 101,894 | 42.90 s | 7.19 GiB | 5.47 GiB |
| 8,192 | 51.65 s | 0.4379 / 0.4348 s | 75,103 | 42.07 s | 7.25 GiB | 20.21 GiB |

All six updates were finite and accepted, with nonempty MLM labels and no dense
fallback. Both final checkpoints committed. Memory values are distinct runtime
counters reported per device (maximum across four devices); they are not a single
combined footprint, and allocator usage alone understates the memory needed for
long-context execution. Only two post-compilation steps were measured per context,
so the throughput figures are smoke observations, not a sustained benchmark.

The downloaded logs were independently revalidated with the same acceptance
checks, including actual input-token count, checkpoint completion and all-device
memory evidence. This qualifies execution/checkpoint writing for these shapes.
It does **not** qualify crash recovery at long context, multi-host long context,
Pallas at these lengths, v6e memory behavior, or downstream model quality.

Frozen overlay SHA256:
`0ce33baa9fed956697aabe6d658ac29a01b0c848c98e8675c5a495342e99aaa1`.
Evidence: `gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/context8/`.
Local aggregate: `artifacts/encoder-validation-0922/context8/acceptance.json`.

The supervisor exited zero. Independent API lists confirmed both the TPU VM and
queued resource were absent. Estimated compute for this allocation is
**$0.54–$0.86**, using the recorded $4.013452/four-chip-hour Spot quote and
observed-READY/full-request windows. Storage/network are excluded; this is not
posted billing or a current credit balance. Both downloaded logs also pass the
strengthened gate requiring the peak reserved-memory counter. The context-related
qualification suite now passes 30 tests; lint, type checks and docs checks pass.


### Evidence integrity and storage cleanup, 2026-09-23

The multi-host aggregator now requires ordered update records from process zero,
matching the trainer's logging contract. It rejects records from other processes,
invalid or zero global batches, non-integer process ranks, incorrect global token
counts, nonfinite or excessive masked-token counts, and nonpositive/nonfinite step
times. Accepted reports include hashes of both update logs as well as the summary
and recovery evidence. Regression coverage exercises these failure modes without
provisioning accelerators. Both downloaded physical multi-host campaigns
(`multirecovery6` and `multixla7`) pass the strengthened checks.

The validation bucket was emptied at the user's request on September 23. GCS
paths above are historical run locations, not currently downloadable evidence.
Downloaded local reports/logs remain under `artifacts/encoder-validation-0922/`;
new aggregate checks are saved as `<run>/acceptance-revalidated.json`. These local
artifacts are not committed to Git. Checkpoints and fixtures must be recreated or
restored before a future cloud campaign. No new TPU allocation was made for this
verification. This does not add v6e or multilingual quality qualification.


### Bounded evidence transfer and early failure

All qualification uploads and recovery-manifest reads now use
`--cloud-timeout-seconds` (default 120 seconds per call), separately from the
training stage timeout. A required CPU-suite, TPU-suite, or released-parity
failure stops subsequent workload stages immediately. Optional Splash diagnostic
failures remain separate from required qualification failures.

A failed final upload marks the local summary as failed and exits unsuccessfully;
remote evidence may be incomplete, so an uploaded summary alone is not proof that
the controller finished successfully. The external supervisor's exit status and
cleanup evidence remain necessary. Regression tests inject upload/read timeouts
and required-stage failures without making cloud calls.

### Long-context evidence integrity

The context acceptance gate requires distinct, nonempty device identities and
finite memory counters, so duplicated records cannot stand in for missing device
evidence. Update records require integer step/token counts and finite, positive
masked-token counts no larger than the input batch. The final checkpoint must
appear after the last update, followed by the device-memory record. Regressions
cover NaN counts, oversized label counts, boolean steps, duplicate/missing device
identities, infinite memory limits, and reordered checkpoint/memory records.

Both retained physical 2K and 8K logs pass these stronger checks; local results are
`artifacts/encoder-validation-0922/context8/context-2048-revalidated.json` and
`context-8192-revalidated.json`. These are rechecks of existing evidence, not new
TPU executions or additional hardware/quality qualification.

The multi-host summary command now also requires the successful controller
receipt (`--controller path/to/workload/summary.json`, or `controller.json` in the
input directory), each worker's `hardware.json`, `interrupt.log`, and raw
`baseline-manifest.json` / `recovery-manifest.json`. It rejects historical bundles
that contain only recovery booleans. Rerun or retrieve missing raw evidence rather
than manufacturing a new qualification from those summaries.

### YAT fusion-boundary diagnostic (September 25)

A single-v5e experiment tested six compiler-boundary placements on a captured
two-layer fixture, keeping BF16 geometry and the existing gradient tolerance.
Attention-output boundaries reduced maximum batch-versus-per-example gradient
discrepancy from 0.003567681 to 0.001989171; all 21 gradient leaves passed.
All-dense boundaries also passed (0.001459762). QKV-only, FFN-output-only and
head-only placements failed. Bias 1, epsilon 0.01 and trainable alpha were retained.

JAX traces contained three warmed calls per placement, with another 20
synchronized calls for timing. Attention-output device-op interval union was
0.193262 ms versus baseline 0.193314 ms across the captured calls. This tiny,
host-overhead-dominated fixture does not establish a throughput improvement.
Forward-versus-autodiff and instrumented-versus-ordinary forward differences
vanished with attention-output boundaries in this fixture, but batch-shape
differences remained. Each placement also changes its reference arithmetic.

This is a candidate for distributed replay, not production qualification or a
default change. Distributed accumulation, full-model throughput, recovery and
quality still require verification. Local evidence is retained in
`artifacts/yat-boundary-tpu-v5e1-0925/RESULTS.md`, with tensor archives, trace
summaries and independent cleanup receipts. The controller exited successfully;
the exact TPU VM and queued resource both subsequently returned `NOT_FOUND`.

To replay a placement through both accumulated-gradient implementations, use
`python -m scripts.replay_mlm_accumulation CAPTURE --boundary-mode attention_output
--output NEW_REPORT.json --profile-directory NEW_TRACE_DIR` on the captured
device count. The default placement is `baseline`; boundaries only affect the
diagnostic model clone. Parameter paths, values, dtype and projection settings
are preserved. The YAT FFN's direct input-kernel operation is not wrapped.

Inspect `replayed_reference_gates` and the `*_replay_vs_per_example` comparisons:
these compare newly executed distributed gradients with a newly executed
per-example reference. The older `*_vs_per_example` fields compare saved capture
gradients and cannot qualify a modified execution. The diagnostic writes reports
even when numerical comparisons fail; successful process exit alone is not a
numerical gate. `--per-example-only` has no distributed gate results. Reference
arithmetic changes with the selected placement; all reports continue to set
`production_qualified` to false.

For automated qualification, add `--require-reference-gates`: both fresh
distributed paths must pass the independent-reference loss and gradient checks.
The report is saved before a failing exit status is returned. This option rejects
`--per-example-only` rather than treating missing distributed results as a pass.
Nonfinite losses fail the gate and are recorded as `null`, with explicit
`loss_finite` flags, so the JSON report remains readable even on numerical failure.

### BF16 distance with wider score/softmax diagnostic (September 25)

A one-v5e primitive experiment compared adaptive BF16 attention with direct BF16
distances plus FP32 softmax, and with direct BF16 distances plus FP32 scores and
softmax. Both candidates passed all eight saved activation-gradient cases under
the existing independent-oracle tolerances; baseline passed four. These reused
development fixtures do not qualify model quality or distributed training.

At batch4/sequence512/heads12/width64, warmed forward-and-backward calls took
1.461 ms globally and 1.813 ms locally for baseline, versus 3.900/5.184 ms for
the wider-score candidate (2.67/2.86 times slower). Temporary allocation fell
from 6,448,128/5,274,624 to 1,095,168/1,159,680 bytes. Twenty synchronized calls
and three traced calls per mask support this isolated tradeoff. The experiment
changes distance evaluation and precision together; it cannot attribute the
slowdown to FP32 alone. Softmax-only candidate timings were similar.

Keep these as experimental numerical references, not training defaults. A
faster fused implementation must preserve their gradient behavior and undergo
physical end-to-end validation. Evidence, raw tensors and six JAX trace summaries
are under `artifacts/yat-precision-tpu-v5e1-0925/`. No v6e, multi-host, long-context
hardware or production-quality qualification is added by this experiment.

### Centered backward and reference selection (September 25)

`scripts.yat_attention_edge_suite.validate_attention_edges(attention, output)`
provides a reusable 126-case gate. The callable accepts
`(q, k, v, segments, alpha, radius, query_tile)`. Call it inside the benchmark's
JAX process; launching a second JAX process against the same TPU can fail with
"TPU is already in use" before any numerical checks execute.

The gate covers global/local/self-only masks, signed and zero alpha, tiles 32/64,
nearby vectors, NaN padding, and constant keys/values both globally and within
individual packed documents. Constant eligible keys require exactly zero query
and alpha gradients; constant eligible values require exactly zero Q/K/alpha
gradients. These assertions supplement the existing oracle tolerances.
Each case saves the device-rounded inputs and actual/reference tensors.
`replay_attention_edges(output)` independently recomputes the FP64 oracle from
those inputs, checks the complete case matrix, and recalculates error limits.

A sweep's successful exit or `correctness_passed` flag may mean that only its
baseline passed. Admission must explicitly require every proposed candidate:

```sh
python -m scripts.yat_benchmark_admission REPORT.json \
  --require centered_reduce --backend cpu
```

This checks report completeness, saved-model coverage, all output/gradient
statuses, both performance masks, and finite timing samples. It does not replace
independent tensor replay, and it does not qualify model quality.

Physical comparison on one v5e chip, batch4/sequence512/heads12/width64:

| Forward + backward | Global | Local radius128 |
|---|---:|---:|
| Centered backward with scalar gathers | 6.383 ms | 7.112 ms |
| Centered backward with masked reductions | 5.275 ms | 6.005 ms |
| Existing wider-score XLA reference | 3.872 ms | 5.179 ms |

The masked-reduction variant is about 17%/16% faster than the centered-gather
variant. JAX profiles confirm that four scalar gathers disappear, reducing
gather time from approximately 1.62/1.61 ms to 0.189 ms per call. The optimized
and gather variants produced identical output/Q/K/V/alpha arrays on all eight
saved-model hardware cases, although CPU compilation showed small Q differences.
Each centered variant passed all 126 physical edge cases and their independent
input-based replays. All three candidates passed their saved-model and full-size
performance oracle gates. The faster XLA reference still has the previously
identified exact-null-gradient failures; its timing is not equivalent numerical
qualification. See `artifacts/yat-reference-tpu-v5e1-0925/` for raw evidence.

These centered candidates retain BF16 distances and coordinate arithmetic with
FP32 score/softmax/coefficient state. BF16 scalar-state experiments passed the
synthetic edge cases but failed a saved-model query-gradient oracle gate (about
11.9% relative error, versus the existing 8% limit). They remain rejected; no
tolerance was relaxed and no production precision default was changed. Full
encoder, optimizer/checkpoint, distributed, and model-quality qualification
remain separate requirements.

The centered backward is also available through the experimental encoder option
`--yat-attention-implementation centered_fp32_scores`. Select
`--attention-score yat_softmax --yat-compute-mode bf16 --dtype bfloat16` and
positive `--yat-attention-block-size` and `--yat-global-attention-block-size`
values. Keep `--yat-softmax-backward factored`; the complete centered backend
provides its own backward. The default implementation remains `standard`.
The implementation option is part of checkpoint identity, and the centered
module is included in the training source hash. Changing implementation during
an exact resume is rejected.

This integration has passed CPU primitive replay, packed/unpacked routing,
larger query-tile checks, and two-layer MLM training with exact optimizer and
checkpoint resume. Its integrated full-encoder TPU throughput and distributed
behavior are not yet qualified by the standalone primitive results above.

The subsequent one-hot key-selection comparison measured 4.983/5.685 ms
global/local versus 5.326/6.001 ms for scalar reductions with a key gather,
an additional 6.4%/5.3% improvement in the same run. It eliminated the remaining
reference gather in the XPlane profile. Both centered variants passed all
126 input-replayed physical edge cases and all saved-model/full-size oracle
gates; saved-model outputs and all gradients were bitwise equal on TPU.
`centered_fp32_scores` now uses that one-hot BF16 MXU selection. Its final source
also passes CPU primitive replay, eight saved-model cases, 12 larger cases and
the two-layer exact-resume test. Evidence is in
`artifacts/yat-mxu-tpu-v5e1-0925/` and
`artifacts/yat-encoder-centered-integration-0925/`.
