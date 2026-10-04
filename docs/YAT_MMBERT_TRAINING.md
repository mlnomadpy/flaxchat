# YAT mmBERT training

> Historical MLM adaptation and launch record. The statements below about
> unlaunched training, credits, pending contrastive stages, and running resources
> apply to their dated attempts. The later embedding stage completed 14,000
> updates; see its [model card](YAT_MMBERT_EMBEDDING_V1_MODEL_CARD.md).
> Continue the trained model using the
> [representation training runbook](REPRESENTATION_TRAINING_RUNBOOK.md).

The native candidate has 307,786,284 parameters: released mmBERT-base tensors
plus one trainable FFN alpha and one trainable attention alpha per layer (44
additional scalars). Initialization copies the pretrained backbone; replacing
GeGLU and dot-product attention changes the function immediately. This is
continued pretraining of a new architecture, not a numerically equivalent import.

`python -m scripts.train_yat_mmbert` is the dedicated entry point. It requires a
local pretrained snapshot and the HF config, enforces 300–500M parameters and
fixes both YAT branches, bias 1, epsilon 0.01, adaptive BF16 distance arithmetic,
BF16 activations/residuals, exact XLA attention and masked MLM projection.
Parameters and optimizer state remain FP32. Alpha starts at 1 and is trainable.
It uses the existing trainer's checksummed inputs, finite-update checks,
checkpoint/resume identity and distributed data loading. It imports no NMN code.
RoPE, alternating local/global attention, padding/document masks and tied
vocabulary weights are preserved. The MLM prediction head retains its GELU;
YAT replaces the transformer FFN branch and attention score.

## First physical validation

Run the related YAT tests on TPU, then a 32-update calibration with the real
307.8M checkpoint and cleaned pilot corpus. Require finite accepted updates,
no dense projection fallback, committed checkpoint and matching resumed state.
Measure compile time, steady-state throughput, peak memory, initial/final MLM
loss and alpha changes. Tiny CPU tests alone do not validate this workload.

```sh
python -m scripts.train_yat_mmbert \
  --config artifacts/mmbert-base/config.json \
  --pretrained artifacts/mmbert-base \
  --data artifacts/encoder-production-0923/fineweb-pilot/prepared/train-clean-v3 \
  --output gs://tpubuilders-flaxchat-validation-0921/yat-mmbert-0924/calibration \
  --steps 32 --batch-size 8 --save-every 16 \
  --learning-rate 2e-5 --mask-probability .15 \
  --mlm-loss-backend pallas --shuffle
```

Use `--preflight-only` to validate inputs without model allocation. The first
hardware target is one host with four Spot v5e chips, not a multi-host campaign.
Keep the cloud attempt bounded to 30 minutes plus the existing cleanup allowance.
For planning, four chips at the published on-demand reference of $1.20/chip-hour
cost $4.80/hour: a 30-minute attempt plus 30-minute cleanup allowance and $2
ancillary reserve is $6.80. This is a conservative planning assumption, not a
Spot quote or hard billing cap; Spot prices vary. Verify rate and remaining
budget before allocation. [Google TPU pricing](https://cloud.google.com/tpu/pricing).

After hardware validation, use measured YAT throughput to choose a fixed horizon
for a 50M-useful-token adaptation pilot, subject to the remaining budget. Do not
substitute conventional mmBERT throughput into its cost estimate. The existing
20-language cleaned corpus is a pilot, not full mmBERT language coverage. Evaluate
matched held-out multilingual MLM before increasing training, and retain a
conventional baseline. Contrastive embedding training is a later stage.

## Cloud prerequisite (applied with user approval)

On September 24 the user approved and we applied the following API enablement
and narrowly conditioned cleanup role. Actual workflow probes verify permissions
before provisioning; the simulator does not support the queued-resource names.
No firewall or email changes were made.

```sh
gcloud services enable policytroubleshooter.googleapis.com --project=tpubuilders
gcloud projects add-iam-policy-binding tpubuilders \
  --member=serviceAccount:flaxchat-validation-cleanup@tpubuilders.iam.gserviceaccount.com \
  --role=projects/tpubuilders/roles/flaxchatValidationCleanup \
  --condition-from-file=infra/tpu/cleanup_condition_yat.json
```

Effective permission verification must pass before provisioning.
Evidence is under `artifacts/yat-mmbert-training-0924/`. Training remains unlaunched
until physical validation and budget admission succeed.

Local validation completed: 111 tests passed, four opt-in distributed cases skipped;
one exact-resume test passed on four simulated CPU devices. The real 307.8M
checkpoint passed a CPU forward and eight-target MLM-loss smoke test on one
512-token corpus row. This does not establish TPU throughput, full-size training
stability or multilingual quality. Targeted lint/type checks pass.

Conversion warning: on the same eight smoke-test targets, conventional mmBERT
loss is 5.71847 and YAT loss is 23.47997. This tiny sample is not a quality
benchmark, but confirms that direct branch replacement changes pretrained
behavior substantially. Require a bounded adaptation run and held-out evaluation
before spending on a larger training campaign; do not call this quality parity.

## September 24 authorized launch update

The user approved the scoped IAM/API changes. Both are applied. Google's installed
CLI returned a legacy permission result and the v3 simulator rejects TPU queued
resource names. Admission now uses actual GET/force-DELETE probes, run as the
cleanup workflow service account against the absent dedicated queue. Existing
resources and permission errors fail admission. The controller checks the exact
deployed workflow source/revision and that execution reached its post-probe
wait step. A live probe passed; project-number canonicalization is covered.

The frozen campaign uses one Spot v5litepod-4, a 5,400-second lease, the existing
1,800-second cleanup allowance, and a $12 planning reservation. An additional
$1.50 allowance is retained for the earlier endurance overrun. Cumulative
planning exposure is $896.5167 of the original $900, not actual billed spend.
No new billing-credit balance is inferred from this figure.

The worker runs YAT tests, 32 full-model updates, committed interruption after
16 updates, resume to 32, and exact final model/optimizer/cursor comparison.
Only then can it launch adaptation. Its horizon is capped at approximately 50M
useful tokens and shortened to fit measured p90 step time with a 1.5 factor and
20-minute evaluation reserve. It refuses fewer than 256 affordable updates.
Matched 256-row development evaluations compare the short calibration and pilot;
this is a smoke evaluation, not production-quality or mmBERT leaderboard parity.

The first physical attempt passed 35 YAT tests and the real input preflight but
was PREEMPTED during the first full-model step (Google maintenance at
2026-09-24 13:45:14 UTC). No completed optimizer update was recorded. Queue and
VM absence were independently confirmed. Its historical $12 reservation remains
in the ledger; separate reconciliation bounds its planning exposure at $5.98,
including request lifetime, 30 extra minutes and $2 ancillary allowance.

A zero-padding optimization now excludes exact zero/zero distance pairs from
near-collision repair, preserving their zero distance and derivatives. It tests
coordinates rather than squared norms so underflowed nonzero inputs still take
the repair path. Forty-four relevant regression tests pass. A small padded CPU
attention backward benchmark improved from 2.327 ms to 0.710 ms with identical
loss and all input gradients; this does not establish TPU speedup or explain
all first-step latency. The retry revalidates the changed kernel on TPU.

The retry has a 3,600-second lease and $9.20 reserve. Including the separate
first-attempt reconciliation, conservative cumulative planning exposure is
$899.6967. This remains a planning bound, not a billing statement. Evidence:
`artifacts/yat-mmbert-retry-0924/`.

## Latest outcome: physical training still unvalidated

The west4 retry was also preempted, during setup (13:56:11 UTC). Queue and VM
absence were independently verified. The alternate east5 request passed cleanup
admission but was denied by Google: this project cannot submit v5litepod-4
requests in us-east5-a. No VM was allocated there; queue and VM inventories
are empty. The audit evidence is in `artifacts/yat-mmbert-east-0924/`.

No full-size optimizer update or adaptation pilot completed. Thirty-five physical
YAT tests passed on the first attempt, before the padding optimization; the
updated padding kernel has local regression evidence only. Multi-host and model
quality qualification remain pending. The supervisor now saves the asynchronous
creation receipt, preserves terminal queue details, and treats a NOT_FOUND create
error as failure rather than resource absence.

The refreshed billing credits page showed **$821.87 marketing credit**, expiring
September 28, 2026. Billing may lag; this is not a cost attribution to these jobs.
Before the east request, conservative cumulative planning exposure was $899.5167
after separately reconciling confirmed deleted attempts. Historical ledgers are
unchanged. These reservations are not actual spend. Further training needs a
stable authorized TPU allocation and a newly checked budget; no automatic retry
or full training campaign is currently running.

## Flex-start v6e campaign

The user authorized Flex-start and durable retention of every checkpoint.
The `yat-mmbert-flex-0924` campaign requests **v6e-4 in us-east5-a** with a
7,200-second overall lease and 900-second capacity wait. Flex-start creation has
an explicit maximum run duration as well as the independent workflow cleanup
lease. It does not silently fall back to Spot. The four chips provide 32 GB HBM
each (128 GB aggregate, not a single shared memory space). Validation starts at
global batch 32, or eight sequences per chip; scaling further requires measured
memory and throughput evidence. Price reference: $5.40 per slice-hour, with a
$16 reservation including additional cleanup and ancillary allowances. The
separate reconciliation retains $839.3567 cumulative planning exposure against
the original $900; this is not a billing total.

Recipe for this bounded pilot:

- Initialize from the released mmBERT-base checkpoint with both YAT branches;
  retain its tokenizer and 512-token sequence length.
- Use the revision-pinned, document-separated 20-language FineWeb2-HQ pilot;
  train on `train-clean-v3`, with held-out validation kept separate. This does
  not yet implement the broader proposed multilingual mixture.
- Token-temperature language sampling exponent 0.5, 15% dynamic MLM masking,
  learning rate 2e-5, cosine decay and warmup for 5% of the horizon capped at
  100 updates. Treat these as experimental adaptation settings.
- Validate 32 full-model updates and committed interruption/resume with identical
  final model, optimizer and cursor. Then use measured step times to fit a pilot
  of up to approximately 50M useful tokens within the remaining lease.
- Save validation checkpoints every 16 updates; pilot checkpoints every 256
  updates and the final update. `--keep-checkpoints 0` preserves **every saved
  checkpoint**, including optimizer and training state, in the campaign GCS
  prefix. It does not mean writing a checkpoint on every optimizer update.
- Compare held-out MLM loss before expanding. The existing smoke comparison
  shows that the architecture swap initially degrades pretrained behavior;
  successful execution alone does not establish a competitive encoder.

Broader language replay, per-language downstream evaluation, and supervised
contrastive retrieval training remain follow-up quality stages. Do not spend
on a larger corpus or longer campaign until this pilot establishes useful
learning and measured cost. See `YAT_ENCODER_RECIPE.md` for the proposed data
mixture and its limitations.

Local checks for this change: 70 passed, four opt-in tests skipped; one additional
save/restore test verifies that all four checkpoints remain and the earliest is
loadable. Focused Ruff and Pyright checks pass. The frozen archive input preflight
also passes. Physical Flex-start results must be read from
`artifacts/yat-mmbert-flex-0924/`; submission is not validation.

While this request queued, an evaluation audit found that the legacy prefix of
256 validation rows covers only Arabic (203) and Czech (53) before overlap
exclusion. The frozen campaign's loss comparison is therefore a limited smoke
check, not multilingual quality evidence. A separate opt-in
`evaluate_encoder --balanced-languages` now round-robins shuffled language rows,
validates full document coverage, and reports realized language counts. Its local
preflight covers all 20 languages with 12–13 rows each before overlap exclusion;
13 sampling/initial-evaluation tests pass. This evaluator change **postdates the
submitted payload**. Run it on retained checkpoints before scaling; it does not
retroactively change any existing result or establish per-language quality.

**Terminal result:** Google accepted the request as `FLEX_START` with maximum run
duration 7,195 seconds, but it stayed `WAITING_FOR_RESOURCES` throughout our
900-second capacity-wait window. The local controller then cancelled it as
configured. Controller exit 1; independent east5 queue and VM inventories are
both empty. No v6e VM, full-model optimizer update, or adaptation checkpoint was
produced. This is a capacity-wait timeout, not a Flex-start permission denial.
Further progress requires a longer bounded queue window or a confirmed capacity
allocation. No controller remains running. The complete checkpoint suite passed
18 tests with two skips. Evidence: campaign `status.json`, `controller.log`,
`request-alpha-status.json`, `cleanup-queues.json`, and `cleanup-nodes.json`.

## Four-source v5e-16 retry (September 24)

The first v5e-16 Flex-start request obtained 16 TPU chips across four worker VMs.
Hardware validation stopped because SSH endpoint indices and TPU runtime process
indices differ. This is a valid permutation, not evidence of missing devices.
The queue and VMs were deleted and their absence independently verified. No
full-model update or model checkpoint was produced by that attempt.

The coordinator now validates a complete runtime-rank permutation and discovers
the sole global training-log writer separately for each stage. It never assumes
SSH worker 0 owns JAX process 0. Per-host evidence stays separate, including when
other workers download the global log for validation. Bounded stage rendezvous,
shared pilot horizon selection, and partitioned global evaluation remain enabled.

The next immutable payload is prepared under `artifacts/yat-mixture-v5e16-0924`.
It implements the proposed **60/20/15/5 expected nonpadding-token recipe**:
FineWeb2-HQ, English FineWeb-Edu, filtered FineWeb2 training text in Hindi/Urdu/
Swahili/Thai/Korean/Bengali, and multilingual Wikipedia reference text. Source
revisions, licenses and document coordinates are retained under
`artifacts/yat-base-mixture-0924`. Wikipedia is reference replay; the selection
is not independently qualified as technical-domain training data. This bounded
sample is an architecture adaptation pilot, not the complete upstream corpora.

`MixtureRows` applies token-count temperature exponent 0.5 within each source and
corrects sampling probabilities for padding occupancy. The manifest fixes source
weights, so changing the recipe invalidates checkpoint resume. Training logs and
campaign reports expose realized source and language tokens. Shares are sampling
expectations, not exact per-minibatch quotas. Sampling is with replacement; hard
per-document repetition caps and production-scale quality qualification remain
open. The pilot remains capped at approximately 50M useful tokens by measured
throughput and lease time. English replay and six missing languages are now in
this prepared recipe; no downstream benchmark test split is an input source.

Documents use deterministic splits and normalized exact deduplication. Complete
training documents sharing an exact 50-token span with validation are quarantined
before tokenization. These checks do not establish semantic decontamination or
audit the original mmBERT checkpoint's pretraining exposure.

Target: v5e-16 Flex-start, global batch 128, two accumulation microsteps, rematerialized
activations, data-parallel replicated model/optimizer, sequence length 512. All
saved GCS checkpoints are retained. Validation precedes adaptation automatically;
physical success must be read from the new campaign evidence, not inferred from
local tests or successful resource allocation.

## Staged scaling and execution diagnostic (September 24)

The four-source attempt produced zero full-model optimizer updates before its
SSH connection broke. Transport reconnection incorrectly treated the existing
command lock as failure. The launcher now detaches one bounded worker job and
reattaches subsequent SSH sessions to its persisted status; it never starts a
second copy. A disconnect/reconnect test verifies one execution.

The follow-up 16-chip diagnostic separates input placement, JAX lowering,
compilation and execution. All four hosts lowered in approximately 18 seconds
and compiled in 91–97 seconds. No update completed during roughly eight minutes
of first execution. This is an execution bottleneck, not evidence that the
compiler takes 47 minutes. Evidence is under
`artifacts/yat-diagnostic-v5e16-0924/diagnosis.json`.

An opt-in `--yat-local-shards` path runs the YAT FFN and attention geometry within
explicit data-parallel `shard_map` regions. Cancellation-repair loops and branch
predicates then act on one chip's local examples; shared parameter gradients
are reduced by the transpose of the shard map. It preserves fixed bias 1,
epsilon .01 and trainable alpha. This is not FSDP: parameters and Adam state
remain replicated. Global batch must divide across devices and accumulation
steps. The evaluator pads small batches with ignored targets for this backend.
Eight four-device CPU tests cover forward/gradient agreement, narrow/wide
features, shard-specific repair, padding/packing, and exact training resume.
Physical speed and stability remain unvalidated until the new campaign passes.
The optimization remains opt-in rather than changing all existing models.

The authorized quality progression is approximately 50M, 200M, then 1B
**cumulative additional useful tokens** after architecture conversion. These are
adaptation/evaluation gates, not a claim that 1B tokens trains a competitive
307.8M model from scratch. A possible 6–10B continuation remains conditional on
learning curves, downstream results, measured cost and remaining credits.

Fresh data preparation is under `artifacts/yat-scale-data-0924`. It samples
additional row groups from the same pinned sources, excludes pilot source
coordinates, and writes resumable chunks with token counts, licenses and
checksums. The first target is 240M raw content tokens to allow filtering before
the 200M stage. Raw downloads are not training-ready: normalized deduplication,
held-out 50-token quarantine and prepared-mixture verification must complete.
The pilot validation documents stay fixed. A subsequent 1B stage requires a
larger fresh tranche; do not loop the original 102M corpus dozens of times.

For a deliberately new stage, `train_encoder` and `train_yat_mmbert` now accept
`--initialize-from-checkpoint PARENT --initialize-step STEP`. This transfers
integrity-checked model weights and records the parent step/metadata hash while
starting a **fresh optimizer, learning-rate schedule and data cursor**. It is
not optimizer-continuous training. Architecture and tokenizer must match, and
the output directory must differ. The optional released snapshot still verifies
geometry/tokenizer, but its weights are not reloaded. Use ordinary `--resume`
for interruption recovery within that new stage; changing data, horizon or
recipe still fails exact-resume identity checks. Do not combine initialization
and resume. A new-stage integration test verifies weight transfer, Adam counter
reset, provenance, changed-dataset admission and strict-resume rejection.

The September 24 credit observation still reads $821.35, expiring September 28;
it is not real-time usage. A separate audited reservation reconciliation under
`artifacts/yat-budget-audit-0924` canonicalizes duplicate campaign/worker ledgers,
charges completed requests for their entire lifetime at conservative rates,
retains full live reservations, and adds $2 ancillary allowance per request.
It additionally counts all $178.65 historical consumption of the marketing
credit and a $25 ancillary allowance, deliberately double-counting posted TPU
usage. That planning bound was $778.56 including the diagnostic reservation,
leaving $121.44 for admission under the user's original $900. None of these
planning numbers represent a Google invoice or permit overspending the credit.

### First full-model chip-local TPU measurement

The `yat-local-v5e16-0924` campaign completed all 32 calibration updates on
16 physical v5e chips across four hosts, with finite losses, accepted updates,
and zero dense MLM projection fallbacks. Steady-state aggregate throughput was
5,931 useful nonpadding tokens/s; median update time was 8.68 seconds. This
establishes executable full-model training, not competitive quality or completed
interruption recovery. Check the campaign summary for those subsequent gates.
At $9.60/hour, extrapolating this calibration gives approximately $22.48/50M,
$89.92/200M and $449.58/1B useful tokens, excluding setup, compilation,
checkpointing, evaluation, storage and interruptions. Larger stages need a fresh
budget admission and throughput/quality review; the old 150K/s baseline does
not describe this YAT configuration. The current bounded lease may train less
than the 50M gate. Report actual useful tokens before claiming a gate complete.

The first expanded prepared corpus is verified: 358,973,214 nonpadding tokens,
379,545 documents and 27 languages, including retained pilot training data.
Its validation token array exactly matches the original held-out set. Simulated
source shares are 59.94/20.03/14.96/5.07 percent. At 200M sampled tokens the
largest expected pool exposure is 0.665 passes; sampling with replacement still
allows individual rows to repeat. Preparation evidence is in
`artifacts/yat-scale-data-0924/prepared-240000000/verification.json`. The larger
1.2B raw-content tranche is still downloading and is not yet training-ready.

A dynamic active-tile loop experiment was rejected by existing reverse-mode
gradient tests and removed. The retained static-loop implementation passes all
25 BF16 YAT tests and eight four-device local-shard tests, plus lint/type checks.
The live frozen TPU payload never included the rejected experiment.

### Bounded JAX profiling

`train_encoder` and `train_yat_mmbert` accept `--profile-dir /tmp/trace`,
`--profile-skip-steps 2`, and `--profile-steps 3`. The window is relative to the
current invocation, including after resume. The default leaves profiling off.
Each process writes its own directory; synchronization occurs within the step
annotation, and normal/exceptional exit closes an active trace. CPU integration
checks actual XPlane output and exact checkpoint recovery with tracing enabled.
View the artifacts with XProf using the JAX profiling documentation. These traces
are measurements, not proof of efficient kernels. A separate one-chip experiment
matches the current four-examples-per-chip microbatch; it cannot diagnose
multi-host communication. The existing multi-host campaign stays frozen.

### September 25 optimization evidence (UTC)

Actual TPU XPlane traces identify the automatically differentiated repair loop
as the main cost in a controlled attention benchmark: one backward while-loop
accounts for about 961 ms of 1,036 ms total device-module time across three
captures. Nested conditional durations overlap that loop and must not be added.
At four examples, 512 positions, 12 heads, 64 features and radius64, measured
forward-plus-backward medians over ten timed iterations were:

| Variant | TPU milliseconds | Compiled temporary memory |
| --- | ---: | ---: |
| Original automatic backward | 345.76 | 220.6 MB |
| Explicit BF16 backward | 31.74 | 192.5 MB |
| Windowed explicit backward | 4.69 | 6.73 MB |

These are synthetic kernel measurements, not a 74x model-training claim.
The windowed output relative L2 error was 1.6e-6 and the largest gradient
relative L2 difference was 0.00309 against the original BF16 reference.
The fixture includes near-collisions and padding. CPU tests additionally cover
document isolation, partial tiles, radius zero, wide windows and all-padding
examples; eleven four-device tests include actual windowed encoder training and
exact interruption-style resume.

`--yat-attention-block-size 64` enables query tiling on local YAT layers only;
global layers keep the dense path unless the separate `--yat-global-attention-block-size` flag is enabled. Both settings default to zero; global query tiling is currently CPU-validated only. The primitive
now uses its explicit backward for narrow attention as well as wide FFNs;
`custom_backward=False` retains the automatic reference. `--donate-state` is
an opt-in model/optimizer buffer reuse path, verified to produce aliasing and
preserve exact CPU resume. The combined local-tiling/donation path has now passed physical multi-host validation (see follow-up below).

The single-chip full-model profiling child was killed with SIGKILL after four
accepted updates, while finishing a three-step trace. A 290 MB XPlane artifact
was preserved; this is a failed full-model run, not a successful qualification.
The exact kill cause was not recovered before VM cleanup. Profiling defaults
are now one captured step, Python tracing disabled and host tracer level1.
Profiled steps are explicitly labeled and excluded from throughput-based budget
planning. Existing frozen campaigns keep their original source and defaults.

See `artifacts/yat-profile-retry-0924/tpu-kernel-summary.json`,
`tpu-kernel-traces-summary.json`, and `math-notes.md` for measurements and
FlashAttention-style online-softmax/backward derivations. A fused streaming
YAT kernel remains pending; the windowed implementation is tiled XLA.

The preserved full-model trace contains three device training modules totaling
3,126.33 ms. Full-size attention repair loops cover 2,616.64 ms (83.7%), with
zero overlap among selected loop intervals. This independently confirms that
attention repair remains the dominant compute target even after introducing the
explicit backward. See `full-trace-attribution.json` and its extraction script;
the enclosing accumulation loop must not be counted again. This attribution
uses the captured execution only and does not qualify the SIGKILL-terminated run.


### September 25 multi-host profiling follow-up (in progress)

The frozen `yat-optimized-v5e16-0925` validation combines narrow explicit VJP,
local query tiles of 64 and model/optimizer donation on 16 v5e chips across four
hosts. Its 32-step calibration completed at 50,182 useful nonpadding tokens/s
(64,601 padded positions/s), about 8.5x the preceding 5,931 useful tokens/s
calibration. These timings exclude compilation, the profiled step and checkpoint
I/O; they are not an end-to-end job throughput or production-quality claim.
All four hosts exported XPlane traces. Exact interruption recovery and evaluation
have now passed on the four physical hosts. Model, optimizer and training-state
manifest comparisons passed after forced interruption and resume. The campaign has a $12 reservation
and independent deadline cleanup; $823.56 is cumulative conservative planning
exposure, not actual billed spend.

Current local code additionally tiles global queries without changing each row's
single full-key softmax. Twelve four-device CPU tests pass, including full-encoder
training and exact resume with both local and global tiles. This newer change is
not part of the frozen multi-host run. It is not a fused streaming FlashAttention
kernel.

Initialization remains a separate quality blocker. A preliminary held-out probe
uses four fixed validation rows truncated to 96 positions and 70 masked targets.
The original mmBERT loss is 1.974; direct YAT conversion gives 17.380. FFN scalar
calibration fitted on separate training rows gives 15.968. Attention temperature
alone gives 18.011, fitted Q/K scaling plus temperature gives 16.568, and combining
both branch calibrations gives 16.367. These tiny probes do not establish general
quality, but none justify promoting this initialization. Gains were fitted on
teacher layer states, so layerwise fit improvements do not guarantee a good
composed student. Calibration is diagnostic only and has not been applied to the
training launch. Bias stays 1, epsilon .01 and alpha remains trainable.

Evidence: `artifacts/yat-initialization-diagnostic-0925/report.json` and
`heldout-attention-probe.json`; the latter preserves all compared variants.


The prior bounded pilot finished 326 updates and saved checkpoint 326. Its
256-row language-balanced held-out evaluation scored 17.2496 over 14,904 masked
targets. Infrastructure passed; `production_quality_qualified` remains false.
The selected optimization regressions pass (67 tests), alongside 12 four-device
CPU tests. Ruff and Pyright pass for the touched model/training modules.


### Four-host trace attribution and rejected streaming prototype

All four XPlane profiles from the optimized validation were retrieved and parsed.
Across 16 devices, the captured training step spans 937.4–940.3 ms. Full-size
attention repair-loop interval unions cover 41.1–62.3% of device step time;
collective intervals cover 12.7–32.5%. The per-device correlation between repair
and collective duration is -0.980. This is consistent with repair-induced
stragglers causing waiting at collectives, not evidence that all collective time
is network transfer. Categories may overlap and must not be added as utilization.
The extraction selects full-size repair loop carriers, so it is not a complete
attribution of all YAT or FFN arithmetic. It measures one captured step only.
See `artifacts/yat-optimized-v5e16-0925/profile-findings.json` and
`profile_attribution.py` for reproducible selection and interval unions.

The campaign now accepts `--yat-global-attention-block-size`; 22 focused campaign,
recipe and attention tests pass. This prepares a physical comparison but does
not establish a TPU speedup from global tiling.

An online-softmax BF16 reference was implemented and rejected for integration.
On a CPU fixture [1,256,2,64], it halves temporary memory but takes about 2.6x
longer than query tiling, with 5.6% Q/K and 38.8% alpha gradient drift. Six tiny
fixtures passed, illustrating why larger stress fixtures are necessary.
The prototype and rejection evidence are preserved solely under
`artifacts/yat-streaming-0925`; production code is unchanged. A fused Pallas
implementation needs a validated backward before promotion.

A branch-isolation probe on the same 70 held-out targets gives loss 16.881 for
FFN-only replacement and 8.940 for attention-only replacement, versus baseline
1.974. Teacher-state FFN scalar calibration gives 14.015; teacher-state attention
scaling gives 16.786. Both substitutions need recovery, with FFN replacement the
larger initial disruption in this small probe. Independent layerwise scaling is
not enough; student-state fitting/distillation requires separate evaluation.


### Student-state output-projection fitting diagnostic

A training-only sequential ridge fit adapts attention-output and FFN-output
matrices using actual student activations and teacher residual targets. It fits
32 training rows (96 positions), with a fixed .01 relative ridge prior toward
pretrained output weights. Alpha uses the earlier training-only FFN scale fit;
attention alpha remains 1, and YAT bias/epsilon remain fixed at 1/.01.

The original four-row/70-target probe improves from 17.380 to 11.518 loss.
A separate 64-row/128-position probe with 1,199 masked targets gives original
teacher 1.593, direct YAT conversion 17.566 and fitted YAT 13.059. Thus the
improvement survives the larger sample, but the student is still not qualified.
Weights and hashes are saved only as diagnostic artifacts, not installed into
training. Source: `artifacts/yat-initialization-diagnostic-0925/fit_student.py`
and `student-fit-report.json`.

The global-query-tiling physical comparison is submitted under
`artifacts/yat-global-v5e16-0925`, with 16 chips, the same batch128/accumulation2,
local/global query tiles64, donation and exact recovery checks. It has no pilot
and a $12 reservation; conservative cumulative planning exposure is $835.56,
not actual billed spend. The frozen payload passed archive preflight. Completion
and performance are still pending; the independent cleanup receipt is retained.


The streaming gradient diagnostic now includes an FP64 mathematical oracle
(`artifacts/yat-streaming-0925/oracle-summary.json`), used only for validation.
Its scalar alpha reduction has a cancellation ratio of 2306.84. Query-tiled and
streaming absolute alpha errors are .0310 and .0677, respectively, against an
oracle value .06372 and total absolute contributions146.995. Consequently the
large relative scalar error must not be interpreted alone as proof of training
failure. A separate FP32 scalar-gradient reduction leaves forward/Q/K/V unchanged
and improves alpha only marginally. Production precision is unchanged; broader
conditioning-aware checks and a dedicated backward remain necessary.


A follow-up sweep over three seeds and alpha .1/1 (six 128-position CPU cases)
compares both BF16 paths against the same FP64 oracle. Worst output errors are
.725% (query) and .739% (streaming); worst Q/K errors are 4.10% and 4.31%.
Worst alpha error normalized by the sum of absolute oracle contributions is
.0653% and .0348%, respectively. Streaming is not uniformly less accurate.
This revises the initial interpretation of the 39% scalar-gradient discrepancy:
it alone does not rule out streaming. The prototype remains unqualified because
it lacks physical performance and training evidence and a validated fused
backward, not because the single scalar relative-error figure proves it invalid.
See `artifacts/yat-streaming-0925/oracle-sweep.json` and `oracle_sweep.py`.


### Fused Pallas forward prototype

`artifacts/yat-streaming-0925/pallas_forward.py` implements tiled online YAT
attention inside a Pallas kernel, with sequential key tiles and parallel
batch/head/query tiles. Scores and direct near-collision repair stay inside the
kernel; it never returns a full score matrix. Nine CPU interpretation tests match
the streaming reference exactly, including multiple key tiles, partial queries,
local masks, document boundaries, all-padding examples and alpha0/.1/1.

Offline export/lowering with explicit abstract v5e and v6e devices succeeds at
sequence512/query32/key128. Target-specific MLIR and receipts pin source hashes
and JAX versions. This catches TPU layout and unsupported-operation failures
without provisioning another device. It does **not** establish hardware compiler
success, performance, VMEM sufficiency, or training correctness. No backward is
implemented and the prototype is not a selectable training backend.

TPU matmul lowering requires FP32 accumulators: BF16 inputs accumulate there and
the result is immediately rounded to BF16. Norms, direct distance repairs, YAT
scores, softmax and online state remain BF16. This accumulator detail must be
preserved in any advertised precision contract.

The active global-query-tiling comparison has logged 27 updates, with 26
non-compilation updates measuring 62,431 useful tokens/s (median .819s). This is
preliminary and excludes checkpoint I/O. Full recovery/evaluation remain pending.


The global-tiling 32-step calibration is complete: 62,592 useful tokens/s
(80,627 padded positions/s), a 24.7% improvement over the prior 50,182 useful
rate, excluding compilation/checkpoint overhead. Forced interruption completed;
resume qualification is still pending at this update. Final training loss was
17.2095, so this is a performance result rather than a quality qualification.

The analytical YAT backward oracle passes nine FP64 autodiff comparisons at
1e-10 tolerance for every Q/K/V/alpha gradient, including alpha zero. It is a
dense mathematical reference, not the fused implementation. The two-kernel Q/KV
reduction design and precision/masking requirements are recorded in
`artifacts/yat-streaming-0925/backward-design.md`; the reference is
`backward_reference.py`. The fused backward and physical Pallas validation remain
outstanding.


### Completed global tiling validation and experimental fused backward

The global-tiling campaign passed all physical four-host stages: calibration,
forced interruption, exact resume manifests and held-out evaluation. TPU node
and queued-resource inventories for us-west4-a are empty after teardown. Evidence
is in `artifacts/yat-global-v5e16-0925/latest-summary.json` and
`cleanup-verification.json`. Held-out loss17.2712 remains unqualified; the speed
improvement does not demonstrate improved model quality.

The validated configuration is selected using `--yat-local-shards`,
`--yat-attention-block-size 64`, `--yat-global-attention-block-size 64`, and
`--donate-state`. General defaults remain unchanged, preserving explicit
execution choices and strict checkpoint identity checks.

The experimental Pallas backward is now implemented in
`artifacts/yat-streaming-0925/pallas_backward.py`. Separate Q and KV kernels keep
output-gradient tiles in VMEM over their sequential reduction axis; they do not
emit quadratic partial-gradient arrays to HBM. Near-collision distance gradients
use direct coordinate differences. Alpha remains differentiable at zero.
Twenty-seven combined forward, backward and FP64-reference tests pass. Backward
CPU tests use explicit BF16-error tolerances, distinct from the strict FP64
mathematical-equation tests. Offline lowering passes v5e and v6e at length512.

This remains artifact-only experimental code: physical compilation, VMEM fit,
performance, larger numerical stress fixtures, custom-VJP integration and real
training/recovery qualification remain outstanding. A lowering receipt is not
hardware execution evidence. The BF16 normalization/backward reduction ordering
is not promised bit-identical to automatic differentiation of the dense path.


### Broader fused-kernel gates and bounded hardware request

The fused-kernel CPU suite now has36 passing cases, including negative alpha
(the trained scalar can cross zero). A frozen-payload interpretation benchmark
also passes forward/Q/K/V smoke gates at lengths128 and256. At length256,
forward drift against query tiling is .278%, Q/K about4.4%, while scalar alpha
qualification remains explicitly false. CPU interpretation is not a TPU speed
prediction.

The one-chip request `yat-pallas-v5e1-0925` is submitted with a $3 reservation,
a guarded30-minute lease and no model training. Cumulative conservative planning
exposure is $838.56, not actual billed spend. The payload is frozen and includes
small and model-shaped kernel compile/timing cases; it must not be modified in
place while live. Evidence and cleanup receipts will be stored in that root.

Long-context CPU interpretation found a real tile-size sensitivity: query32/key128
has .720%, .939% and3.503% forward drift at lengths512/2048/8192 respectively.
At8192, key256 reduces drift to .995%, key512 to1.029%; both pass the declared2.5%
forward smoke gate. Forward/backward offline v5e lowering succeeds at8192/key256.
These are single-fixture forward checks, not long-context backward or production
quality qualifications. The active hardware payload remains unchanged (key128,
maximum length512). See `long-forward.json` and `long-forward-key-tiles.json`.


### Physical compiler failure and long-context backward gate

The one-chip `yat-pallas-v5e1-0925` test failed while physically compiling the
forward kernel: Mosaic's infer-vector-layout pass rejected a BF16 vector32 to
32x1 column reshape. Offline lowering did not run this later hardware compiler
pass, so it was insufficient to establish physical compilability. No fused
kernel speed result was produced. The controller has initiated cleanup; retain
`tpu-benchmark.log` and `tpu-summary.json` as failed evidence.

Long-context interpretation with key256 also fails the Q/K backward smoke gate:
roughly10.8% drift at2048 and15.6% at8192 against the query-tiled reference. A
separate compensated BF16 accumulation candidate barely changes these errors
(10.8%/15.6%), so compensation alone is not a fix. These are reference comparisons,
not claims about exact mathematical error. Forward accuracy does not establish
backward or training accuracy. The compensated candidate is not integrated.

A separate layout-only forward candidate (`pallas_forward_layout.py`) widens
values only during broadcasting, then returns to BF16 before arithmetic. It
introduces no FP32 distance arithmetic. Twelve CPU interpretation tests and
offline lowering pass, but physical compilation is still unverified. The frozen
failed hardware payload remains unchanged. Further work must address both the
hardware layout constraints and backward normalization accuracy before training.


### Mathematical reference revises the backward diagnosis

A bounded-query FP64 oracle at lengths512/2048/8192 shows the initial long-context
smoke comparison was not a reliable accuracy verdict: it treated the existing
BF16 automatic backward as ground truth. At8192, that existing path has15.9–16.0%
Q/K relative error versus FP64, while the Pallas analytical backward has2.1–2.4%.
At2048, the respective ranges are11.2–11.4% and1.8–1.9%. Compensation and row-stat
recomputation did not close the inter-implementation difference because the
reference derivative's BF16 rounding was a substantial source of that difference.
These conclusions are limited to the recorded fixtures. See
`artifacts/yat-streaming-0925/long-backward-oracle.json`.

The harness now factors the BF16 softmax Jacobian explicitly as
`p * (g - sum(p*g))`. Forward arithmetic and values are unchanged; the automatic
reference is retained through `softmax_bf16(..., custom_backward=False)`, which
also supports forward-mode AD. Factoring the derivative reduces Q/K error against
the oracle to about1.7%/3.0%/2.1–2.4% at512/2048/8192. Alpha absolute error at8192
improves from.0732 to.0328 (oracle gradient.0913); scalar relative error remains
conditioning-sensitive. This is an accuracy improvement on fixtures, not a model
quality claim. All distance and softmax gradient arithmetic remains BF16.

Fifty-four focused tests and twelve four-device training/exact-resume tests pass;
Ruff and Pyright pass. HLO tests check absence of FP32 operations in the BF16
softmax gradient. This new numerical change has **not yet been physically TPU
validated** and changes source identity; do not resume an old checkpoint silently.
Previously frozen physical campaigns remain evidence for their original source.
Diagnostic scripts now explicitly select the preserved automatic derivative when
reproducing historical comparisons.

The failed one-chip fused-kernel resource and queue are now confirmed deleted;
see `artifacts/yat-pallas-v5e1-0925/cleanup-verification.json`. The fused kernel's
hardware layout failure remains unresolved by physical evidence, independently
of its improved agreement with the mathematical gradient oracle.


The explicit softmax-backward physical campaign is now submitted as
`artifacts/yat-softmax-v5e16-0925`: same16-chip/four-host batch128/accumulation2,
local/global query tiles64, donation, interruption/resume and evaluation checks.
Its frozen archive preflight passed. The $12 reservation brings conservative
planning exposure to $850.56; this is not billed spend. The independent cleanup
receipt is in its attempt directory. No model pilot is included.

The layout-only Pallas repair now covers both forward and backward
(`pallas_forward_layout.py`, `pallas_backward_layout.py`). Twenty-four CPU tests
pass, and the repaired backward lowers offline for v5e. Values are widened only
for layout broadcasting and rounded back before arithmetic. Physical compilation
is still unverified; offline export does not execute the failing hardware
infer-vector-layout pass. Neither layout candidate is integrated into training.

### BF16 softmax offset stress test (September 25, 2026 UTC)

The factored VJP remains sensitive to common cotangent offsets when rounded probabilities do not sum exactly to one. An experimental first-coordinate-centered, mass-normalized VJP passed six CPU tests (constant null direction, FP64 softmax oracle, BF16-only gradient HLO). It is **not integrated**: the composed attention oracle at 8,192 tokens worsened Q relative gradient error from 2.10% to 10.85% and alpha absolute error from 0.03282 to 0.25806 in the tested fixture. Isolated softmax tests are insufficient to qualify this change. Reproduction and results: `artifacts/yat-streaming-0925/test_softmax_centered.py`, `centered_softmax_oracle.py`, `centered-candidate-status.json`.

The frozen softmax16 physical run uses the earlier factored VJP, not this rejected candidate. Its hardware/preflight evidence confirms four hosts and 16 TPU devices; training/recovery completion was still pending at this inspection.

A second experimental VJP uses a probability-weighted center followed by a residual correction, all in BF16. Six isolated softmax tests pass. Across 18 attention fixtures (two seeds, three input scales, three positive/negative alpha values), worst Q/K relative error improves from 11.36% to 4.66%; the 8,192-token fixture retains approximately 2.1%/2.4% Q/K errors. However, alpha absolute error regresses from 0.08185 to 2.37128 in one scale-1/alpha-1 fixture. It remains **artifact-only, not integrated or physically qualified**. See `artifacts/yat-streaming-0925/refined-candidate-status.json` and `refined-softmax-sweep.json`. Further work must distinguish cancellation in scalar alpha reduction from errors in score gradients; Q/K improvements alone do not qualify training.

FP64 contribution analysis clarifies the refined candidate’s scale-1 alpha regression: the oracle gradient is -101.777526, so its 2.37128 absolute error is 2.33% relative (the factored variant is 0.0804%). The sum of absolute contributions is 7,699.15, giving a cancellation ratio of 75.65. The independent NumPy oracle agrees within 4.3e-14. This is a real regression, not a sign failure; conditioning alone does not localize its source. Evidence: `alpha-condition-sweep.json`. The live frozen softmax16 run has now completed calibration successfully; recovery and heldout evaluation are pending.

### Fused local-attention grid optimization (September 25, 2026 UTC)

The artifact-only Pallas prototype now has a local grid that skips tiles outside the radius in forward, Q backward, and KV backward. For length 8,192, radius 128, query tile 32 and key tile 128, forward/Q-backward key visits fall from 64 to 4 per output tile; KV-backward query visits fall from 256 to 13. This is a work-count reduction, **not measured TPU speedup**. Sixteen forward and fifteen backward CPU interpretation tests pass, including exact equivalence to the full-scan prototype on document boundaries, padding and non-aligned windows. Both forward and backward offline v5e lowering pass at 8,192 tokens. Physical Mosaic compilation, hardware execution and training integration remain unverified. Evidence: `artifacts/yat-streaming-0925/local-grid-status.json`.

The refined BF16 softmax backward microbenchmark (64 rows, 512/2,048/8,192 columns) is 1.45–1.53x slower than the factored variant on CPU, with about 1.97x temporary memory. This is not a TPU or whole-model performance prediction. It remains experimental. Evidence: `softmax-cpu-cost.json` in the same artifact directory.

The frozen softmax16 run completed its resume subprocess successfully after the intended SIGKILL. Exact-resume manifest verification, heldout evaluation and cleanup still require the final summary. A separate bounded one-chip v5e test of the revised Pallas layouts and local grids has been submitted under `artifacts/yat-pallas-local-v5e1-0925`, with a $3 reservation and 1,800-second lease. Frozen archive CPU preflight passed. Conservative cumulative planning exposure is $853.56; this is not actual billed spend.

### Frozen factored-softmax physical result

The softmax16 four-host/16-chip validation completed successfully (controller exit 0). Independently downloaded checkpoint manifests match exactly for model, optimizer, and training state after the step-16 SIGKILL and resume to step 32. Measured throughput is 61,887.85 useful tokens/sec, compared with 62,592.40 in the prior global-query-tiling run; this is not a demonstrated speed improvement. Balanced-language heldout evaluation covers 256 rows and 14,904 masked targets, with loss 17.333903. Physical correctness is qualified for this tested configuration; production quality remains false. Evidence: `artifacts/yat-softmax-v5e16-0925/latest-summary.json` and `recovery-verification.json`.

### Local fused kernel hardware result and FFN distillation probe

The one-chip local-kernel run compiled and executed the initial global forward case (batch 1, length 128, 2 heads): Pallas median 0.1389 ms and 315,392 temporary bytes, reference 0.1446 ms and 1,084,416 bytes. The backward compilation then failed at the BF16 `[32,128]` to `[32,128,1]` reshape used in distance-repair weighting. **The physical numerical gate did not complete, and no longer local-attention case ran.** No training qualification or whole-model speedup follows. The frozen source is preserved; a separate `pallas_backward_local_layout2.py` candidate widens only for broadcast layout and casts back before arithmetic.

A training-only single-layer FFN distillation diagnostic uses 8 training rows and 8 separate heldout rows from the training pool, 64 positions each. After 96 Adam updates, heldout relative squared output error falls from 0.97385 (raw conversion) to 0.04789; least-squares alpha scaling alone yields 0.70921. Bias 1 and epsilon 0.01 remain fixed, alpha is trainable, and YAT distance arithmetic remains BF16. This is teacher-state layer-0 approximation, **not composed-student or end-to-end MLM quality**. Reproducible script, weight hash and output: `artifacts/yat-initialization-diagnostic-0925/distill_ffn.py`, `ffn-distillation-report.json`, and `ffn-distilled-layer0.npz`.

The saved layer-0 distilled FFN was evaluated inside the original encoder on 32 heldout validation rows (128 positions; 561 masked targets; seed 2917). Original mmBERT loss: 1.570533; raw layer-0 YAT replacement: 3.044110; distilled layer-0 YAT replacement: 1.689960. All remaining FFNs and attention blocks are original mmBERT. This supports further progressive-conversion tests but does not qualify an all-YAT encoder. Evidence: `artifacts/yat-initialization-diagnostic-0925/distilled-ffn-mlm-probe.json`; weights and validation manifest hashes are recorded.

A layout2 hardware-test payload is frozen and passes archive preflight, under `artifacts/yat-pallas-layout2-v5e1-0925`. Its benchmark now persists results after each path and checks forward numerical agreement before starting backward compilation, so a later failure no longer discards earlier measurements. It was prepared but not yet submitted at this entry.

### Progressive FFN conversion diagnostic

A sequential conversion of all 22 FFNs fits each branch to the teacher residual using the current student activations, leaving all attention unchanged. The 8-training-row/96-update-per-layer experiment yields MLM loss 12.444589 on the same 32-row/561-target diagnostic set, versus raw all-FFN conversion 15.970797 and teacher 1.570533. This is an improvement but remains far from usable; later-layer branch approximation largely fails. This repeatedly used set is diagnostic, not a fresh qualification holdout. A separate larger local experiment now uses 128 training rows, 16 probe rows, and 512 updates per layer to test the small-sample limitation. No cloud training is launched by these scripts.

The prior local-kernel TPU and queue were confirmed absent. The layout2 kernel test is now submitted under `artifacts/yat-pallas-layout2-v5e1-0925` with a $3 reservation; cumulative conservative planned exposure is $856.56, not actual spend.

The larger progressive FFN fit (128 training rows, 512 updates/layer) improves diagnostic MLM loss to 8.6250 versus the small fit’s 12.4446, but teacher loss remains 1.5705. On 64 separate validation rows disjoint from that diagnostic set (1,185 targets), teacher/raw-all-FFN/distilled-all-FFN losses are 1.3712/16.7005/9.1895. All attention remains original; production qualification is false. The larger sample does not solve accumulated conversion error. Evidence: `artifacts/yat-initialization-diagnostic-0925/progressive-conversion-conclusion.json`.

Primary-source method review suggests testing end-to-end progressive replacement rather than assuming independent branch fits compose: [BERT-of-Theseus](https://arxiv.org/abs/2002.02925) increases stochastic module replacement probability during training; [Deterministic Continuous Replacement](https://arxiv.org/abs/2511.18670) anneals a deterministic blend and reports a limited single-seed controlled study; [TinyBERT](https://arxiv.org/abs/1909.10351) uses Transformer-layer distillation. Applying these to native YAT is an untested inference, not a literature-backed guarantee of mmBERT parity. Any transition must end at pure YAT branches, with fixed bias 1, epsilon 0.01 and trainable alpha, and must pass fresh heldout evaluation before long training.

The layout2 TPU run passed the initial forward numerical smoke check (relative L2 error 0.00313548, shape 1x128x2x64). Backward compilation progressed beyond the prior reshape and failed on scalar BF16 `aa*2`. A separate layout3 candidate changes this to multiply the vector by 2 before multiplying by alpha; no FP32 distance arithmetic is introduced. Larger/local cases and backward execution remain unqualified.

The experimental `replacement.py` implements a continuous whole-block transition with exact teacher/student endpoints and student-only differentiation, preserving input gradients through frozen teacher modules. Two CPU tests pass for endpoints, padding and finite gradients. A 96-update, two-layer randomly initialized smoke test ending at pure YAT reduces logit MSE from 4.976e-7 to 3.039e-7 with learning rate 1e-4; teacher parameter hashes remain unchanged. The first 1e-3 attempt regressed to 1.288e-6 and is preserved in `replacement-training-smoke.log`. This fixed-batch toy result is **not a pretrained quality or TPU training result**.

The replacement prototype now reuses shared frozen outer modules (embeddings, norms and MLM head), eliminating a duplicate vocabulary projection when those objects are shared. Student differentiation is limited to transformer blocks; both teacher and student NNX DiffState filters explicitly mark shared outer parameters frozen to avoid inconsistent-aliasing errors. The corrected 96-update tiny shared-state run reduces pure-student logit MSE from 4.976e-7 to 9.371e-8 with teacher hashes unchanged. This remains a fixed-batch optimization smoke, not pretrained adaptation. Regression tests cover shared-state updates and independent student graph/state reconstruction.

Layout2 TPU and queued resource were confirmed deleted. Layout3 hardware validation is submitted under `artifacts/yat-pallas-layout3-v5e1-0925`, with a $3 cap and 30-minute guarded lease; conservative planned exposure is $859.56. No long model training has been launched.

### Pretrained replacement plumbing and controlled follow-up

A local 16-step run uses the real pretrained snapshot, prior progressively distilled FFN initialization, native YAT attention, shared frozen outer modules, trainable student transformer blocks, and a masked-logit KL objective. It reaches pure YAT with finite updates and unchanged teacher parameters. On an 8-row/16-target truncated heldout diagnostic, teacher/initial-student/final-student MLM losses are 1.8121/13.0526/14.4041: **the short transition worsens quality**. This is plumbing evidence, not an accepted training recipe. Report: `artifacts/yat-initialization-diagnostic-0925/pretrained-replacement-report.json`.

A matched local comparison now uses the same initialization, 32 training rows, 8 heldout rows, learning rate 3e-5 and 128 updates per condition: a 96-update replacement ramp versus direct pure-YAT distillation. Both end at pure YAT and train only student transformer blocks. It is intended to test whether the curriculum contributes, not to assume it helps. Reports will be `pretrained-replacement-ramp-report.json` and `pretrained-replacement-direct-report.json` in the same directory.

### Fused forward/backward physical smoke completed

The layout3 run passed physical forward and Q/K/V smoke checks for four fixtures, through local length 2,048 and batch4/length512/heads12. This is not alpha-gradient or model-training qualification. At batch4/512/12, fused backward is 14.8438 ms versus reference 4.33024 ms (3.43x slower), with temporary bytes 516,096 versus 6,619,648. The fused implementation remains experimental. A separate conditional-backward candidate skips coordinate repair when no pair is sensitive; its 15 CPU tests and offline v5e lowering pass, including query tile128 at length8192. Hardware performance of this candidate remains unmeasured.

The matched pretrained comparison completed: initial heldout loss 12.032419; 96-step-ramp final loss 13.138935; direct-distillation final loss 10.083004 after 128 updates each. Data indices, initialization, teacher baseline and update counts match, and teacher parameters are unchanged. This tiny 8-row/16-target diagnostic favors direct distillation for the next broader test, but does not establish general schedule superiority or production quality. Evidence: `artifacts/yat-initialization-diagnostic-0925/pretrained-replacement-comparison.json`.

### Persisted direct-distillation weights and broader evaluation

The direct-distillation run was reproduced and exported as checked, weights-only transformer-block parameters. The diagnostic export validates checksum, exact configuration, complete parameter paths, shapes, dtypes and finite values before applying any update; two round-trip/rejection tests pass. It explicitly does not support optimizer resume. On 64 validation rows disjoint from the listed earlier diagnostic rows (1,070 masked targets, 128 positions), original teacher/initialized-all-YAT/trained-all-YAT losses are 1.377852/13.042000/10.738688. This supports an improvement beyond the tiny training diagnostic but remains far from teacher quality. Source weights and validation hashes are recorded in `artifacts/yat-initialization-diagnostic-0925/pretrained-direct-broader-probe.json`.

The prior layout3 TPU and queue are confirmed absent. A bounded conditional-repair/query-tile sweep is submitted at `artifacts/yat-pallas-conditional-v5e1-0925`, with query tiles32/64/128 and a length8192 local-attention case. A one-step JAX trace is captured separately from timing for the largest batch/query128 backward case. Archive preflight passed. Reservation $3; conservative cumulative planned exposure $862.56, not actual billed spend.

### Experimental fused custom VJP

`artifacts/yat-streaming-0925/pallas_attention_vjp.py` wraps the fused forward and analytic backward as a JAX custom VJP. It saves row maxima/normalizers and the forward output alongside Q/K/V, and passes them to backward rather than recomputing forward. Six CPU tests pass for exact equivalence to the explicit kernel backward, signed/zero alpha, padding and local/global masks; a patched callback verifies that backward does not call forward. The wrapper passes offline v5e lowering at local length8192/query128. Physical custom-VJP execution, model integration, alpha qualification and training remain unverified. It is not enabled in the encoder.

### Conditional repair / tile sweep and JAX trace

Physical v5e smoke checks passed for query tiles32/64/128 at batch4/length512/heads12 and query128 at local length8192. At length512, fused backward falls from 8.869 ms (conditional query32) to 5.540 ms (query64) and 5.076 ms (query128), versus 4.341 ms for the query-tiled XLA reference. At length8192/batch1/heads2, fused forward/backward are 0.777/2.183 ms versus reference 1.535/4.594 ms. These are isolated kernel measurements, **not model-training speedups**; alpha and training qualification remain incomplete.

The separately captured query128/length512 JAX profile covers 4.864 ms of device module time; 4.785 ms belongs to three Pallas-named operations. Their durations are approximately 1.356 ms forward, 1.598 ms Q backward, and 1.830 ms KV backward, identified by result shapes. Layout copies are a small part of this capture. The new `scripts/summarize_jax_trace.py` summarizes interval unions and warns that categories overlap and collective intervals can include waiting. It was tested on real completed XPlane files; 16 trace-summary/CI-scope tests, Ruff and targeted Pyright pass.

A new static alignment bound uses gcd(query_tile,key_tile) to reduce visited local tiles safely. For query/key128 and radius128 it visits3 rather than4; the reverse KV grid similarly reduces4 to3. Thirty-three CPU coverage/forward/backward tests pass, and offline v5e backward lowering at length8192/query128 passes. This tighter-grid candidate is not yet measured on hardware. Evidence: `artifacts/yat-streaming-0925/window_tiles.py`, `tight-grid-tests.log`, and `artifacts/yat-pallas-conditional-v5e1-0925/trace-summary.json`.

### Profile-driven custom-VJP validation and numerical probes

The conditional TPU and queue are confirmed absent. A new single-chip v5e test combines the tighter static window bounds with saved-forward custom VJP state. It compares synchronized forward and `value_and_grad` timings after compilation/warmup, and captures one separate JAX trace. The frozen run is `artifacts/yat-pallas-vjp-v5e1-0925`, reservation $3, guarded lease30 minutes, conservative cumulative planning exposure $865.56 (not actual billed spend).

A new FP64 mathematical-oracle diagnostic (CPU only) exercises local/global masks, padding, alpha -0.1/0/0.1/1, and identical values. Identical values expose small nonzero Q/K/alpha gradients from BF16 probability/reduction rounding in both the query-tiled reference and fused candidate, although exact attention is independent of Q/K/alpha in this case. This remains a numerical qualification gap. The FP64 oracle is test-only; kernel distance arithmetic stays BF16. Report: `artifacts/yat-streaming-0925/tight-vjp-oracle.json`.

An experimental value-centering wrapper restores exact constant-value output and zero Q/K/alpha gradients in eight CPU fixtures. However, its first-token anchor worsens random-input forward relative error to as much as 8.1% versus approximately 1% for the uncentered fused kernel. It is **rejected as a default**; no production implementation changes were made. Report: `artifacts/yat-streaming-0925/centered-value-oracle.json`.

### Tight-grid custom VJP on physical v5e

The frozen custom-VJP run passed physical forward/Q/K/V smoke gates for global length512, batch4/heads12 local length512, and heads2 local length8192. The benchmark reference uses query tile32; production tile64 and other reference tile choices must be measured before any claim of best implementation. Synchronized `value_and_grad` medians:

| Fixture | XLA tile32 | Fused | Ratio |
| --- | ---: | ---: | ---: |
| Global B1/L512/H2 | 0.6075 ms | 0.4065 ms | 1.49x |
| Local B4/L512/H12 | 5.3492 ms | 4.4144 ms | 1.21x |
| Local B1/L8192/H2 | 5.9668 ms | 1.7381 ms | 3.43x |

At local length512, compiled temporary storage falls from7,103,488 to516,096 bytes, but fused forward alone remains slower (1.3688 versus1.1793 ms). Q/K relative differences versus BF16 reference reach4.2% in this fixture; alpha relative difference is2.35%, and alpha is still **not qualified**. These are isolated kernels, not end-to-end model training throughput or quality results.

The separate JAX capture reports4.1997 ms of device module time, including4.1164 ms of Pallas-named operations. Forward/Q-backward/KV-backward are1.1535/1.3993/1.5630 ms. Outputs, trace and machine-readable summary are under `artifacts/yat-pallas-vjp-v5e1-0925`.

Mean-value centering improves the small random-input error versus the first-token anchor but fails packed-document and padding isolation in BF16. Perturbing another document changes target-document output by up to0.04565; perturbing padding changes valid output by0.03394 in the probe. Both centered candidates are rejected. The uncentered kernel has exactly zero differences and cross-document/padding gradients in this probe. Four new harness regression fixtures enforce document/padding isolation in mixed and BF16 modes, for local and global attention. All12 windowed-attention tests and targeted Ruff pass. Evidence: `artifacts/yat-streaming-0925/centering-isolation-probe.json`.

The custom-VJP run controller exited0 and both its TPU VM and queued resource are confirmed absent (`cleanup-verification.json`). A matched tile32/64/128 follow-up is prepared and its frozen CPU preflight passed at `artifacts/yat-pallas-tiles-v5e1-0925`; it has not been submitted or reserved.

### Matched reference-tile sweep and independent gradient oracle

The physical follow-up compares the same fused value-and-gradient kernel against XLA query tiles32/64/128 at identical shapes and masks, with20 synchronized timed calls after warmup. All Q/K/V smoke gates pass. Results supersede the tile32-only speedup interpretation:

| Fixture | XLA32 | XLA64 | XLA128 | Fused128 | Fused vs best tested XLA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global B1/L512/H2 | .5779 ms | .6415 ms | .8062 ms | .4116 ms | 1.40x |
| Local B4/L512/H12 | 5.3611 ms | 6.1573 ms | 9.6666 ms | 4.4246 ms | 1.21x |
| Local B1/L8192/H2 | 5.9874 ms | 3.5605 ms | 2.1146 ms | 1.7274 ms | 1.22x |

The long-sequence3.43x figure was against tile32, not the best reference. The current evidence supports approximately22% more isolated-kernel throughput over the best tested reference at8192. It does not justify changing the production model's tile policy from a microbenchmark: tile64 previously performed best in an end-to-end training measurement. Separate batch4/512 JAX traces show5.131 ms XLA device module time versus4.199 ms fused, with4.116 ms inside the three fused operations. Nested XLA while/conditional event times overlap and must not be added. Reports: `artifacts/yat-pallas-tiles-v5e1-0925/results/benchmark.json` and `trace-summary.json`.

A reusable NumPy/FP64 attention-gradient oracle now lives in `tests/yat_attention_oracle.py`; it uses direct coordinate differences and independent analytic derivatives. Six finite-difference fixtures validate signed/zero alpha and local/global masks, plus an exact constant-value fixture. Six BF16 fixtures compare actual Q/K/V/alpha gradients to this oracle; scalar-alpha error is normalized by the sum of absolute analytic contributions to account for cancellation. These are bounded regression tolerances, not full model-quality or fused-alpha qualification. The13 oracle tests plus windowed/CI suites total39 passes; targeted Ruff passes. CI selects the oracle tests when the helper changes.

A diagnostic FP32 scalar-alpha reduction (leaving BF16 geometry, forward and Q/K/V unchanged) improves7 of8 random-input fixtures but does not resolve constant-value null-direction errors. It remains artifact-only, not a production change. Report: `artifacts/yat-streaming-0925/fused-alpha-reduction-comparison.json`.

### Gradient scale sweep and probability-selected centering

A CPU interpretation sweep compares fused and current XLA attention against the independent NumPy FP64 oracle over 32 fixtures: local/global masks, Q/K scales 0.05/0.2/1/3, signed/zero alpha, padding and a valid near-collision pair. The initial separate random-input sweep accidentally placed the near-collision pair in padding; it is retained as `fused-gradient-scale-random.json`, and the corrected valid-pair sweep is `fused-gradient-scale-sweep.json`. Neither is a physical TPU qualification.

The corrected sweep finds a global scale0.05/alpha0.1 case with only 0.54% forward error but Q/K gradient errors of 15.5%/13.4% for fusion and 24.7%/21.5% for current XLA. This is a shared numerical weakness, not solely a fused-kernel regression.

A new backward-only prototype anchors the cotangent to the key with maximum stored probability, then subtracts its probability-weighted mean using BF16 arithmetic and the rounded probability mass. Forward values are unchanged. In the same 32 fixtures, maximum Q/K relative errors fall to 4.07%/4.02%; the identified worst case falls to 2.77%/1.71%. This does not universally solve alpha conditioning: the maximum alpha error normalized by absolute analytic contributions remains 1.18%. No production default changed.

Six additional softmax tests pass, covering lengths 17/512/8192, exact constant-cotangent zero gradients, offset stress and unchanged forward values. Offline v5e lowering passes at length8192/tile64; hardware compilation and timing remain unverified. Evidence: `artifacts/yat-streaming-0925/softmax_maxcenter.py`, `maxcenter-sweep-comparison.json`, `maxcenter-tests.log` and `windowed-maxcenter-tpu-v5-lite-8192-k64.json`.

The matched-tile controller exited successfully, and both its TPU VM and queued resource are confirmed absent in `artifacts/yat-pallas-tiles-v5e1-0925/cleanup-verification.json`. No further cloud run was launched during the scale-sweep investigation.

### Max-centered softmax: physical TPU and actual-activation checks

The single-chip v5e comparison completed successfully at `artifacts/yat-maxcenter-v5e1-0925`. It uses a $3 reservation and guarded 30-minute lease; cumulative conservative planned exposure is $871.56, not billed spend. Across the 32 physical numerical fixtures, current/max-centered worst Q errors are 7.65%/2.22% and K errors are 6.64%/2.27%. Forward outputs are exactly identical and all outputs/gradients are finite. Maximum alpha error normalized by absolute contributions falls from 1.377% to 0.665%. These TPU results differ from CPU arithmetic results and remain bounded synthetic evidence, not a model-quality qualification.

Best measured value-and-gradient times, current versus centered: global B1/L512/H2 0.598/0.632 ms; local B4/L512/H12 5.348/5.816 ms; local B1/L8192/H2 2.112/2.351 ms. Centering costs approximately 6–11% in these isolated operators. Separate batch4 traces show 5.131 versus 5.598 ms device module time. Production defaults remain unchanged.

A separate CPU probe captures actual Q/K/V from layers 0/7/14/21 of the existing directly distilled student, one validation row at 128 positions and heads 0/5. Layer0 current Q/K gradient errors versus the independent FP64 oracle are 30.2%/54.4%; centering barely changes them. Partial-intermediate substitution shows that BF16 scores with otherwise exact geometric derivatives reproduce 30.3%/54.4%, implicating forward-score rounding in this case. Isolated dot-product, distance and final-score rounding each contribute; their effects are nonlinear and should not be added. Other layers are less sensitive. This one-row CPU probe does not establish TPU model-wide error. Evidence: `real-attention-gradient-probe.json`, saved `real-attention-layer-*.npz` fixtures, and `score-quantization-attribution.json` under `artifacts/yat-initialization-diagnostic-0925`.

Algebraic score reassociation was also tested diagnostically. Dividing before multiplying reduces layer0's score-induced K error from54.4% to30.8%, but it does not eliminate it and is not uniformly better across layers. No score formula changed in production.

The matched 64-row/1,070-target MLM evaluation checks whether switching the same frozen student's YAT geometry to mixed precision immediately recovers quality. BF16/mixed losses are 10.738688/10.761568. Mixed precision does not rescue this checkpoint's quality; this does not establish that precision has no effect during training. Weight/data hashes and identical rows are preserved in `pretrained-direct-precision-probe.json`. The poor pretrained conversion remains a separate quality problem.

### Core API integration of the optional backward method

The TPU-tested algorithm is now available through `softmax_bf16(..., backward_mode='max_centered')` and `windowed_yat_attention(..., softmax_backward_mode='max_centered')`. The original factored backward remains the default. The new option is explicitly rejected with the automatic-reference mode or mixed-precision attention, where it would otherwise be ignored. Encoder configuration and trainer selection are not yet wired; any future training selection must be recorded in training identity rather than silently changing resume semantics.

Twenty-one focused softmax/windowed tests pass, including the independent near-collision gradient oracle and masked-cotangent isolation. The preceding broader primitive/oracle run passed65 tests before the two new windowed-option fixtures were added. Encoder/recipe tests pass22 cases; the12 sharding cases require a separate four-device process. Targeted Ruff and Pyright pass. The frozen TPU prototype remains archived separately from this subsequent core API integration.

The max-centered TPU run controller completed, and its TPU VM and queued resource are confirmed absent in `artifacts/yat-maxcenter-v5e1-0925/cleanup-verification.json`. No new TPU was allocated for API integration.

The separate four-device CPU run completed: all12 local-sharding tests pass. This checks CPU partitioning compatibility; it is not a substitute for physical multi-host validation of the new opt-in method.

### Encoder/trainer integration and four-host validation submission

`EncoderConfig.yat_softmax_backward` and `--yat-softmax-backward` now select the optional method through dense, query-tiled and chip-local attention. Both `train_encoder` and `train_yat_mmbert` support it; the TPU validation harness forwards it. Configuration validation rejects a non-BF16/non-YAT selection. The mode is stored in the resolved encoder configuration and checkpoint identity. Tests prove exact resume rejects switching methods, and stage initialization does not classify the gradient strategy as an execution-only override. Older configuration dictionaries infer only the original factored default; source/config identity checks remain strict.

The two new end-to-end CPU fixtures perform dense/tiled training, interruption, exact model/optimizer recovery and mode-change rejection. The related training/recipe/stage suite passes15 tests. The expanded four-device CPU suite passes18, including both gradient methods through local sharding and model training/resume. CI/pilot checks pass19; targeted Ruff/Pyright pass. The artifact-only weights loader accepts the omitted legacy factored field but rejects an explicit strategy mismatch (three tests pass).

A frozen full-model run is submitted at `artifacts/yat-maxcenter-v5e16-0925`: v5e16 across four hosts, batch128, accumulation2, local/global query tiles64, donation, chip-local YAT and max-centered backward. It requests one bounded calibration profile, exact kill/resume validation and heldout evaluation, then stops via `--validation-only`. Archive preflight passed. Reservation $12, 30-minute guarded lease, cumulative conservative planned exposure $883.56; this is not actual billed spend. Physical training/recovery/performance results remain pending.

### Broader regression run and centered-FFN architectural ablation

The post-integration encoder/training/snapshot/reference/stage suite passes89 tests with8 skips. Skips are guarded multi-device/multiprocess/physical-TPU or optional-reference checks, not a claim of full hardware coverage; the separate four-device suite already passed18 cases. Log: `artifacts/yat-maxcenter-v5e16-0925/cpu-encoder-regression.log`.

While the frozen four-host run provisions/compiles, an independent CPU architectural ablation tests a centered FFN gate `K(x,w) - 1/(||x||²+||w||²+.01)`. It retains bias1 and epsilon.01 within K but **changes the gate function** and is not an algebraic optimization of the existing model. Its equivalent factorized form, with `z=x·w` and `S=||x||²+||w||²+.01`, is `z*(z+2+2/S)/(||x-w||²+.01)`. A NumPy FP64 identity test passes. The subtraction permits signed gate responses, unlike the original nonnegative kernel factor.

On layer0 teacher activations (8 calibration and8 heldout training-pool rows), scalar-fitting the original/centered branch gives heldout relative squared errors70.9%/11.7% at the original prototype scale. The initial comparison fits a scalar after projection; the follow-up applies the fitted alpha in the actual BF16 gate before projection. A 22-layer scalar calibration is preserved in `centered-ffn-calibration.json`.

On the existing32-row/561-target validation probe, teacher/single-centered-FFN/all-centered-FFNs MLM losses are1.57053/1.70463/12.20049. All attention remains original mmBERT. The single-layer fit is useful initialization evidence, but composition fails badly: this architecture is **not adopted**, and no cloud training or production code was changed for it. Reports and source are under `artifacts/yat-initialization-diagnostic-0925`, including `centered-gate-ablation.json` and `centered-ffn-mlm-probe.json`.

### Max-centered full-model calibration and profiling caveat

The 16-chip/four-host run completed 32 calibration updates and saved checkpoint32. The official measurement excludes compilation and the captured profile step: 30 steps yield51,675.85 useful tokens/s. The immediately following step took4.882s; a separately reported sensitivity using steps5–32 yields59,884.73 useful tokens/s (median0.8405s), compared with61,887.85 in the prior factored run. This sensitivity does not replace the official metric or prove profiler causality. Independent per-host trace export can shift waiting into the next collective step; a post-export cross-host barrier is a candidate instrumentation fix to verify.

Measured writer-chip peak allocation is4.57GB, reserved4.96GB, with16.91GB available. Compilation reports a different temporary-memory accounting and must not be equated with observed HBM peak. Exact interruption/resume and heldout evaluation remain in progress. The new method remains opt-in, with the factored default unchanged. An additional focused run of trace-summary, BF16 softmax-mode and FP64-gradient-oracle tests passes24/24. Evidence: `artifacts/yat-maxcenter-v5e16-0925/steady-throughput.json`.

The four-host validation subsequently finished successfully (all worker exits0, controller exit0). Independently downloaded checkpoint32 manifests match model, optimizer and training state after forced interruption at16 and resume. Heldout evaluation covers256 rows/14,904 masked targets with loss17.295873; this short run does not establish production quality. The TPU and queued resource were both independently confirmed absent; storage evidence remains.

All four XPlanes were downloaded and summarized using the public JAX profile reader. Across16 devices, module interval unions range776.19–777.37ms; collective unions67.20–218.70ms (8.65–28.14%). These include possible waiting and overlap with other categories. Existing Pallas-named operations total32.90ms; this is not evidence that fused YAT attention is integrated. Reports: `profile-findings.json` and per-host `*.xplane.summary.json`.

After the frozen run completed, `TrainingTrace` gained a cross-host barrier after successful trace export. It keeps export skew within the explicitly profiled step instead of the next timed collective. Exception cleanup does not initiate the barrier. Local profiler/summary checks pass11 tests, including export ordering, failure cleanup, real trace generation and donated-state exact resume; Ruff/Pyright pass. This instrumentation change has **not** yet been revalidated on physical multi-host TPU. The factored gradient remains the default.

Shape-matched attention distance-repair loops occupy352.74–501.69ms (45.4–64.6% of device module time). Their variability is inversely associated with collective waiting, consistent with workload imbalance. The match uses the current48-head/batch and64-query tile shapes and must be reviewed if those shapes change. This makes reducing/regularizing repair work the next kernel priority; the experimental fused implementation still requires actual-activation, alpha-gradient and full-model qualification.

### Uniform-reduction four-host qualification submission

A frozen v5e16/four-host validation is submitted under `artifacts/yat-uniform-v5e16-0925`, using the factored softmax default, uniform block32 forward repair, the all-zero-sensitive-cotangent backward shortcut and synchronized profile export. Batch128, accumulation2, query tiles64 and donation remain unchanged. Production pilot training is disabled.

The validation harness now runs ten targeted numerical regressions on every worker before the full-model preflight: dispatch-invariant distance, zero/small cotangents, and near-collision packed-document output/Q/K/V/alpha isolation. CI mapping includes the corresponding suites;14 CI selection tests and15 recipe/pilot tests pass. Archive preflight passed, and the independent cleanup guard is verified. Physical results are pending.

Reservation$12 brings the conservative post-reconciliation planning envelope to$898.56 of$900, leaving$1.44 admission headroom. This is not actual billed spend or a current credit balance; further paid runs require a fresh reconciliation. The preceding one-chip benchmark is confirmed cleaned up.

### September25 credit and admission reconciliation

Authenticated billing for the account linked to `tpubuilders` shows$805.60 remaining on the Marketing TPU credit, expiring September28, versus$821.35 at the previous observation (posted decrease$15.75). Its original grant was$1,000; this is **not** the balance when this task started, and historical consumption is not attributed to this task. The separate Google.org grant is excluded. The account overview reports September1–24 gross costs$82.64 and net$0 after savings; this is account-wide and may lag usage.

A new, separately preserved reconciliation inventories both nodes and queues in all six previously used zones, rejects unaccounted live resources, and keeps the active uniform-reduction run's full reservation. Completed absent resources use their entire request-to-deletion lifetime plus$2 ancillary allowance each. It deliberately double-counts posted credit consumption and retains an additional$25 allowance. The resulting conservative exposure is$855.87, leaving$44.13 for future admission under the original$900 cap. Neither number is actual invoiced task spend. All historical ledgers and the active run's original budget are unchanged. Evidence: `artifacts/yat-budget-audit-0925/reconciliation.json` and `billing-observation.json`.


Separate attention initialization is opt-in through `--yat-attention-alpha 0.1`,
with `--yat-alpha 1` retaining the FFN initialization. Alpha remains trainable.
A bounded CPU diagnostic improved adaptation, but this setting is not yet
physically TPU-qualified or production-quality-qualified. See
`artifacts/yat-alpha-adaptation-0925/RESULTS.md`. Defaults remain unchanged.


Final eight-chip alpha0.1 validation passed: 32 updates, 78,209 useful tokens/s over30 warm unprofiled steps, exact model/optimizer/training-state recovery after committed-step16 SIGKILL, and held-out MLM loss9.194311 on256 rows/14,904 targets. Both TPU VM and queue were independently verified absent at07:49:02 UTC. Profiles and evidence remain in storage. This qualifies the bounded single-host implementation/recovery check, not model quality, every scale, or the altered setting on physical multi-host. See `artifacts/yat-alpha-v5e8-fixed-0925/RESULTS.md`.
