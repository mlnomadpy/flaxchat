# Production encoder qualification work log

The active objective is v6e execution, larger-scale qualification, sustained
physical multi-host training of the repaired backend, and production multilingual
model quality. Passing one requirement does not close the others.

## Required evidence

| Requirement | Evidence needed | Current state |
|---|---|---|
| v6e | Physical device identity, repaired loss/gradient gates, complete training horizon, exact recovery and cleanup | Current repaired v6e-4 request is waiting for capacity with a verified cleanup guard; earlier attempts are cleaned up |
| Larger scales | Frozen-source runs for the outstanding scale matrix, complete worker inventories, memory/performance measurements and matched batch comparisons | Prior v5e 16/32/64/128/256-chip gaps remain; batch 64 cannot be called matched scaling above 64 devices |
| Sustained repaired multi-host backend | At least 30 measured minutes of uninterrupted baseline training on multiple physical hosts, finite accepted updates, committed midpoint fault, exact final recovery and cost/cleanup evidence | Four-host/16-device Pallas baseline completed 16,994 accepted updates and 39.17 measured minutes; exact recovery and final all-worker/cleanup audit remain pending |
| Production multilingual quality | Actual candidate checkpoint; validated reference implementation; representative multilingual pretraining; locked train/dev/test separation; complete downstream classification, retrieval and token-level evaluation; long-context evaluation; multiple seeds and per-language results | Reference fidelity and 50M-token continued-MLM pilot passed; matched initial-weight comparison and full downstream acceptance remain open |

The larger-scale matrix remains the one in
[the scale results](ENCODER_SCALE_RESULTS_2026_09_23.md). Smaller successful slices
must not stand in for untested larger slices. A matched strong-scaling series uses
a fixed divisible batch; a different batch is a separate experiment. Capacity
failure is not a successful hardware test.

The 30-minute sustained threshold is a minimum for this infrastructure milestone,
not a days-long stability claim. The new two-hour runner ceiling does not alter
IAM, the cleanup workflow, or its pre-existing maximum lease. Every attempt still
reserves its whole lease plus 30 minutes of deletion time before allocation.
The sustained workload calibrates 100 real updates, freezes an even update horizon
with a 25% duration margin and checks that baseline plus recovery fit the remaining
lease. It still requires the measured 30 minutes to pass. Estimated duration does
not replace measurement. The worker receives its actual remote timeout so the
plan accounts for time already spent waiting for capacity and setup.

## Reference implementation fidelity

`python -m scripts.validate_encoder_reference --mode export --snapshot SNAPSHOT
--oracle ORACLE` freezes Transformers eager/FP32 outputs from released weights.
Run `--mode verify` with the same arguments and `--output REPORT.json` separately.
The validator checks valid-token hidden states, mask-aware pooling, and complete
256,000-entry logit vectors at two positions per row. The fixture includes eight
languages, padding, masked input tokens and local-attention boundaries at lengths
32, 128 and 256. Thresholds are declared in code before measurement.

The September 23 CPU run passed all nine comparisons. Maximum absolute errors:
hidden states 0.0015762, pooled vectors 0.0000308, logits 0.0014858; elementwise
tolerances were `atol=0.002`, `rtol=0.001`. Oracle, exporter source, weight/config/
tokenizer hashes and raw result are retained in `artifacts/encoder-qualify-0923/`.
These tests do not establish BF16 fidelity at every topology or 8K quality.

## Production evaluation path

Default planning assumes continued pretraining from the released mmBERT-base
snapshot, pending the user's requested preference. From-scratch reproduction is
not assumed achievable within the remaining credit. This does not lower the
quality requirement: the actual resulting checkpoint still needs evaluation.

Use development splits for all recipe selection. The earlier 512-per-language
XNLI test probe has already been observed and must not become a development set.
Retain that historical regression. Final reports must disclose prior exposure,
report untouched XNLI examples separately, and include fresh independent tasks.
A small frozen ridge probe is not the published fine-tuned XNLI protocol.

The [mmBERT paper](https://arxiv.org/html/2509.06888v1) evaluates classification,
question answering, structured prediction and embeddings. Reproduce task adapters
and fine-tuning protocols before comparing published numbers. Plan a matched
released-model reference and candidate with three seeds; lock dataset revisions,
language lists, exact splits and per-task minimum scores before evaluating the
candidate's final test results. A benchmark pass is scoped to its tasks/languages,
not a certificate for all 1,833 languages or every production domain.

The source inventory pins XNLI, XTREME, FineWeb2-HQ and mmBERT decay dataset
revisions in `dataset-sources.json`. FineWeb2-HQ spans 20 languages; broader
coverage needs additional data. Its reported decoder-model token efficiency does
not establish equal savings for encoder MLM. See the
[dataset card](https://huggingface.co/datasets/epfml/FineWeb2-HQ).

Full XNLI development inputs are retained with checksums: 392,702 English training
pairs and 2,490 validation pairs in each of 15 languages. This preparation did
not read test data. Task fine-tuning and final quality acceptance are still pending.

## Cost controls

This phase has a $150 conservative reservation cap. The first v6e attempt reserves
$14.60 using the on-demand ceiling, a 40-minute lease and a separate ten-minute
capacity limit; Spot billing may be lower. The credit UI displayed $859.92 at
phase start and expires September 28, 2026; posted usage can lag. The separate
program credit is not added to this balance. Previous phase spending/reservations
are retained separately. Preserve evidence before deleting temporary cloud data;
retain the existing seven-day soft-delete policy.
The subsequent 16-chip attempt reserves $50, bringing this phase's reservations
to $64.60. It timed out and queue/node absence was verified. It uses a two-hour maximum lease and a six-minute capacity wait. Its
frozen training bundle is `source-v3.tar.gz`; no in-flight source replacement is
permitted. Capacity and completion are recorded in the campaign ledger.

The v5p fallback reserves $85.0667, bringing this phase to $149.6667 of $150.
It also timed out waiting for capacity. Reservations are not measured spend;
the node was subsequently observed in DELETING state. No further allocation fits
this phase cap. Verify queue and node absence before closing the phase.

## End-to-end classification

The new classification adapter uses mean pooling, the pretrained ModernBERT
prediction head and a new classification layer. Preparation preserves sentence
pair separators, refuses silent truncation, and pins source, tokenizer, token and
label hashes. Training supports linear warmup/decay, deterministic epoch shuffling,
finite-update rejection and exact optimizer/cursor resume. Evaluation rejects train
splits, duplicate split/language entries and mismatched dataset identities before
constructing the model.

Prepare each official split/language independently:

```bash
python -m scripts.prepare_encoder_classification --source pairs.jsonl --tokenizer SNAPSHOT/tokenizer.json --output DATA --dataset facebook/xnli --revision PINNED_REVISION --split train --language en
python -m scripts.finetune_encoder_classifier --config SNAPSHOT/config.json --pretrained SNAPSHOT --data TRAIN --output CHECKPOINTS --steps 12272 --warmup-steps 737 --batch-size 32 --learning-rate 2e-5
python -m scripts.finetune_encoder_classifier --mode evaluate --output CHECKPOINTS --checkpoint-step 12272 --eval-data VALIDATION_LANGUAGES --batch-size 32 --report development.json
```

The training example covers approximately one epoch of 392,702 English rows.
Pass each validation directory as a separate argument. Candidate encoders can use
`--encoder-checkpoint` and an explicit `--encoder-step`. This is an evaluation
capability, not evidence of production quality or a selected final recipe.

Local tests compare logits, loss, classifier and embedding gradients against
Transformers, exclude padded rows, reject corrupt labels, and compare uninterrupted
and resumed model/optimizer/cursor manifests in FP32 and BF16. The 13 classifier
tests pass on four simulated CPU devices. Physical TPU classification, full
multi-seed task training and final quality acceptance remain outstanding.

The updated full CPU suite passes: 861 passed, 19 skipped, five deselected;
79.71% total coverage and all per-module floors pass. Lint, type checking,
documentation links and the additional CI-policy checks pass. Logs are retained
in `artifacts/encoder-qualify-0923/`. Skipped/deselected tests are not hardware
qualification evidence.

All six temporary cloud source files were matched to local copies by size and
CRC32C, then deleted. The phase prefix has no live objects; the seven-day
soft-delete retention policy is unchanged. The v5p queue/node teardown is still
pending (SUSPENDING/DELETING at the last read), and subsequent reads confirmed NOT_FOUND for both. The controller recorded
resource_absent; no compute remains from that phase.


## Next physical phase

The previous turn made progress: classifier implementation/CI, full CPU validation,
data preparation and storage cleanup. Subsequent authoritative reads confirmed
both the v5p node and queue are absent. The new phase is capped at $100 and starts
with one v6e-4 request reserving $29 at $10.80/hour, including the two-hour lease,
30-minute teardown allowance and $2 ancillary reserve. The capacity wait is now
30 minutes within that lease. This is a new request after verified deletion, not
a replay prompted by an observation timeout. No TPU allocation runs concurrently.

The credit UI was rechecked: $859.90, expiring September 28. Carry the entire
previous $149.6667 reservation as unresolved exposure because posted billing can
lag. Do not interpret the $0.02 displayed balance movement as this phase's cost.
The separate program credit is not combined with this balance.

Frozen source and recipe live in `artifacts/encoder-production-0923/`. The v6e
workload first runs repaired projection and released-model gradient gates plus
2,000-update exact recovery, then classifier resume tests and a 100-update real
model calibration. It runs the fixed 12,272-step English XNLI recipe only when the
90th-percentile measured step time, 25% margin, all 37,350 development examples
and a 600-second checkpoint/compile allowance fit the remaining lease. A failed
estimate does not silently shorten the training horizon. The result remains a
single-seed development baseline, not final production acceptance.

This first development recipe is LR 2e-5, batch 32, sequence length 256, seed 17,
737 warmup steps, 12,272 total steps, mean pooling and zero classifier dropout.
All final recipe comparisons use development data. Test data remains untouched
by this phase. Three seeds, candidate continued-pretraining, independent tasks,
long-context behavior and per-language acceptance remain necessary.

The initial parallel source upload stalled and was terminated. Its bounded retry
also reached its deadline for the 1.1 GB base archive; the 38 MB harness/data
overlay completed. The exact retained base generation was restored, copied inside
GCS and matched to the local archive by CRC32C and size, then the temporary
restored copy was deleted again. All four input objects were verified before
TPU submission. This transfer delay incurred no TPU allocation.


## Distributed classifier recovery coverage

The additional localhost two-process test passes in FP32 and BF16. It trains an
uninterrupted reference, independently kills both workers after committed step 2,
then resumes to step 5. Raw model, optimizer and cursor inventories match exactly.
Worker logs establish two processes/two devices and both committed-fault markers.
The CPU shared-filesystem override is explicit and rejected on TPU; physical
multi-host checkpoints still require GCS. CI now executes these two regressions.

The focused local classifier/CI/policy suite passes 45 tests with two opt-in tests
skipped, and both opt-in distributed tests pass separately. These changes postdate
the immutable in-flight v6 bundle; they are not attributed to that hardware run.

## Multilingual continuation data

A pinned FineWeb2-HQ pilot is being exported across all 20 available language
subsets. It samples three shards per language and one row group per shard using
seed 17, reading only text and provenance columns. Source revision, shard metadata,
row-group coordinates, document IDs, language, quality score and output hashes are
retained. This avoids downloading unused embeddings. Sampling is stratified by
shard; documents do not have globally uniform inclusion probabilities. It is a
bounded continuation pilot, not the full corpus or all 1,833 mmBERT languages.
No test split is used. Export must finish before tokenization, deduplication and
held-out-data checks; no model-quality claim follows from preparing these inputs.

The export completed and was independently checked: 60,000 documents across
20 languages, all output hashes matching, 60,000 unique IDs and 60,000 unique
NFKC/whitespace-normalized texts. Deterministic document hashing assigned
58,791 training and 1,209 held-out documents, with both splits present
in every language and zero exact normalized overlap. Split policy, per-language
counts and file hashes are retained in the pilot split manifest. Near-duplicate
and downstream contamination checks have not yet been performed; tokenized
training inputs and model-quality results are still pending.

The complete document-window preparer now preserves source coordinates and
language accounting, fails instead of silently dropping rows at its limit, and
passes seven regression tests plus lint/type checks. The pilot produced 140,671
training rows (56,343,462 nonpadding tokens; 78.23% occupancy) and 3,073 held-out
rows (1,254,263 nonpadding tokens; 79.72% occupancy), all length 512. These are
prepared inputs, not trained model results; token-span and near-duplicate audits
remain pending. Changes are not injected into the frozen v6 request.

### Exact-span contamination audit and quarantine

The row-aware audit verifies actual token equality after hash matches, excludes
special/padding tokens, never crosses row boundaries, and reports every affected
training row rather than only the first training witness. Regression tests cover
hash collisions, both boundary directions, repeated training matches, chunk sizes,
empty inputs and randomized comparison with brute force. The focused corpus,
contamination and CI suite passes 31 tests; lint and type checks pass.

The original pilot has 412 training rows in 131 documents sharing a 50-token span
with held-out MLM rows. Another 68 rows in 53 documents share a 13-token span with
XNLI development rows across the 15 languages. These sets contain 184 distinct
training documents. All were excluded whole, preserving 58,607 training documents
and leaving both evaluation datasets unchanged. The shorter-span matches can
include common phrases; exclusion is conservative, not proof of copied benchmark
examples. Counts, witnesses, hashes and quarantine IDs are saved in
`artifacts/encoder-production-0923/fineweb-pilot/overlap-audit.json` and
`split/quarantine.json`.

Rebuilding and independently re-auditing the cleaned prepared set completed: zero
matching rows at both configured thresholds. The cleaned set has 139,621 rows
and 71,485,952 input positions; use `prepared/train-clean`, not the original
`prepared/train`, for any future pilot. This exact-span
check does not establish absence of semantic/edited near-duplicates, cross-window
matches or contamination elsewhere in the pretrained model's original corpus.
No continuation training has run on this pilot and no production-quality gate is
satisfied by these data checks alone.

### Harness overlap command and next physical attempt

`scripts.audit_encoder_overlap` now provides the reusable equivalent of the
pilot audit. It checks prepared token hashes, tokenizer/special-token identity,
held-out split labels, and contiguous complete document provenance; it records
all affected training rows and whole document IDs, and exits 1 on overlap.
It does not modify either dataset. Example:

```bash
python -m scripts.audit_encoder_overlap \
  --training artifacts/encoder-production-0923/fineweb-pilot/prepared/train-clean \
  --candidate artifacts/encoder-production-0923/fineweb-pilot/prepared/validation \
  --config artifacts/mmbert-base/config.json --span-tokens 50 \
  --output artifacts/encoder-production-0923/fineweb-pilot/harness-overlap-clean.json
```

This command passed on the cleaned pilot. Its focused audit/contamination/CI suite
passes 30 tests, and lint/type checks pass. Exact-span limitations remain as above.

The v6e request exhausted its 30-minute capacity wait without running workloads.
Both its exact queued-resource name and node name returned NOT_FOUND after the
controller verified teardown. This is a capacity failure, not v6e qualification.
The next attempt is `flaxchat-validation-production-0923-sustain16-0`, Spot
`v5litepod-16` in `us-west4-a`, targeting 16 devices across four physical hosts.
It has a two-hour lease, six-minute capacity wait, independent cleanup guard,
measured 30-minute baseline requirement, and exact recovery requirement. Its
$50 conservative reservation brings the phase total to $79/$100; reservations
are not posted spend. At 18:41 UTC its authoritative queue state was PROVISIONING.
The frozen encoder/backend/trainer/scale-validator sources match the current
repaired versions; the newly added corpus auditor is not injected into this run.


### v6b terminal update and BF16 YAT

The us-central1-b v6e-4 attempt reached hardware, but its workload exited 1 after
5.4 seconds before uploading evidence. The controller terminated with
`resource_absent`; both the exact queued resource and TPU VM independently
returned NOT_FOUND. No successful v6e qualification follows from this attempt.
The current phase retains $99.90 of $100 as a conservative reservation, not a
measurement of billed spend. No replacement allocation was created.

The newer BF16-only YAT forward option was verified on simulated CPU devices
only. It postdates the frozen TPU workload and is materially slower in the CPU
microbenchmark. See ENCODER_TRAINING.md for limits and measured drift.


### v6b startup failure diagnosed and repaired

The retained local `v6b-cloud/attempt-0/workload/worker-0.log` records
`ModuleNotFoundError: No module named 'scripts'` while the nested artifact
entrypoint imports `scripts.workload_deadline`. This was an import-path failure,
not a failed TPU numerical check. `gcp_tpu_run.worker_command` now explicitly sets
`PYTHONPATH` to the frozen checkout for both the entrypoint and its subprocesses,
overriding a stale inherited value. Worker summaries now link the retained log.
A real shell/subprocess regression checks nested imports and child imports from a
checkout path containing spaces, with a conflicting inherited package. A new
physical run is still required; no prior evidence is retroactively qualified.


### Complete-document contamination check

The new `scripts.audit_encoder_documents` retokenizes checksum-verified complete
source documents without window truncation, verifies exact spans after hashing,
and never joins different documents or deletes interior special tokens. It found
**three additional matching training documents** in the previously row-clean
58,607-document source, against the 1,209-document MLM validation split at width
50. Their witnesses are in `fineweb-pilot/document-overlap-clean.json`. The prior
row-only pass remains valid within its scope but is insufficient for clearance.
Do not use that training fixture for a quality pilot until the three documents
are quarantined, the fixture rebuilt, and both audits rerun. Evaluation remains
unchanged. Semantic/edited overlap and imported-model contamination remain open.


### Replacement corpus verified

The replacement `fineweb-pilot/prepared/train-clean-v2` contains 58,604 documents,
139,552 rows and 55,817,200 nonpadding tokens. Three entire documents (69 rows)
were removed; every retained row was verified identical to the prior fixture.
The source, tokenizer and held-out inputs remained unchanged during verification.
Both the row audit and complete-source-document audit now report zero exact
50-token overlaps against the MLM validation split. The actual trainer's CPU
input preflight passed with the released mmBERT configuration, batch 64,
`xla_local`, and language exponent 0.5. No model training or TPU allocation was
performed for these checks. Use this v2 fixture for future pilots. This clears
the observed exact-span data issue, not semantic overlap or production quality.


### Current-source v6e attempt

Billing UI rechecked: marketing credit $852.72, expires September 28; the
separate $1,000 program credit is not combined with this budget. Prior $249.57
conservative reservations remain intact. A separate $20 phase reserves $18.20
for one v6e-4 Spot attempt, keeping total reserved exposure about $267.77 within
the original $900 budget. This is reserved exposure, not measured spend.

`flaxchat-validation-v6-current-0923-0` in us-central1-b is waiting for capacity.
The current overlay passed isolated frozen-source import preflight and includes
the repaired launcher environment and adaptive BF16 YAT. It has a 3,600-second
lease, 600-second capacity wait and verified cleanup workflow execution
`09f80583-2a95-4576-9ad8-559d99bcc3a2`. No physical qualification is yet established.
Run plan and source identities are under `artifacts/encoder-v6-current-0923`.


### Combined worktree regression verification

The full non-accelerator suite completed with 932 passed, 21 skipped and 5
accelerator tests deselected. Coverage is 80.47%; all risk-based module floors
passed. Repository lint and type checks passed. The single warning is an upstream
Starlette/httpx test-client deprecation. This is CPU regression evidence only;
the same v6e request remains waiting for resources with its controller live.
Evidence is under `artifacts/encoder-v6-current-0923/`.


### Current-source v6e attempt terminal result

The ten-minute capacity wait expired without hardware. Controller exit 1 records
`Spot capacity wait budget exhausted` and then `resource_absent`. Independent
queue and TPU VM describes both returned NOT_FOUND. No physical workload ran,
and the $18.20 reservation remains conservative exposure rather than measured
spend. All four top-level qualification requirements remain open.


### Current-source sustained multi-host attempt

One v5litepod-16 Spot request, `flaxchat-validation-sustain-current-0923-0`, is
waiting for resources in us-west4-a. The frozen current overlay is unchanged from
the v6e attempt and passed the full CPU suite. The workload calibrates the repaired
Pallas backend, freezes an even horizon, requires at least 1,800 measured baseline
seconds, injects a committed midpoint interruption and compares exact recovery.
Expected topology is 16 chips across four physical hosts. Sixty planner, scale,
launcher and deadline preflight tests passed. Physical evidence remains pending.

The attempt has a 7,200-second lease and 900-second capacity wait. Cleanup guard
execution `687a735d-eb41-4772-8e8c-3e51f2dbb126` was verified before provisioning.
Its $50 reservation brings retained cumulative exposure to about $317.77, within
the original $900 budget. No prior reservation was released or treated as actual
spend. Evidence and the concrete launch plan are under
`artifacts/encoder-sustain-current-0923/`.

### Sustained run: coordinator failure and repaired retry preparation

The `sustain-current-0923` controller is terminal (exit 75). Exact GCP queue and
VM describes both return NOT_FOUND. The controller ledger records resource_absent
before preempted_retry_eligible; the latter classification followed explicit
cleanup after a planner failure and must not be interpreted as proof of an
independent Spot preemption.

Calibration completed 100 accepted Pallas updates on 16 chips / four physical
hosts, with 14.085 measured seconds and no dense fallback. This is insufficient
for the 30-minute sustained gate. Worker 2 carried the JAX-primary training log;
the previous wrapper incorrectly selected SSH worker 0.

The permanent endurance runner downloads every calibration log and selects the
single verified runtime-primary log. A mocked end-to-end coordinator regression
covers the worker-2-primary case and frozen duration plan. The owning supervisor
now opts into cancelling peer transports when a worker fails. Standalone launcher
semantics remain unchanged for intentional fault/recovery workflows. Targeted
runner, launcher, scale and operations tests: 77 passed; affected lint/type checks
passed. A new immutable payload is being preflighted for a bounded $50 attempt;
prior $317.77 conservative reservations remain retained. Reservations are not
reported billing costs. v6e, sustained training and production quality remain open.

### Pinned multilingual bitext evaluation inputs

The complete official `mteb/tatoeba-bitext-mining` test set is now local at
`artifacts/encoder-retrieval-0923/raw`: 112 shards and 88,877 aligned pairs, pinned
to revision `4f65d2990c55534b6aa84eeb7da78b1f13c97426`. The inventory records each
upstream URL, SHA256, compressed size and pair count. Independent rereading
verified every checksum, row count and complete shard coverage against the
pinned repository metadata. No scores were computed, examples printed, model
settings selected or training data modified. The dataset card states CC-BY-2.0.

This supports a later held-out cross-language bitext evaluation. It does not
replace full retrieval, token-level, classification or long-context testing.
Before scoring, freeze candidate/reference checkpoints and the upstream MTEB
protocol (including duplicate handling, similarity and metric aggregation).
Our generic retrieval scorer has not yet been shown equivalent to that protocol;
do not label its output an official Tatoeba/MTEB score. Extend training-overlap
audits to these inputs before any resulting production-quality claim.

### Upstream bitext protocol check

Pinned MTEB code commit `3cd2041234f3126416bf3659a34b4dec846dc1c0`
selects dataset revision `69e8f12da6e31d59addadda9a9c8a2e601a0e282`, which differs
from the repository-head revision downloaded above. All 112 canonical shards
were downloaded separately and independently verified: their compressed bytes
match the head revision exactly. Canonical inventory:
`artifacts/encoder-retrieval-0923/mteb-pinned/inventory.json`; code checksums:
`artifacts/encoder-retrieval-0923/upstream/inventory.json`.

The official task's primary score is weighted multiclass F1 over paired corpus
row indices, with zero-division classes assigned zero. Pair accuracy is also
reported but is not interchangeable with F1 when predictions collapse onto one
target. `flaxchat.bitext.bitext_metrics` now implements precision, recall, F1 and
accuracy with support weighting and bounded observed-class memory. Thirteen
unit cases cover collapsed predictions, repeated gold targets, large corpus IDs,
perfect/incorrect predictions and invalid evidence. Lint and type checks pass.
Full MTEB parity still requires its search/tie behavior, model adapter and frozen
inference settings; this metric implementation is not a leaderboard certificate.

Independent metric-reference check: 100 random synthetic cases matched
scikit-learn 1.9.1 exactly (maximum absolute error 0). Evidence is
`artifacts/encoder-retrieval-0923/metric-parity.json`. No held-out dataset scores
were computed during this comparison.

### Full bitext input preparation

`scripts.prepare_encoder_bitext` now prepares pinned gzip JSONL shards for the
checkpoint embedding exporter. It preserves every aligned pair and duplicate,
rejects empty content, oversize inputs, changed checksums and existing outputs,
and writes atomic query/corpus arrays plus index-aligned judgments. No sentence
is silently truncated or filtered. CI selects the preparation, metric and
checkpoint-export regression tests for this script.

All 112 benchmark-pinned shards / 88,877 pairs were prepared with the local
mmBERT tokenizer at length 512. The actual maximum including special tokens is
209; no input was truncated. Independent prepared-data loading verified every
array against the encoder configuration, every source revision/split, every
row-ID alignment and all output hashes. Evidence:
`artifacts/encoder-retrieval-0923/prepared-verification.json`.
The preparation/retrieval/CI regression suite passed 48 tests; affected lint and
type checks passed. No checkpoint embeddings or held-out scores were computed.
These changes postdate the in-flight frozen TPU payload.

### Multi-host timeout resolved; alternate-zone v6e attempt

The repaired multi-host controller exited 1 after its 900-second capacity bound.
No setup or training began. Its ledger recorded resource_absent, and independent
GCP describes confirmed both queue and VM NOT_FOUND. The $50 reservation remains
retained rather than being treated as measured spend or refunded budget.

A fresh v6e-4 Spot attempt now targets `us-central1-a`, after the accelerator-type
API confirmed support. Controller session 58375, request
`flaxchat-validation-v6-alt-0923-0`, cleanup workflow execution
`6909c427-13ea-422f-849c-7c11300b836c`. Bounds: one-hour lease, 600-second capacity
wait, $18.20 reservation under a $20 phase cap. All prior $367.77 reservations
remain retained; cumulative reservations are $385.97 of the original $900.
The frozen payload includes the latest YAT implementation and synchronized
benchmark, and passed isolated import and trainer-input preflight. This is an
attempt in progress, not v6e qualification or a model-quality result.

### Full CPU regression after endurance and retrieval changes

The current worktree passed 974 tests, with 21 skipped and 5 accelerator tests
explicitly deselected. Coverage is 80.71%; all risk-based per-module floors passed.
Repository-wide lint and type checking passed. Evidence is retained under
`artifacts/encoder-v6-alt-0923/{full-tests.log,coverage.json,coverage-floors.log,lint.log,types.log}`.
This includes the new endurance coordinator, YAT tile optimization, checkpoint
retrieval export, bitext metrics and input preparation. It does not convert
skipped/deselected physical-device tests into passes or establish model quality.

### Alternate-zone v6e outcome

The `us-central1-a` v6e attempt reached its 600-second capacity limit without
starting worker setup or training. Controller 58375 exited 1; the ledger recorded
resource_absent. Independent exact queue and VM describes both returned NOT_FOUND.
No v6e performance or correctness result was produced. Cumulative reservations
remain $385.97, distinct from billing spend. The v6e, larger-scale, sustained
multi-host and production-quality requirements remain open.

### Checkpoint-to-bitext evaluation path connected

The prepared-data → checkpoint embedding export → nearest-neighbor ranking →
weighted bitext metrics path is now connected through
`scripts.evaluate_encoder_bitext`. Campaign scoring rejects missing language
subsets, altered embedding checksums, misaligned gold pairs, mixed model/pooling
identities and changed dataset revisions or splits. It records predictions and
per-subset metrics rather than only a final mean. The existing actual-checkpoint
restore test now checks the resulting F1 as well as pooled embeddings/rankings.
Targeted regression suite: 48 passed; affected lint/type checks passed. These
checks use synthetic/tiny checkpoint fixtures, not held-out benchmark model
scores. Official MTEB tie parity and physical model quality remain unqualified.

### Bitext overlap found and quarantined

A full-document scan of `train-clean-v2` against all 177,754 prepared bitext
sentences found exact 13-token spans in three training documents. Only 45,260
benchmark sentences contain an eligible 13-token span; this is not clearance of
shorter or semantic overlaps. The original imported checkpoint corpus remains
unaudited. Evidence: `artifacts/encoder-retrieval-0923/training-overlap.json`.

Those three whole training documents (170 rows) were removed into the fresh
`fineweb-pilot/prepared/train-clean-v3` fixture. It contains 58,601 documents,
139,382 rows and 55,730,904 nonpadding tokens. Independent verification checked
that every retained token row is byte-identical and all held-out inputs remain
unchanged. The quarantine record is `fineweb-pilot/split/quarantine-v3.json`.
A fresh complete-source audit is running; do not mark its result passed until
`training-overlap-clean.json` is produced and inspected.

The fresh audit subsequently completed: zero 13-token span witnesses across all
58,601 remaining training documents and all 177,754 bitext sentences. Report
`artifacts/encoder-retrieval-0923/training-overlap-clean.json` was inspected and
passed. This closes the detected exact-span issue under the stated coverage
limits, not the broader production-quality requirement.

### Longer-window v6e request

After confirming the prior queue and VM absent, a single new v6e-4 Spot request
was submitted in `us-central1-b`: `flaxchat-validation-v6-window-0923-0`, controller
74196. It reuses checksum-pinned, preflighted source
`8ea08b94ad5cfd26bc14bdd47080b051cb1d97c44a0b59c63ef844f14b50c66a` and the same
qualification checks. Capacity wait is now bounded at 1,800 seconds within a
7,200-second lease, giving the provider a longer allocation window. The verified
cleanup guard execution is `76646737-83a4-4353-b678-a567fd6d7de4`.
Reservation: $29 under a $30 cap; all prior $385.97 reservations are retained,
for $414.97 cumulative against the original $900 limit. No additional dataset or
source upload was needed. Submission is not physical qualification.

### Clean-corpus continued-training pilot recipe

`artifacts/encoder-quality-pilot-0923/plan.json` freezes a development baseline:
released mmBERT weights, GeGLU, 1,972 updates, batch 64, seed 17, language token
exponent 0.5, LR 2e-5, cosine decay, 119 warmup updates, BF16 compute/FP32 residuals,
and the repaired masked Pallas loss. The actual deterministic sampling schedule
covers 50,013,874 nonpadding tokens, including special tokens, over 20 languages.
It visits 82,707 unique rows in 126,208 draws; maximum repetition is eight.
This is replacement sampling, not one complete corpus epoch or a hard repetition
cap. Weight, tokenizer, data and compute-source hashes are pinned.

The real trainer's input preflight passed on `train-clean-v3`. No training or
new TPU allocation was performed for this recipe. Physical calibration on this
corpus is required before estimating cost: sparse numeric-fixture timings do not
represent its masked-target density. This baseline enables a controlled later
YAT comparison; it does not replace the YAT candidate, multiple quality seeds,
held-out downstream evaluation, or production acceptance.

### Real-corpus pilot cost gate (2026-09-23)

`scripts.plan_encoder_quality.plan` now requires a physical TPU calibration log
with an exact input identity (training recipe, tokenizer, corpus manifest, source
and initial weights), topology, Pallas backend, and an exact per-update nonpadding
token schedule. The trainer emits the identity in both its input preflight and
run configuration. Obtain the expected identity from a separate preflight on the
target TPU runtime; a CPU runtime identity must not be substituted.

The frozen 1,972-update pilot includes the first 100 expected nonpadding counts
in `artifacts/encoder-quality-pilot-0923/plan.json`. Calibrate using its full
training arguments plus `--stop-after 100` and a separate calibration output.
Collect every worker log and select the actual runtime primary, not SSH rank 0.
Pass that log's records, the independently checked preflight identity, the frozen
100 counts, topology, remaining lease, remaining compute budget, and the chosen
whole-slice hourly planning rate to the planner.

The estimate charges a full fresh 1,972-step run after calibration using p90
step time with 25% headroom and a 900-second allowance for compilation,
checkpoints, evaluation and teardown. It does not credit the first 100 steps as
resumed work. It rejects a recipe that exceeds the remaining lease or compute
budget; it never truncates the training horizon. Already spent calibration,
storage and other ancillary costs must remain accounted separately in the cloud
ledger. The allowance is not a measured evaluation duration, and the result is
not actual billing or quality acceptance. No physical density calibration or
quality training has run yet. The live v6e payload is unchanged and predates
this planner, identity logging, and latest YAT custom backward.

### v6e timeout and 32-chip attempt — 2026-09-23 21:51 UTC

The 30-minute v6e capacity window ended without setup/training. Controller 74196
exited 1; the ledger recorded resource absence. Independent exact queue and VM
queries both returned NOT_FOUND for `flaxchat-validation-v6-window-0923-0`.

One 32-chip v5e Spot request is now being submitted in us-west4-a:
`flaxchat-validation-sustain32-0923-0`, controller 85186. It uses eight processes,
Pallas, the repaired coordinator, a minimum 1,800 measured seconds and exact
fault recovery. The two-hour guard is verified (execution
`58afd183-372f-4154-9946-f310b5dd4559`); capacity wait is capped at 600 seconds.
Frozen payload SHA256: `efdb77c0d4bd5611a258ee9883ddc9348d6cfef4e34a5ff278c7b6e1e1e8a4f2`.
Its isolated input preflight passed. Source includes the latest YAT backward and
trainer identity logging, but this endurance workload trains the GeGLU baseline
on the numeric infrastructure fixture; it does not qualify YAT or model quality.

Reservation: $98 at the conservative $38.40 whole-slice hourly planning rate,
including cleanup and ancillary allowance. All older reservations remain counted:
$512.97 cumulative against the original $900 budget. These are reservations,
not posted spend. Physical qualification is still pending.

### 32-chip terminal result — 2026-09-23 22:04 UTC

Controller 85186 exited 1 after the 600-second capacity window expired. No
worker setup or training started. Cleanup took approximately 164 seconds after
the timeout; the ledger recorded resource absence. Independent queue and VM
queries both returned NOT_FOUND for `flaxchat-validation-sustain32-0923-0`.
The v5e-16 repaired attempt, both recent v6e attempts, and this v5e-32 attempt
all ended with capacity timeouts and resource absence. No hardware qualification
follows from these attempts. All reservations remain retained ($512.97 total),
not converted into reported spend.

The next independent leg is a smaller TPU real-corpus calibration for the frozen
multilingual quality pilot. Preparing and measuring that workload can advance
quality/cost evidence; it cannot substitute for the v6e, larger-scale, or sustained
multi-host requirements. No new allocation has been launched for that leg yet.

### Bounded real-corpus quality worker prepared — 2026-09-23

`scripts.validate_encoder_quality_pilot` now validates all frozen input hashes,
forces TPU execution, runs an independent full-recipe input preflight, then
calibrates 100 updates on the real clean-v3 corpus. The frozen 1,972-step horizon
and schedule remain unchanged during calibration. The cost gate requires exact
per-step nonpadding counts and matching input/runtime/source/model identities.
Only a complete fresh pilot that fits the remaining lease and compute allowance
is permitted. It checks accepted updates, final checkpoint, and the exact frozen
50,013,874-token horizon, then evaluates the entire development validation pool
(subject to reported exact-row/duplicate exclusions). Evaluation inputs are also
checksum-pinned. Neither success nor a development loss is production acceptance.

39 focused planner/orchestrator/CI-scope tests passed, plus lint and types.
The source/data archive passed isolated data preflight and an extracted full
seed-17, language-exponent-0.5, cosine-schedule recipe preflight. It contains
106,063,866 bytes; SHA256
`76997722583552ff9e16f9a6a70141fc95d9f38fc3be337cb8c286215f399e84`.
`artifacts/encoder-quality-run-0923` holds the frozen launch, budget and receipts.
Upload precedes allocation. Proposed target is single-host v5e-4, two-hour lease,
ten-minute capacity wait, $14 reservation including cleanup/ancillary allowance.
All prior $512.97 reservations remain retained; proposed cumulative reservation
is $526.97. These are spending bounds, not posted billing or actual estimates.

The real-corpus archive upload completed. Controller 17775 submitted
`flaxchat-validation-quality-real-0923-0` in us-west4-a; cleanup execution
`b15b8a0e-51b8-4d8d-afaf-b8b1512b019d` is verified. Its two-hour lease expires
at Unix 1790208742.58; capacity wait is 600 seconds. Reservation $14 brings the
retained cumulative total to $526.97. No physical calibration result exists yet.

### Real-corpus preflight failure and repair — 2026-09-23

The v5e-4 request reached ACTIVE, with one physical worker; setup passed in
75.79 seconds. The workload failed during child preflight before any training
update. Published `preflight.log` reports `The TPU is already in use by process
with pid 6199`. The parent worker imported `flaxchat.encoder_data` for file hashing;
Python first ran `flaxchat.__init__`, which initialized JAX through `common.py` and
claimed the TPU. This is a harness defect, not a capacity failure or a model result.

The orchestrator now hashes inputs with the standard library and imports no JAX
or flaxchat package. A fresh-process regression test forces `JAX_PLATFORMS=tpu`
and asserts both modules remain absent after importing the worker. Forty focused
tests passed with lint and types. The corrected frozen archive also passed this
import check and full extracted recipe preflight. New archive SHA256:
`387d877d991d873feadbd945381baba2f6875a5e821f745dd9178fdb9c6a9e5e`.

The failed allocation is being cleaned up by controller 17775. The corrected
payload is uploading under `encoder-quality-fixed-0923`; no replacement may
launch before independent queue and VM absence and the previous cleanup receipt.
A future retry retains the old reservation and adds another $14, bringing the
conservative total to $540.97, not actual billed spend.

The failed quality controller exited 1 after cleanup. Both exact queue and VM
queries independently returned NOT_FOUND. The corrected upload completed and
controller 23273 submitted `flaxchat-validation-quality-fixed-0923-0` in
us-west4-a. Guard `75ac5a01-7873-41c7-b684-20612a2ca432` is verified, expiring
at Unix 1790209376.73. Reservation remains $14 for this new attempt; retaining
all earlier reservations gives $540.97 cumulative. Physical preflight and
calibration are pending; the previous failed attempt completed no updates.

The corrected v5e-4 attempt reached hardware. Setup passed in 74.97 seconds;
physical child input preflight passed with JAX 0.11.1/libtpu 0.0.46.1. The child
trainer's run_config reports TPU backend, four devices, one process, Pallas,
seed 17, language exponent 0.5 and the fixed 1,972-step cosine recipe. Calibration
has initialized; no completed optimizer update or measured throughput was yet
observed at this inspection. This verifies the coordinator ownership repair
through child initialization, not the full pilot or any quality requirement.

### Physical real-corpus calibration passed — 2026-09-23

All 100 calibration updates were ordered, finite, accepted and free of dense
projection fallback; checkpoint 100 was committed. Every nonpadding count matches
the frozen real-corpus sampling schedule. After excluding compilation, 99 updates
took 31.8768 seconds: 78,837 nonpadding tokens/s, 101,768 input positions/s and
11,756 masked targets/s. Median update: 0.317909 seconds; p90: 0.318563 seconds.
These are actual four-chip v5e measurements on clean-v3, not sparse-fixture rates.

The full fresh 1,972-update pilot passed its cost/time gate and was observed at
step 50 with accepted updates and no dense fallback. Its conservative remaining
estimate was 1,685.26 seconds and $2.247 compute at the $4.80 whole-slice planning
rate. This includes 25% timing headroom and 900 seconds of allowance, excludes
already spent calibration, and is not posted billing. Evidence is under
`artifacts/encoder-quality-fixed-0923/evidence/`. Pilot completion, development
evaluation, and broader production-quality acceptance are still pending.

### Real-corpus pilot completed and resources removed — 2026-09-23

All four stages passed: physical preflight, calibration, full 1,972-update pilot,
and evaluation restored from committed checkpoint 1972. The pilot consumed
50,013,874 nonpadding tokens across 20 languages, exactly matching the frozen
sampling schedule, with no dense projection fallback. Excluding the first
compile, measured throughput was 78,626 nonpadding tokens/s over 635.77 seconds.
Development evaluation used all 3,073 rows and 186,981 masked targets, seed 2026;
masked-token loss was 0.950978. The candidate is continued-MLM GeGLU mmBERT,
not a YAT model. No matched initial-weight score or production-quality claim yet.

Checkpoint: `gs://tpubuilders-flaxchat-validation-0921/encoder-quality-fixed-0923/run/pilot/1972`.
Evidence: `artifacts/encoder-quality-fixed-0923/evidence/`. Controller finished
successfully; its ledger records resource_absent. Fresh independent queue and
VM describes returned NOT_FOUND. Cumulative retained reservations: $540.97,
not actual billing. The v6e, larger-scale, sustained multi-host, and production
quality gates remain open.

Before the next v6e request, inspection found its old coordinator imported
flaxchat in the parent, the same TPU-ownership defect repaired in the quality
worker. `scripts/validate_encoder_v6.py` runs import preflight in a short-lived
CPU child and forces TPU in physical probe/workload children. A fresh-process
regression checks that neither JAX nor flaxchat is initialized in the parent.
The new frozen overlay includes current YAT changes and collision-stress
benchmarks; it must pass extracted preflight before allocation.

The repaired v6e archive passed extracted trainer preflight and uploaded before
allocation. SHA256 `bf5afa5da23803126b78627628c2b29bc0284d58d83b91c17911b0e73ad6ff62`.
Controller 16058 submitted `flaxchat-validation-v6-repaired-0923-0` in
us-central1-b. Fresh describe: WAITING_FOR_RESOURCES. Guard execution
`dc8e6085-2845-4d57-bc05-84bb75f41b15`, deadline Unix 1790211744.13.
Capacity wait 1,800 seconds, lease 7,200 seconds; new reservation $29 brings
retained cumulative reservations to $569.97, not posted spend. This payload
includes the current YAT backward optimization and collision stress benchmark.

The MLM evaluator now supports `--initial-pretrained SNAPSHOT` alongside a
candidate `--checkpoint`. It uses the same checkpoint architecture, tokenizer,
training-pool identity, held-out selection, seed and masking policy, but loads
only the exact initial weight shards recorded in checkpoint metadata. It rejects
missing/mismatched hashes, load-time changes, and YAT architecture substitution.
Reports identify the parameter source and selected-row digest. The initial-weight
physical score remains pending; this evaluator revision is not in the active
v6e archive. New evaluation/CI tests: 20 passed. Existing trainer regression plus
initial-identity tests: 40 passed, 4 hardware-dependent skips. Ruff/Pyright passed.

### Matched MLM protocol prepared — 2026-09-23

`artifacts/encoder-mlm-pair-0923/plan.json` freezes both evaluation commands for
checkpoint root `.../encoder-quality-fixed-0923/run/pilot`, explicit step 1972,
all 3,073 validation rows, batch 8, seed 2026, plus local source/data hashes.
The candidate must be re-evaluated alongside the initial weights: the older
0.950978 report lacks the new matching fields and cannot silently stand in.

The evaluator now selects checkpoint metadata once and restores that selected
step, preventing a concurrent newer checkpoint from changing the evaluated
parameters. `--checkpoint-step` permits explicit selection. Reports record
actual runtime, model source hashes, evaluator hash, selected-row digest and
parameter provenance. The comparator rejects mismatched protocols, reversed
roles, missing identities, nonfinite loss, and empty evaluations. It reports
regressions as regressions and never marks development MLM as production quality.

Regression run: 95 passed, 4 hardware-dependent skips. After adding runtime/source
matching and explicit restore-step assertions: 65 focused tests passed, with
Ruff and Pyright clean. Physical paired evaluation remains pending. Controller
16058 is confirmed live; the existing v6e request remains WAITING_FOR_RESOURCES.
No replacement or additional TPU allocation was made.

### Matched evaluation payload ready — 2026-09-23

`scripts/validate_encoder_mlm_pair.py` runs a physical four-device inventory and
the two evaluators in separate child processes, with one bounded deadline and
reserved evidence-upload time. Parent imports never initialize JAX. Failures
persist evidence; changed inputs, partial rows, wrong parameter roles, mismatched
steps/seeds/checkpoint roots, and non-TPU reports cannot qualify the comparison.
A successful comparison still sets production quality false, even if MLM improves.
Worker, comparator and CI tests: 65 passed; Ruff/Pyright passed.

The paired payload is uploaded under `encoder-mlm-pair-0923`; archive SHA256
`9969650eec00c4b3158db17d309312b1399007bb5081a3d39b55272a5f7848ce`.
Extracted trainer/data preflight passed. An independent archive inspection verified
all 127 pinned source/data inputs and both commands, step 1972 and all 3,073 rows.
No TPU is allocated for this payload yet. Proposed v5e-4 lease: one hour, ten-minute
capacity wait, $9.20 reservation ($579.17 cumulative only if submitted). The
launcher checks the current v6 request's terminal cleanup receipt before launch;
independent queue/VM absence must also be confirmed. Actual posted spend unknown.

### Token-level adapter added — 2026-09-23

The quality audit identified a missing trainable token-level head. The new
`EncoderTokenClassifier` maps final hidden states to token labels with a linear
FP32 head and no dropout. It explicitly masks padding logits and loss; negative
labels exclude special/subword positions. Statistics provide token-weighted
loss and valid/correct counts, with zero gradients at ignored positions. Invalid
active class IDs produce a nonfinite loss for finite-update rejection.

Tests compare loss/gradients to an independent selected-token reference, check
all-ignored batches and invalid labels, and verify nonzero encoder/head gradients
in FP32 and BF16. Combined token/classification tests: 18 passed, 3 hardware skips;
Ruff/Pyright passed. This is an adapter, not a trained NER system. Word/subword
alignment, a checkpointed fine-tuning runner, entity span F1, dataset preparation,
and physical multilingual evaluation remain necessary. The adapter postdates both
the live v6 archive and the prepared matched-MLM archive; neither qualifies it.

### NER metrics and word alignment — 2026-09-23

`flaxchat/ner.py` adds prefix-BIO exact entity-span micro precision/recall/F1,
with separate sentence boundaries, per-language reports, explicit counts and
zero-division behavior. It follows the non-strict CoNLL-style BIO behavior of
[seqeval](https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py),
not strict IOB2. Native metrics matched pinned seqeval 1.2.2 on 1,000 randomized
cases, including per-language scores and extracted spans. Ten oracle fixtures
are retained in the test suite. Oracle dependencies were installed only under
`/tmp/flaxchat-seqeval-check`; the runtime metric has no seqeval dependency.
Evidence: `artifacts/encoder-ner-0923/seqeval-parity.json` and its reproduction script.

First-subword alignment uses tokenizer word IDs and rejects missing, reordered,
noncontiguous, or out-of-range words. The higher-level `align_encoding` also checks
Encoding overflow, catching a partly truncated final word even when every word ID
is still present. Tests use a real WordPiece tokenizer and cover that failure.
Padding/special/noninitial-subword labels are ignored. Combined NER/head tests:
23 passed; Ruff/Pyright passed. Corpus preparation, fine-tuning integration and
physical downstream results remain open. These additions postdate the frozen
v6e and matched-MLM payloads and do not change either queued experiment.

### v6e capacity timeout; paired MLM submitted — 2026-09-23

The repaired v6e attempt exhausted its 1,800-second capacity window without
reaching hardware. Controller16058 exited1; the ledger recorded resource_absent.
Both exact queued-resource and VM describes independently returned NOT_FOUND.
This is no v6e qualification evidence. Retain its $29 reservation in accounting.

The uploaded paired MLM archive was independently checked against Cloud Storage
size and MD5 (generation1790205278603371; 99,756,989 bytes). With predecessor cleanup
verified, controller25982 submitted `flaxchat-validation-mlm-pair-0923-0`, v5e-4,
us-west4-a. Cleanup execution `c270f4b1-aafa-4b35-a4ec-48a1ebf39e96` is verified;
lease deadline Unix1790210002.399. One-hour lease, 600-second capacity wait, $9.20
new reservation; cumulative retained reservations $579.17, not actual spend.
Physical inventory and paired scores are pending. No overlapping TPU allocation.

### Physical matched MLM comparison passed — 2026-09-23

The comparison ran on four physical TPU v5 lite devices, one process. Both
starting-weight and checkpoint1972 evaluations completed with returncode0.
All 3,073 development rows and 186,981 masked targets were scored with seed2026,
mask probability0.15, identical runtime/config/source/tokenizer/row identities.
The candidate log explicitly restores checkpoint1972. Independent local comparison
recomputation and frozen archive source-hash checks passed.

Starting mmBERT loss: **1.0140968702**. Pilot loss: **0.9509779851**.
Absolute decrease: **0.0631188852**; relative decrease: **6.2241%**.
The candidate score exactly reproduces the earlier pilot evaluation. This is
matched development MLM evidence for the 50M-token continued-MLM GeGLU candidate,
not YAT, per-language improvement, downstream performance, or production acceptance.
Evidence: `artifacts/encoder-mlm-pair-0923/evidence/` and `audit.json`.

The worker finished successfully and cleanup started. At this observation the
queue is SUSPENDING and VM DELETING; controller25982 remains live. Do not treat
cleanup as complete or start another allocation before independent absence checks.

### Paired evaluation cleaned up; four-host endurance submitted — 2026-09-23

The paired MLM controller exited0. Its ledger records resource_absent at Unix
1790207025.235; fresh independent queued-resource and VM describes returned
NOT_FOUND. The audited matched loss reduction remains 6.2241%; production quality
remains unqualified.

A refreshed 16-chip/four-host endurance payload passed extracted Pallas trainer
preflight and uploaded before allocation. SHA256:
`4663af4dde6b0a7b709ebfb06626e04863dc4b302620409ea8326d8d3d5bed5b`.
The payload includes current trainer/source and test fixtures. The coordinator and
Pallas backend themselves match their already tested revisions. Controller41234
submitted `flaxchat-validation-sustain-next-0923-0` in us-west4-a, with guard
`022cf36d-ddd2-44fa-9917-89c92df0143f`; lease deadline Unix1790214312.764.
Two-hour lease, 900-second capacity wait, $50 reservation; retained cumulative
reservations $629.17, not billed spend. No overlapping allocation.

Acceptance is unchanged: complete physical inventory, 100-step calibration from
the actual JAX primary, frozen full horizon that fits the remaining lease,
at least 1,800 measured seconds of uninterrupted baseline, committed midpoint
failure and exact final recovery. Capacity or calibration alone cannot qualify it.

### Four-host setup passed; NER preparation implemented — 2026-09-23

The 16-chip queue reached ACTIVE. Setup inventory reports v5litepod-16, Spot,
four selected workers and returncode0 on every worker. The four calibration
children started; their complete logs and the selected-primary endurance plan
are still pending. Setup alone does not prove the 30-minute or recovery gates.

`scripts/prepare_encoder_ner.py` now prepares one pinned dataset revision,
language and official split. It writes checksummed token, label and word-ID arrays,
with first-subword labels and explicit ignored positions. It rejects complete or
partial-word overflow, max-row truncation, malformed labels, and source/tokenizer
changes during preparation. Atomic publication prevents partial datasets. The
loader checks array hashes, alignment semantics, class inventory, token policy
and total word coverage before model allocation. Tests include actual WordPiece
splitting, partially truncated final words, tampered arrays with recomputed hashes,
and changed-source publication rejection. Combined NER/CI tests:46 passed;
Ruff/Pyright passed. A real benchmark corpus, checkpointed NER fine-tuning and
physical downstream scores remain pending. These local additions do not alter
the in-flight frozen endurance payload.

The current four-host physical calibration completed 100 ordered, accepted Pallas
updates with no dense projection fallback. All four worker logs are retained under
`artifacts/encoder-sustain-next-0923/calibration-evidence/`. The actual primary
was worker0 in this allocation. Median measured update:0.132512313 seconds.
The shared frozen plan selects16,980 baseline updates, requires1,800 measured
seconds, and estimates5,400.12 seconds for baseline plus recovery and900 seconds
of overhead allowance. This plan fits the remaining lease but is not acceptance;
the measured sustained baseline and exact final recovery remain to be verified.
Frozen plan: `artifacts/encoder-sustain-next-0923/endurance-plan.json`.

The sustained baseline is now executing; a read-only observation saw accepted
update664/16980, no dense fallback, 0.1370-second update. Its sparse fixture had
2,131 nonpadding tokens out of32,768 input positions in that update, so input-position
throughput must not be presented as real-corpus throughput. All four hardware logs
were independently checked: TPU backend,16 global devices,4 processes, and distinct
process indices{0,1,2,3}. Audit: `artifacts/encoder-sustain-next-0923/hardware-audit.json`.
The full measured-duration and exact-recovery gates remain pending under live
controller41234 and the existing cleanup guard; no new allocation was created.
# Interrupted sustained run: evidence preservation repair

The next attempt, `encoder-sustain-progress-0923`, is provisioning with the repaired
payload (SHA256 `909d89e32350ab38cffffae897e5d8e4def1abb7dcbea747eeff00aeb45eb2f2`).
Its live controller is recorded in `artifacts/encoder-production-0923/continuation.json`;
consult current cloud evidence rather than treating this status as permanent.

Matched downstream development protocol is frozen in
`artifacts/encoder-xnli-matched-0923/plan.json`: released mmBERT versus pilot step
1972; seeds 17/29/43; 392,702 English training rows; 37,350 development rows across
15 languages. All 51 data/snapshot files were hashed and prepared data semantics
validated. Each run uses 12,272 updates, batch 32, length 256, LR 2e-5 and 737 warmup
updates. Calibration must preserve that full schedule; an insufficient lease
must not silently shorten it. No physical classifier result is available yet.
This is an explicitly defined one-epoch-equivalent development comparison,
not replication of a published XNLI training protocol or production acceptance.

The `encoder-sustain-next-0923` four-host Pallas run calibrated successfully,
but its baseline did not finish. Last observed update was 3,153/16,980. All four
SSH transports exited 255; the controller exited 1 and both queue and node were
independently confirmed absent. Cloud audit termination reason is `UNKNOWN`;
preemption is not established. No sustained-duration or exact-recovery pass.

The GCS inventory contained hardware evidence but no baseline log because stage
logs were uploaded only after completion. `scripts/stage_progress.py` now sends
partial logs to each worker's `progress/` prefix every 60 seconds and emits an
SSH heartbeat. Upload failures do not kill training; subprocess deadlines still
kill and reap the child. Partial snapshots never mark a stage successful. Final
publication and existing measured-duration/recovery gates remain required.
This repair has 52 passing targeted progress/scale/CI tests and passes lint/type
checks; it is not yet physically validated. No new TPU was allocated for it.

## Matched downstream evidence and remaining gates — 2026-09-23

The full seed17 XNLI development pair has finished. Both final checkpoints are
at update12,272. Independent review checked all accepted updates (baseline resumed
at101, candidate started at1), identical training/evaluation provenance, and all
37,350 development examples across15 languages. Macro accuracy is77.9331% for
released mmBERT and77.4056% after the50M-token continued-MLM pilot: a decline of
0.5274 percentage points. Fourteen languages declined; Urdu improved. The lower
MLM development loss therefore has not established improved downstream quality.
This candidate uses GeGLU/dot-product attention, not YAT. One seed is insufficient
for a stable quality conclusion; no production qualification is claimed.
Evidence: `artifacts/encoder-xnli-pair17-0923/terminal-audit.json` and its downloaded
baseline/candidate/comparison reports. Controller exited0; queue and VM absence
were independently verified.

Seed29 remains live; latest downloaded snapshot reached accepted update6,179 of
12,272 in baseline training. Seed43's first attempt timed out waiting for capacity,
with controller exit1 and independently verified queue/VM absence. Its retry uses
the identical frozen archive and full protocol, a fresh queue/evidence prefix,
a two-hour lease, a900-second capacity wait and automatic cleanup. Controller
57292 and its budget record are in `artifacts/encoder-xnli-pair43-retry-0923`.
Retained reservations total$867.17 of the original$900 limit; this is not actual
billed spending. Consult continuation.json and live cloud state for later status.

The latest v6e and eight-host attempts also timed out before training and were
cleaned. They provide no physical qualification. Sustained multi-host duration
and exact recovery, larger-scale execution, v6e execution, three-seed downstream
results, and broader multilingual task quality remain open.

## Complete-seed reporting

`scripts/summarize_encoder_classifier_seeds.py` aggregates raw baseline/candidate
XNLI reports only when every seed in the frozen plan is supplied exactly once.
It revalidates each pair, rejects changed source/runtime/data/recipe across seeds,
and reports mean, sample standard deviation, minimum and maximum for macro and
per-language paired changes. These descriptive statistics are not significance
claims, and the report never marks production quality qualified. The actual
single completed seed was checked and correctly rejected as an incomplete
campaign. Twenty-seven combined pair/aggregation tests, Ruff and Pyright pass.

Invoke after all three full pairs finish, passing `--plan PLAN`, one
`--pair SEED BASELINE_JSON CANDIDATE_JSON` per frozen seed, and `--output REPORT`.
The output hashes all input files for audit. The independently checked training
logs and physical runtime inventory remain separate required execution evidence.
Do not substitute this report for bitext, NER, retrieval, long-context or broader
production-quality gates.

The alternate-zone v6e attempt in us-central1-a exhausted its capacity wait before
hardware execution. Controller exit1 and independent queue/node absence are
recorded in `artifacts/encoder-v6-alternate-0923/terminal-audit.json`. It establishes
no v6e qualification. The32-chip/eight-host retry uses the same frozen repaired
payload and unchanged30-minute measured baseline and exact recovery requirements.
Its live state must be read from the controller/cloud, not inferred from this text.

Budget reconciliation is recorded separately from historical reservations in
`artifacts/encoder-production-0923/reservation-reconciliation.json`. Only three
completed, independently absent v5e requests were revised to conservative lifetime
bounds: full request duration plus1800seconds at$1.20/chip-hour and the original
ancillary allowance. All v6e and other reservations were retained. Including the
new$98 eight-host reservation, planning exposure is$880.24 against the original
$900 limit; historical reservations total$989.17 and remain intact. Neither number
is actual billed spending. The latest observed marketing credit is$848.53; delayed
billing and project-attribution limits remain. No additional credit pool was added.

## Full three-seed XNLI development result

All six classifier runs completed the frozen12,272-update schedule. Final
checkpoints and all37,350 development examples in15 languages were independently
verified per run. Seed17 baseline resumed the verified100-step calibration;
all other classifier runs started at update1. All three comparison controllers
exited0 and their exact queue/VM names were independently confirmed absent.

| Seed | Released baseline accuracy | Continued-MLM candidate accuracy | Change(pp) |
| --- | ---: | ---: | ---: |
|17|77.9331%|77.4056%|-0.5274|
|29|77.3896%|77.8313%|+0.4418|
|43|76.9639%|77.3199%|+0.3561|
|Mean|77.4288%|77.5190%|+0.0901|

Sample standard deviation of the paired changes is0.5366percentage points.
The small mean change relative to observed seed dispersion does not establish
reliable improvement. Mean changes are negative for Arabic, Bulgarian, Swahili
and Vietnamese. Urdu improves in each seed. No statistical significance or
production acceptance is claimed. The50M-token continued-MLM candidate uses
GeGLU/dot-product attention; this is not a YAT quality result or published-recipe
replication. Lower development MLM loss did not yield a clear XNLI improvement.

Raw-provenance-checked aggregate and input hashes:
`artifacts/encoder-xnli-matched-0923/three-seed-summary.json`. Per-run audits and
cleanup evidence reside in the corresponding encoder-xnli-pair artifact roots.
The v6e, larger-scale, sustained multi-host/recovery and broader multilingual
quality requirements remain open. No new production-quality conclusion follows
from completing this one development benchmark.

## Reusable bitext embedding inference

`EmbeddingSession` in `scripts/export_encoder_retrieval.py` now retains one restored
model, parameter digest and compiled pooling callable across subset exports. Its
checkpoint constructor accepts an explicit step. `from_pretrained(snapshot,
config=...)` provides the released baseline using the same requested inference
configuration; it rejects architecture changes, verifies weight/config/tokenizer
identity during loading, and records released origin separately from a checkpoint.
The old single-subset `export` API remains available. Input checks, atomic export,
full row coverage and numerical-output validation remain in place.

CPU tests verify two consecutive subset exports require one checkpoint restore,
one parameter digest and one inference trace for matching shapes, with outputs
matching direct pooling including a padded final batch. Actual tiny safetensors
weights are loaded through the released adapter and checked against independently
loaded pooling outputs and parameter identity. Ten exporter/bitext evaluation
tests pass, with Ruff and Pyright clean. No physical throughput improvement or
held-out model score is claimed. The full matched campaign worker and lease
calibration remain required before TPU execution.


## Complete matched bitext execution harness

`scripts/export_encoder_bitext_campaign.py` now runs one role of the frozen
bitext plan with one loaded model shared across every subset. Baseline and
candidate run in separate processes. Preflight verifies the inventory, every
prepared input hash, dataset revision, subset counts, sequence length, and released
weight hash before model allocation. The current real preflight verified all
560 files, 112 subsets and 88,877 pairs; evidence is
`artifacts/encoder-bitext-matched-0923/campaign-preflight.json`.

A final `report.json` is published only after every export, full scoring, and
another frozen-input verification pass. `progress.json` records partial subsets
and is never completion evidence. Existing output directories are refused.
The runner records load/export/scoring duration, batch size, source/configuration
identity and available devices; it explicitly does not claim all-device
utilization. Its cooperative between-subset deadline requires an external worker
hard timeout and the existing resource-lease cleanup for paid execution.

`scripts/compare_encoder_bitext_campaign.py` audits the original exported arrays,
manifest identities, released-versus-checkpoint origins, matched configuration,
tokenizer, sources and runtime. It recomputes all rankings before accepting the
recorded scores and returns per-subset and macro deltas. Completion is not official
MTEB parity or production acceptance. Tiny real released-safetensors/checkpoint
integration tests cover full paired execution; interruption, frozen-input
mutation, incomplete coverage and altered scores are rejected.

No full-model bitext inference has run yet. Next: bounded TPU timing calibration,
then the complete matched campaign if its conservative estimate fits the budget.
No additional cloud resource was allocated for this implementation.


## Physical bitext campaign submitted

`artifacts/encoder-bitext-pair-0923` contains the frozen 788-file payload, extracted
CPU preflight, launch command and $14 reservation. Overlay SHA256:
`4d41c158d67e13f43b7ae5efadffc13ca3156e6e61fa76d624ecdbdec85dbfee`.
The exact Spot request `flaxchat-validation-bitext-pair-0923-0` was observed
PROVISIONING in us-west4-a, accelerator v5litepod-4. Cleanup execution
`a4833739-675c-4e7c-90f4-daeaed8430c9` is armed with deadline 1790226069.8875241.
Submission is not hardware or quality qualification.

The stdlib parent first runs baseline/candidate calibration in separate TPU
processes. Each exports the two largest complete subsets without scoring them.
The planner retains first-call compilation, floors every full subset at the
measured warm-subset cost, adds a 1.5× pair margin and 600 seconds for scoring,
and separately reserves 360 seconds for final evidence publication. A pair that
does not fit the remaining lease is rejected. Full campaigns and CPU comparison
run only after passing that check. Model execution remains single-device; four
available TPU devices are not a claim of four-device utilization.

43 relevant CPU tests pass, with targeted Ruff and Pyright clean. Conservative
planning exposure after reservation is $894.2367 of $900; historical reservations
remain recorded separately and actual gross spending remains unreconciled.


## Parallel qualification attempts after reservation audit

Fresh west4 queue/node inventories verified that the completed eight-host retry
and classifier calibration resources remain absent. Their conservative planning
bounds count their entire request lifetime plus 1,800 seconds at on-demand rates,
plus the original ancillary reserve. The two reductions total $72.70. Historical
ledgers remain unchanged and this is not posted-spend reconciliation; see
`artifacts/encoder-production-0923/reservation-reconciliation-v2.json` and its
immutable inventory snapshots.

Bitext is on active v5e hardware, exporting the full baseline after calibration.
A separate four-host v5e-16 attempt uses the same frozen repaired-backend payload
and unchanged full30-minute/recovery acceptance criteria. Relevant current source
hashes match that payload. A separate v6e-4 attempt is submitted in us-east1-d,
which advertises the type and has a published regional pricing row. It includes
the latest compact-pair YAT implementation and passed extracted-archive preflight.

Active attempts: `encoder-bitext-pair-0923` ($14),
`encoder-sustain16-bounded-0923` ($50), and `encoder-v6-east-0923` ($24).
All have independent cleanup leases. Aggregate conservative planning exposure
is $895.5367 of $900, while historical reservations total $1,077.1667 and are not
actual spending. No qualification is inferred from a live request or export
progress; complete physical evidence and final cleanup audits remain required.


## Independent endurance auditor hardening

The post-run auditor now requires exactly one successful integer controller rank
per SSH worker; duplicate/missing ranks and boolean return codes cannot qualify a
run. It also rejects primary training-update records in non-primary JAX worker
logs and invalid topology/horizon/batch/duration inputs. SSH ranks are not assumed
to equal JAX ranks; each inventory is validated separately. 47 scale/endurance
planner tests pass, including malformed-controller and duplicate-update fixtures.
This local auditor postdates the running payloads and will inspect their unchanged
raw artifacts after completion; it does not modify any live worker.


## Baseline bitext completion and numerical portability finding

The released baseline finished all112 subsets/88,877 pairs in 734.70 seconds on
TPU. Its immutable exports were manually copied to GCS while candidate export
continued. A worker durability gap was repaired locally: future campaigns publish
each completed model before starting the next and avoid overwriting completed
objects during final publication. This patch postdates the active frozen worker;
33 relevant tests and targeted lint/types pass.

Independent local float32 ranking recomputation changed two of 88,877 predictions
(ben-eng row725 and srp-eng row476). Every embedding checksum matched. Python
`math.fsum` cosine checks found distinct candidates with true margins about
1.52e-7 and 2.05e-7, consistent with float32 score accumulation changing ranking.
The exact legacy cross-platform audit failed and remains recorded as failed.

An explicit float64 scoring option was added without changing the legacy default
or the running frozen campaign. High-precision full recomputation matches every
remote baseline prediction and metric. The optional reports bind the complete
scoring source, including `flaxchat/retrieval.py`.48 relevant tests pass, including
near-tie, row-order, and block-size cases. Float64 reduces this numerical risk but
is not a universal bitwise-portability guarantee.

Baseline subset-macro F1 is 0.301584049274632 and pair accuracy is
0.338995098163219 under this supplied embedding recipe. These are not official
MTEB acceptance or production qualification. Evidence:
`artifacts/encoder-bitext-pair-0923/baseline-high-precision-audit.json`,
`baseline-portability-diagnostic.json`, and `baseline-float64-supplemental.json`.
Candidate comparison and full endurance/recovery remain pending. The v6e east1
attempt hit its capacity deadline without hardware; controller exit1 and exact
queue/node absence are independently recorded in its terminal audit.


## Native matched bitext pair completed

The full worker completed both calibrations, both112-subset exports, and the
matched comparison; controller exit0 and exact request/VM absence are independently
verified. Native subset-macro F1: baseline0.301584049274632,
candidate0.30558368490264315 (delta+0.39996356280111445 percentage points).
Accuracy: baseline0.338995098163219, candidate0.34066836327821093.
The candidate array download and independent float64 rescore are complete.
Every prediction for both models matches the native reports, and input,
embedding, model and report hashes were independently verified. The original
float32 reports remain unchanged; their separate cross-platform float32 parity
failure remains recorded. See `artifacts/encoder-bitext-pair-0923/pair-high-precision-audit.json`.
This is the GeGLU continued-MLM candidate, not YAT training-quality evidence,
official MTEB parity, or production acceptance. Sustained16 remains live.

Across the 112 language subsets, F1 improves on 61 and regresses on 51. The
largest losses are Tamil (−3.87 percentage points), Bulgarian (−3.57), and
Norwegian Nynorsk (−3.04). Subset sizes differ; these are observed differences,
not significance claims or production acceptance. Full per-language results are
preserved in `artifacts/encoder-bitext-pair-0923/language-deltas.json`.

The latest four-host, 16-device baseline observation verifies 13,448 consecutive
accepted updates and 1,851.62 measured seconds with finite losses and no dense
projection fallback. It has crossed the 1,800-second duration gate, but has not
finished the frozen 16,994-update baseline or interruption/recovery protocol.
It remains unqualified until the complete protocol and exact final-state
comparison pass. Evidence: `artifacts/encoder-sustain16-bounded-0923/baseline-progress-audit-4.json`.

## Useful-token throughput and alternate v6e attempt

A later observation verifies 15,676 accepted baseline updates and 2,164.81
measured seconds. This sparse endurance fixture processes 237,267 padded input
positions/s but only 16,832 non-padding tokens/s (7.09% non-padding). Neither
number is a representative production-corpus benchmark. The independent auditor
now labels the legacy throughput as padded input positions and reports useful
token throughput separately. Partial or invalid non-padding counts fail the
audit; legacy logs with no counts report that metric as unknown. The change
postdates both live payloads and does not change training arithmetic. All 52
focused scale/planner tests pass; Ruff and type checking of the changed source
pass. Direct type checking of the test file also exposes existing heterogeneous
fixture typing errors; it is not reported as a clean test-file type check.

Evidence: `artifacts/encoder-sustain16-bounded-0923/baseline-progress-audit-6.json`.
The entire baseline and exact recovery remain pending.

One alternate v6e-4 Spot attempt is live in us-east5-b, using the verified
`encoder-v6-east-0923` payload and a new output prefix. Its capacity deadline is
600 seconds, lease is 5,400 seconds, and cloud cleanup guard is recorded under
`artifacts/encoder-v6-east5-0923/cloud/attempt-0/guard.json`. The observed queue
state is WAITING_FOR_RESOURCES; no hardware qualification is claimed.

After reconciling the independently cleaned bitext and east1 v6e attempts, the
new $24 reservation brings conservative planning exposure to $897.7767 of the
original $900 budget. Actual billed spend and current credit balance remain
unknown. Original ledgers are unchanged; reconciliation evidence is in
`artifacts/encoder-production-0923/reservation-reconciliation-v3.json`.

## Full sustained baseline completed; recovery protocol running

The primary worker's completed baseline log independently passes the frozen
16,994-update horizon, 1,800-second minimum, finite accepted updates, no dense
fallback, and final committed-checkpoint gates. Measured duration is 2,350.49
seconds (39.17 minutes). The physical inventory reports four processes and 16
TPU devices, with the Pallas loss backend. Useful throughput is 16,808 non-padding
tokens/s; padded input throughput is 236,898 positions/s. This remains a sparse
fixture, not a production-corpus performance claim.

The worker reports baseline exit0 and has started the separate interruption run.
The campaign is not yet qualified: committed midpoint SIGKILL, resumed training,
exact final-state equality, all-worker completion and cleanup are still required.
Evidence: `artifacts/encoder-sustain16-bounded-0923/baseline-completion-audit.json`.
The prepared `audit_completed.py` in that directory collects final worker evidence
only after controller success and runs the independent frozen-protocol audit.

## Follow-up CPU regression and v6e cleanup

The east5 v6e request exhausted its 600-second capacity wait without entering
training. Controller exit1 and exact queue/VM absence are independently recorded
in `artifacts/encoder-v6-east5-0923/terminal-audit.json`. The reservation is
reconciled using the same conservative method as prior attempts, with historical
ledgers unchanged. Planning exposure is now $883.0167, leaving $16.9833 within
$900; this is neither a posted bill nor a current credit balance.

Broader CPU verification found an optional PyTorch import preventing all test
collection in the JAX environment. The benchmark preflight now imports its
PyTorch adapter only when executed; the sole PyTorch-dependent test explicitly
skips when that optional dependency is absent. Six framework-independent tests
remain active. No packages were installed or upgraded.

After that repair, 1,192 tests pass, 24 skip, and five slow/accelerator tests are
deselected (240.99 seconds). Repository Pyright reports zero errors; targeted
Ruff passes. This run does not measure coverage or qualify physical hardware.
Evidence: `artifacts/encoder-regression-followup-0923/verification.json`.
Production language coverage has been requested from the user and remains
undecided; pilot coverage must not silently become the production requirement.

## NER checkpoint training and entity evaluation

The existing first-subword BIO preparation and token-classification head are now
connected to the checkpointed trainer. Select `--task token_classification` in
`scripts.finetune_encoder_classifier`; the sequence-classification default is
unchanged. Token loss is weighted by active first-subword labels, not padded
positions. The recipe records the label inventory, alignment policy and absence
of pooling. It uses the shared optimizer, finite-update gate, global batch
sharding and exact-resume cursor checks.

Example commands after preparing pinned train and validation splits:

```sh
python -m scripts.finetune_encoder_classifier --task token_classification \
  --config CONFIG.json --data NER_TRAIN --pretrained SNAPSHOT \
  --output CHECKPOINT_DIR --steps STEPS --batch-size BATCH
python -m scripts.finetune_encoder_classifier --mode evaluate \
  --output CHECKPOINT_DIR --checkpoint-step STEP --eval-data NER_VALIDATION \
  --batch-size BATCH --report ner-validation.json
```

Evaluation dispatches from the checkpoint family. It reconstructs words only at
verified first-subword positions, checks full row/word coverage, excludes padded
final-batch entries, preserves word-level references/predictions, and reports
exact BIO entity counts/F1 overall and per language. It rejects training splits,
mixed validation/test reports, duplicate language/split inputs, changed dataset
or tokenizer identity, reordered/changed label inventories, and overwriting an
existing report. Array hashes are revalidated before publication.

This adds the missing execution path, not a production-quality result. Real NER
data, a frozen matched baseline/candidate protocol, physical TPU validation and
held-out multilingual scores remain required. Current live TPU bundles predate
this addition; no source was changed inside their running jobs.

NER adapter verification: 62 related CPU tests pass and three optional distributed
checks skip; both FP32/BF16 exact-resume and full-word evaluation cases also pass
with four simulated CPU devices. Targeted Ruff and Pyright pass. CI includes the
new NER tests in the multi-device command and selects them for trainer/evaluator
changes. Evidence: `artifacts/encoder-regression-followup-0923/ner-verification.json`.
The earlier 1,192-test full-suite result predates this NER addition.

## Committed interruption and resumed training observed

The primary worker finished all 8,497 pre-interruption updates, committed the
midpoint checkpoint, printed the explicit fault-injection marker and exited with
the expected SIGKILL status (-9). The retained remote checkpoint manifest has
step 8,497 and nonempty model, optimizer and training-state inventories.

The resumed process starts at step 8,498. Independent comparison verifies that
all pre-interruption updates and the first 320 resumed updates (through 8,817)
exactly match baseline losses, token/masked-token counts, learning rates and
acceptance/fallback decisions. This is partial recovery evidence, not the final
qualification: all workers must finish 16,994 and their final raw manifests must
match baseline exactly, followed by independent resource-cleanup verification.
Evidence: `artifacts/encoder-sustain16-bounded-0923/midpoint-and-partial-resume-audit.json`.

## September 24: terminal endurance failure and cleanup permission gap

The sustained run is terminal, not still training. All four controller SSH
workers returned 124; recorded resume progress ends at step 11,920 rather than
16,994, and no final recovery checkpoint exists. Independent raw-log comparison
matches all 3,423 resumed updates (8,498–11,920) exactly against baseline across
loss, accepted-update flag, fallback flag, learning rate and token counts.
The full baseline completed 16,994 updates over 2,350.49 seconds. These are useful
partial results, but sustained final-state recovery remains unqualified.
Transport timeout and runtime telemetry do not establish the cause of stalled
progress. Evidence: `terminal-audit.json` and `terminal-resume-prefix-audit.json`
under `artifacts/encoder-sustain16-bounded-0923/`.

Independent inventories confirm this attempt's queue and VM absent. Cloud logs
show the cleanup workflow principal repeatedly denied `tpu.nodes.delete` in
us-west4-a; local-controller credentials ultimately removed the resources.
The observed conditional IAM grants cover us-east5-a, us-central2-b and
us-central1-a, not this attempt's zone. An ACTIVE workflow was insufficient.

The local guard now binds the execution to the currently deployed workflow
revision and service account, then requires Policy Troubleshooter's overall
CAN_ACCESS result for tpu.nodes.get/delete on the exact queued-resource name at
expiry and at the end of the 30-minute retry period. Unknown/denied results,
unavailable APIs and changed workflow revisions block allocation. It checks
remaining lease time again after permission queries. These checks assess the
current policy at two timestamps, not a guarantee against future IAM changes or
arbitrary time-dependent conditions between them.

The project's Policy Troubleshooter API was observed disabled. No IAM policy or
API activation was changed. New allocations remain blocked pending verifiable
cleanup access. Validation: 52 guard/operations/transport tests pass; targeted
Ruff and source Pyright pass. No new TPU allocation was made for this repair.


September 24 follow-up: the user approved the YAT-scoped cleanup grant and API
activation. Direct v3 testing showed that Policy Troubleshooter rejects TPU
queued-resource names; the guard was corrected to use live TPU API probes from
the actual workflow principal before allocation. Source/revision/lease checks
bind the proof to the reviewed guard, and audit logs confirm deletion permission.
See `docs/YAT_MMBERT_TRAINING.md` for the new model campaign, first Spot preemption,
verified cleanup, padding optimization, and bounded retry. This supersedes the
permission-simulator admission approach described above.
