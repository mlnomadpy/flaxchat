# Continuing YAT embedding representation training

Current October 6 continuation: [overnight run and receipts](representation-v2-2026-10-05/overnight-1006/README.md).
The prior physical run trained through step500. This continuation preserves its
optimizer, sampler and20,000-step schedule while extending the execution horizon
to60,000 under a12-hour TPU lease. The explicit pinned
`--resume-quality-migration` changes finite quality regressions to report-only;
all data/model/optimizer identity remains checked. This is contrastive
representation training, not additional MLM. See the dated status receipt for
current physical acceptance and progress.

Checkpoint managers must not enable generic Orbax temporary-directory cleanup
in a GCS root containing nested best checkpoints. Metadata/restore readers are
read-only; protected starting checkpoints remain outside rotation.


Current October 5 continuation: [representation-v2 campaign](YAT_REPRESENTATION_V2_CAMPAIGN.md).
This supersedes the dated parent-materialization status below: all 181 public
parent tensor leaves are authenticated, while restoration on TPU and current
QAT execution remain unqualified. The first corrected physical request exhausted
its Spot capacity window; queue/node absence is verified. Separate public releases
are authorized for each completed and validated stage.

Latest reconciled audit: [all open issues, current evidence and tool priorities](SYSTEM_AUDIT_CURRENT_2026-10-01.md). Earlier dated findings below remain historical where superseded.


Latest requested parallel review: [audit, new gaps, tools and validation scope](PARALLEL_SYSTEM_AUDIT_2026-10-01.md). All 29 open issues were refreshed; historical counts below remain dated evidence.

Canonical operating guide as of 2026-09-30. The objective is to improve the existing model's representations for multilingual text and code retrieval. This guide supersedes the launch procedure in `TPU_TRAINING_CHECKLIST.md`; that file records earlier historical runs and is not a live resource ledger. Audit findings are in [evaluation-review.md](audit-2026-09-30/evaluation-review.md). The versioned agent workflow is [flaxchat-training](../skills/flaxchat-training/SKILL.md).

The [October 1 follow-up audit](SYSTEM_AUDIT_2026-10-01.md) supersedes the implementation status below where noted. New fixes are local, pass selected model-free checks, and remain physically unqualified. The [issue checklist](ISSUE_READINESS_CHECKLIST.md) records the current remaining defects and separates training, scaling and release gates.

## Current model and the quality decision

The published embedding-v1 stage is described in [its card](YAT_MMBERT_EMBEDDING_V1_MODEL_CARD.md), with a separate [PyTorch conversion card](YAT_MMBERT_EMBEDDING_V1_PYTORCH_MODEL_CARD.md). It continued existing trained YAT weights for 14,000 updates at global batch 128 on MS MARCO, MIRACL, and CodeSearchNet. The cards report English MTEB mean 52.78, common eight-category multilingual mean 48.21, and CoIR mean 43.45. These values are historical measurements, not evidence of a currently active training resource or a matched EmbeddingGemma comparison.

The mean improvement conceals a Tatoeba decline from 52.72 to 35.27, spanning 108/112 language pairs. The next stage should test multilingual replay while preserving retrieval and code performance. This is a hypothesis about representation retention, not a proven causal diagnosis. Keep the current published checkpoint immutable. A candidate must pass declared development criteria before a large continuation allocation; a final test suite is for reporting, not repeated mixture selection.

## Implemented tools and proposed upgrades

| Capability | Available implementation | What remains |
| --- | --- | --- |
| Embedding fine-tuning | [train_yat_embedding_finetune.py](../scripts/train_yat_embedding_finetune.py): physical TPU guard, manifest-selected sources, global relevance-aware hard-negative InfoNCE, mixed or homogeneous batches, language-temperature replay, Orbax exact stage resume and authenticated parent import | Physical qualification of changed semantics; curriculum and scalable global-negative strategies |
| Data preparation | [prepare_yat_embedding_finetune.py](../scripts/prepare_yat_embedding_finetune.py): pinned registry pair/triplet sources, explicit truncation, vocabulary/special-token/content manifests, exact code identities and mixture-wide held-out quarantine | Benchmark contamination beyond exact text; source quality review |
| Development validation | Bounded declared held-out retrieval probes for every source, finite MRR/Recall gates, best-checkpoint retention and persisted quality state | Language-stratified retrieval/bitext and pinned STS gates implemented, including per-language regressions; physical recovery/metric acceptance and production suite quality remain unproven |
| Physical TPU tests | [validate_test_suite.py](../scripts/validate_test_suite.py) runs isolated modules with JUnit, timeout and GCS evidence; [validate_gcp_multihost.py](../scripts/validate_gcp_multihost.py) covers its declared workloads | Dedicated hard-negative, cache and trainer recovery tests now exist; the new [embedding multi-host worker](EMBEDDING_MULTIHOST_ADMISSION.md) checks actual distributed training. Physical execution, sustained interruption and each changed recipe/topology remain unqualified |
| Full public checkpoint evaluation | [evaluate_yat_public_mteb.py](../scripts/evaluate_yat_public_mteb.py), [subset evaluator](../scripts/evaluate_yat_mteb_subsets.py), [split evaluator](../scripts/evaluate_yat_mteb_splits.py), merge helpers: finite score checks, expected row coverage and source/package provenance are implemented | Immutable task revision/inventory and interpreter/numerical environment checks are implemented; physical task receipts and matched EmbeddingGemma adapter remain pending |
| Recovery | [checkpoint.py](../flaxchat/checkpoint.py), immutable stage metadata, explicit `--resume` | Check changed embedding trainer interruption behavior and topology transitions on the actual physical topology |
| Profiling | [profiling.py](../flaxchat/profiling.py), [summarize_jax_trace.py](../scripts/summarize_jax_trace.py) | Scheduled trace wired into embedding trainer; input, gradient, token and checkpoint metrics added. Physical traces and throughput still required |
| Cloud operations | Guarded GCP lifecycle plus [representation_run.py](../scripts/representation_run.py) sealed manifest launch/worker, [build_runtime_bundle.py](../scripts/build_runtime_bundle.py) Linux frozen wheelhouse and [cloud_accounting.py](../scripts/cloud_accounting.py) posted billing/retained storage | Fresh GCE offline dependency install passed; TPU install/execution and physical disconnect acceptance remain. Actual training output-path binding and installed inventory revalidation are implemented; [deployment guide](INFRASTRUCTURE_RUNS.md) |
| Publishing | [Flax publisher](../scripts/publish_yat_embedding_from_gcp.py), [PyTorch publisher](../scripts/publish_yat_torch_from_gcp.py), [TPU parity tool](../scripts/validate_yat_torch_parity.py) | Generic authenticated stage identity is implemented. Full runtime provenance and bounded campaign scheduling are implemented; physical matrix coverage and verified upload remain pending |
| Larger contrastive batches | Current trainer uses replicated parameters/optimizer with globally sharded batch arrays | Pair-only static update avoids the third encoder, and hosts materialize only their local rows. Opt-in `--encoder-chunk-size` gradient caching is implemented with a full global denominator and recomputed encoder VJPs. FSDP remains unimplemented; all paths need physical loss/gradient and throughput measurement |
| Teacher/MRL/quantization | No integrated embedding-v1 objective for these features identified in this audit | Distillation, Matryoshka dimensions and quantization are future work; do not pass imagined flags |

Before using any command, inspect its current parser or help in an environment that does not execute model work, and inspect the launcher manifest. The table documents code availability, not successful qualification of every topology or version. Historical evaluation `run_yat_*` shell scripts still contain fixed paths, steps, bucket prefixes and model IDs; treat them as receipts/examples. Full-training and setup wrappers now require the immutable manifest route documented in the deployment guide.

## Stage admission and lineage

1. Identify the intended parent by immutable Hub revision or durable checkpoint path, step, weight/config/tokenizer SHA-256, and publication/export receipt. Use existing trained weights. Keep parent, new-stage output, reports and release prefixes distinct.
2. Decide **exact resume** versus **new stage**. Exact resume requires the same data/model/objective/optimizer/schedule/source identity and restores the cursor. Changing the recipe or stage schedule uses a new output and an explicit optimizer reset/import policy. Check `start < stop <= total_steps`; an already-completed stage must not appear to train successfully without updates.
3. Freeze a stage manifest containing source commit plus dirty source snapshot digest, dependency/runtime versions, YAT settings, parent identity, dataset revisions/content hashes, tokenizer, splits, sampler/seed, objective, effective global batch and negative pool, lengths, optimizer/schedule, checkpoint retention, quality criteria and evaluation manifest. All hosts must agree. Amendments create a new manifest identity.
4. Define a finite run segment: planned useful exposures and steps, wall-time limit, attempt limit, cost cap, evidence location and independent cleanup deadline. Keep failure/retry records and a durable last-success checkpoint. No open-ended retry loop.

The current trainer's parser implements `--parent-public`, `--parent-manifest`, optional `--parent-checkpoint` and `--parent-step`, repeated `--data`, `--source-weights`, `--output`, `--steps`, optional `--stop-after`, `--batch-size`, `--learning-rate`, `--temperature`, `--warmup`, `--weight-decay`, `--seed`, `--save-every`, `--eval-every`, `--keep-checkpoints`, `--distributed` and `--resume`. These are existing interfaces, not a ready-to-run new recipe. Additional interfaces now include `--batch-policy`, `--language-exponent`, `--dev-max-rows`, `--max-dev-regression`, `--profile-dir`, `--profile-skip` and `--profile-steps`, `--encoder-chunk-size` and repeated `--sts-dev NAME=DIR`. These implementations remain unqualified on physical TPU until new receipts exist. A production run manifest should not claim those features merely because an older trainer has related helpers.

## Representation data and selection

Use held-out development examples for early decisions and keep final benchmark tests isolated. Pin their revisions and exact query/document relevance groups. Audit training data against both before training, including normalized text, duplicate groups, known translations and code identifiers/content where feasible. Record exact overlap and unresolved semantic/pretraining exposure separately.

Test bitext replay with existing multilingual pairs and review language distributions, query/document direction and repetition. Do not assume every translated pair is retrieval supervision. Homogeneous dataset batches can make in-batch negatives more useful; preserve cross-language tasks explicitly. Relevant alternative documents and translations must never become false negatives because source-local IDs differ.

Use a small development suite with English retrieval, multilingual retrieval, cross-language bitext alignment, STS and code retrieval. Before the run, record baseline values, acceptable regressions, and the selection rule. Report per-task/per-language values, macro means, uncertainty where feasible, truncation and domain slices. Loss may diagnose learning but cannot select a representation model on its own. After a candidate passes development gates, run the final pinned suite once for reporting; reserve fresh final data if earlier tests influenced model selection.

## Physical TPU qualification and profiling

Model forward/backward, precision, convergence, throughput and sharding verification run on physical TPUs. Local manifest/schema/documentation and orchestration checks do not execute the model and are appropriate. The user's preference excludes CPU model timing or validation as evidence for TPU choices.

Required coverage for a changed training path includes hard-negative loss and gradients, duplicate/relevance masking, no-explicit-negative batches, empty/all-padding behavior, finite alpha/weight gradients, actual lengths and BF16 numerics, deterministic sampling, interrupted optimizer/cursor recovery and a short real-data update sequence. For physical multi-host work, require all-host source/manifest agreement, collective completion, unique local row ownership, global-negative semantics, durable checkpoint commit, interruption/resume and transport-disconnect behavior. Mark every device family/chip/host combination passed, failed, skipped, unavailable or not run. A one-host or localhost multiprocess result does not qualify physical multi-host training.

Profile a warmed-up representative training step, including host preparation, transfer, forward/backward, collective operations, optimizer and checkpoint overhead. Block until device work completes when timing asynchronous JAX operations. Use [JAX profiling](https://docs.jax.dev/en/latest/profiling.html) and the trace summary helper. Record actual non-padding tokens and pairs per second, negative pool size, HBM and host RAM, compilation time, and checkpoint seconds. Preserve raw traces and source/runtime identities. Adopt a faster kernel or layout only when the full training step improves with loss/gradient and recovery behavior intact.

Gradient accumulation alone does not expand InfoNCE negatives across microbatches. Gradient caching must compute the intended full-batch embedding loss and reproduce its encoder gradients with correct RNG/rematerialization and global semantics. Validate on TPU before reporting a larger effective negative pool. FSDP may save state memory but can increase communication; existing helper code is not proof of embedding-trainer integration.

## Evaluation receipt contract

Pin benchmark package version, task names, dataset revisions, expected splits and subsets, metrics/scales, evaluation code/runtime hashes, model revision/files, batch/length/padding, prompts, pooling, normalization and backend/topology. A task is complete only with finite required scores and exact expected row coverage. Extra, missing, duplicate, conflicting, unknown or wrong-revision rows fail. Partial suites must retain explicit coverage and never receive a full-suite label.

Aggregate first using each task's prescribed subset/split policy, then the declared task/category policy. Keep multilingual eight-category comparison and native nine-category mean separately named. Avoid a single mean concealing bitext loss. The current historical assembler fixes 41 English and 131 multilingual tasks; these counts describe its pinned inventory, not all future benchmark registries. CoIR and MTEB Code are distinct protocols. Paper-reported mmBERT/EmbeddingGemma numbers are context until actual matched candidate/baseline receipts exist. Use each competitor's documented prompts and report both default and controlled-length protocols if they answer different product questions.

## Budget, resources and cleanup

Verify the account/project explicitly; do not rely on an old browser tab or ambient default. Check current resource/queue ownership and outstanding allocations before requesting another. Preflight quota/type exposure is eligibility, not proof of free physical capacity. Verify flex-start/Spot behavior and regional support from [Cloud TPU queued-resource documentation](https://docs.cloud.google.com/tpu/docs/queued-resources) for the selected configuration.

Build a cap from current [TPU pricing](https://cloud.google.com/tpu/pricing), chip count, maximum billable lifetime, boot/storage/network and any supporting VM costs. Record pricing timestamp and whether the charge is estimated, posted gross cost, credit applied or net due. Use [Cloud Billing export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) or provider billing evidence for actual charges; billing latency and unavailable exports must remain visible. A runtime estimate is not actual spending, and a historical credit balance is not today's remaining balance.

Verify independent cleanup before allocation and leave checkpoint/sync margin before its deadline. Bound provisioning attempts, setup, workloads and retries. If capacity attempts exhaust the manifest budget or a deterministic error repeats, preserve the failure, release owned resources and fix the cause. Do not extend a lease beyond existing authorization silently. A lost SSH connection warrants status/reattachment before re-execution.

After completion or failure, verify queue and TPU node deletion independently, remove owned temporary VM/disk resources, and inventory storage objects including versions/soft-deleted data where applicable. Preserve parent weights, published artifacts and needed checkpoint/evidence prefixes. Delete storage only within the user's authorized scope and record retained versus removed objects. Zero compute consumption and empty storage are separate claims, each requiring evidence.

## Release and handoff

Export the selected durable checkpoint with all tensor/config/tokenizer/source identities and SHA-256 values. Preserve exact checkpoint step and parent lineage. A new training stage gets a new public identity when publication is requested; existing models remain reproducible.

JAX and TorchXLA forwards must run sequentially in separate environments because they own incompatible TPU runtimes. Bind parity receipts to source and target weights, configs, tokenizer, loader code, input fixture, dependency versions, actual device identity and comparison tolerances. Existing eight-input/128-token parity is narrow implementation agreement. Expand to varied batch sizes, actual production lengths, padding/masks, languages/code and meaningful retrieval corpus before claiming broader equivalence. Verify remote upload revision and bytes/hashes; remove authorized short-lived upload tokens and VM afterward.

Handoff must include stage state, last committed checkpoint/cursor, data exposure, measured representation regressions/improvements, TPU scope and JUnit skips/failures, profile evidence, charges versus estimates, release revisions and independently checked resource/storage cleanup. Update the implemented/proposed table when code changes. Record one next decision and stop repeating audits without new evidence.

## Current implementation admission tools

Before allocation, run `python -m scripts.preflight_yat_embedding_stage` with actual parent/data arguments to authenticate public files, parent checkpoint semantics and the global prepared mixture. Rebuild historical vocabulary-missing manifests with `python -m scripts.qualify_yat_embedding_mixture`; a corrected recipe is a new stage, never an exact resume of altered data. `scripts/prepare_embedding_sts_dev.py` prepares pinned development-only STS data.

The current selection score is the equal-source macro mean of aggregate MRR or STS Spearman. Every language has a separate regression gate, without multiplying a source's vote by its language count. The parent's baseline is retained at best step 0. Evaluation cadence, probe policy and tolerance belong to exact-resume identity.

## Implemented admission fixes and remaining qualification

The October 1 reconciliation records local fixes for complete required-array checksums, raw negative-valid agreement, failed-quality-gate persistence, a nonmutating committed checkpoint reader and complete parity runtime provenance. Selected model-free checks pass; physical and actual-data acceptance remain outstanding. The reader verifies recorded metadata integrity and namespace stability, not tensor restoration. The embedding qualification lock now pins Safetensors; the frozen corrected runtime passed physical v5e setup. Selected acceptance is 14 passes and two cache optimizer-state failures; see the [physical follow-up](audit-2026-09-30/physical-qualification-2026-10-01.md). See the [current audit](SYSTEM_AUDIT_2026-10-01.md) for source evidence and GitHub mapping. These are not established defects in historical public weights.

Shared admission uses the same trainer flags plus required `--expected-devices`, `--expected-processes` and `--receipt`. A missing or failing gate is not waived by relaunching the same stage. Fix the applicable defect, run the relevant metadata checks, freeze one revision, then perform one bounded physical qualification. Independent controllers must not allocate separately against a local budget ledger.

## Follow-up data and fault tools

The [independent development interface](INDEPENDENT_REPRESENTATION_DEVELOPMENT.md) documents the actual preparation arguments, exposure schema and conservative whole-bundle rejection policy.

Prepare candidate development exclusions first, then pass `--development-exclusions DIR` to every future training preparation. Exact text and aligned upstream groups are quarantined before deduplication, splitting and tokenization. MIRACL groups now share `miracl:{upstream_id}` across languages, preserving language separately; this requires a newly prepared recipe and new stage. Existing stage data must not be rewritten for exact resume.

`scripts.prepare_embedding_retrieval_dev` builds an independent tokenized retrieval/bitext/code bundle. Admit it through `--retrieval-dev NAME=DIR` in shared preflight and training. The loader recomputes candidate, token, future-training quarantine and declared historical parent-exposure evidence. The trainer records aggregate and per-language `independent/NAME` metrics in the same persisted baseline/best/recovery gate. Known-stage exposure checks do not prove complete historical, MLM, translation or semantic cleanliness. Actual production data and restored-parent evaluation remain required; metadata acceptance alone is insufficient.

The latest cache diagnostic exhausted its bounded capacity wait and ran no model tests. Both its queue and TPU node were independently verified absent. The previous physical acceptance remains 14 passed and two optimizer-state failures. Shape-matched ordinary autodiff and named Adam-moment diagnostics are implemented; do not relax the original required assertions. See the [follow-up consolidation](audit-2026-09-30/consolidated-followup-2026-10-01.md).

Use the [pinned candidate development recipe](REPRESENTATION_DEVELOPMENT_RECIPE.md), [programming-language provenance](CODE_LANGUAGE_PROVENANCE.md) and [exact overlap auditor](EMBEDDING_BENCHMARK_OVERLAP.md) with actual authenticated inputs. Preparation, parent exposure and cross-source quarantine remain explicit gates. The isolated [peer-loss campaign](audit-2026-09-30/embedding-peer-loss-campaign.md) and [attachment-loss campaign](audit-2026-09-30/embedding-attachment-loss-campaign.md) require an already leased physical multi-host slice; their metadata checks do not establish physical fault recovery.

## Production-quality and release safeguards follow-up

Shared admission now defaults to `--training-scope production`, requiring loaded,
authenticated independent retrieval, bitext and code plus STS. Declared defaults
are64 selected rows,16 per language and16 distinct queries/documents; these are
operational coverage floors, not statistical confidence or measured quality.
Qualification fixtures explicitly use `--training-scope qualification`.
Actual programming-language slices enforce regressions without extra macro votes.
Production STS must match raw pairs/scores/source in the same candidate bundle
and recomputed parent exposure; token arrays are recomputed against the parent
rather than trusting valid-shaped hashed files. This policy changes immutable
stage identity and requires an explicit new stage, never silent exact-resume migration.

Publication now requires v4 complete row/shape/backend scopes and retained private
raw evidence, which is rehashed and numerically recomputed before upload. Campaign
and replay have explicit host RAM/evidence budgets; campaign scratch is separate.
Bounded NPY headers reject unreasonable shapes before loading. The full advertised
context/batch matrix remains required. Physical acceptance of these changes is
still outstanding; no metadata pass upgrades an old physical or publication receipt.

## Concrete prerequisite tools after the collision result

The native MIRACL scan completed with three exact-text collisions in the broader exclusion inventory, not necessarily the selected cap. `scripts.filter_representation_development` now creates a separate bounded filtered candidate from authenticated complete collision details, preserving the original and replayable proof. The initial scan predates detailed output. The subsequent actual detail-producing scan/filter/re-scan completed: see [retained controller](audit-2026-09-30/native-miracl-filter-01-controller.json) and the current audit for its byte authentication and zero known-history exact overlap. No filtered production candidate has been physically accepted.

`scripts.prepare_enriched_parent_export` creates the actual parent import prerequisite from independently pinned public artifacts and original committed JSON, streaming Safetensors leaf authentication without model execution. Its finite near-cloud template is in the [parent identity report](audit-2026-09-30/production-parent-public-identity-2026-10-01.md). The first actual materialization authenticated public weights but failed enrichment. Its stable-field comparison is fixed locally; corrected materialization and physical TPU import remain gates. CI routing identifies their metadata checks; physical embedding tests remain manual and cannot be counted as local model acceptance.

## Current implementation follow-up: lifecycle, telemetry and confidence

The [data-job lifecycle](BOUNDED_DATA_JOBS.md) promotes source/ownership/deadline/terminal-status checks to versioned tooling. Explicit session authorization plus an actual GCS read is necessary after the observed Cloud Shell credential-proxy failure; account ACTIVE alone did not establish access. It does not provision TPUs or qualify model behavior.

[Global telemetry](EMBEDDING_TELEMETRY.md) now reports actual all-host processed/useful exposures and invocation timing across evaluation/checkpoints. It requires physical TPU cadence/performance qualification and changes source identity; never bypass exact-resume source guards. [Query-level confidence reports](EMBEDDING_UNCERTAINTY.md) are reporting-only and require retained authenticated per-query results. They do not change quality selection or convert selected-positive probes into full-corpus benchmarks.

## Current transfer and audit decision

The [bounded transfer diagnostic](ARTIFACT_TRANSFER_DIAGNOSTIC.md) is implemented
and literal-byte tested. The actual parent06 GCS copy timed out before enrichment;
no current enriched parent or physical import is established. Measure the executor
transfer before selecting a changed bounded download strategy. Keep full original
SHA/lineage and leaf authentication. The [current issue inventory](audit-2026-09-30/issue-acceptance-2026-10-01-current.md)
was refreshed at21:46UTC; all29 issues remain open.

[Full-corpus reporting](FULL_CORPUS_RETRIEVAL.md) now provides authenticated native
retrieval metrics and a staged TPU block scorer. Do not relabel the selected-positive
probe or an unqualified top-k receipt as an official full-corpus benchmark. Physical
encoder integration, scorer parity and complete corpus traversal are still needed.
Inserting host RAM/scratch preflight remains the reviewed workload owner's duty;
the generic supervisor does not insert it automatically.

## Complete heldout inputs for representation-v2 replay

The portable development archive contains historical training rows and manifests,
not every historical development file. Use the generation-pinned six-source
heldout supplement from [the repaired preparation plan](representation-v2-2026-10-05/training-data-plan.md),
passing its exact `--heldout-sha256`. The old three-pair supplement alone fails.
The preparer verifies all six original dev files before scanning training rows.
Persist completed prepared data before stage preflight; a later failure must not
force another full preparation. Retain full worker/controller diagnostics before
cleanup and distinguish a running controller from executed model steps.

The corrected complete production preparation and shared stage preflight passed
on 2026-10-06 UTC: 2,165,175 rows, eleven sources, zero declared exact heldout
overlaps, and 48 Linux metadata/lifecycle checks. The data VM and auto-delete
boot disk were removed after retaining verified artifacts.
[The launch receipt](representation-v2-2026-10-05/repair-training-launch.json)
records the single replacement training controller, immutable source/data and
provider deadline. Its physical TPU qualification and training state must be
read from the live campaign; the data pass alone does not establish model execution.
