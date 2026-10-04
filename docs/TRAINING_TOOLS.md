# Tools for continued representation training

Latest reconciled audit: [all open issues, current evidence and tool priorities](SYSTEM_AUDIT_CURRENT_2026-10-01.md). Earlier dated findings below remain historical where superseded.


Subsequent [implemented safeguards, verification and guarded diagnostic outcome](audit-2026-09-30/implementation-followup-2026-10-01.md) supersede the earlier implementation gaps while retaining their physical acceptance requirements.

Latest requested parallel review: [audit, new gaps, tools and validation scope](PARALLEL_SYSTEM_AUDIT_2026-10-01.md). All 29 open issues were refreshed; historical counts below remain dated evidence.

Updated October 1, 2026. This inventory supports the
[training runbook](REPRESENTATION_TRAINING_RUNBOOK.md) and
[system audit](SYSTEM_AUDIT_2026-09-30.md). It distinguishes available components
from integrations that still need implementation and physical TPU validation.
No cloud resources, billing exports, or additional services are provisioned by
this document.

Current status is in the [follow-up audit](SYSTEM_AUDIT_2026-10-01.md).
Preflight, relevance masking, language replay, exposure tracking, development
selection, gradient caching and stricter receipts now have implementations.
Their selected model-free checks pass, but they remain physically unqualified; the original audit's proposed
integration list below describes priorities rather than proof of absence.

## Use and consolidate what exists

| Purpose | Available tool or component | Work needed for the embedding stage |
|---|---|---|
| Model training | JAX, Flax NNX, Optax; `scripts/train_yat_embedding_finetune.py` | Implemented continuation/data/quality/cache contracts; failed-gate persistence now implemented; physical acceptance remains |
| Prepared data | Hugging Face Datasets, Tokenizers, NumPy mmap arrays; preparation scripts | Identity/relevance/provenance/exposure implemented; full array hashes/raw flag checks implemented; real-data and scalable contamination screening remain |
| Durable recovery | Orbax; `flaxchat/checkpoint.py` | Broader source/runtime identity, authenticated parent lineage, interruption and cross-topology embedding tests |
| Cloud execution | `gcloud`, guarded supervisor/runner/Workflows; `scripts/representation_run.py` manifest builder and launch adapter | Fresh-VM install, physical controller-loss/topology/peer failure qualification; [deployment guide](INFRASTRUCTURE_RUNS.md) |
| Runtime reproducibility | Existing accepted lock; `scripts/build_runtime_bundle.py`, hashed source/input/overlay/wheelhouse, offline exact installation and runtime receipt | Capture and physically qualify separate current embedding train/eval and PyTorch/XLA Linux bundles; old validation freeze does not qualify new runtimes |
| TPU performance | `flaxchat/profiling.py`, JAX profiler, trace summarizer | Scheduled profiling/metrics integrated; collect warmed physical traces and measure compile, input, compute, collectives and checkpoints |
| Quality evaluation | Pinned MTEB adapter, retrieval evaluators, saved receipts | Finite scores, immutable inventories and development gates implemented; physical metric qualification and matched EmbeddingGemma adapter pending |
| Export and publication | Safetensors, Hugging Face Hub, PyTorch/XLA port | Artifact/zero-vector guards implemented; full provenance/case scheduling implemented; execute physical matrix and verify remote inventory |
| Cost and retention | Shared campaign ledger option, whole-slice price evidence; `scripts/cloud_accounting.py` read-only billing and live/version/soft-delete inventory plus protected cleanup plans | Existing export permission/schema and posted-cost reconciliation; scoped inventory reads already exist, stable cleanup verification remains; remaining promotional balance still needs provider evidence; no automatic destructive cleanup |
| Issue tracking | Connected GitHub tools; local `gh` when authenticated | Issues with evidence and acceptance criteria; no automatic training or expensive CI dispatch from issue creation |

Earlier local `gh` authentication failed; the latest refresh succeeded once and
then hit transport resets. The GitHub connector independently returned all 29 open issues. This is an operator authentication
observation, not a defect in the training model or a reason to replace credentials
automatically.

## Add integrations in this order

1. **Strict input and continuation preflight.** Extend existing validators to the
   embedding schema and parent checkpoint. Check shapes, vocabulary, padding,
   source/runtime identities and cursor consistency before an expensive run.
2. **Representation development gate.** Evaluate pinned multilingual bitext,
   retrieval, STS and code development sets. Preserve best checkpoints and final
   test isolation. Record per-language regressions instead of selecting solely
   by an overall mean or training loss.
3. **Shared data registry and negative-quality audit.** Store source revision,
   language, task, license, split, IDs and relevant-document relations. Add replay
   and same-source batches without losing exact data-cursor replay.
4. **Profile-driven batch scaling.** Measure replicated training first. Compare
   FSDP only when it helps the measured memory/throughput tradeoff; add gradient
   caching when a larger contrastive pool cannot fit with full activations.
   Ordinary gradient accumulation combines updates but does not create the same
   cross-microbatch negative pool. See the
   [gradient-cache paper](https://aclanthology.org/2021.repl4nlp-1.31/).
5. **One campaign owner and release validator.** Consolidate setup, train, evaluate,
   export and publish stages while keeping their environments separate. Each stage
   gets an immutable manifest, terminal status and authenticated artifact inventory.

Inspect the current parsers: several integrations above now have CLI flags and
metadata tests, while their physical acceptance remains outstanding. Teacher distillation,
Matryoshka objectives and long-document curricula belong after the correctness
and quality gates; choose them through measured development improvements.

## Optional external tools

- **XProf / TensorBoard profile plugin:** inspect TPU execution, communication and
  memory. Capture short post-compilation windows and keep traces with the run.
  Use an existing local viewer first; creating a viewer VM is a separate resource
  decision. [Google TPU profiling](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm)
  and [JAX profiling](https://docs.jax.dev/en/latest/profiling.html).
- **Cloud Billing export to BigQuery:** reconcile posted usage, credits and
  adjustments by project/time/SKU, with run attribution where supported. Billing
  data can lag and exporting now does not guarantee all historical jobs are
  recoverable. Check existing export availability before enabling a new service.
  [Billing export documentation](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery).
- **W&B or TensorBoard metrics:** optional visualization of quality, alpha values,
  gradients, data exposure and throughput. Durable JSON/GCS evidence remains the
  source of truth; an experiment dashboard is not required to launch.
- **A disk-backed deduplication and retrieval index:** select an appropriate index
  when corpus scale exceeds the current Python sets. Evaluate memory, near-duplicate
  quality and negative recall before adding a specific backend. Keep preprocessing
  separate from the TPU model process.

No paid experiment platform, cluster orchestrator, or new model framework is a
prerequisite for this continuation. The largest gaps are in integrating and
qualifying components already present.

## Tool admission evidence

Before a new run, record the active GCP account/project, resource identifiers,
current pricing basis, observed credit balance and observation time. Old account
preferences and model paths guide discovery but do not prove current billing,
resource absence, capacity or ownership. A request that queues successfully is not
evidence that a region has available capacity.

Freeze dependency versions from an accepted runtime separately for JAX and
PyTorch/XLA, and capture source/data/checkpoint hashes. For an integration change,
require physical TPU output/gradient/recovery evidence appropriate to that change.
Static checks and data-only tooling can run without TPUs. Do not run unrelated CPU
model loops as a substitute for the user's TPU target.

## Immediate tool decisions after the fresh review

Use the existing 8-device physical test runner for objective/cache/recovery qualification; use `scripts.validate_embedding_multihost` for embedding-specific physical multi-host checks after admission. Its four-step resume fixture does not qualify sustained training or abrupt worker loss. Use `scripts.run_yat_parity_campaign` and its bounded `scripts.run_yat_parity_case` workers to resume verified missing cases; aggregate physical acceptance remains unrun.

Shared preflight now uses the stdlib-only committed checkpoint reader. Raw-data/array-inventory integrity and failed-gate persistence have passing model-free checks. Actual tensor restoration and training qualification remain physical TPU work. The generic validation lock lacks Safetensors; the workload-specific `infra/tpu/embedding-qualification-lock.txt` supplies it and its frozen bundle installed successfully on v5e. Current model and production configuration acceptance remain separate.

Use JAX profiling/XProf after correctness passes; use an available Billing export for posted cost reconciliation. W&B, a cloud-global reservation service, a deduplication index, and teacher generation remain optional and conditional on actual scale needs. The reviewed MIRACL/publishing wrappers now delegate to frozen manifests. Any remaining historical floating setup is not an accepted runtime recipe. No new tool installation or service enablement was performed by this audit.

## October 1 concrete follow-up integrations

The [parent exposure inventory builder](PARENT_EXPOSURE_INVENTORY_BUILDER.md) now matches retained stage metadata to actual prepared manifests and bounds historical raw authentication. The [terminal campaign receipt](audit-2026-09-30/supervisor-terminal-campaign-receipt.md) retains capacity/setup/workload failure phase and cleanup observations even before worker readiness. Both have model-free checks; actual historical raw scanning and physical/controller failure qualification remain separate.

The subsequent consolidation adds authenticated development exclusions to training preparation, independent retrieval/bitext/code tokenization and `--retrieval-dev NAME=DIR` admission/evaluation. Parent-exposure proofs are recomputed against explicitly declared known stages; complete historical cleanliness remains unresolved. MIRACL upstream alignment groups are shared across languages for new preparation. Peer recovery now requires durable worker terminal evidence, and attachment stamps publish atomically. See the [follow-up receipt](audit-2026-09-30/consolidated-followup-2026-10-01.md) for verification and the capacity-limited diagnostic outcome.

`infra/tpu/embedding-qualification-lock.txt` supplements the older generic lock
with pinned Safetensors for the embedding export/trainer fixtures. The Linux
wheel registry hash, stable-ABI compatibility and effective dependencies were
checked without local model execution. On October 1, the corrected frozen
runtime installed offline on a physical 8-chip v5e TPU, passed pip dependency
checks and the required-import/device setup checks. The model workload is a
separate acceptance gate. Its headless interpreter transformation preserves
all retained files and excludes only the optional terminal database; upstream
and derived hashes are retained. See the [qualification log](audit-2026-09-30/physical-qualification-2026-10-01.md).

Programming-language provenance and exact exposure reporting are implemented
([guide](CODE_LANGUAGE_PROVENANCE.md)); absent upstream labels remain unknown.
The new bounded SQLite [overlap auditor](EMBEDDING_BENCHMARK_OVERLAP.md) uses
the training identity policy and explicit pinned benchmark inventory. Actual
production files must be supplied before any overlap conclusion.

The [pinned candidate development recipe](REPRESENTATION_DEVELOPMENT_RECIPE.md)
and bounded streaming exporter verify source metadata and keep final benchmark
splits out of tuning. Actual preparation, parent-exposure checks, cross-source
quarantine and physical acceptance of the implemented independent development integration remain required.

The [peer-loss campaign](audit-2026-09-30/embedding-peer-loss-campaign.md) is
an isolated adapter for an already leased physical multi-host slice. It must
prove durable commit, all original remote workers drained, and exact resumed
state; metadata checks are not physical fault acceptance. Controller and
transport loss remain separate unqualified requirements.

Current posted-cost discovery (October 1): the `azettaai` project has billing enabled on account `012CAE-712976-475489`. Its read-only BigQuery dataset list returned no datasets and no next page. [Receipt](audit-2026-09-30/billing-export-discovery-2026-10-01.json). This does not inventory other possible export projects or establish zero spend; no query ran and no service/export was enabled.

## Implemented after the current audit

- [Bounded data-job controller](BOUNDED_DATA_JOBS.md): source-pinned detached supervisor, once-only ownership and immutable terminal execution/cleanup receipts; observation never starts another worker. Linux integration qualification remains separate from local tests and model acceptance.
- [Global embedding telemetry](EMBEDDING_TELEMETRY.md): exact useful/processed exposure across hosts, slowest-host timing and invocation overhead; physical collective cadence and overhead remain unqualified.
- [Query uncertainty reporting](EMBEDDING_UNCERTAINTY.md): bounded deterministic query-mean and paired-delta bootstrap on authenticated scalar score inputs; no model or checkpoint-selection changes. Full-corpus evaluation and competitor adapters remain pending.

## Current transfer and audit decision

The [bounded transfer diagnostic](ARTIFACT_TRANSFER_DIAGNOSTIC.md) is implemented
and literal-byte tested. The actual parent06 GCS copy timed out before enrichment;
no current enriched parent or physical import is established. Measure the executor
transfer before selecting a changed bounded download strategy. Keep full original
SHA/lineage and leaf authentication. The [current issue inventory](audit-2026-09-30/issue-acceptance-2026-10-01-current.md)
was refreshed at21:46UTC; all29 issues remain open.

## Actionable tool matrix after parallel review

| Tool | Current implementation | Missing acceptance / tracker |
|---|---|---|
| `bounded_data_job` | Versioned Linux controller; four real lifecycle scenarios passed | External pin enforced/tested; changed-source Linux and whole-controller/provider loss remain. #51/#55 |
| `diagnose_artifact_transfer` | Bounded stream progress and literal-byte tests | Actual executor SDK/version plus range measurement; full delivery/enrichment still pending. #40/#52/#55 |
| Global exposure telemetry | Integrated all-host useful/processed counters and invocation overhead | Physical one/multi-host cadence, correctness and overhead. #49/#54 |
| Query bootstrap | Authenticated paired/mean uncertainty reporting | Actual scored query inputs; aligned translations need grouped resampling. #47/#58 |
| Full-corpus retrieval reporter | Authenticated native metrics and reporting-only uncertainty | Actual complete snapshots and results; top-k alone cannot prove search coverage. #47/#57/#58 |
| Bounded TPU top-k scorer | Staged single-host block scorer retaining query×k | Physical encoder integration, block/tie/partial parity and full corpus traversal. #57/#59 |
| Named cache diagnostics | Reporting-only embedding/cotangent/moment boundaries | Execute original five mandatory TPU nodes without relaxing bounds. #49/#50 |
| Matched competitor adapter | Receipt comparison exists; actual faithful adapter still missing | Pinned mmBERT/EmbeddingGemma tokenizer/prompts/pooling and measured results. #58/#61 |
| Posted-cost reconciliation | Read-only provider/export adapter exists | Authorized usable export or authenticated rows, billing windows and run attribution. #54 |

Guides: [retrieval metrics and TPU integration](FULL_CORPUS_RETRIEVAL.md),
[transfer diagnosis](ARTIFACT_TRANSFER_DIAGNOSTIC.md),
[query uncertainty](EMBEDDING_UNCERTAINTY.md),
[global telemetry](EMBEDDING_TELEMETRY.md).
Use existing JAX profiling/XProf before custom kernels. No new paid dashboard or
cluster framework is required. The [official JAX profiler](https://docs.jax.dev/en/latest/profiling.html),
[TPU profiling guide](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm) and
[Billing export guide](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery)
were refreshed for this review. Export availability/timing is not historical
per-model charge attribution.

The observer fix passed61 focused metadata tests with one explicitly Linux-only
macOS skip. Four earlier Linux lifecycle passes remain scoped to the old source
freeze; fresh changed-source Linux acceptance is pending. No TPU model or cloud
execution occurred in this review.

Actual continuation: the changed observer/controller source has now passed its
[Linux lifecycle and external-pin acceptance](audit-2026-09-30/bounded-data-job-linux-selftest-02/selftest.json),
with four separate terminal cleanup observations. Earlier statements that fresh
changed-source Linux acceptance was pending are superseded for this exact source;
whole-controller/provider/TPU faults remain separate. The measured-transfer worker
started under a finite timeout; its result is not yet established.
