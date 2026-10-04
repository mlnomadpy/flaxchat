# System audit: representation training readiness

Latest requested parallel review: [audit, new gaps, tools and validation scope](PARALLEL_SYSTEM_AUDIT_2026-10-01.md). All 29 open issues were refreshed; historical counts below remain dated evidence.

Reconciled October 1, 2026 after three independent parallel reviews of the current working tree. This is the current status; the [September 30 audit](SYSTEM_AUDIT_2026-09-30.md), [earlier follow-up](audit-2026-09-30/followup-before-reconciliation-2026-10-01.md), and source-review addenda preserve historical findings. The intended product is improved multilingual and code embeddings continued from existing YAT weights, with fixed bias 1, epsilon 0.01 and trainable alpha.

**Subsequent physical follow-up:** the frozen embedding runtime installed on an
8-chip v5e TPU. Selected physical tests returned **14 passed, 2 failed, zero
skipped**. Objective and trainer recovery cases passed; both encoder-cache
optimizer-state comparisons failed. Both attempts' owned TPU VMs and queues
were independently verified deleted. The [follow-up log and raw evidence](audit-2026-09-30/physical-qualification-2026-10-01.md)
supersede the earlier unrun status for these selected cases only. Production
parent restoration, actual data quality, sustained multi-host recovery,
v6e/scales and release acceptance remain open.

## Decision and next gate

The [next-gates progress receipt](audit-2026-09-30/next-gates-progress-2026-10-01.md) adds physical independent-gate definitions, a bounded parent inventory builder, terminal campaign receipts and actual matching historical GCS manifests. Selected model-free verification now totals 171 tests and 68 subtests. Raw production rows and the new physical nodes remain unqualified.

**Later consolidation:** all 29 open issues were refreshed again. Independent development preparation, recomputed parent-exposure evidence, future-source quarantine, cross-language MIRACL group identity and stricter fault ownership/status checks are now implemented. The selected model-free suite passed **162 tests and 62 subtests**. A separate one-attempt cache diagnostic exhausted capacity before setup; its owned queue and node were independently verified absent. It does not change the 14-pass/two-failure physical result above. The [consolidated follow-up](audit-2026-09-30/consolidated-followup-2026-10-01.md) records scope and remaining gates; historical counts and inventories below remain dated snapshots.

The system has substantially improved, but a substantial training continuation is not yet qualified. **The five reviewed P1 defects now have local implementations and passing model-free checks:** failed-quality-gate persistence, complete required-array checksums, raw negative-valid agreement, nonmutating committed-checkpoint admission, and complete parity runtime identity. This does not close their physical acceptance requirements. Freeze a complete source/runtime/data revision, then run one bounded physical single-host qualification. Multi-host and release acceptance apply before their respective stages, not as prerequisites for a training-only correctness run.

GitHub was refreshed during this review: **29 open issues**, including all 22 original audit findings, six older issues and [tracker #61](https://github.com/mlnomadpy/flaxchat/issues/61). New observations extend existing issues; they do not justify duplicate issues or premature closure. The [full issue checklist](ISSUE_READINESS_CHECKLIST.md) includes every open issue and its remaining gate; the [raw inventory](audit-2026-09-30/issue-refresh-final-2026-10-01.json) preserves issue bodies and links.

## Reviewed findings, implemented fixes and remaining acceptance

| Priority | Current finding | Existing issues | Implemented change; remaining acceptance |
|---|---|---|---|
| P1 | Failed quality evaluation is omitted from saved state; exact resume checks baseline against itself and can bypass the failed gate | #47, #50 | Implemented authenticated last evaluation/decision and v2 gate policy. Physical intermediate/horizon failed-gate restart tests remain unrun |
| P1 | Only manifest-listed checksums are verified; a required loaded array may lack a checksum | #42 | Implemented complete canonical train/dev inventory and omission/mutation rejection. Actual frozen recipe and physical trainer acceptance remain |
| P1 | Authenticated raw negative presence is not compared with `negative_valid` | #39, #42 | Implemented row agreement and pair/triplet flip rejection before allocation. Physical objective/gradient evidence remains |
| P1 | Optional parent-checkpoint preflight constructs a writable cleanup-enabled Orbax manager | #40, #42, #52 | Implemented stdlib-only committed JSON reader with integrity/stability checks and model-free fixtures. Retained step-14,000 metadata passed; tensor restoration still needs TPU |
| P1 | Parity computes full provenance but retains only packages/source hashes | #53, #58, #60 | Implemented v3 complete backend identity and guarded receipt checks. Current physical matrix and publication acceptance remain |
| P2 | `recall_at_k` is actually first-relevant hit rate for multi-positive queries | #47 | Implemented true fractional recall, versioned v2 metric policy and multi-positive metadata fixtures. Physical metrics remain |
| P2 | Historical MIRACL and upload wrappers still have floating installs/source paths | #52, #55 | Implemented equivalent manifest routes for reviewed MIRACL/publishing wrappers. Real execution/release acceptance remains |
| P2 | Resources lack the run labels expected by billing attribution | #54 | Implemented normalized run/stage labels and provider read-back enforcement. Allocated-resource labels passed physical provider read-back; posted export reconciliation remains |
| P2 | A signal after child exit during final upload can prevent the status receipt | #55 | Implemented status-before-upload, absent-process handling and passing signal-window fixture. Physical controller-loss evidence remains |
| P2 | Full parity matrix remains one monolithic test despite a bounded case worker | #60 | Implemented bounded campaign with current-identity/raw-hash verification and missing-case scheduling. Physical coverage remains |
| P2 | Generic parity still assumes layer 10 exists | #53, #60 | Implemented depth-derived first/middle intermediate schema. Current physical model-depth cases remain |
| Conditional | Campaign cap coordinates one shared filesystem, not independent machines | #54 | One campaign owner now; cloud reservations only if separate controllers are introduced |

Detailed exact line evidence, effects and acceptance are in the independent [training/data review](audit-2026-09-30/training-review-2026-10-01.md), [infrastructure review](audit-2026-09-30/infrastructure-review-2026-10-01.md), and [evaluation/release review](audit-2026-09-30/evaluation-review-2026-10-01.md). The primary reviewer checked the cited current code paths. No historical corruption of public weights is established by these findings.

## What is implemented and what is still unvalidated

Shared resolved-stage admission now covers sampler, STS, language and cache configuration. Stage-global evaluation cadence, best/recent commit reconciliation and actual optimizer-counter checks are implemented. Recognized trainer output paths are bound to the manifest; setup execution is separate from physical acceptance; installed runtime inventory and bounded capacity waits are checked. Evaluation revisions/inventories, finite scores, zero-vector parity guards and generic stage export identity are implemented. These earlier defects should not remain listed as absent fixes.

Dedicated physical tests now define hard-negative reference loss/gradients, full NNX cache gradients/optimizer updates, direct/cache pair/triplet recovery, off-cadence stops and commit-window faults. The [embedding-specific multi-host worker](EMBEDDING_MULTIHOST_ADMISSION.md) checks physical rank/device ownership, global negative gradients and four-update GCS exact resume. These are test definitions, **not executed qualification**. Sustained physical multi-host training, abrupt host/transport loss, v6e and all-scale measurements remain unproven for this revision.

The development tools implement deterministic language probes and pinned STS, but an actual representative retrieval/bitext/STS/code development mixture still needs preparation and qualification. Programming-language provenance is now preserved separately, with explicit unknowns when the pinned CodeSearchNet release has no labels; global preparation retains large Python identity/relevance structures; replay with replacement must be reported as exposure, not epochs. A disk-backed identity index/CSR representation becomes useful when measured host memory demands it.

Historical card scores remain English MTEB 52.78, multilingual eight-category 48.21, CoIR 43.45 and MIRACL 46.37. This review did not rerun them. Tatoeba declined from 52.72 to 35.27, so continuation must measure retention. mmBERT paper columns and EmbeddingGemma comparisons are not matched local measurements. CodeSearchNet/CoIR overlap is unresolved. Actual pinned multi-subset/multi-split MTEB output must qualify the strengthened receipt schema. Historical eight-input, 128-token PyTorch parity is narrower than the planned physical matrix and does not confer separate PyTorch benchmark scores.

## Verification and operational scope

Final reconciliation passed **124 selected model-free tests and 48 subtests**, including standalone stage/contract/checkpoint-reader tests. No model forwards/backwards or CPU model benchmarks ran. Documentation and skill validation results are recorded in the [final audit receipt](audit-2026-09-30/audit-delivery-2026-10-01.json). These selected tests do not establish repository-wide coverage or physical acceptance.

Read-only provider inventory around 10:56 UTC found no GCE instances project-wide in `azettaai`, and no queued resources or TPU VMs in `us-west4-a`. This is regional TPU evidence, not proof about every region, storage, or final posted cost. This audit allocated no compute, deleted no storage, enabled no service and dispatched no GitHub workflow. Previous builder installation and retained storage receipts are historical; current balance/spend was not refreshed here.

## Prioritized next work

1. The frozen embedding qualification runtime now passes physical setup using its Safetensors lock and headless interpreter. Diagnose the two cache optimizer-state failures before repeating the physical acceptance gate; keep a separate Torch/XLA environment.
2. Run physical single-host objective, full encoder-cache gradient/update and trainer recovery acceptance, including failed quality gates and off-cadence stops.
3. Prepare actual pinned retrieval/bitext/STS/code development data, preserve programming-language provenance and audit CodeSearchNet/CoIR overlap. Establish parent baselines and regression thresholds before substantial continuation.
4. Qualify sustained physical multi-host training and abrupt host/controller/transport loss; then measure warmed throughput, memory and cost across intended scales. v6e and all scales remain unvalidated.
5. Execute current physical parity and real MTEB receipt/merge/resume acceptance; run matched mmBERT/EmbeddingGemma measurements before comparison claims. Verify new publication at an immutable Hub revision.
6. Reconcile posted billing and explicit storage retention. A remaining credit balance, reserved exposure and actual charges are separate measures.

## Tools and upgrades

Keep JAX/Flax NNX/Optax, Orbax, prepared mmap data, Hugging Face datasets/tokenizers, Safetensors, MTEB, GCS, pytest/JUnit and the existing guarded supervisor/Workflows lease. Installed command presence was checked for `gcloud`, `bq`, `gh`, `git`, `uv` and Python; presence does not prove permissions. GitHub connector reads work; writes return 403, so the authorized Chrome session supplies issue updates without changing credentials.

Use [JAX profiling](https://docs.jax.dev/en/latest/profiling.html) and [XProf on TPU](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm) for short warmed physical traces after correctness passes. Use an available [Cloud Billing export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) for actual posted cost; estimates, applied credits and remaining balance are different quantities. No replacement framework, Kubernetes deployment or paid dashboard is required. W&B, cloud-global budget reservations and a preprocessing index are conditional tools, not prerequisites.

The [GitHub tracker reconciliation](https://github.com/mlnomadpy/flaxchat/issues/61#issuecomment-5930414571) was published and verified by connector read-back. The [training skill](../skills/flaxchat-training/SKILL.md), [runbook](REPRESENTATION_TRAINING_RUNBOOK.md), [tool inventory](TRAINING_TOOLS.md) and issue checklist are updated to these current distinctions. The skill preserves existing user authorization, continuation lineage and TPU-only model qualification, and adds failed-gate persistence, complete array authentication, raw flag agreement and complete parity provenance. Documentation/source changes remain local until separately committed and pushed.
