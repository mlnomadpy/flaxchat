# Parallel system audit and continuation plan

Latest reconciled audit: [all open issues, current evidence and tool priorities](SYSTEM_AUDIT_CURRENT_2026-10-01.md). Earlier dated findings below remain historical where superseded.


Subsequent [implemented safeguards, verification and guarded diagnostic outcome](audit-2026-09-30/implementation-followup-2026-10-01.md) supersede the earlier implementation gaps while retaining their physical acceptance requirements.

October 1, 2026. Three independent agents reviewed training/data, infrastructure,
and evaluation/release; the coordinator refreshed GitHub and reconciled source
evidence, retained physical results, actual data receipts and documentation.
The goal remains improving the existing YAT multilingual/code embedding weights.
This is a source and evidence audit, not proof that every repository path or
every cloud region was exercised.

**Decision:** finish the bounded correctness and independent quality gates before
substantial continuation. The harness has many implemented safeguards, but two
physical cache optimizer-state checks remain failed. Sustained multi-host,
production parent restoration, representative development coverage, current
release acceptance and posted billing remain open.

## Every open issue

The live GitHub connector returned **29 open issues**. The
[original bodies and URLs](audit-2026-09-30/issue-refresh-request-2026-10-01.json)
and [complete acceptance checklist](ISSUE_READINESS_CHECKLIST.md) cover all of
them. A successful CLI read was followed by transport resets; the connector
provided the saved refresh. Existing #39–#60 cover the new observations; #61
coordinates continuation. No duplicate issue is needed, and no issue was closed.
Decoder release/demo and cross-framework benchmarks (#1/#11/#19/#23) remain
separate acceptance work; Kaggle #31 remains deferred for this GCP priority.

## Findings and work order

| Priority / stage | Gap | Required outcome | Existing issues |
|---|---|---|---|
| P1, correctness | Two cache Adam-state comparisons failed | Execute named-leaf full-batch, shape-matched ordinary autodiff and cache diagnostic; explain/fix discrepancy under original bounds | #49, #50 |
| P1, production quality | Independent retrieval/STS lists may be empty at generic admission | Enforce declared production task coverage while retaining explicit narrow infrastructure qualification | #47, #61 |
| P1, continuation | Parent metadata is authenticated; actual current parent tensors are not restored on the new accepted path | Restore the retained parent on TPU with exact tokenizer/config/lineage and explicit optimizer policy | #40, #41, #42 |
| P1, data | Historical raw integrity is verified, but actual candidate exposure/full lineage is not | Prepare authentic candidate rows, recompute exposure and future-source quarantine; disclose inherited MLM/semantic uncertainty | #44, #47 |
| P2, quality | Programming-language labels lack regression slices; tiny language minima are only validity checks | Measure practical per-task/language coverage and code-language retention without double-counting macro selection | #47, #48, #57 |
| P1 before publication | Publisher aggregate parity contract is weaker than durable campaign verification | Require complete rows/shapes/raw output binding, then execute current physical matrix and immutable remote verification | #53, #60 |
| P2 before long-context parity | Full intermediates and comparison temporaries lack a memory/evidence budget | Bound host RSS/scratch/evidence bytes or stream comparisons at the largest advertised context | #49, #60 |
| P1 before scaling | Sustained physical multi-host and abrupt faults lack receipts | Qualify exact topology, disjoint batches, global negatives, durable recovery and owned cleanup | #12, #50, #51, #55 |
| P2, measurement | Actual posted costs and retained storage are not reconciled | Separate posted usage, credits, reservations, live/version/soft-delete storage and observed inventory scope | #54, #56 |

The detailed independent reviews contain file evidence and qualification limits:
[training/data](TRAINING_DATA_AUDIT_2026-10-01.md),
[infrastructure](INFRA_AUDIT_2026-10-01.md), and
[evaluation/release](EVALUATION_AUDIT_2026-10-01.md).
No finding establishes corruption of the historical public model.

## Tools required

Keep JAX/Flax NNX/Optax, Orbax, pinned Hugging Face data/tokenizers, Safetensors,
prepared mmap arrays, physical pytest/JUnit, immutable runtime bundles, GCS,
the guarded supervisor and its independent lease. The embedding lock includes
Safetensors and installed on v5e; a separate Torch/XLA environment is required.

Use short warmed [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)
and [XProf TPU traces](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm)
after correctness passes. Measure input, compute, collectives, checkpoint time,
HBM, host RSS and scratch before deciding on FSDP or custom YAT kernels.
Use an authorized existing [Billing export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery)
for posted cost attribution. These are measurement integrations, not a need for
a new training framework, Kubernetes or a paid dashboard.

The GCS raw scanner and parent inventory builder need the verified historical
producer policy passed through both tools. Cloud Shell is optional for data-only
work near the bucket; [its compute is free](https://cloud.google.com/shell/pricing),
but ancillary storage/transfer exposure still needs reservation. Disk-backed
deduplication/sparse relevance, teacher distillation, Matryoshka objectives and
long-document curricula are conditional improvements selected through measured
quality or memory bottlenecks.

## Verified evidence

- Selected local verification: **211 model-free tests, 76 subtests passed**.
  This checks metadata/admission/orchestration contracts; no CPU model forward,
  backward or benchmark ran.
- Historical frozen physical v5e-8: **14 passes, two cache optimizer-state
  failures, zero skips**. Its global FP32 environment override does not describe
  every fixture: the cache encoder explicitly uses BF16 compute and FP32
  residuals. Neither all-BF16 production nor newer source is qualified by it.
- The [actual GCS scan](audit-2026-09-30/cloud-parent-scan-01/scan.json) verified
  **2,075,961,764 bytes and 1,694,512 rows** across CodeSearchNet, MIRACL and
  MS MARCO, matching original manifests and generation-pinned raw hashes.
  Runtime/controller receipts are retained. Candidate exclusions were absent:
  overlap is unknown and `admission_ready` is false.
- [Independent cleanup](audit-2026-09-30/cloud-parent-scan-01/cleanup-verification.json)
  found the owned scan processes and scratch absent. This does not assert the
  entire Cloud Shell environment is deleted or every cloud resource is absent.
  No new TPU or GCE VM was allocated for this audit.
- The [campaign ledger](audit-2026-09-30/cloud-parent-scan-01/campaign-ledger.json)
  retains prior reservations and terminal scan state. Reservations are not
  posted charges; actual billing remains unknown.

## Skill and docs delivery

The [training skill](../skills/flaxchat-training/SKILL.md) now covers production
quality coverage, programming-language retention, effective precision scope,
raw parity/memory evidence and policy propagation. The runbook, issue checklist,
tool inventory and development recipe point to these current distinctions.
Existing fixed YAT bias 1, epsilon 0.01, trainable alpha, continuation lineage,
TPU-only model qualification and paused expensive workflows remain in force.

Next concrete gate: diagnose the failed cache optimizer comparisons on a frozen
physical TPU scope. Prepare candidate development/exposure in parallel without
using final benchmarks for tuning. Release defects gate publication; multi-host
requirements gate scaling. Neither requires blocking a narrow single-host
correctness diagnostic. Keep implementation, physical acceptance and quality
improvement as separate states. Changes remain local until committed/pushed.

Independent skill forward-testing confirmed the production coverage and scope distinctions. A separately frozen direct-training path can be qualified with cache disabled; this leaves cache acceptance failed and does not waive production quality or topology gates.

## Latest bounded follow-up

The [native MIRACL candidate scan](audit-2026-09-30/native-miracl-data-01/README.md)
finished with three exact-text collision rows in the broader exclusion inventory;
independence admission remains blocked. Original parent JSON and the public
step-14000 weight identity are now authenticated separately; actual TPU tensor
import remains unrun. New full-manifest binding, bounded collision diagnostics
and separate capacity/startup budgets passed **67 model-free tests and eight
subtests**, plus Ruff and diff checks. The repository and installed training skill
are synchronized and validate. These follow-ups do not close physical issues.

The [collision/parent/comparison follow-up](audit-2026-09-30/collision-parent-comparison-followup-2026-10-01.md) adds executable prerequisites and 97 passing model-free tests with 15 subtests. Actual data filtering, parent materialization, TPU diagnosis and matched evaluation remain unexecuted; source hashes are retained separately from prior physical evidence.
