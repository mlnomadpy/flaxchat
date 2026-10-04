# Infrastructure audit for continued YAT representation training

Independent read-only review of current source and preserved receipts on October 1,
2026. This review allocated no cloud resources, changed no cloud configuration,
deleted no storage, and executed no model on a CPU. It covers infrastructure,
checkpoint admission, deployment identity, budgeting and cleanup. Training/data
and evaluation/release reviews supply their own acceptance evidence.

**Decision:** the main infrastructure safeguards have implementations, but a large
production continuation is not yet qualified. The immediate requirement is to
finish the targeted physical correctness and parent-restoration gates, rather than
install a different orchestration framework. Open issues should retain their
implemented-versus-qualified distinction.

## Evidence and priorities

| Priority | Remaining risk or acceptance gap | Current evidence | Required outcome |
| --- | --- | --- | --- |
| P1 | Two cache optimizer-state comparisons still fail | [Physical summary](audit-2026-09-30/physical-qualification-02/summary.json): 14 passed, two failures, no skips; [diagnostic result](audit-2026-09-30/cache-diagnostic-03/result.json): capacity exhausted before model execution | Run the prepared named-leaf/Adam diagnostic on a physical TPU; preserve the original reference tolerances until evidence explains the discrepancy; rerun the affected acceptance nodes after any fix |
| P1 | Actual production parent state has not been restored by the current accepted runtime | [Committed reader](../flaxchat/checkpoint_metadata.py) verifies bounded, stable metadata/identity JSON without JAX or Orbax; its module explicitly excludes tensor-byte authentication | Restore the retained parent tensors and optimizer state on TPU; verify manifest hashes, counters, tokenizer/model identity and intended continuation semantics |
| P1 | Changes after the historical freeze lack equivalent physical receipts | Historical summary binds source SHA `08e966d68222dcf1848765ce517c33f230180cf21fff0523c6209e0452f0ebc7`; current worker rechecks source, interpreter, package inventory and qualification in [representation_run](../scripts/representation_run.py) | Freeze the final current source/runtime/data identity and bind new physical results to that identity; reuse historical results only for their exact scope |
| P1 for scaling | Sustained multi-host training and abrupt host/transport/controller faults remain unqualified | [Multi-host validator](../scripts/validate_embedding_multihost.py) defines rank/device ownership, global gradients and four-step GCS resume, and explicitly excludes abrupt process/transport failures | A sustained physical multi-host campaign with durable checkpoints and selected worker/controller/transport faults; independently verify owned queue and TPU VM absence after cleanup |
| P2 | Posted dollars are not reconciled to run identities | [Accounting adapter](../scripts/cloud_accounting.py) supports parameterized bounded BigQuery queries; [RunLedger](../flaxchat/operations.py) keeps posted usage null until reconciliation | Query an existing authorized export with known schema/interval and scan cap, retain raw rows/freshness and map run/resource labels to reservations; report gross cost, signed credits and net cost separately |
| P2 | Storage listing cannot establish an atomic or current empty bucket | `storage_inventory` performs three sequential live/version/soft-delete reads and preserves their limitation; its `consistent` flag detects only a particular observed-live discrepancy | Scope the inventory, protect parent/best/latest/evidence generations, use stable or bounded repeat observations, and verify retention/soft-delete expiry after any separately authorized cleanup |
| Conditional | A local campaign ledger does not coordinate independent machines | `RunLedger` uses a filesystem lock and atomic local replacement; reservations remain conservatively retained | Keep one campaign owner/shared ledger now. Introduce cloud compare-and-swap reservations only if independent controllers are introduced |

These are acceptance gaps, not evidence of corruption in an already published
model. The capacity-only diagnostic adds no numerical result. Neither a timeout
observed by a local client nor a missing log justifies another allocation before
the original request's state and owned resource cleanup are resolved.

## Safeguards verified in current source

- The paid Spot route reserves whole-slice exposure, verifies an independently
  executing cleanup lease before provisioning, records provider labels, and bounds
  capacity/setup/workload/cleanup operations. [Supervisor](../scripts/gcp_spot_supervisor.py)
  and [guard](../scripts/gcp_cleanup_guard.py) implement these contracts.
- Deployment zone must match the pricing evidence's region; accelerator,
  provisioning model and whole-slice hourly rate must also agree. Switching regions
  cannot silently preserve an unrelated rate in
  [manifest admission](../scripts/representation_run.py).
- Trainer output arguments must match the declared stage checkpoint namespace.
  Setup command success is distinct from physical qualification; workload admission
  requires an explicit qualification receipt contract.
- Source/runtime artifacts have SHA256 bindings. Workers revalidate source, lock,
  interpreter binary/version, installed package inventory and required qualification
  identity. Version inventory does not authenticate every installed package byte
  against manual same-version edits.
- Workload exit status is persisted before final best-effort telemetry upload.
  Supervisor [terminal campaign receipts](audit-2026-09-30/supervisor-terminal-campaign-receipt.md)
  retain phase, original failure and cleanup uncertainty; a SIGKILL cannot guarantee
  a final local write, so the independent guard and provider absence check remain
  necessary.
- Explicit-step checkpoint metadata admission does not construct a writable
  cleanup-enabled Orbax manager. Authenticating recorded model-state hashes is
  distinct from actually restoring and hashing tensors on TPU.
- Free compute may reserve zero hourly compute with a positive ancillary
  reservation. A zero-total reservation remains rejected. This does not make
  associated storage/transfer free or establish posted cost.

## Qualification limits

The historical physical campaign is selected single-host v5e evidence, not a
production-quality, sustained multi-host, v6e or all-scale qualification. Its
environment overrides include `FLAXCHAT_DTYPE=float32` and highest matmul precision,
but the cache encoder fixture explicitly sets BF16 compute and FP32 residuals in
[the physical test](../tests/test_embedding_gradient_cache_physical_tpu.py).
Therefore neither “all FP32” nor “all BF16 production is qualified” describes that
campaign correctly. Validate the actual intended model precision, batch, sequence
length, shapes and optimizer configuration before throughput or scale claims.

The preserved diagnostic receipt independently reports queue and node
`NOT_FOUND` for its exact project, zone and resource name. This is owned-resource
absence evidence at its observation time, not a current project-wide, every-region,
storage or billing inventory. A data-only Cloud Shell scan has a separate owned
job/scratch cleanup scope; its completion must not be described as deleting the
entire Cloud Shell environment or as physical model qualification. See the
[scan admission](audit-2026-09-30/cloud-parent-scan-01/admission.json) and root-owned
terminal receipts for that job's current result.

## Tools required and tools to defer

| Purpose | Use now | Evidence still needed |
| --- | --- | --- |
| Immutable deployment | Existing `gcloud`, manifest builder, hashed source/wheelhouse/interpreter and separate JAX versus Torch/XLA environments | Current frozen revision accepted on the chosen physical TPU |
| Recovery and cleanup | Existing supervisor, Workflows guard, Orbax state manifest and committed metadata reader | Actual parent restore and fault-specific owned resource absence |
| Physical correctness | Existing pytest/JUnit TPU suite and named cache diagnostic | Failing optimizer comparisons resolved, actual intended precision/configuration checked |
| Performance | Existing JAX trace/profiling integration and XProf viewer | Short warmed TPU traces identifying input, compute, collectives and checkpoint stalls; measured cost per useful exposure |
| Actual cost | Existing accounting adapter plus access to an existing BigQuery billing export | Authorized table/schema, scan cap, posted rows, export freshness and resource/run attribution |
| Data integrity near GCS | Existing generation-pinned streaming scanner and authenticated parent inventory builder | Actual candidate non-overlap and complete eligible parent lineage, separately from raw-byte coverage |

No new paid dashboard, Kubernetes cluster or replacement model framework is
required. A disk-backed preprocessing index is justified when measured corpus
memory makes the existing Python identity/relevance structures unsuitable. A
cloud-global budget store is justified only with multiple independent controllers.
Teacher generation and additional training objectives should follow correctness
and independent quality selection, rather than consume the remaining campaign
budget before those gates pass.

## Ordered next actions

1. Preserve current successful data-only scan receipts, while keeping raw integrity,
   candidate exposure checks and production-model readiness separate.
2. Resolve the named physical cache diagnostic without allocating repeated capacity
   attempts or weakening reference comparisons speculatively.
3. Freeze final source/runtime and restore the actual retained parent on TPU, then
   run the selected correctness/recovery suite on the intended configuration.
4. Prepare independently eligible development data and establish parent retention
   thresholds before substantial continuation.
5. Qualify sustained multi-host/fault behavior before expanding topology; profile
   and choose scale based on measured throughput, memory and cost per useful row.
6. Reconcile posted billing and a scoped storage retention inventory; retain
   conservative reservations while actual amounts remain unknown.

This document review did not rerun tests. Use the root controller's current test
receipt for aggregate metadata counts, and physical raw/JUnit receipts for model
acceptance. Historical review counts are dated snapshots and must not be summed.

## Subsequent parity capacity admission implementation

[The bounded parity campaign](../scripts/run_yat_parity_campaign.py) now requires
explicit positive `--host-ram-budget-bytes`, `--scratch-budget-bytes` and
`--evidence-budget-bytes`. Run it with `--preflight-only` before cloud allocation:
this reads configuration/model-file sizes, emits the full required matrix and
its estimate, and launches no identity probe or model. A one-case lease still
admits retention for the complete matrix across resumptions. It does not reduce
advertised maximum context or required batch coverage to fit a smaller budget.

For 72 fixture rows, context length L and actual configured width H, define
E=72×L×H, K=2+number of retained intermediate indices. The output per backend
is O=4×(K×E+72×H) bytes. The conservative host estimate is
`runtime reserve + 3×model file bytes + 3×O + 64×E`. The three output sets cover
chunk retention/concatenation/archive lifetime; the 64 bytes per sequence element
allow for float64 norm casts/squares and FP32 differences/products/quantile
temporaries with overlapping lifetimes. Two backend archives, token inputs and
1 MiB metadata allowance per case are summed over the complete matrix for scratch
and evidence budgets. Default runtime reserve is 2 GiB and is explicitly adjustable.

For width 768, context 32,768 and two retained layers, this gives **27.000 GiB
per backend output**, **54.010 GiB per case of retained evidence**, and
**191.001 GiB host RAM plus three model-file copies**. This is a conservative
capacity estimate for the current full-output implementation, not a measured peak
or a device-HBM feasibility result. Larger full-output campaigns may require a
high-memory worker or a separately validated streaming implementation.

Before actual identity/model subprocesses, the campaign checks actual accumulated
evidence bytes, available scratch and Linux `MemAvailable` when exposed. It records
the estimate policy/basis and host observation separately, then checks retained
evidence after each new case. On platforms without `/proc/meminfo`, RAM remains an
operator capacity declaration; the receipt states that limitation. Model/runtime
installation storage is outside the additional-evidence scratch estimate and must
be included in the worker's separate provisioning plan. The estimate does not
bound framework/device allocations or guarantee future available RAM.

This subsequent implementation passed **19 targeted model-free tests** across
resource admission and campaign integration, with Ruff and diff checks passing.
No cloud allocation or model execution was performed.
