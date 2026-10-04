# Current infrastructure, evaluation and release audit

Read-only source/evidence review on October 1, 2026. This reviewer made no cloud
requests, allocations, restarts, publications or CI changes and ran no model or
numerical tests. No applicable `AGENTS.md` was found in the workspace. This report
supplements [infrastructure](INFRA_AUDIT_2026-10-01.md) and
[evaluation](EVALUATION_AUDIT_2026-10-01.md) reviews; references below identify
current implementation rather than proving physical acceptance.

The harness has substantial admission and release protections. **A large production
continuation remains blocked by physical correctness and parent qualification.**
Buying a replacement orchestration or monitoring platform would not resolve these
gates. The latest preserved cache diagnostic attempt05 exhausted capacity before
model execution; it adds owned-resource absence evidence, not a numerical pass.

## Current implementation reconciliation

The original rows below describe the earlier review snapshot. Subsequent
[lifecycle/telemetry/parent evidence](audit-2026-09-30/lifecycle-telemetry-parent-followup-2026-10-01.md)
supersedes implementation-absence claims: bounded data jobs have four actual
Linux lifecycle cases passed; global telemetry is integrated; query uncertainty
is implemented. Their remaining physical/provider scopes remain open.
Parent04 exposed the tokenizer bound defect, now fixed/tested; parent05/06 failed
transport before enriched materialization. A range diagnostic is prepared and
literal-byte tested, not yet executed remotely. Full-corpus reporter and staged
TPU scorer exist; the physical encoder and matched competitor runs do not.
See [current tools](TRAINING_TOOLS.md) for implementation versus acceptance.

A fresh P2 source finding is tracked under #51/#55: observer callers can omit the
externally retained spec hash. The enforcement fix and tamper tests now pass; no lifecycle pass is inferred for changed source from the earlier
Linux result. No new duplicate GitHub issue is needed.

## Concrete findings and acceptance work

| Priority | Finding and source | Required work and acceptance |
| --- | --- | --- |
| P1 | Historical qualification02 has 14 passes and two Adam-state failures; diagnostic05 [delivery](audit-2026-09-30/cache-diagnostic-05/delivery.json) records no remote model execution | Run the frozen named-leaf diagnostic on physical TPU, attribute gradient/moment error and rerun original acceptance nodes without weakening tolerances. Source changes or capacity receipts cannot close #49/#50. |
| P1 | Real step14000 public bytes now authenticate, but actual preparation01 failed and did not reach leaf authentication or TPU import: [worker](audit-2026-09-30/parent-materialize-01/worker.json) | Current source stability check (`scripts/prepare_enriched_parent_export.py:167`) excludes read-only access-time changes while retaining identity/size/mtime/ctime checks. Treat that fix as local implementation until a new immutable preparation passes. Then qualify actual tensor restoration and explicit optimizer continuation policy on TPU. Downloading weights is insufficient for #40–#42. |
| P1 before scale | Multi-host admission is a four-update synthetic encoder scope, explicitly excluding abrupt process/transport faults (`scripts/validate_embedding_multihost.py:255`, `:290`) | Sustained intended-model multi-host training, exact row/device ownership, global-negative correctness and checkpoint recovery under selected worker/controller/transport faults. Preserve process inventories and independently verify owned queue/node absence. #12/#50/#51/#55 remain physical work. |
| P1 quality | Matched receipt validation exists; it is explicitly not a model runner (`scripts/compare_embedding_receipts.py:1`, `:224`) | Add pinned mmBERT and EmbeddingGemma physical evaluation adapters with official per-model prompts, tokenizer and pooling. Freeze common task/split/subset inventories, scoring source and native scale before scoring; preserve per-language/category regressions and independent parent comparison. Receipt comparison alone provides no new scores. |
| P1 before release | Parity v4 complete receipt and private raw replay are implemented (`scripts/release_contract.py:75`, `scripts/parity_evidence.py:43`), but a current full physical matrix is absent | Execute both isolated JAX/TorchXLA backends across required contexts/batches and72 rows, validate retained raw evidence and independently verify uploaded immutable revision. Historical small parity receipts cannot qualify current release or changed source. |
| P2 | Full parity outputs remain memory-intensive even after capacity admission (`scripts/run_yat_parity_campaign.py:50`, `:72`) | Measure actual peak host RAM/device memory and retained bytes on admitted hardware. Consider streaming exact comparisons, preserving all current gates including exact quantile semantics. Estimate admission explicitly says device HBM and physical peak are unqualified. |
| P2 operations | Root-owned data jobs have useful durable receipts but their execution controllers remain temporary artifacts; current skill describes policy, not a reusable checked-in data-job lifecycle | Integrate a versioned bounded data-job controller with unique ownership marker, generation-bound sources, independent deadline, terminal worker state, observation-only reconnect and exact owned-process/scratch absence receipt. A disconnected SSH attachment must never create a new job. Model-free lifecycle tests should cover worker failure with SSH exit0 and uncertain reconnect; no new paid infrastructure required. |
| P2 accounting | Existing BigQuery adapter supports bounded parameterized queries (`scripts/cloud_accounting.py:17`, `:129`), but current billing export is disabled and project-level UI amounts are not model attribution | Establish an authorized export for future accounting, or import authenticated provider cost-table rows with explicit invoice/filter scope. Keep usage, promotional credits, net amount, current credit balance and safety reservations separate. Preserve unattributed services rather than allocating them arbitrarily to training. |
| P2 storage | Inventory consists of three sequential reads, not an atomic snapshot (`scripts/cloud_accounting.py:34`, `:79`); cleanup is proposal-only (`:87`) | Inventory scoped live/noncurrent/soft-deleted generations with bounded repeat observations, protect parent/best/latest/evidence lineage, verify retention/holds and owned deletions. A successful prefix cleanup cannot establish empty project-wide storage or zero charges. |
| Conditional | `RunLedger` locks local files (`flaxchat/operations.py:23`) | One root controller/shared ledger is appropriate now. Use cloud compare-and-swap reservations only before independent controllers on separate machines; parallel reviewer agents need no new budget database. |

## What is already implemented

The Spot supervisor never retries timed-out create calls as fresh allocations
(`scripts/gcp_spot_supervisor.py:49`), checks queue and VM absence after deletion
(`:73`), and treats transport uncertainty as unknown. Separate capacity/startup
waits fit under the global lease; whole-slice reserve includes1800 seconds of
cleanup allowance (`:247`). The cloud guard and conditional IAM scope admission
remain required before provisioning. Exact node labels bind billing run identity.
Manifest admission checks zone/pricing/provisioning agreement and immutable source
hashes (`scripts/representation_run.py:153`).

Training already records input latency, accepted update/loss, gradient norm,
masking diagnostics and local useful-token count (`scripts/train_yat_embedding_finetune.py:379`).
It blocks on loss readiness before reporting step latency, exposes JAX traces and
records checkpoint latency (`:384`, `:407`). The next performance integration
should aggregate **global nonpadding tokens**, sustained warmed step distributions,
per-rank stalls, evaluation/checkpoint overhead and actual cost per useful pair/token.
Local token counts cannot be interpreted as global multi-host throughput. XProf
and existing JAX trace hooks should identify an actual bottleneck before a new
kernel, topology or cache strategy is adopted.

The newer quarantine filter deliberately preserves `parent_exposure_checked=false`
(`scripts/filter_representation_development.py:163`): authenticated candidate
collision filtering is not complete inherited-training exposure or semantic
decontamination. Deployment gates must retain that distinction.

## Tool integration priorities

1. **Use existing tools first:** `gcloud`, independent Workflows cleanup guard,
   immutable GCS generations, Orbax committed checkpoints, physical pytest/JUnit,
   pinned MTEB, isolated JAX/TorchXLA and JAX profiling/XProf.
2. **Add reusable data-job lifecycle tooling:** owned markers/deadlines, durable
   worker and launcher receipts, reconnect/cleanup verification and concise
   transport diagnostics. Temporary job scripts should become reviewable source.
3. **Add actual competitor evaluation adapters:** current receipt comparator is
   ready to consume complete results; it cannot generate them. Query-level retained
   scores permit paired uncertainty estimates when the metric supports them.
4. **Add accounting/throughput reconciliation:** provider-posted rows plus resource
   labels, billed windows and useful-exposure counters. Export activation is a
   separate cloud configuration action, not evidence of recovered historical data.
5. **Optimize parity evidence only after preserving gates:** streaming row/chunk
   comparisons can reduce host/storage cost, but require independent equivalence
   checks of exact existing metrics before replacing full-output replay.

Kubernetes, another paid observability service, new CPU model tests and automated
expensive GitHub checks are unnecessary for these immediate gates. Existing checks
should stay paused until the user re-enables them.

## Actual evidence and limits

The preserved September [cost-table observation](audit-2026-09-30/posted-september-cost-observation-2026-10-01.json)
shows $212.95 for tpubuilders and $112.52 for azettaai, total$325.47, with promotional
credits excluded but spending discounts/tax included. This is whole-project
September usage; October is excluded, invoice finality and CSV download are
unverified, and it is not the model's A-to-Z cost.

Preparation01 downloaded the authentic1,231,164,480-byte weight artifact and other
pinned public files, then failed enrichment. Worker return1 with attachment
return0 demonstrates why durable worker status is authoritative. The launcher
reports cleanup, while independent absence verification must come from the root's
subsequent observation. This report does not claim a currently running job is
terminal or that all resources/storage are clean.

No tests were rerun by this reviewer. Previous model-free contract receipts prove
their dated source scope; only fresh physical receipts can establish current
numerical, topology, performance, parity or production-quality acceptance.

The observer fix passed61 focused metadata tests with one explicitly Linux-only
macOS skip. Four earlier Linux lifecycle passes remain scoped to the old source
freeze; fresh changed-source Linux acceptance is pending. No TPU model or cloud
execution occurred in this review.
