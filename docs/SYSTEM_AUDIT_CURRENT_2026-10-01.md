# Current full system audit and training readiness

October 1, 2026. Three parallel reviewers inspected training/data, infrastructure/evaluation/release, and every open GitHub issue. This is a source and retained-evidence audit, not exhaustive execution of every configuration. The current goal is to improve the existing YAT multilingual/code embedding weights.

## GitHub inventory

All **29 open issues** were refreshed with full bodies/comments at 21:46 UTC. No membership changes or duplicate new issue were found. The [complete inventory](audit-2026-09-30/open-issues-2026-10-01-current.json) and [every-issue acceptance table](audit-2026-09-30/issue-acceptance-2026-10-01-current.md) distinguish implementation from acceptance. Existing issues cover the findings below. Costly GitHub workflows remain undispatched.

## What must be resolved before substantial continuation

| Order | Gate | Actual state | Acceptance / issues |
|---|---|---|---|
| 1 | Authenticate/import the real parent | Public step-14000 bytes authenticated in earlier attempts; atime/tokenizer bounds fixed locally; latest retained-GCS copy timed out before enrichment | Corrected bounded materialization, then every-leaf physical TPU import; explicit optimizer policy. #40–#42/#52 |
| 2 | Qualify the chosen training path | Historical v5e-8: 14 passes, two cache Adam-state failures, zero skips | Named-leaf diagnosis under original bounds for cache; separately frozen direct path can qualify with cache disabled. #49/#50 |
| 3 | Independent production development | Native MIRACL filtering/re-scan completed; zero exact overlaps against supplied history; full four-domain plan incomplete | Authenticate retrieval, bitext, STS and code inputs; parent baselines, practical coverage, quality/regression and best/recovery gates on TPU. #44/#47/#48/#61 |
| Before multiple hosts | Sustained scaling/recovery | Implemented bounded fixture and fault tools; intended production topology unqualified | Global negatives, disjoint row ownership, sustained updates, worker/controller/attachment fault scopes and exact recovery. #12/#50/#51/#55 |
| Before publication | Current conversion/evaluation release | Receipt/provenance/finite-score safeguards implemented; full current physical matrix absent | Required contexts/batches/raw parity, pinned task inventory, immutable upload verification. #53/#57–#60 |

A changed recipe is a new stage initialized from trained weights, never a silent exact resume or a random restart. Fixed YAT bias 1, epsilon 0.01 and trainable alpha remain unchanged. Release and multi-host gates apply to those actions; they do not prevent a bounded single-host training-only qualification.

## Newly sharpened gaps

- **Development retrieval scope:** current bounded probes rank selected positive passages. Add separately pinned full-corpus and mixed-language distractor evaluation for product retrieval claims. Keep these metrics separately named. #47/#57/#58.
- **Quality confidence:** operational minima of 64 rows/16 per language are coverage checks, not reliable evidence of small improvements. Add query-level uncertainty reporting and sufficient actual coverage; version selection-policy changes. #47/#61.
- **Preparation scalability:** Python row/identity/relevance structures and dense membership storage can dominate host RAM/scratch. Measure the intended recipe before adding a disk-backed index or CSR relevance. #44/#49.
- **Whole-slice efficiency:** local useful-token counts do not report global multi-host throughput. Global counters and invocation overhead are now implemented; qualify their physical cadence and connect posted cost per useful exposure. #49/#54.
- **Cloud data-job lifecycle:** recent working controllers live in temporary directories. Their bounded ownership, immutable source, durable worker status and cleanup are now versioned; qualify the changed-source controller and whole-controller faults. SSH exit0 can accompany worker failure. #51/#55/#61.
- **Competitor evaluation:** receipt comparison exists; a faithful mmBERT/EmbeddingGemma runner does not. Add pinned adapters and model-specific prompts/pooling before superiority claims. #57/#58/#61.

Detailed file-line evidence: [training/data/quality](TRAINING_AUDIT_CURRENT_2026-10-01.md), [infrastructure/evaluation/release](INFRA_RELEASE_AUDIT_CURRENT_2026-10-01.md).

## Tools and upgrade priorities

Keep JAX/Flax/Optax, Orbax, frozen wheelhouses, Safetensors, mmap preparation, physical pytest/JUnit, pinned MTEB, GCS and the independent cloud cleanup guard. Their qualification scope is in the runbook. No replacement framework or paid dashboard is required.

1. Complete existing parent/direct-training/quality gates.
2. Qualify the implemented near-cloud data-job lifecycle and global exposure/time accounting in their remaining production scopes.
3. Collect short warmed [JAX/XProf traces](https://docs.jax.dev/en/latest/profiling.html) and [device-memory analysis](https://docs.jax.dev/en/latest/device_memory_profiling.html). Keep unprofiled and full-stage timing separate.
4. Integrate full-corpus evaluation, uncertainty and actual competitor runners.
5. Establish authorized [Billing export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) or scoped authenticated provider rows for future accounting. Current export is disabled; activation does not prove recovery of all old runs.

FSDP, custom YAT/Pallas kernels, teacher distillation, refreshed hard negatives and Matryoshka dimensions are conditional future work. [Cached and Matryoshka objectives](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html) are available in other frameworks; that does not make them integrated or TPU-qualified here. Choose isolated new stages based on measured quality or bottlenecks.

## Actual data, cost and cleanup evidence

The [completed native MIRACL controller](audit-2026-09-30/native-miracl-filter-01-controller.json) reports scanner/filter/recheck return0 and 674.88 seconds. The downloaded [scan/filter/recheck receipts](audit-2026-09-30/native-miracl-filter-01/receipt-verification.json) match controller SHA-256 values. Re-scan authenticated 2,075,961,764 bytes /1,694,512 historical rows and found zero exact overlaps for the new candidate. `admission_ready` remains false: parent export/full known-stage binding, independent admission and physical quality remain. Undisclosed contrastive, inherited MLM and semantic/translation exposure remain unresolved. The candidate raw export remains retained remotely; this audit did not download/revalidate all candidate files locally.

The [independent cleanup observation](audit-2026-09-30/current-owned-data-cleanup-2026-10-01.json) found both original parent/filter timeout PIDs, ownership-tagged processes and scratch absent. Receipt/source directories remain; this is not a project-wide clean-resource or empty-storage claim. No new training/TPU allocation occurred in this audit. Latest retained public embedding checkpoint is step14000; no new model update is established by data-only results.

Actual preliminary [September billing observation](audit-2026-09-30/posted-september-cost-observation-2026-10-01.json): tpubuilders $212.95, azettaai $112.52, total $325.47. Promotional credits were excluded; spending discounts/tax included. This covers whole projects, excludes October, and is not per-model A-to-Z cost; invoice finality/CSV were unverified. A credit balance or reservation is a different quantity.

## Skill and validation delivery

The [training skill](../skills/flaxchat-training/SKILL.md) is now a concise mode router with detailed qualification procedures in a supporting reference. It preserves lineage, TPU-only model verification, failed gates, budget/ownership and stage-specific acceptance. It now explicitly covers selected-positive retrieval scope, whole-stage efficiency, read-induced atime changes and worker-vs-SSH authority. An independent reviewer forward-tested changed-recipe continuation and failed-worker/multi-host scenarios; both selected the correct gates. The runbook, tool inventory, index and readiness docs point here while retaining dated prior evidence.

No CPU model tests, expensive CI or new TPU campaigns were run for this audit. Both skill copies passed schema checks; report/skill links and diff checks passed. Forty-two selected metadata tests passed (parent preparation, cache diagnostics and receipt comparison). An initial unittest invocation discovered no tests and was not counted; pytest executed the selected tests. Retained-evidence byte verification and these checks are not model qualification. Repository changes remain local; the installed skill is synchronized. Posting the prepared detailed GitHub comment was rejected by automatic approval review because of its cloud/cost/data/infrastructure payload; no new remote issue/comment was created. The current inventory was refreshed read-only.

**Next concrete gate:** measure the failed parent transfer on the existing executor, then use that evidence for one changed bounded full-transfer/enrichment attempt. Physical parent/direct-training qualification follows; cache-enabled training requires its original numerical acceptance. Complete independent bitext/STS/code preparation in parallel. Substantial continuation still requires actual production quality gates.

## Subsequent concrete implementation/execution progress

The [lifecycle, telemetry and actual-parent follow-up](audit-2026-09-30/lifecycle-telemetry-parent-followup-2026-10-01.md) adds versioned lifecycle tooling with actual Linux acceptance, global exposure/invocation telemetry, query-level uncertainty and metadata-only routes. The tokenizer-specific JSON-size defect is fixed and metadata-tested. Parent05 failed external transport and parent06 timed out on the retained-GCS weight copy before enrichment. The parent remains unimported on TPU. Original historical audit acceptance is retained.

## Current prioritized execution and tool gaps

The current working tree already contains bounded data jobs, global telemetry,
query uncertainty and a transfer diagnostic. Their guides are linked from the
[tool inventory](TRAINING_TOOLS.md); earlier proposals above do not imply they
are still absent. Four actual Linux lifecycle scenarios passed. Telemetry needs
physical collective/cadence/overhead acceptance; bootstrap confidence needs real
authenticated per-query results; transfer diagnostics need an actual executor
measurement. No platform replacement is needed.

1. Measure the parent transfer on the existing near-cloud executor, without
   downloading model bytes through the local connection. Authenticate the full
   artifact before enrichment and all181 leaves before physical import.
2. Qualify the chosen direct training path with the real parent. In parallel,
   execute the new reporting-only cache boundary diagnostic; keep the original
   two optimizer failures and their tolerances. Cache-enabled continuation waits
   for cache acceptance; direct qualification does not claim a cache pass.
3. Complete independent retrieval, bitext, STS and code development coverage,
   per-language/programming-language parent baselines and regression thresholds.
   Full-corpus retrieval must prove actual corpus encoding/search coverage.
4. Run a bounded new training stage from the retained weights and preserve
   distinct best/recovery checkpoints. Profile warmed physical steps and entire
   stage cost before changing kernels, sharding or batch strategy.
5. Qualify sustained physical multi-host and fault recovery before scaling;
   current full parity and immutable release checks apply before publishing.

The next gate is a measured artifact-transfer decision, not another identical
copy retry. The [independent training readiness review](audit-2026-09-30/cache-direct-parent-independent-readiness-2026-10-01.md)
records the cache/direct/parent distinction. A complete system audit includes
the decoder/tag/demo/framework issues as separate work; embedding progress does
not close them. Kaggle remains deferred under the stated GCP priority.

## Latest independent findings and implementation

The parallel infrastructure reviewer found optional expected-spec authentication
in the data observer; the CLI/API now require the external pin and tamper tests pass under #51/#55.
The evaluation reviewer implemented [authenticated native retrieval reporting](FULL_CORPUS_RETRIEVAL.md)
and a staged bounded TPU scorer. Exhaustive scalar fixtures establish reporting
semantics; scalable top-k reports explicitly remain unqualified for full search.
Actual YAT encoder integration, physical scorer parity, corpus traversal and
matched competitor execution remain required. These map to existing #47/#57–#59.
No duplicate issue, cloud allocation, publication or expensive workflow was
needed to complete this review.

The observer fix passed61 focused metadata tests with one explicitly Linux-only
macOS skip. Four earlier Linux lifecycle passes remain scoped to the old source
freeze; fresh changed-source Linux acceptance is pending. No TPU model or cloud
execution occurred in this review.

[Final requested-audit verification receipt](audit-2026-09-30/requested-parallel-audit-delivery-2026-10-01.json): root ran125 metadata tests plus28 subtests; one Linux-only macOS skip. Initial collection attempts failed on unavailable JAX/NumPy and ran no tests; the successful scope excluded model execution. Skill schema and installed/versioned equality passed, and8 selected documentation files have no missing local links (Jekyll template destinations excluded). Current-source physical TPU acceptance remains pending.

Actual continuation: the changed observer/controller source has now passed its
[Linux lifecycle and external-pin acceptance](audit-2026-09-30/bounded-data-job-linux-selftest-02/selftest.json),
with four separate terminal cleanup observations. Earlier statements that fresh
changed-source Linux acceptance was pending are superseded for this exact source;
whole-controller/provider/TPU faults remain separate. The measured-transfer worker
started under a finite timeout; its result is not yet established.

## Issue closure scope correction

The latest read-only refresh at21:59UTC still lists29 open issues. Original
acceptance review found #39 physically verified and #46 implemented/tested,
pending integration/issue updates. #43/#45 have narrow event/identity test gaps.
Earlier checklist rows over-expanded these into full production qualification;
that work belongs to #44/#47/#50/#61. See the corrected
[every-issue table](audit-2026-09-30/issue-acceptance-2026-10-01-current.md).
Training remains at the retained step14000; no current production model update
is established by metadata, lifecycle or preparation progress.

Actual [transfer result and decision](audit-2026-09-30/transfer-diagnostic-01/decision.md):
4,980,736bytes in180seconds, then the finite range deadline; exact owned processes
and scratch independently absent. This measured stalled Cloud Shell path is
unsuitable for full parent delivery. The next gate is a changed delivery/executor
path with original full authentication, not another unchanged timeout extension.
No TPU/GCE was allocated by the probe.

Full-corpus YAT encoding integration is now implemented with authenticated restored
leaf bytes and complete source/runtime/corpus pins.61 focused reporter/admission/
CI-routing metadata tests passed. Ten physical scorer cases are defined but unrun;
actual encoder/scorer parity and complete real-corpus benchmarks remain pending.
