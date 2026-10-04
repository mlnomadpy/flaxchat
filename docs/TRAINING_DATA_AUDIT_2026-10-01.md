# Training and data audit — October 1, 2026

Read-only review of the current working tree for the requested parallel system
audit. This addendum extends the [current system audit](SYSTEM_AUDIT_2026-10-01.md)
and [issue checklist](ISSUE_READINESS_CHECKLIST.md); it does not overwrite their
dated evidence. No model was executed, no hardware was allocated, and no CPU
model test or benchmark was used. The coordinator owns the fresh GitHub snapshot
and current cloud scan receipts.

## Decision

The harness has useful implementations for continuation, authenticated inputs,
quality-state recovery and independent development. It has not yet demonstrated
a production continuation improving the retained parent's representations under
the new policy. The next model gate is the cache diagnostic and bounded physical
single-host qualification; actual independent data preparation and parent
exposure can proceed without a model allocation. Multi-host qualification is
required before multi-host scaling, not before a single-host correctness test.

## Prioritized findings

| Priority | Finding and file evidence | Required action and existing issue |
|---|---|---|
| P1 | Two cache optimizer-state comparisons failed in the retained physical receipt, while earlier objective/gradient/weight checks passed. `embedding_gradient_cache.py:51–59` accumulates VJPs in FP32 and casts the completed result to parameter dtype; differing reduction shapes are a diagnostic hypothesis, not an established cause. | Run the named-leaf, shape-matched ordinary autodiff/cache/full-batch diagnostic under the original bounds. Do not waive Adam-state failures. #49, #50. |
| P1 | Independent retrieval and STS are opt-in: `embedding_stage.py:176–206` admits empty lists and writes empty coverage into its receipt. An ordinary stage may therefore use only source-internal development. | Add a stage-specific production quality policy with mandatory declared task coverage; keep bounded infrastructure fixtures eligible under an explicit qualification scope. Policy presence must not substitute for authenticated actual rows. #47, #61. |
| P1 | Matching committed metadata and historical raw coverage do not establish that the actual parent tensors restore correctly or that the candidate probes were unseen. `embedding_dev_exposure.py:54–65` explicitly retains unresolved MLM, semantic/translation and undeclared-stage exposure. | Restore the intended production parent on TPU, recompute candidate exposure against retained known stages, and document remaining historical uncertainty. Raw integrity alone cannot admit a clean candidate. #40, #44, #47. |
| P2 | Programming-language provenance is retained at `embedding_data.py:134–136` and in exposure counters, but `train_yat_embedding_finetune.py:282–303` evaluates only natural-language slices. This cannot detect a regression concentrated in one programming language. | Add programming-language development slices when real labels exist, pin the selection policy and avoid double-counting their votes in checkpoint selection. The stripped CodeSearchNet source's unknown labels must remain unknown. #47, #48. |
| P2 | Language selection guarantees only two retrieval rows per represented language (`embedding_quality.py:103–138`); STS raises this to three. These are validity minima, not evidence of a reliable production quality estimate. | Declare practical minimum query/candidate coverage per task/language and retain sample size with scores; choose larger coverage using observed metric stability. Tiny fixtures remain correctness tests. #47, #57. |
| P2 | Relevance is mmap-backed, but all raw row records, hashes, groups, language lists and relevance sets are still retained in Python (`embedding_data.py:42–125`). Dense mmap storage is rows × maximum positive width (`:139–146`), even for sparse relations. | Measure preparation peak RSS and temporary disk on the actual intended recipe before scaling. Add disk-backed identity construction and sparse relevance only when that measurement requires it. #44, #49. |
| P2 | The embedding trainer constructs only a `("data",)` mesh with replicated parameters/state (`train_yat_embedding_finetune.py:140–174`). FSDP is not integrated. YAT attention explicitly requires XLA (`encoder.py:98–101`); Splash is not interchangeable with its kernel. | Profile the accepted path on TPU. Integrate state sharding or a new YAT kernel only after a measured bottleneck and physical loss/gradient/recovery acceptance. #49, #50. |

The two new policy observations above are quality gaps, not proof that historical
published weights were corrupted. No issue should close merely because a helper
exists or an audit recommends a change.

## Implemented versus physically accepted

| Component | Present implementation | Evidence limitation |
|---|---|---|
| Parent/data admission | Published file hashes, fixed YAT settings, complete prepared-array schema, raw negative-valid agreement and committed metadata reader | Actual current parent tensor restoration and real recipe acceptance remain open. |
| Representation quality | Retrieval, STS, per-natural-language regression checks, baseline/best/recovery state and independent `--retrieval-dev` | Actual representative independent data and production TPU baseline/continuation are not qualified. |
| Sampler and exposure | Mixed/homogeneous source policies, replayable language sampling, checkpointed unique/repeated pairs and token counts | Replacement exposure is not an epoch count; current real recipe/scales need measurement. |
| Objective and cache | Complete relevance masking and opt-in global-loss encoder cache | Historical selected v5e-8 receipt: 14 pass, two cache Adam-state failures, zero skips. This does not qualify current changed source. |
| Physical recovery tests | Off-cadence/failed-gate/commit-window fixtures and multi-host/fault campaign definitions | New independent fixture, sustained physical multi-host faults, v6e and every intended scale remain unvalidated. |
| Historical data evidence | Bounded GCS scan, immutable producer adapter, exposure builder and candidate checks | Integrity scan without candidate exclusions cannot prove non-overlap or full historical lineage. |

## Tools to use and integrations to defer

Use the existing stdlib metadata readers, generation-pinned GCS streaming scan,
authenticated candidate exporter and parent inventory builder for data work near
the bucket. Pass the verified historical producer policy through both scan and
inventory construction when required. Reuse the pinned embedding runtime,
physical pytest/JUnit runner, named diagnostic receipts and guarded campaign
ledger for model acceptance.

After correctness passes, collect short warmed JAX/XProf traces and measure
input, encoder, contrastive denominator, collectives, checkpointing, HBM, host RSS
and scratch disk. Those measurements select the useful optimization. No new
framework or paid dashboard is needed for this gate. Teacher distillation,
Matryoshka objectives, longer-document curricula, mined-negative refresh and
FSDP are separate measured improvements, not currently accepted capabilities.

## Documentation corrections for the coordinator

`REPRESENTATION_DEVELOPMENT_RECIPE.md` still describes independent integration
as a required adoption task without explaining that `--retrieval-dev` now exists;
actual production adoption remains outstanding. `TRAINING_TOOLS.md` retains an
older statement that the embedding-specific lock needs completion, followed by
a newer paragraph recording its successful physical installation. Reconcile
these present-tense inventories while preserving the historical physical
14-pass/two-failure receipt and the scope of later data-only work.

Verification for this review was source inspection and cross-checking the
retained audit/physical receipts. No fresh training result, model quality number,
throughput number, cost or broad test-pass claim is introduced here.
