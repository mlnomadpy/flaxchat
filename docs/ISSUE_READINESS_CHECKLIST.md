# Open issue and training readiness checklist

**October6 update:** [fresh25-open-issue reconciliation](issue-review-2026-10-06/README.md). The29-issue counts below are historical. Issues#39,#43,#45,#46 are already closed; current single-host TPU acceptance is11passes with real checkpoint continuation.

Latest reconciled audit: [all open issues, current evidence and tool priorities](SYSTEM_AUDIT_CURRENT_2026-10-01.md). Earlier dated findings below remain historical where superseded.


Latest requested parallel review: [audit, new gaps, tools and validation scope](PARALLEL_SYSTEM_AUDIT_2026-10-01.md). All 29 open issues were refreshed; historical counts below remain dated evidence.

Refreshed October 1, 2026. All 29 issues below remain open. Code availability is not acceptance. See the [current audit](SYSTEM_AUDIT_2026-10-01.md) for exact findings and the [raw GitHub snapshot](audit-2026-09-30/issue-refresh-final-2026-10-01.json) for original acceptance text.

## Work order

Latest [GitHub refresh](audit-2026-09-30/issue-refresh-followup-2026-10-01.json) still contains 29 open issues. Independent development tooling and cross-source quarantine are implemented and pass metadata checks, but actual data/parent evaluation remains open. The new cache diagnostic ran no model tests because capacity was unavailable; queue/node cleanup was verified. See the [consolidation](audit-2026-09-30/consolidated-followup-2026-10-01.md).

- [x] Implement required-array checksum coverage, raw negative-valid agreement, failed-quality-gate resume and nonmutating optional checkpoint admission; selected model-free checks pass. Physical acceptance is separate.
- [ ] Complete physical single-host acceptance: the frozen v5e-8 fixture passed 14 checks and failed two cache optimizer-state comparisons; new diagnostics preserve the failure and original bounds.
- [ ] Prepare actual multilingual/bitext/STS/code development data and verify quality against the retained parent.
- [ ] Qualify intended physical multi-host topology, then measure sustained recovery and warmed scaling before allocating a larger continuation.
- [ ] Continue existing weights under the declared quality/budget/checkpoint policy; retain best and recovery independently.
- [x] Implement full parity provenance, depth-aware intermediates and bounded per-case scheduling.
- [ ] Execute physical parity and real benchmark protocol acceptance before publication or competitor-superiority claims.
- [ ] Reconcile posted costs and retention; commit/release/close issues only against their actual acceptance.

## Every open GitHub issue

| Issue | Title | Remaining acceptance |
|---|---|---|
| [#1](https://github.com/mlnomadpy/flaxchat/issues/1) | Repository quality and release-readiness upgrade plan | Repository-wide reproducibility, tagged compatibility and decoder release evidence; separate from embedding-only qualification. |
| [#11](https://github.com/mlnomadpy/flaxchat/issues/11) | [P1] Cut v0.1.1 with the TPU initialization fix and acceptance evidence | Version is 0.1.1 locally; tag, artifact provenance and clean-release acceptance still need verification. |
| [#12](https://github.com/mlnomadpy/flaxchat/issues/12) | [P1] Run and publish a physical multi-host TPU acceptance suite | Physical multi-host acceptance: topology portability and actual interrupted/host-loss recovery; new worker is a bounded fixture. |
| [#19](https://github.com/mlnomadpy/flaxchat/issues/19) | [P2] Publish a small checkpoint and automated inference demo | Tagged small decoder checkpoint and clean inference demonstration; existing YAT publication alone does not satisfy it. |
| [#23](https://github.com/mlnomadpy/flaxchat/issues/23) | [P1] Execute the matched flaxchat, nanochat, and MaxText benchmark protocol | Matched decoder flaxchat/nanochat/MaxText runs; baseline plans remain pending. |
| [#31](https://github.com/mlnomadpy/flaxchat/issues/31) | [P1] Make FineWeb Kaggle TPU training independent of live PyArrow streaming | Kaggle-specific prepared input acceptance; deferred from GCP continuation priority. |
| [#39](https://github.com/mlnomadpy/flaxchat/issues/39) | [P1] Exclude all known relevant documents from the mined-negative denominator | Four tiny physical complete-relevance loss/gradient cases passed on v5e-8; actual prepared-recipe and distributed acceptance remain. |
| [#40](https://github.com/mlnomadpy/flaxchat/issues/40) | [P1] Authenticate parent checkpoint semantics before a fresh embedding stage | Nonmutating committed-parent reader implemented and metadata checked; physical import semantics remain. |
| [#41](https://github.com/mlnomadpy/flaxchat/issues/41) | [P1] Include sampler, imported kernels, and numerical runtime in exact-resume identity | Physical exact-resume source/runtime/sampler agreement and migration evidence. |
| [#42](https://github.com/mlnomadpy/flaxchat/issues/42) | [P1] Validate prepared embedding token schemas and padding against the encoder | Checksum/flag/reader implementations pass metadata checks; actual recipe and physical training acceptance remain. |
| [#43](https://github.com/mlnomadpy/flaxchat/issues/43) | [P2] Reject resume cursors beyond stop-after and emit explicit completion status | Physical cursor/update agreement and explicit already-completed versus failed status. |
| [#44](https://github.com/mlnomadpy/flaxchat/issues/44) | [P1] Unify cross-source duplicate identity and train/dev quarantine | Current-mixture overlap receipt, benchmark contamination review and production preparation budget. |
| [#45](https://github.com/mlnomadpy/flaxchat/issues/45) | [P2] Use code-aware text identity instead of casefolding code | Actual re-prepared code identities and immutable new-stage dataset evidence. |
| [#46](https://github.com/mlnomadpy/flaxchat/issues/46) | [P2] Make tokenizer truncation policy explicit in embedding preparation | Pinned preparation policy and real-data truncation receipt. |
| [#47](https://github.com/mlnomadpy/flaxchat/issues/47) | [P1] Gate embedding continuation on representation quality and preserve best checkpoints | Failed-gate persistence and fractional recall implemented; actual dev suite and physical selection/recovery remain. |
| [#48](https://github.com/mlnomadpy/flaxchat/issues/48) | [P1] Generalize replayable dataset/language sampling for representation continuation | Pinned multilingual replay, programming-language provenance and measured unique/repeated exposure. |
| [#49](https://github.com/mlnomadpy/flaxchat/issues/49) | [P2] Qualify embedding scaling and remove avoidable encoding and input overhead | Physical cache parity, traces, HBM/host memory, scaling and quality at equal global pool. |
| [#50](https://github.com/mlnomadpy/flaxchat/issues/50) | [P1] Add direct physical-TPU coverage for hard-negative embedding training and recovery | Execute defined objective/cache/trainer/fault and failed-gate resume tests, then physical multi-host. |
| [#51](https://github.com/mlnomadpy/flaxchat/issues/51) | [P1] Unify GCP launch paths behind an independently verified cloud lease | Physical controller-loss/transport cleanup and independently verified queue/node absence. |
| [#52](https://github.com/mlnomadpy/flaxchat/issues/52) | [P1] Lock embedding training/evaluation environments and source overlays | Frozen embedding runtime including Safetensors installed and passed physical v5e setup; separate Torch/XLA runtime and complete model acceptance remain. |
| [#53](https://github.com/mlnomadpy/flaxchat/issues/53) | [P1] Bind PyTorch parity to published artifacts and reject nonfinite gate values | Full identity/depth fixes implemented; physical parity and immutable remote release inventory remain. |
| [#54](https://github.com/mlnomadpy/flaxchat/issues/54) | [P2] Replace device-based cost guesses with verified slice rates and billing reconciliation | Actual run/stage provider labels read back successfully; posted billing reconciliation remains; one campaign owner unless cloud reservation added. |
| [#55](https://github.com/mlnomadpy/flaxchat/issues/55) | [P2] Use manifest-driven launches and bounded telemetry uploads | Reviewed wrapper/signal fixes implemented; physical lifecycle and workflow acceptance remain. |
| [#56](https://github.com/mlnomadpy/flaxchat/issues/56) | [P2] Track retained storage and cleanup plans across training stages | Explicit retention policy, stable inventory verification and posted storage costs; no inferred empty bucket. |
| [#57](https://github.com/mlnomadpy/flaxchat/issues/57) | [P1] Require finite scores and exact split/subset coverage for complete evaluations | Real pinned ordinary/subset/multi-split task receipt and merge/resume acceptance. |
| [#58](https://github.com/mlnomadpy/flaxchat/issues/58) | [P2] Bind evaluation receipts to source, runtime and aggregation protocol | Parity retains full identity; actual emitted protocol evidence remains. |
| [#59](https://github.com/mlnomadpy/flaxchat/issues/59) | [P1] Route embedding evaluator and PyTorch changes to appropriate validation | Actual evaluator/merge/port integration and physical route acceptance; no skipped tests counted. |
| [#60](https://github.com/mlnomadpy/flaxchat/issues/60) | [P2] Expand physical-TPU conversion parity beyond eight 128-token inputs | Scheduling/depth schema implemented; complete physical matrix remains. |
| [#61](https://github.com/mlnomadpy/flaxchat/issues/61) | [P1] Track representation-training continuation readiness after September 30 audit | Reconcile these gates and improve parent representations without language/code regression. |

The next training-only gate does not require prior release or multi-host acceptance. The scaling gate applies before multi-host use; the release gate applies before publication. Retry only when code/admission changed or one specifically identified live execution needs reattachment. Expensive workflows remain undispatched.

## Latest implemented safeguards, still awaiting physical acceptance

Production coverage enforcement and programming-language regression slices now
address the additional #47/#48/#61 observations. Production STS candidate/parent
binding and raw/token recomputation are implemented. #53/#60 now require v4
complete scopes, authenticated raw numerical replay and explicit memory/evidence
admission. Bounded candidate preparation and CI routing are implemented. Selected
model-free verification passed235 tests and97 subtests. The historical physical
result remains14 passes/two cache optimizer-state failures; no issue closes on
these local implementation checks alone.
