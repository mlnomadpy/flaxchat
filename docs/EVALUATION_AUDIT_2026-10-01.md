# Evaluation, conversion and release audit — 2026-10-01

This review covers the current worktree's evaluation, PyTorch parity, publication and test admission paths. It supplements the earlier [evaluation review](audit-2026-09-30/evaluation-review-2026-10-01.md), retaining its distinction between implemented contracts and physical acceptance. The initial review changed only this document; the authorized implementation follow-up below subsequently changed the relevant contracts and fixtures. No model execution, cloud allocation, dataset download or remote publication occurred.

## Priorities before the next representation stage

| Priority | Work required | Concrete acceptance |
| --- | --- | --- |
| P1 | Resolve the two actual gradient-cache optimizer-state failures before production continuation | Named-leaf, three-path physical TPU diagnostics using the existing bounds; authenticate results against the frozen revision and committed parent |
| P1 | Qualify the new evaluator on actual pinned datasets and emitted result rows | Ordinary, multilingual-subset and multi-split physical examples, then exact resume/merge and deliberate missing-row rejection |
| P1 | Establish independent representation improvement over the parent | Held-out retrieval, multilingual/bitext or STS and code measurements; parent and candidate use identical immutable evaluation data; no training/dev overlap claim from manifests alone |
| P1 product evidence | Add matched mmBERT and EmbeddingGemma embedding comparison | Official model-specific prompts/tokenizers/pooling with common immutable task inventories, explicit context/truncation policies, per-task and per-language differences |
| P2 | Harden release receipt schema and independently bind retained parity evidence | Missing rows/shapes/output hashes/backend scopes fail release admission; altered retained output evidence fails before upload |
| P2 | Bound expanded parity host memory and retained bytes | Maximum-length, width and layer-count estimate checked before execution; actual physical resource measurements and a bounded evidence strategy |
| P3 conditional | Clarify cleanup ownership for direct publisher API callers | The CLI already removes tokens on every parsed invocation; direct `publish(args)` callers must own early-failure cleanup or gain the same guarantee |

Existing GitHub issues #53 and #57–#61 cover related publication, evaluation, tests, parity and quality work. These are mappings to earlier issue records, not a fresh GitHub state verification or a reason to close an issue.

## Current implementation strengths

The evaluator pins MTEB version and immutable dataset revisions, validates exact split/subset result inventories, rejects nonfinite scores and preserves model/source/runtime/device/batch identity. The Arrow fast path explicitly falls back for logical row selections rather than reusing the unfiltered physical table (`scripts/evaluate_yat_public_mteb.py:35`). Evaluation remains physical-single-host-TPU-only (`scripts/evaluate_yat_public_mteb.py:150`).

The v3 parity pipeline preserves effective backend numerical settings and guarded deployment receipt hashes. Intermediate layers derive from configured depth rather than a fixed layer ten. The actual comparison rejects collapsed/shape-inconsistent outputs and verifies zero masks. The durable campaign verifies saved raw output hashes, sidecars and input bytes before reusing cases (`scripts/run_yat_parity_campaign.py:13`). Partial completion is explicitly insufficient for release. Separate JAX and TorchXLA subprocesses avoid sharing incompatible TPU runtimes.

Authenticated exports preserve committed source metadata, manifest, config, tokenizer and weights. Conversion/publication no longer hardcodes the original training steps. The publisher checks the immutable uploaded revision's file bytes and refuses to overwrite an existing intended fresh model repository (`scripts/publish_yat_torch_from_gcp.py:21`). These are implemented protections, not proof that a new production stage has passed them.

The physical suite admission rejects mixed pass/skip inventories, missing required nodes and duplicate test identities. Metadata tests are useful for these contracts but cannot accept physical numerics, sustained multi-host recovery, every topology, serving speed or model quality.

## Newly identified implementation gaps

### EVAL-C1 — P2: publication admission accepts a reduced parity receipt

`scripts/validate_yat_torch_parity.py:177` checks actual output arrays against 72 rows and declared tensor shapes, and the campaign verifies retained `.npz` hashes. However, `scripts/release_contract.py:57` and `scripts/release_contract.py:87` do not require the receipt's `scope.rows`, metric `shape`, backend output hashes or backend run scopes. The accepted fixture in `tests/test_release_contract.py:13` omits those fields. The publisher calls `validate_release` on aggregate JSON without consuming the retained raw evidence.

This does **not** mean the normal physical comparison accepts collapsed arrays. It means the final publication boundary has a weaker evidence schema than the producer/campaign. A stale or manually reduced report can pass that boundary without expressing the advertised fixture coverage or retained raw identities.

Require a versioned complete case schema at release admission: exact fixture/row policy, metric dimensions derived from context and configured width, equal backend scopes, backend output SHA-256 and an authenticated durable evidence inventory. Add metadata mutation tests for each missing or altered field. Require the publisher to validate that evidence inventory or a separately authenticated qualification bundle before remote writes; avoid copying large raw tensors into the public model repository unnecessarily.

### EVAL-C2 — P2: maximum-context parity has no host-memory/evidence-size admission

The JAX path retains all chunk outputs and concatenates all 72 rows for embedding, representative layers, hidden states and pooled vectors (`scripts/validate_yat_torch_parity.py:94`). Torch also retains full chunk outputs (`scripts/validate_yat_torch_parity.py:134`). The comparison then materializes full FP32 casts/deltas, while shape/zero checks compute FP64 norm temporaries (`scripts/validate_yat_torch_parity.py:190`, `scripts/validate_yat_torch_parity.py:220`). Case deadlines bound time but do not bound these bytes.

For illustration, one float32 hidden-state array at 72 rows × 8,192 tokens × 768 width is about 1.69 GiB. Five such arrays approach 8.44 GiB per backend before pooled outputs, duplicate concatenation buffers and comparison temporaries. This is arithmetic, **not a measurement of this model or an OOM observation**. Derive the real estimate from the authenticated config and actual intermediate count, record both host RAM and evidence storage allowances, and reject impossible admissions before allocating.

Prefer chunked raw evidence or streaming sufficient statistics, retaining deterministic query/corpus pooled vectors and exact row identities. Preserve exact gates, including an explicitly defined p99 calculation; do not silently replace the current quantile with an approximation. Maximum-context physical acceptance still needs actual observations.

### EVAL-C3 — P3 conditional: direct publisher API has a narrower cleanup guarantee than its CLI

`scripts/publish_yat_torch_from_gcp.py:21–48` parses/validates release files and reads/rejects an empty token before entering `publish(args)`'s local `try/finally`. However, `main()` at line75 already wraps that entire call in unconditional token cleanup. **The supported CLI therefore covers these early failures; there is no demonstrated CLI credential leak.** The narrower function guarantee matters only if another caller invokes `publish(args)` directly without its own cleanup wrapper.

Document the direct function's cleanup ownership, or give it the same guarantee if direct API use is supported. Add model-free early-failure tests at the CLI boundary as useful coverage; this is conditional hygiene, not a current production launch blocker. Delete only the explicitly supplied owned temporary file, never a general credential store.

## Measurement and tooling gaps

`scripts/run_matched_benchmarks.py:79` compares FlaxChat, nanochat and MaxText **trainer adapters**. It is not an mmBERT/EmbeddingGemma frozen-embedding benchmark adapter. Reuse the existing evaluation contract for a new matched embedding campaign; do not describe this trainer tool as solving the competitor gap.

`scripts/audit_embedding_overlap.py` already accepts pinned benchmark inputs and checks bounded exact text/code identities. The missing piece is a complete actual benchmark extraction/inventory and exposure audit over the production lineage. Its own limitations correctly retain semantic, translation, near-duplicate and earlier-stage exposure uncertainty. CodeSearchNet-family training and CoIR-family evaluation require particular attention. A successful raw byte/hash/count scan alone does not establish benchmark non-overlap.

Keep three sets separate: sealed final benchmarks, independent development retrieval/representation sets used for checkpoint selection, and training-time diagnostics. Evaluate a matched parent before continuation and preserve task/category regressions, not only an improved average. Paired bootstrap intervals on query-level retrieval scores are useful where retained judgments and task metrics permit them; they are additional uncertainty reporting, not a substitute for independent data.

Needed tools are already mostly available: pinned MTEB, official competitor adapters, GCS generation/SHA-bound exposure inventory, separate JAX/TorchXLA environments, the durable parity campaign, exact JUnit admission, and guarded GCP cleanup/accounting. Add the matched embedding adapter, complete parity qualification-bundle validator, byte/RAM admission and query-level uncertainty reporting. No new paid monitoring service is needed for these fixes. Keep expensive GitHub workflows paused until explicitly re-enabled.

## Evidence scope and verification

The retained qualification02 receipt has 14 passing physical nodes and two failing gradient-cache **optimizer-state** comparisons. It is not an all-passing trainer qualification. The historical eight-input PyTorch receipt is not acceptance of the new 72-row context/batch matrix. Historical public model-card scores are not newly rerun or matched competitor scores. Earlier actual MTEB package metadata inspection did not load the datasets or emit physical evaluation score rows.

This reviewer ran only model-free contract tests:

```text
.venv/bin/python -m pytest -q tests/test_release_contract.py tests/test_evaluation_contract.py tests/test_evaluation_integration_metadata.py tests/test_tpu_validation.py
28 passed, 9 subtests passed
```

The initial successful tests confirmed the contracts under their then-current fixtures. No numerical tolerance or physical result was changed by this audit.

## Authorized implementation follow-up

EVAL-C1 is locally remediated with parity receipt format **v4**. Release admission now requires the exact72-row fixture policy, complete equal backend scopes/output identities and correctly dimensioned tensor metrics matching configured width. The Torch publisher requires a private durable evidence directory outside the public release tree. It authenticates each case, token file, raw backend file and sidecar, recomputes every existing comparison gate, verifies all claimed metrics against the result, and rejects evidence changed during verification. Only a hashed evidence inventory and capacity receipt are added to the public release; large raw tensors stay private.

Before any array load, bounded NPY headers must declare the exact expected shape and int32 token/float32 output dtype, with matching actual body bytes. ZIP member inventories, actual uncompressed member sizes and actual retained file sizes are admitted. A synthetic tiny-member/huge-shape regression rejects before `np.load` is called. This authenticates retained implementation evidence; it is not independent attestation that a manually fabricated entire dataset of receipts ran on physical hardware.

EVAL-C2 is locally remediated in coordination with infrastructure's shared case estimator. The campaign and publication replay require explicit capacity declarations, inspect actual retained bytes, and check worker available RAM where `/proc/meminfo` is available. Publication reuses the same conservative temporary-memory formula with zero model-load copies. It requires `--parity-evidence`, `--host-ram-budget-bytes` and `--evidence-budget-bytes`; the guarded publication workflow binds a separate hashed `parity_evidence_target` and requires those flags. Operator capacity declarations remain explicit if a platform has no available-RAM observation. Actual physical peak memory and maximum-context acceptance remain pending.

The historical fixed-bucket conversion launcher is retired. Its replacement requires a verified manifest selecting the bounded durable campaign and explicit source, target, isolated interpreter, evidence and capacity flags. Conversion is a separate authenticated stage. It cannot implicitly overwrite the old GCS release or turn a single 128-token case into a full qualified matrix.

Follow-up verification: **34 model-free tests and19 subtests passed** across release, evaluation contract/integration, physical-suite admission and workflow metadata fixtures. Tests cover reduced receipt fields, reduced raw tensors with consistently rewritten hashes, altered metrics in both saved and published reports, insufficient replay budgets, and malicious NPY headers. Ruff `F,E9` and shell syntax checks passed. Full default Ruff also reports existing compact-statement style violations in the historical metadata tests; this follow-up did not reformat those unrelated fixtures. Current physical v4 acceptance, matched benchmarks and public upload are still unverified.
