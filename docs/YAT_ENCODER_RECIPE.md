# YAT multilingual encoder: evidence and experiment recipe

Status: research recommendation, September 23, 2026. No YAT encoder quality or
TPU speed advantage is established. Preserve the running physical qualification
workload and the canonical ModernBERT baseline.

## Decision

Start from released 307M mmBERT, preserve its tokenizer, positional geometry and
checkpoint-compatible backbone. Evaluate exact YAT in both FFN and attention with
independent toggles. Choose the final implementation by held-out multilingual
quality per dollar and inference latency, not kernel novelty or asymptotic cost.
A full scratch competitor is a separate, substantially larger compute project.
An embedding leaderboard win does not establish a universally better base encoder.

## Literature baseline

[mmBERT](https://arxiv.org/html/2509.06888v1) uses approximately 3T tokens:
2.3T base, 600B context extension, 100B decay, expanding 60 to 110 to 1,833
languages. Masking declines 30%, 15%, 5%; FineWeb language sampling exponents
are 0.7, 0.5, 0.3. Later stages raise data quality. Base decay variants are merged.
Its embedding comparison uses 1.25M MS MARCO hard triplets for one epoch.
These are useful controls, not evidence that compressing the recipe into a tiny
run reproduces the published base model.

[FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) covers 20 languages.
Its reported data-efficiency gains are from decoder experiments, not proof of
encoder gains. It cannot supply all languages or the paired supervision needed
for multilingual embeddings.

[Multilingual E5](https://arxiv.org/html/2402.05672v1) combines weakly supervised
pairs with labeled retrieval/NLI data, hard negatives and teacher distillation.
Its large-scale pair count is not our budget target. We adopt the distinction
between general language adaptation and task-specific embedding supervision.

## Proposed data recipe (hypothesis, not a published optimum)

Use the imported checkpoint's tokenizer unchanged. Initially preserve 512-token
windows for adaptation and 256-token sequences for the controlled embedding
pilot; test longer lengths separately and do not claim 8K quality without tests.

If architecture conversion needs MLM adaptation, start with this token mixture:

| Share | Source role |
| --- | --- |
| 60% | FineWeb2-HQ across its 20 languages |
| 20% | High-quality English web/reference text (DCLM/FineWeb-Edu candidates) |
| 15% | Filtered FineWeb2 for missing target languages, including Hindi, Urdu, Swahili, Thai, Korean and Bengali |
| 5% | Curated multilingual reference/technical text |

Pin source revisions and verify licenses, coverage and filtering before ingestion.
These proportions are explicit experimental choices. Within multilingual groups,
sample by retained token counts raised to 0.5, with an exposure/repetition cap;
compare 0.3 only if low-resource validation motivates it. Log realized tokens per
language. The opt-in language sampler now implements the fixed exponent and token-share
accounting; hard repetition caps and automatic exponent annealing remain open.
Compare 15% versus 5% MLM masking at matched input-token and compute budgets;
5% produces fewer supervised targets and must not be assumed more efficient.
Start with a measured 50M–200M-token architecture pilot; scale only on gains.
The existing ~56M-token corpus is a pipeline pilot, not the full recipe.

For embedding training, first reproduce the 1.25M-triplet control with the same
recipe for baseline and YAT. Then test 5M–10M curated examples, subject to a measured
cost cap. A proposed sampling mix is 50% retrieval/QA with mined negatives,
20% cross-lingual parallel or comparable pairs, 20% entailment/paraphrase, and
10% diverse domain/instruction retrieval. Use official training splits; examples
may come from MS MARCO, MIRACL, Mr. TyDi, NLI and suitable translation corpora.
Do not ingest a benchmark's development/test examples. Verify and record each
source's reuse rights rather than assuming a mixture shares one license.

Use InfoNCE, normalized pooled embeddings, globally gathered negatives with tested
gradients, duplicate/false-negative masks, and gradient caching if necessary.
One or two mined negatives per query is the first cost-controlled setting.
Cache teacher scores where licensed/available; avoid spending the budget on
unbounded synthetic generation. Cross-lingual pairs supplement, rather than replace,
retrieval relevance supervision. Matryoshka dimensions are a later ablation.

## Historical implementation audit (before native YAT integration)

At the time of this initial audit, `flaxchat/encoder.py` had GeGLU and standard
bidirectional attention without YAT. Native YAT is now implemented; see
[YAT mmBERT training](YAT_MMBERT_TRAINING.md) for the current entry point. Installed `nmn` is version 0.3.4; the
repository's Torch YAT port is older. The local package contains exact, fused,
and approximate attention paths, but presence does not establish suitability.

1. `constant_alpha=True` means sqrt(2) in installed NNX, versus 1 in our Torch
   port. Specify numeric alpha and model-format version explicitly before parity
   tests or checkpoint conversion. Do not silently reinterpret old checkpoints.
2. Installed Pallas YAT L1 attention exposes causal control, but no segment IDs,
   padding mask or local-window controls. Packed encoder integration requires
   these semantics in forward AND backward before use.
3. The spherical performer uses features `(a dot x)^2 / sqrt(P)` with unit-sphere
   random anchors. For unit inputs their expected inner product is proportional
   to `1 + 2*(x dot y)^2`, not just `(x dot y)^2`. Orthogonal inputs thus retain
   positive similarity that a global rescaling cannot remove. A 100,000-anchor
   CPU probe produced orthogonal/self ratio about 0.332; the target is zero.
   Evidence: `artifacts/yat-encoder-design-0923/probe.json`. Do not use this as
   a faithful approximation of exact spherical YAT. Clipped random features and
   finite quadrature introduce further approximation choices needing validation.

## Recommended exact architecture candidates

FFN candidate: retain the GeGLU parameter shapes. Replace the GELU branch with
`YAT(x, W_a)` and retain the signed linear gate `x W_g` and down-projection.
This is a gated-YAT proposal, not an existing proven optimum. Use an explicit
positive epsilon floor and logged scale parameters. Also test an ungated
YAT-plus-linear FFN with 1.5 times the GeGLU intermediate width for approximately
matched matrix parameters (2*h*m versus 3*h*f). Compare cost as well as size.

Attention candidate: preserve learned Q/K/V/output projections, RoPE and alternating
local/global bidirectional attention. The first exact spherical candidate uses
`u=q/max(norm(q), delta)`, `v=k/max(norm(k), delta)` and
`score=(u dot v)^2 / (max(norm(u)^2+norm(v)^2-2*u dot v, 0)+epsilon)`.
Using actual norms handles near-zero normalization safely. Mask scores first,
then divide by row sum; zero-mass/all-masked rows must return zero with finite
gradients. This L1 operator differs from softmax applied to YAT scores: test the
latter as a separate control. A common positive multiplicative score scale cancels
under exact L1 normalization; do not add a meaningless learnable temperature.

Use BF16 matrix operands with FP32 accumulation, distance/norm calculation,
normalizers and reciprocal initially. Compare against an FP32 reference around
near-coincident, antipodal, orthogonal and zero vectors. Very small epsilon can
magnify errors; positivity alone is not a stability guarantee. Keep the validated
residual precision policy until measured evidence supports changing it.

[SLAY](https://arxiv.org/html/2602.04915v1) studies spherical YAT and linearization.
[Aether](https://arxiv.org/html/2603.12276v1) reports an unnormalized YAT FFN and
attention decoder experiment and normalization ablations. These are distinct
operators/settings; its normalization failure is not proof that normalization
can never coexist with any YAT variant. Test normalization placement explicitly.
Do not remove all normalization from an imported encoder based on that claim.

Build dense reference and gradients first, then tiled Pallas exact attention
with segment/local masks and recomputation backward. Stock Splash attention
cannot be assumed to implement this changed scoring/normalization. Exact tiling
saves attention-matrix memory but remains quadratic for global layers; approximate
linear attention comes only after a correct, measured reference.

## Conversion and decision gates

For a pretrained conversion, temporarily blend old and new block outputs with an
explicit schedule from zero to one. Train new branches using block-output/hidden
state distillation plus MLM, then evaluate the fully converted model. A zero gate
blocks task-loss gradients to the new branch, so use its separate distillation
loss or advance the schedule; do not leave it frozen accidentally. Teacher
inference and dual branches add cost and must be measured.

Run baseline, YAT-FFN-only, YAT-attention-only, and both with identical data,
seeds, tokenizer, evaluation and comparable budgets. First use short pilots, then
repeat promising candidates over three seeds. Compare both equal-token quality
and equal-dollar quality; parameter matching alone is insufficient.

Success requires held-out multilingual MTEB/retrieval gains with per-language
reporting, plus XNLI and token-level NER/POS retention, longer-context checks,
stability and inference/memory measurements. Fix benchmark versions and exclude
development-selected results from final test claims. Report prior checkpoint
training exposure separately from the newly audited data. No leaderboard gain,
speedup, or production readiness is established by this proposal.

## First implementation

The `yat_glu` FFN candidate is now implemented as an opt-in encoder configuration,
with fixed numerator bias 1, fixed epsilon 0.01, and trainable per-block alpha
initialized to 1 (the requested revision supersedes the initial fixed-alpha variant). It preserves matrix layout
and the linear gate while adding one scalar parameter per block, computes sensitive geometry in FP32, and records the variant
in checkpoint identity. Dense formula and gradient tests match direct Euclidean
differences. FP32/BF16 local training and exact resume pass; changing FFN type at
resume is rejected. Attention remains the canonical baseline; neither fused YAT
attention nor distillation conversion is implemented by this change. The baseline
code description above records the state at the start of this investigation.

The focused baseline/reference/YAT suite passed 38 tests (3 skipped); the YAT
formula and training/resume tests also passed all 9 cases on four simulated CPU
devices. These are CPU checks, not physical multi-host TPU evidence.

The exact raw-YAT/softmax attention control is now available as
`--attention-score yat_softmax`. It keeps fixed bias 1, epsilon 0.01 and a
separate trainable scalar alpha per block. It preserves the canonical masks and
RoPE, and uses a dense XLA reference with expanded norms. The spherical/L1 and
fused-Pallas candidates in the research plan remain unimplemented. Combined
YAT FFN+attention CPU training and exact resume pass; no quality advantage is claimed.
