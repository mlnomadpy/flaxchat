# Bidirectional encoder training

FlaxChat has a separate ModernBERT-compatible encoder path. The existing GPT
engine and chat stages remain causal. Use the encoder commands below, not
`scripts.pretrain` or `flaxchat.initialize`, for masked language modeling (MLM).

## Implemented

- Bidirectional global attention every third layer and symmetric local attention.
- ModernBERT's split-half RoPE, first-layer norm exception, GELU GLU blocks,
  prediction head, tied token embeddings, and decoder bias.
- Float32 master parameters and optimizer state. Compute and residual precision
  are independently configurable. The trainer defaults to BF16 compute and FP32
  residuals; the model constructor retains FP32 for reference diagnostics. BF16 numerical
  comparisons are diagnostic, not downstream quality measurements; block rematerialization; chunked,
  rematerialized vocabulary projections for loss. The default one-document-per-row
  attention mask is linear in sequence length. XLA attention itself can still have
  quadratic memory/work; this is not a claim of FlashAttention-like efficiency.
- Explicit segment IDs and optional position IDs for packed-document isolation.
  The data preparer deliberately uses one document per row, with truncation.
- Deterministic BERT 80/10/10 masking keyed by seed, step, and stable row ID.
  Padding and special tokens are excluded from targets and random replacements.
- Data-parallel training on a global mesh with replicated parameters and optimizer.
  Loss is normalized by the global selected-token count, not per-device means.
- Integrity-checked Orbax checkpoints including cursor and optimizer update count.
  Resume checks architecture, horizon, batch size, masking, tokenizer, data manifest,
  learning rate, seed, and implementation hashes. Empty-mask batches skip updates.
- Strict local safetensors import, with names, shapes, finiteness, and tied weights
  validated before changing live parameters. No remote model code is executed.
- Deterministic held-out MLM evaluation, weighted by selected-token count, excluding
  exact unpadded training rows and duplicate validation rows. This is not substring,
  near-duplicate, semantic, or benchmark decontamination.
- `ModernBert.encode(...)` token representations and `ModernBert.pool(...)` masked
  mean pooling. Pooling does not turn a base MLM into a trained retrieval model.

## Use a pretrained mmBERT snapshot

Install `flaxchat[encoder]` to read safetensors (the Pixi environment also includes
it). Obtain a local snapshot of `jhu-clsp/mmBERT-base` pinned to an immutable HF
revision, including `config.json`, `tokenizer.json`, and all safetensors shards.
The pinned mmBERT release currently supplies `pytorch_model.bin`. Convert it
locally with the weights-only PyTorch loader before TPU setup:

```bash
python -m scripts.convert_encoder_checkpoint artifacts/mmbert-base
```

Conversion requires PyTorch and safetensors (both available in Pixi), records
source/output checksums, checks tied embeddings and finite values, and refuses
to overwrite an existing safetensors file. Keep that snapshot's tokenizer: the trainer checks its checksum against the data.
The example uses `artifacts/mmbert-base` as the snapshot location.

Export a bounded, revision-pinned selection of FineWeb2-HQ to JSONL with one `text`
field per document. Select language proportions before this step and keep separate
training and held-out documents. Do not download the supplied embeddings for MLM.
Prepare data on CPU before renting TPU time:

```bash
python -m scripts.prepare_encoder_data \
  --source artifacts/train.jsonl \
  --tokenizer artifacts/mmbert-base/tokenizer.json \
  --output artifacts/encoder-train --sequence-length 512 --max-rows 10000 --mask-token-id 4

python -m scripts.prepare_encoder_data \
  --source artifacts/heldout.jsonl \
  --tokenizer artifacts/mmbert-base/tokenizer.json \
  --output artifacts/encoder-validation --sequence-length 512 --max-rows 1000 --mask-token-id 4

python -m scripts.train_encoder \
  --config artifacts/mmbert-base/config.json \
  --pretrained artifacts/mmbert-base \
  --data artifacts/encoder-train --output artifacts/encoder-checkpoints \
  --steps 100 --batch-size 8 --save-every 25 \
  --learning-rate 2e-5 --mask-probability 0.15 \
  --dtype bfloat16 --residual-dtype float32 --loss-chunk-size 128

python -m scripts.evaluate_encoder \
  --checkpoint artifacts/encoder-checkpoints \
  --train-data artifacts/encoder-train --data artifacts/encoder-validation \
  --max-rows 1000 --output artifacts/encoder-evaluation.json
```

For BF16 matrix multiplies with FP32 residuals, use
`--dtype bfloat16 --residual-dtype float32`. For BF16 embeddings, normalization
outputs, and residual activations as well, use
`--dtype bfloat16 --residual-dtype bfloat16`. Normalization reductions, loss,
master parameters and optimizer state retain FP32 for numerical stability.
This is BF16 activation training, not literally all-BF16 arithmetic or storage.
Changing either precision setting changes checkpoint identity and cannot silently
resume a run with a different policy. Compare against a reference with the same
precision policy, then assess held-out loss and TPU throughput before promotion.

Append `--resume` to the same training command after interruption. `--stop-after`
can exercise a controlled early stop without changing the declared horizon.
Checkpoint format 3 stores the complete NNX optimizer state, including its
update counter as well as Optax moments and schedule counters. Format 2 remains
readable for model-only loading, but full optimizer resume requires an explicit
migration with a verified NNX update counter: older checkpoints omitted that
counter, and checkpoint labels need not equal optimizer update counts.
The initial training policy is fixed learning rate and sequential cyclic rows;
this is continued MLM training, not a reproduction of mmBERT's full recipe.
Scratch training accepts native `EncoderConfig` JSON without `--pretrained`;
`configs/encoder/tiny.json` is an architecture example, whose tokenizer must match.

## Distribution and GCP

The global batch must divide evenly across all devices. On multiple TPU workers,
launch the same command on every worker with `--distributed`, identical local data
and snapshot files, and a common `gs://` checkpoint directory. An explicitly shared
filesystem is also supported through `--shared-local-checkpoints`; ordinary
worker-local directories are not shared.

Use `--attention-backend xla` for multi-device training. The experimental Splash
backend accepts lengths divisible by 128 on a single TPU device; multi-device
Splash is rejected because automatic partitioning is not supported by this path.
Physical TPU correctness/performance must be measured before claiming qualification.

The training command does **not** provision or delete TPUs. Run it under the
existing external Spot budget/deadline and cleanup supervisor. JSON step logs
include loss, selected tokens, total tokens, update acceptance, and elapsed time;
the first step in each process is marked `includes_compilation` and should be
excluded from steady-state throughput. Checkpoint time is logged separately.
The training-step rate excludes checkpoint I/O and is not end-to-end billed
throughput. Startup logs record compute/residual precision and device topology. Do not reuse GPT benchmark rates to estimate
encoder cost. Start with a short 512-token pilot, measure throughput and checkpoint
time, and then decide whether a larger batch/context is worthwhile.

## Validation and limits

```bash
pixi run python -m pytest tests/test_encoder.py tests/test_encoder_training.py -q
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  pixi run python -m pytest tests/test_encoder_training.py -q
FLAXCHAT_RUN_DISTRIBUTED_CPU=1 \
  pixi run python -m pytest tests/test_encoder_training.py -k two_process -q
# On an actual TPU device:
pixi run python -m pytest tests/test_encoder.py -m accelerator -q
```

Tests compare tiny-model forward outputs, MLM loss, and an attention-weight gradient
against Hugging Face ModernBERT; check future-token visibility, local windows,
padding, document boundaries, chunked-loss gradients, rematerialization, masking,
and exact training restart. The reference test requires PyTorch and Transformers
(included in Pixi); minimal installations skip it.

The TPU verification workload defaults to a 512-token BF16-compute/FP32-residual
pilot. Use `--lengths 128 512 2048 8192` only for a deliberate context sweep.
Use a fresh GCS prefix per campaign; residual variants have separate checkpoint
paths. A successful smoke/recovery summary explicitly does not qualify model
quality. Reference files require matching snapshot checksums; regenerate older
files without provenance using `scripts.validate_encoder_checkpoint reference`.
Comparing different precisions requires `--diagnostic`, whose report has
`passed: null` and preserves the numerical gate result without asserting parity.

See [precision evidence and decision](ENCODER_PRECISION.md).

Remaining work before a production mmBERT campaign: released-weight BF16 downstream
quality qualification, broader physical multi-host/topology acceptance, encoder FSDP,
nonzero dropout, supervised classification/retrieval training and downstream
benchmarks, stronger decontamination, and corpus/language-mixture scheduling.
Single-host physical TPU and projection measurements are recorded in
[TPU verification](ENCODER_TPU_VALIDATION.md) and
[projection benchmarks](ENCODER_PROJECTION_BENCHMARK.md). Those execution and
numerical checks do not establish production quality. No cloud resources are needed for the
local tests above.

Architecture references: [mmBERT model/config](https://huggingface.co/jhu-clsp/mmBERT-base),
[Hugging Face ModernBERT](https://github.com/huggingface/transformers/tree/main/src/transformers/models/modernbert).


Before provisioning a TPU, run the training command with `--preflight-only` on
CPU to validate local configuration, tokenizer, and data identity without creating
checkpoints or allocating model parameters. This does not validate model execution
or checkpoint resume. The preparer explicitly protects `--mask-token-id` even when
the tokenizer JSON marks that token non-special, as the released mmBERT tokenizer
does. Use the model config's mask ID when preparing other snapshots.

TPU validation separates CPU regression checks from accelerator tests and rejects
CPU fallback. BF16 Splash diagnostics use native matmul precision; the FP32
reference gate uses highest precision. After a documented full CPU-suite pass,
`--cpu-tests focused` reruns the affected encoder/launcher tests and records that
narrower scope in the report.

## Experimental masked-position projection

`train_encoder --mlm-projection masked` compacts selected MLM positions before
the prediction head and full-vocabulary softmax. Inference without labels still
returns dense logits. The default remains `dense` until physical TPU throughput
and training validation qualify the optimization.

Compaction is per row, with static capacity equal to one quarter of sequence
length rounded up to `loss_chunk_size` (capped at sequence length). At length
512 and chunk size 128, the head projects 128 rows per sequence instead of 512.
Ignored and padding targets are removed before compaction; unused buffer entries
have zero loss weight. If any row exceeds capacity, the entire batch uses the
dense path, preserving all Bernoulli-selected targets and their normalization.
The objective is unchanged; floating-point summation order can differ.

`--mlm-projection-capacity N` optionally sets this per-row capacity independently
of `--loss-chunk-size`; it requires masked projection and is capped at sequence
length. Omitting it preserves the existing rule. The resolved value is part of
checkpoint identity, so changing it during exact resume is rejected.

For example, sequence length 128 with chunk size 32 normally reserves 32 targets per row;
capacity 64 retains chunk size 32 computation while reducing whole-batch fallback at
large batch sizes. Under independent 15% masking of 128 eligible positions,
capacity 32 has about 0.105% per-row overflow probability, which becomes about 42%
for 512 rows because any row triggers fallback. These are analytic estimates;
padding and special tokens lower the eligible count. Measure actual fallback,
memory, throughput and numerical drift before selecting a production capacity.

Local tests compare loss and every parameter gradient for FP32 and BF16,
empty masks, padding, and overflow; they also verify exact checkpoint resume and
sharded global loss. See the [matched TPU measurements](ENCODER_PROJECTION_BENCHMARK.md) for measured
speedups and their limits. Benchmark both modes
at identical model, batch, sequence, mask seeds, precision, and optimizer settings,
separating compilation and checkpoint time from steady-state throughput. Record
overflow frequency, peak memory, and end-to-end cost per input token. Projection
mode is part of checkpoint identity; switching modes during resume is rejected.

This follows [ModernBERT sparse prediction](https://huggingface.co/docs/transformers/v4.52.1/en/model_doc/modernbert).
Static selection sizes are needed under [JAX JIT](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html).
Vocabulary-tiled fused cross-entropy is available experimentally with
`--mlm-loss-backend pallas --mlm-vocab-tile 1024` on TPU. The `xla_full` backend
provides an explicitly sharded unchunked XLA control. Both support masked
projection. Checkpoint evaluation replicates restored parameters onto the same
data mesh and pads partial evaluation batches without counting padding labels.
Backend and tile size are checkpoint identity fields. The custom TPU kernel is
inspired by [Cut Your Losses](https://arxiv.org/abs/2411.09009); its published GPU
implementation is not a drop-in TPU kernel. Sampled softmax and low-rank decoder replacements
change the objective or architecture and are not enabled for checkpoint replication.

## Measured TPU projection setting

For the tested four-chip v5p slice at batch 64 × sequence 512,
`--mlm-projection masked --mlm-loss-backend xla_full` reached 252K input tokens/s,
versus 233K for Pallas and 53K for the dense baseline. Both optimized backends
passed the released-model numerical gates after sharing the head transform
before loss chunking. This is a synthetic throughput measurement, not a
production-quality training result. See [the full protocol, costs and remaining
qualification](ENCODER_PROJECTION_BENCHMARK.md).

The September 23 real-text campaign adds stricter limits to that recommendation:
Pallas fails the unchanged loss-parity gate on its XNLI fixture, and `xla_full`
at batch 64 exceeds four-chip v5e HBM. Canonical chunked `xla` completes 500
updates and exact interruption recovery on that v5e slice. See the
[current qualification report](ENCODER_SCALE_RESULTS_2026_09_23.md); earlier
synthetic passes do not qualify an arbitrary dataset, batch or TPU generation.

### Snapshot preflight and numerical identity

Before model construction, `--preflight-only` checks that safetensors shards
exist, have readable structure, match their index, and contain no duplicate tensor
names. Workers compare weight hashes and numerical runtime identity alongside the
recipe and dataset. Weight import still validates architecture-specific names,
shapes and finite values. Preflight success alone is not model-quality validation.

Encoder resume now checks the recorded runtime as part of recipe identity. Earlier
checkpoints without that field require an explicit compatibility/migration path;
changing runtime versions must not silently claim exact continuation.

## Repair options and precision contract (2026-09-23)

The decoder uses BF16 matrix operands with **explicit FP32 projection output and
FP32 bias/loss reductions**. FP32 residuals and master parameters remain the
recommended defaults. The previous cast-to-BF16-then-FP32 expression did not have
consistent lowering: on the measured TPU, XLA retained FP32 logits while Pallas
rounded them. This repair makes the contract explicit in both implementations;
it is not a relaxation of parity thresholds. Source identity changes deliberately
prevent silent resume across this numerical change.

`--mlm-loss-backend xla_local` scans vocabulary projection chunks **inside each
data device**, rematerializes their logits during backward, and globally sums the
loss numerator and target count. It keeps FP32 master-weight gradient accumulation.
It is an opt-in, memory-bounded alternative to `xla_full`; canonical `xla` remains
the default until the physical comparison is complete. `xla_full` can still exceed
HBM on large vocabulary/batch combinations.

Optional recipe controls:

- `--accumulation-steps N`: global batch must divide by `N * global_device_count`.
  Microbatch gradients are weighted by their selected MLM token count, not by the
  number of microbatches. The optimizer advances once per effective batch;
  entirely empty-mask batches do not update or advance its schedule.
- `--lr-schedule cosine --warmup-steps N --final-lr-ratio 0.05`: schedule advances
  with accepted optimizer updates. Warmup must be shorter than the declared
  training horizon. Defaults retain constant learning rate.
- `--shuffle`: deterministic epoch permutations, including batches crossing an
  epoch boundary, reconstructed from the checkpoint cursor and seed. This is
  uniform row sampling, not a language-mixture curriculum.
- `prepare_encoder_data --long-document-policy window`: retain successive
  tokenizer overflow windows from long documents instead of dropping their tails.
  Windows never combine different documents; this is not document packing.

These settings participate in checkpoint identity. Exact resume requires the
same horizon, batch, schedule, accumulation and data-order policy. A stop before
the restored cursor is an error; an already completed stop emits
`already_completed`. Preparation reports padding utilization, and training reports
both input positions/sec and **nonpadding tokens/sec**; compare cost with the
latter when corpora have different lengths.

A small XNLI-premise fixture is suitable for infrastructure tests, not continued
pretraining. The observed XNLI test subset must not select learning rates or data
mixtures. Use representative, licensed multilingual training text and a separate
development split, lock acceptance thresholds before evaluation, and retain an
untouched downstream test set. These recipe controls do not establish production
quality, encoder FSDP, or qualification of every TPU size.

## Experimental YAT-gated FFN

`--ffn-type yat_glu` replaces the GELU feature branch with an exact YAT
transformation while retaining the signed linear gate and output projection:

```text
s = x @ W_a
d = max(sum(x*x) + sum(W_a*W_a, axis=input) - 2*s, 0)
FFN(x) = W_out [ (alpha * (s+1)*(s+1) / (d + 0.01)) * (x @ W_g) ]
```

Here `x` is the existing block's normalized FFN input. `W_a` and `W_g` are
halves of the existing input projection; matrix parameter names and shapes stay
unchanged. Each YAT block adds one trainable FP32 scalar alpha. `geglu` remains the default. Reusing pretrained weights initializes a
new architecture; it does not preserve the original checkpoint's predictions.
The existing attention, residual precision and normalization remain in place.

Add these flags to an otherwise validated `scripts.train_encoder` command:

```text
--ffn-type yat_glu --yat-epsilon 0.01 --yat-alpha 1.0
```

Equivalent encoder config fields are `ffn_type`, `yat_epsilon`, and `yat_alpha`.
Omitted CLI options preserve values in the config. The numerator bias is fixed at 1 and epsilon is fixed at 0.01; neither
is a trainable parameter. Alpha is a trainable scalar per block, initialized
from `yat_alpha` (default 1). The config records `yat_bias=1` and
`yat_alpha_trainable=true`; incompatible settings are rejected for YAT GLU.
Numeric initialization avoids conflicting boolean-alpha conventions in older
implementations. This initialization does not guarantee GeGLU activation magnitude
or superior quality. Existing checkpoints of the earlier fixed-alpha formula
require explicit conversion; exact resume does not reinterpret them.

BF16 operands use FP32 dot accumulation, squared norms, distances and division;
the gated output is cast back to the compute dtype. Checkpoint identity records
all configuration fields and rejects changing FFN type during exact resume.
This is a dense JAX implementation, not a fused Pallas FFN or YAT attention.
Formula/gradient checks and FP32/BF16 training with exact checkpoint resume have
passed locally. Physical TPU performance and downstream quality remain untested.

The implementation is internal JAX/Flax; it does not import `nmn`. A combined
input projection computes both feature and gate dots in one GEMM. The distance
uses squared-norm expansion, avoiding a token-by-input-by-neuron difference
array. The elementwise geometry is rematerialized in backward while keeping the
GEMM outside that boundary. Standard autodiff preserves the clamp subgradient
and BF16 cast semantics, avoiding an unverified hand-written backward rule.
A CPU forward/backward comparison against the unrematerialized arithmetic passed;
compiled temporary allocation was unchanged at the measured shape. This is not
evidence of a TPU speed or memory improvement. The fixed-bias/trainable-alpha
revision passes 10 formula, gradient, training and exact-resume tests on four
simulated CPU devices, plus lint and type checks.

## Language-aware corpus sampling

For corpus exports with verified document/language provenance, use
`--language-exponent 0.5`. A language with `n` retained nonpadding tokens gets an
expected token share proportional to `n**0.5`. Row-selection probabilities are
corrected for mean row occupancy, so short padded rows do not distort the token
mixture. The option requires an explicit training split and verified document
hash, row coverage, language inventory and actual token counts. It cannot be
combined with epoch shuffle.

Sampling is with replacement and reproducible from seed, global step and global
batch size. All workers select the same global IDs before taking their own slice.
Resume identity includes the exponent. The `language_sampling` event records
target shares and expected repeated row exposure; each training event records
realized nonpadding token counts by language. These counts include special tokens
consistently with the existing nonpadding throughput metric. Repetition is reported,
not hard-capped; this is a fixed per-run sampling policy, not a complete annealed
multi-stage language curriculum.

## Experimental exact YAT attention

Add `--attention-score yat_softmax --ffn-type yat_glu` to enable both native
YAT components. Attention computes:

```text
distance(q,k) = max(sum(q*q) + sum(k*k) - 2*(q @ k), 0)
score(q,k) = alpha_attention * ((q @ k) + 1)**2 / (distance(q,k) + 0.01)
attention = softmax(masked_scores) @ values
```

The bias is fixed at 1 and epsilon at 0.01. Each block has a separate trainable
FP32 attention alpha initialized from `yat_alpha`. This is softmax of raw YAT
scores, not spherical normalization, linearized attention or L1 normalization.
Softmax makes a common trainable score scale meaningful; it would cancel under
exact L1 normalization. Standard projection/RoPE and local/global layer scheduling
are retained. Masks preserve document boundaries, padding and local windows;
padded queries and all-padding rows return zero with finite gradients.

The exact XLA reference uses one QK contraction plus squared norms, avoiding
five-dimensional explicit pairwise differences, but still materializes quadratic
attention scores. It rejects the Splash backend rather than silently using
ordinary dot-product attention. It does not depend on `nmn`. No fused or
linear-time implementation is implied. The existing `dot_product` score remains
the default. Matrix weights can be imported from mmBERT, while the added alpha
parameters initialize locally; this changes predictions and needs adaptation.

The combined FFN/attention suite passed 16 tests on four simulated CPU devices,
including independent distance/gradient checks, masks, padding, BF16 training and
exact checkpoint resume. These results do not establish physical TPU correctness,
throughput, long-context memory feasibility or downstream model quality.


### Experimental BF16-only YAT forward arithmetic

Select `--dtype bfloat16 --yat-compute-mode bf16` with
`--ffn-type yat_glu` and/or `--attention-score yat_softmax`. The native implementation
in `flaxchat/yat.py` computes squared differences in feature blocks of eight,
avoiding cancellation in the squared-norm identity. Distances, reductions,
YAT numerator/division and attention softmax use BF16 array arithmetic. Bias stays
fixed at 1; fixed epsilon 0.01 rounds to its BF16 representation (0.010009765625).
Alpha remains trainable, with an FP32 master parameter cast for forward evaluation.
Master weights, optimizer state and the configured residual dtype are unchanged.
This is not a claim that TPU hardware dot-product accumulators use BF16 internally.

Twenty YAT tests passed on four simulated CPU devices, including StableHLO checks
with no FP32 types in the isolated forward primitives, near-collision distances,
finite gradients, masking and exact training checkpoint resume. This is **not TPU
qualification**. On CPU, a 128-token, width-768/intermediate-1152 FFN forward/backward
microbenchmark measured 144.6 ms versus 2.29 ms for mixed arithmetic, and 23.99 MB
versus 13.07 MB compiler-reported temporary memory. Relative forward L2 drift was
1.11%, and input/weight gradient drift 2.41%/2.17%, against mixed arithmetic using
the same BF16 operands. These are one synthetic workload's results, not quality
or TPU throughput predictions. Keep `mixed` as the production default: direct
BF16 distances are an experimental numerical reference, **not a speed improvement**.
A tiled TPU kernel and physical measurements are still needed before promoting it.


A subsequent same-input CPU comparison separated algorithm choice from dtype:
mixed norm identity measured 2.29 ms; BF16 norm identity 3.25 ms; blocked BF16
direct differences 146.91 ms (forward/backward, same 128/768/1152 shape).
BF16 norm identity had 0.77% relative forward L2 drift, versus 1.11% for direct
BF16 differences. However, the BF16 identity rounded the true squared distance
between [64,64,1] and [64.5,64,1] from 0.25 to zero. Thus it is a promising fast
candidate, not a validated replacement. The measured slowdown mainly reflects
explicit pairwise distance work rather than a requirement of BF16. This probe
lives in `artifacts/yat-encoder-design-0923/benchmark_bf16_identity.py`; the fast
BF16 identity has not been promoted into the production model.


### Adaptive BF16 YAT distance

Use `--dtype bfloat16 --yat-compute-mode bf16_adaptive` to enable a faster native
BF16 implementation for YAT FFN and/or attention. Bias 1, epsilon 0.01 and the
trainable alpha semantics remain unchanged. No NMN dependency is introduced.
The mode is recorded in model configuration and checkpoint compatibility checks.

`--yat-ffn-compute-mode bf16_adaptive` instead overrides only the YAT FFN.
For example, keep `--yat-compute-mode bf16` for centered attention while testing
the adaptive FFN. Omission preserves the shared mode. The override requires a
YAT FFN and rejects a direct-only backward tile when selecting adaptive mode.
It is covered by CPU composition and exact-resume tests; changing it on resume
is rejected. Physical performance and sustained numerical behavior of this
composition still require validation.

It reuses the combined FFN projection (or QK contraction), computes norms in BF16,
and uses direct coordinate differences when the computed distance is at most
one eighth of the sum of squared norms. This threshold is a numerical heuristic,
not a proven error bound. A global conditional skips all fallback work when no
pair is sensitive. Otherwise 16-by-64 tiles selectively recompute distances,
with autodiff and rematerialization rather than an approximate custom gradient.
For feature widths up to 128, at least one quarter of tiles requiring repair
switches to a single dense direct pass. Wider FFN inputs retain the tiled path
to avoid reserving a large dense backward buffer. Do not vmap these conditionals
without remeasuring: batching can evaluate both branches.

CPU forward/backward measurement at 128 tokens, width 768, intermediate 1152:
mixed 2.38 ms, direct BF16 151.77 ms, adaptive BF16 3.39 ms (about 45x faster than
direct BF16, still slower than mixed). Relative output L2 drift against mixed
fell from 1.11% to 0.77%. Compiler temporary allocation was 15.83 MB versus
23.99 MB direct and 13.07 MB mixed. The near-collision example's distance is
correctly recovered as 0.25. Isolated forward StableHLO has no FP32 types; master
parameters, optimizer and residual policy are unchanged.

Performance depends on fallback frequency. With one injected close FFN pair,
adaptive took 36.64 ms versus 162.48 ms direct. Random attention (B1/L128/H4/D64)
took 1.80 ms versus 6.04 ms direct; equal Q/K took 5.94 ms versus 5.79 ms direct.
Attention temporary allocation increased to 7.60 MB versus 6.55 MB direct due
to the fallback branches. These synthetic CPU probes establish neither TPU
speed nor model quality. `mixed` remains the default; `bf16` retains the original
direct reference. Physical TPU qualification of `bf16_adaptive` remains open.

Final verification: 27 YAT tests passed on four simulated CPU devices, including
sparse/dense fallback gradients, partial tiles, broadcast batches, compiler dtype
checks and exact training resume; 39 baseline/CI tests passed with 3 optional
skips. Lint and type checks passed for the changed implementation.


### Retrieval metric evaluation

`python -m scripts.evaluate_retrieval_embeddings --data <directory> --output <report.json>`
scores precomputed encoder embeddings against the full supplied corpus in bounded
query/document blocks. The directory contains float `queries.npy` and `corpus.npy`
matrices and a `manifest.json` with:

- `format: "flaxchat-retrieval-embeddings-v1"`, `split: "validation"` or `"test"`;
- nonempty `dataset`, pinned `revision`, `model_identity`, and `pooling` strings;
- `queries_sha256`, `corpus_sha256`, ordered unique `query_ids` and `document_ids`;
- `qrels`: query ID to document ID to nonnegative integer relevance grade;
- `languages`: one nonempty language label for every query.

Every query must have a positive judgment and all judged documents must exist
in the corpus. Missing queries/documents, truncated rankings, duplicate IDs,
zero embeddings and nonfinite embeddings are rejected. Inputs are rehashed after
evaluation to detect mutation. Cosine normalization precedes ranking; exact score
ties break by ascending document ID. The implementation computes Recall@k,
MRR@k and linear-gain nDCG@k, with query means, per-language means and an unweighted
language macro. Linear gains follow the [trec_eval nDCG convention](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c);
tie handling is explicit and no complete pytrec_eval parity claim is made.
Unjudged documents count as nonrelevant; held-out queries are never silently omitted.

Fourteen CPU tests cover dense-ranking equivalence across block sizes, ties,
graded gains, language aggregation, invalid judgments and checksum tampering.
This is metric infrastructure: the report explicitly keeps `quality_qualified`
false. Model identity is supplied provenance, not independently verified
checkpoint execution. Real checkpoint embedding export, representative benchmark
runs, confidence intervals and production acceptance remain required.


### Export embeddings from a harness checkpoint

`python -m scripts.export_encoder_retrieval --checkpoint <checkpoint-root>
--queries <prepared-queries> --corpus <prepared-corpus> --judgments <task.json>
--output <fresh-export-directory> --batch-size 8` restores a pinned checkpoint
step, exports mean-pooled nonpadding token states (including special tokens),
and writes the checksum-pinned inputs consumed by the retrieval evaluator.

The task JSON supplies `dataset`, `revision`, held-out `split`, a nonempty
`tokenization_policy`, ordered `query_ids`/`document_ids`, `qrels` and `languages`.
Both prepared-row manifests must match that dataset/revision/split and the
checkpoint tokenizer/special-token policy. Supply one query or document per row;
this exporter does not reconstruct multi-window documents. Record any truncation
in the tokenization policy. Empty inputs, invalid judgments and identity mismatches
are rejected before model export. The destination must be fresh and is published
only after all batches and input-rehash checks succeed.

The exporter restores the exact step selected when reading metadata, hashes actual
model parameters, and combines their digest with encoder configuration, pooling
policy and implementation digests in `model_identity`. This distinguishes weights
used with different numerical/kernel settings. It currently supports one process
and XLA-attention checkpoints. It does not train a retrieval objective or claim
benchmark/model quality. A real tiny-checkpoint end-to-end test verifies restored
outputs, partial-batch padding, evaluation rankings and provenance rejection.

### Complete bitext evaluation from checkpoint exports

Prepare each pinned bitext shard with `scripts.prepare_encoder_bitext`, then use
`scripts.export_encoder_retrieval` with that subset's `queries`, `corpus`, and
`judgments.json`. Keep one output directory per subset under a shared embedding
root. Score every declared subset in one command:

```bash
python -m scripts.evaluate_encoder_bitext \
  --embeddings /path/to/checkpoint-embeddings \
  --inventory artifacts/encoder-retrieval-0923/prepared-inventory.json \
  --output /path/to/bitext-report.json
```

The scorer requires all declared subsets, matching pair counts, one checkpoint
identity and pooling policy, the pinned dataset revision, and the official test
split. It verifies embedding checksums and original one-to-one pair alignment.
Reports retain predicted corpus indices, per-subset support-weighted precision,
recall, F1 and accuracy, plus the unweighted subset mean. Missing or inconsistent
subsets fail instead of silently producing a partial average.

This is a native harness evaluation with deterministic document-ID tie breaking.
Upstream MTEB uses torch.topk, whose tie behavior has not been matched here; the
report therefore explicitly sets `official_mteb_parity=false` and
`quality_qualified=false`. Checkpoint execution provenance, contamination checks,
and production acceptance criteria remain separate requirements. Do not use the
held-out test results to select model hyperparameters.


Experimental local accumulation: pass `--local-gradient-accumulation` with `--accumulation-steps N` to accumulate per-device masked-token numerators and FP32 gradients before global reduction. This preserves global target weighting, including empty microbatches, and records the mode in checkpoint identity. It keeps model parameters replicated. CPU correctness and exact resume are tested; physical TPU throughput, memory and full-model BF16 drift remain unqualified. The default accumulation mode is unchanged.

Checkpoint manifest transfers use16MiB by default on a single TPU device and1MiB elsewhere. Override with `--checkpoint-manifest-chunk-mib` (1–64). A307.8M-parameter physical v5e comparison reduced paired local checkpoint save time from80.38s to65.90s while preserving exact model/optimizer/training state and exact interrupted resume. This does not establish GCS or multi-host performance. The checkpoint format and default restore hash validation are unchanged. Evidence: `artifacts/yat-checkpoint-full-tpu-v5e1-0925/analysis.json`.

`--host-gc-diagnostics` observes Python collection durations without changing collector behavior. `--nnx-jit-partial` binds the model and optimizer once to reduce per-step Python traversal. It remains opt-in and checkpoint-identity protected: use the same flag when resuming.

Physical v5e single-chip qualification on September 25 used a 307.8M-parameter adaptive YAT encoder, batch 8, sequence length 128, with 32 synthetic training steps and interruption/resume at step 16. Baseline, candidate and resumed model/optimizer/training-state manifests matched exactly. Across 28 unprofiled warm steps, aggregate throughput increased from 7,582 to 9,394 tokens/s (23.9%); median step time fell from 123.79 to 108.86 ms. No outliers were removed. The three profiled TPU training modules covered 316.98 versus 316.74 ms, indicating a host-overhead improvement rather than faster device kernels. Candidate warm steps recorded zero GC collections, versus 281 in the baseline.

Compilation-flagged first steps still took about 98 seconds. Two checkpoint saves took 78.59 seconds baseline versus 80.81 seconds candidate; this dispatch change did not demonstrate faster checkpointing or lower total cost for the short qualification run. These results qualify the measured single-chip configuration, not multi-host, v6, long-context or sustained model quality. Evidence: `artifacts/yat-nnx-partial-tpu-v5e1-0925/analysis.json` and the paired training trace summaries.

The primitive `centered_yat_attention(..., compensate_kv=True)` exposes an experimental BF16 correction for cross-query K/V accumulation; the encoder configuration does not enable it. An 18-case physical v5e study preserved forward/Q/alpha results and passed padding and constant-value null-gradient checks. It reduced error against a diagnostic accumulation reference for global attention, but worsened local V error by up to 17.5%; at 8,192 tokens it ran about 4.6% slower and increased compiler temporary buffers from 64 MB to 153 MB. Keep the default `False`; this is not a general speed or accuracy improvement. See `artifacts/yat-kv-compensation-tpu-v5e1-0925/RESULTS.md` for the reference's limits and outstanding qualification.

A subsequent five-case TPU diagnostic using shared materialized BF16 block gradients reduced global accumulation error and did not worsen local error. A follow-up matched the full-kernel and shared-buffer inputs at twelve heads, verified their hashes, and tested an optimization barrier at the K/V block-result boundary. The local V discrepancy persisted, so that boundary change was removed. Shared-buffer improvement does not establish integrated-kernel accuracy; compensation remains opt-in and unqualified for production training. See `artifacts/yat-kv-boundary-tpu-v5e1-0925/analysis.json`.

Experimental `--fsdp N` partitions divisible matrix parameters and optimizer arrays along their first axis, with distinct batch rows across both data and FSDP mesh axes. Initialization allocates directly into those layouts; checkpoint restore uses the live target layouts. FSDP is included in checkpoint identity, and layout checks run after initialization/restore and before saves. The replicated path remains the default. This mode currently rejects `--local-gradient-accumulation`, `--yat-local-shards`, and `--nnx-jit-partial`, whose existing assumptions have not been qualified together with this mesh.

Four physical v5e devices passed the small-encoder foundation checks for FSDP factors 1, 2 and 4, but state sharding increased communication and latency at that small size. Production trainer integration, full-size restart parity, pretrained import, and sustained multi-host behavior require further TPU qualification. The full-model trainer validation is tracked under `artifacts/encoder-fsdp-trainer-tpu-v5e4-0925`; a submitted run is not a passing result.
