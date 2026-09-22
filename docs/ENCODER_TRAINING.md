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
quality qualification, physical multi-host encoder acceptance, encoder FSDP,
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
