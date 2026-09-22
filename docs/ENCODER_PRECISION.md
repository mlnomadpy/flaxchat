# Encoder precision evidence and qualification

Reviewed September 22, 2026. No production precision policy has been qualified
on physical TPU for the new encoder path.

## Primary sources

- [mmBERT base training configuration](https://github.com/JHU-CLSP/mmBERT/blob/main/configs/mmBERT-base.yaml)
  uses `precision: amp_bf16`, with LayerNorm. This is automatic mixed precision,
  not evidence that parameters, optimizer state, and all residuals are BF16.
- [Google Cloud TPU guidance](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
  recommends BF16 activations with FP32 weights and gradients, and occasional FP32
  baselines. TPU matrix multiplication uses BF16 products with FP32 accumulation.
- [MaxText configuration](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml)
  defaults to BF16 activations, FP32 weights and gradients, and casting logits
  to FP32. This is a useful TPU precedent, not an encoder-specific guarantee.
- [ModernBERT normalization implementation](https://github.com/AnswerDotAI/ModernBERT/blob/main/src/bert_layers/normalization.py)
  provides both standard LayerNorm and a Triton implementation with an explicit
  residual precision option. There is no universal requirement for FP32 residuals.
- [JAX matrix multiplication precision](https://docs.jax.dev/en/latest/201/precision.html)
  distinguishes array dtype from internal multiplication precision. A diagnostic
  FP32 baseline on TPU must explicitly request highest matmul precision.
- [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)
  explains lost small weight updates under nearest rounding and studies stochastic
  rounding and compensated summation.
- [Stochastic Rounding for LLM Training: Theory and Practice](https://arxiv.org/abs/2502.20566)
  studies BF16 optimizer updates with stochastic rounding, including distributed
  execution. Its reported savings are results for its experiments, not estimates
  for this harness. Learning-rate choice matters, especially for small updates.

## Selected baseline and experiment

Use BF16 dense/attention compute while retaining FP32 master parameters, gradients,
Adam state, normalization reductions and loss. Keep residual precision configurable.
The trainer now defaults to BF16 compute with FP32 residuals. This is our
conservative engineering baseline for the limited validation budget, not a claim
of completed physical TPU qualification. BF16 residuals remain an explicit
efficiency experiment. The model constructor's FP32 default is retained for
reference diagnostics.

This choice preserves BF16 matrix multiplies while avoiding a simultaneous
change to residual rounding and optimizer storage. We will only promote BF16
residuals after their measured memory/throughput benefit justifies any observed
held-out-loss difference. Pure BF16 optimizer state is deferred.

The existing diagnostic compared BF16 against FP32 on four multilingual examples.
Changing embedding, normalization output and residual precision together reduced
hidden-state relative L2 difference from about 7.5% to 2.9%. This did not isolate the
residual addition as the cause, and does not measure downstream model quality.

Qualify policies using matching-precision upstream references and a fixed-data
training comparison: identical weights, masks, batches, schedule, and token budget.
Measure held-out MLM loss per language, gradient/update finiteness, loss spikes,
checkpoint/resume consistency, peak TPU memory, compile time, steady-state tokens
per second, and dollars per fixed token budget. Multiple seeds and downstream
evaluation are needed before claims of quality equivalence.

BF16 optimizer/master-state storage with stochastic rounding is a separate
experiment. Do not obtain it by blindly casting the current Adam state to BF16.
Do not infer total-memory or cost halving from halving residual tensor storage.

## Local checks after exposing residual precision

Both residual modes pass the finite-gradient and activation-dtype tests, with
FP32 master parameters verified: focused encoder suite 20 passed, 1 skipped,
1 accelerator test deselected. Ruff and implementation Pyright checks pass.

A CPU BF16 reference using PyTorch autocast plus explicit BF16 embedding/LayerNorm
output hooks was compared with JAX BF16 activations on the released checkpoint.
Hidden-state relative L2 difference was 1.75%, selected-logit difference 0.76%,
loss difference 0.51%, and first-QKV-gradient difference 3.82%. The strict
pointwise gate still failed. These hooks define a comparable experimental policy;
this is not the original CUDA training stack. This four-mask diagnostic neither
qualifies physical TPU execution nor establishes quality equivalence.


## Baseline implementation verification

The selected baseline is exercised by exact model/optimizer/cursor resume tests
alongside FP32 and BF16-residual variants. Resume rejects precision-policy changes.
Four simulated CPU devices pass the integration checks; two real CPU processes
pass a BF16-compute checkpoint/restart check. These establish CPU distributed
behavior only, not TPU numerical behavior or TPU performance.

Startup and step logs record precision, device topology, compilation-inclusive
first steps, steady-state token rates and separate checkpoint duration. The TPU
workload defaults to 512 tokens and reports smoke/recovery scope explicitly.
Physical TPU measurement and multilingual held-out quality evaluation remain
required before a production campaign.

## Projection consistency

The prediction head now applies its BF16 dense/GELU and FP32 normalization once
before vocabulary-loss chunking. Repeating that transformation inside the scan
produced backend-dependent rounding and gradient accumulation. A released-weight
TPU regression reduced the fused-versus-XLA relative gradient difference from
7.93% to 0.69%, without changing the 3% acceptance threshold. The residual/master/
Adam policy is unchanged. Full-model BF16-versus-FP32 differences remain diagnostic
and require held-out/downstream assessment; numerical agreement between backends
is not evidence of model-quality equivalence. See the
[projection report](ENCODER_PROJECTION_BENCHMARK.md) for the frozen evidence.
