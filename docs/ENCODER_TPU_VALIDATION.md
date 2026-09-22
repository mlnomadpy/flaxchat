# Encoder TPU verification — 2026-09-22

Project: `tpubuilders`; zone: `us-east5-a`; Spot `v5p-8` (four chips, one VM).
Model: `jhu-clsp/mmBERT-base`, revision
`c5955035435e2bf121cde7f3c8863ef52ff35d82`, 307,786,240 parameters.
Policy: BF16 compute, FP32 residuals/master parameters/Adam state; XLA attention.

## Completed evidence

- Physical topology probe: four TPU devices, one process, JAX 0.11.1, Flax 0.12.9.
- CPU suite on the TPU VM: 596 passed, 8 skipped, 5 accelerator tests deselected;
  coverage 78.92%; all configured per-module coverage floors passed.
- Encoder tests on TPU, including precision variants and exact restart: passed.
- Encoder Splash forward/gradient comparison: passed.
- GPT Splash forward/gradient checks: all four passed after precision isolation.
- Released-weight FP32 output/loss/gradient parity on TPU: passed.
- Released 307.8M-parameter checkpoint: six BF16 training updates at global batch
  four, sequence 512, all finite and accepted; step-6 GCS checkpoint committed.
- Recovery follow-up: physical TPU probe, held-out evaluation, SIGKILL after
  committed step 3, and resume to step 6 all passed. Model, optimizer, and training
  state manifests exactly match the uninterrupted baseline.
- Held-out fixture: 16 rows, 1,177 selected tokens, masked-token loss 0.0573377.
  These synthetic fixtures validate execution only, not production quality.
- BF16-versus-FP32 diagnostic: hidden relative L2 2.02%, logits 0.46%,
  loss 1.83%, first-QKV gradient 3.18%. Strict pointwise gate failed. This is
  numerical drift evidence, not a BF16 parity or model-quality qualification.

## Issues exposed and corrections

| Issue | Correction | Verification |
| --- | --- | --- |
| Single-worker launcher exported distributed coordinator variables; a CPU subprocess test inherited the coordinator and the parent aborted. | Only export distributed variables for multiple workers; isolate CPU validation environments. | Local launcher regressions and subsequent full CPU suite passed. |
| Global highest matmul precision forced invalid FP32 operations inside the BF16 GPT Splash kernel. | Use native precision for BF16 Splash diagnostics; keep highest precision for the FP32 reference gate. | Local regression and four physical TPU checks passed. |
| Released tokenizer labels model mask token 4 as non-special. | Explicit mask ID in data preparation and manifest; protect it from masking targets/random replacement. | Regression, released-data preflight, and six physical TPU updates passed. |
| Controller monotonic time lagged the cloud wall-clock deadline during the third attempt; capacity waiting left insufficient time for recovery tests. | Bound every operation by the earlier monotonic and wall-clock deadline; retain independent cloud cleanup. | Local deadline/launcher suite: 21 passed. Cloud guard deleted the third slice at expiry. |

The first two attempts were stopped by validation and their queues and VMs were
confirmed absent. Neither completed model training. Failure logs are retained.
A fourth bounded attempt reused the committed baseline and passed recovery checks.
Its supervisor exited successfully; both queue and TPU VM absence were independently
confirmed. All four campaign allocations have been cleaned up.
Production quality is not established
by the short synthetic multilingual fixtures used for this verification.

## Throughput interpretation and experimental optimization

The five post-compilation updates in the released-model pilot measured roughly
28.6–30.4K input tokens/s (2,048 tokens per step). First compilation/update took
42.1 seconds; saving the checkpoint took 41.6 seconds. This six-step pilot is a
correctness check, not a sustained throughput qualification.

The prior GPT pilot recorded 197,176 steady-state input tokens/s, but used 12
layers, vocabulary 50,257, and batch 16 × sequence 1,024 (16,384 tokens per step).
The encoder uses 22 layers, vocabulary 256,000, and batch 4 × sequence 512.
The models, vocabulary projections, batch sizes, and parameter sharding differ;
these rates do not isolate a precision-related regression.

An opt-in `--mlm-projection masked` path was added locally after freezing the
recovery workload. It projects static buffers of selected positions, with a
dense overflow fallback that drops no labels. At sequence 512 / chunk 128 it
reduces head rows fourfold when capacity is sufficient; whole-model speedup is
unmeasured. Local encoder/trainer suite: 32 passed, one multiprocess test skipped,
one accelerator test deselected; an additional four-device CPU sharding parity
test passed. The frozen physical TPU results above do not validate this new path.
Physical multi-host encoder validation and a sustained masked-projection TPU
benchmark remain outstanding.

## Artifacts and cost accounting

Evidence is retained under `artifacts/encoder-validation-0922/` and the private
GCS campaign prefix `gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/`.
Every attempt uses the independent cloud expiry guard and a conservative $25 cap.
The recorded Spot quote is $1.003363 per chip-hour ($4.013452 per four-chip hour),
checked September 22. Entire resource-lifetime estimates intentionally include
non-billable waiting/deletion periods and are not posted billing or credit usage.

## Later projection follow-up

The preceding recovery campaign predates projection optimization. Subsequent
physical masked/fused measurements, the shared-head rounding fix, and its
released-model gradient regression are recorded in
[projection benchmarks](ENCODER_PROJECTION_BENCHMARK.md). Physical multi-host
encoder acceptance and downstream model quality remain separate requirements.
