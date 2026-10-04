# Lightweight YAT versus mmBERT speed benchmark

Compare actual public YAT embedding v1 and mmBERT-base weights in the same FlaxChat JAX harness on one physical TPU device. This is an inference implementation comparison, not upstream GPU/FlashAttention serving speed or model-quality evaluation.

The fixed matrix is batches 1 and 8 by sequence lengths 128 and 512, with three warmups and ten synchronized repetitions. Timing includes encoder forward, mean pooling and FP32 L2 normalization. It excludes download, weight conversion/loading, tokenization, compilation, and host output copies. First-call compile-and-execute latency is reported separately. Inputs are identical deterministic nonpadding token IDs, so the matrix measures fully occupied shapes rather than natural-text length distributions. Preserve released YAT arithmetic and fixed bias 1, epsilon 0.01, and trained alpha.

Use `python -m scripts.benchmark_embedding_speed_tpu` once per model in separate processes under a finite external lease. Both model revisions and full original weight SHA256 values are required. The mmBERT PyTorch file is converted with the existing safe tensor-only converter; no CPU model forward is permitted. Actual model/source/input/runtime/hardware identities and every timing sample are retained in per-model JSON receipts. Partial receipts are not a completed comparison.

The October 3 campaign is bounded to one Spot v5e-8 allocation in us-west4-a, with an independent cloud cleanup lease and maximum $10 reservation including cleanup and ancillary allowance. One chip executes the benchmark; whole-slice cost and one-chip throughput must not be confused. Physical execution completed successfully for both models. All eight cases produced finite normalized vectors. Ten warmed synchronized samples per shape were retained.

| Batch | Tokens | YAT median ms | mmBERT median ms | YAT latency / mmBERT |
|---:|---:|---:|---:|---:|
| 1 | 128 | 14.88 | 13.02 | 1.14x |
| 8 | 128 | 23.10 | 14.53 | 1.59x |
| 1 | 512 | 19.94 | 14.02 | 1.42x |
| 8 | 512 | 59.85 | 32.81 | 1.82x |

At batch 8, length 512, throughput is 133.7 sequences/s for YAT and 243.8 for mmBERT. YAT first calls took 14.5–18.1 seconds including compilation; mmBERT 6.2–14.9 seconds. This narrow synthetic-input speed comparison establishes neither model quality nor GPU performance. See `benchmarks/results/yat-mmbert-light-speed-20261003/` for original receipts. Posted run charges remain unknown; $10 was a reservation ceiling, not actual spend.

The supervisor verified both the exact benchmark queue and node absent after deletion. The independent final inventory confirmed the same absence. Protected model weights and small result receipts were retained.
