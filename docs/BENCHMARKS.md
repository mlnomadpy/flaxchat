# Benchmarks

Performance claims should include the Git revision, JAX/backend versions,
device kind and count, shapes, warmup count, measured iterations, precision,
and raw machine-readable output. Correctness gates run before timing.

The attention benchmark emits JSON suitable for archiving:

```bash
python -m benchmarks.attention --backend=xla --sequence-length=2048
python -m benchmarks.attention --backend=splash --sequence-length=2048
```

Do not compare compile-inclusive first-call latency with warmed execution.
Use identical query/key/value tensors for backend comparisons, and report both
global causal and sliding-window configurations. The TPU test suite separately
checks XLA/Splash output and query-gradient parity at sequence length 128,
including grouped-query heads and variable-length right-padded batches.
