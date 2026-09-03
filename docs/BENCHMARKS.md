# Benchmarks

Performance claims should include the Git revision, JAX/backend versions,
device kind and count, shapes, warmup count, measured iterations, precision,
and raw machine-readable output. Correctness gates run before timing.

The attention benchmark emits JSON suitable for archiving:

```bash
python -m benchmarks.attention --backend=xla --sequence-length=2048
python -m benchmarks.attention --backend=splash --sequence-lengths 1024 2048 4096 8192
python -m benchmarks.speculative --max-tokens=16 --draft-steps=4
```

Do not compare compile-inclusive first-call latency with warmed execution.
Use identical query/key/value tensors for backend comparisons, and report both
global causal and sliding-window configurations. The TPU test suite separately
checks XLA/Splash output and query-gradient parity at sequence length 128,
including grouped-query heads and variable-length right-padded batches.
The emitted memory estimate is compiled argument + output + temporary bytes;
it is a comparable XLA estimate, not a device-wide HBM watermark. Speculative
results report exact greedy agreement, proposal acceptance, main-model call
count, steady-state throughput, and measured speedup. Keep regressions even
when speedup is below 1.0; hardware/model pairing determines the break-even.

## Trainer comparisons

`benchmarks/baselines/` defines matched flaxchat, MaxText, and nanochat run
controls. Replace every `REQUIRED` revision, run on the same hardware, and
capture each result with all fields enforced by `benchmarks.compare`:

```bash
python -m benchmarks.compare artifacts/flaxchat.json artifacts/maxtext.json artifacts/nanochat.json
```

The command fails closed if hardware, device count, precision, parameter count,
sequence length, global batch size, or validation metric differs. It never
silently presents unmatched historical numbers as an apples-to-apples result.
