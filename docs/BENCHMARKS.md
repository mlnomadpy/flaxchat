# Benchmarks

Performance claims should include the Git revision, JAX/backend versions,
device kind and count, shapes, warmup count, measured iterations, precision,
and raw machine-readable output. Correctness gates run before timing.

The attention benchmark emits JSON suitable for archiving:

```bash
python -m benchmarks.attention --backend=xla --sequence-length=2048
python -m benchmarks.attention --backend=splash --sequence-lengths 1024 2048 4096 8192
python -m benchmarks.speculative --max-tokens=16 --draft-steps=4
python -m benchmarks.training_scaling --device-counts 1 2 4 8 \
  --mode strong --global-batch-size 32 --trials 3 \
  --output artifacts/training-scaling.json
python -m benchmarks.training_scaling --device-counts 1 2 4 8 \
  --mode weak --per-device-batch-size 8 --trials 3 \
  --output artifacts/training-weak-scaling.json
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

The training-scaling command reports strong scaling (fixed global work) and
weak scaling (fixed per-device work) separately. Each point records repeated
trial timings with median and population dispersion, alongside
compile time, steady tokens/sec, a conventional `6*N*tokens` MFU estimate,
compiled memory, synchronous Orbax checkpoint latency, loss samples, and
efficiency relative to the one-device result. This does not substitute for a
physical multi-host run; the output labels that limitation explicitly.
The default declared efficiency floor is 0.50. Every point includes a
`meets_efficiency_threshold` boolean and the top-level record lists device
counts below the floor in `efficiency_regressions`; use
`--minimum-efficiency` to change the predeclared floor.

## Trainer comparisons

`benchmarks/baselines/` defines pinned flaxchat, MaxText, and nanochat run
controls. Every plan remains labeled `pending_matched_run` until executed on
the declared hardware. Capture each result with all fields enforced by
`benchmarks.compare`:

```bash
python -m benchmarks.compare --protocol benchmarks/protocol.yaml \
  artifacts/flaxchat.json artifacts/maxtext.json artifacts/nanochat.json
```

The executable matched suite creates one byte-tokenized TinyStories batch
artifact and passes its SHA-256-bound arrays to all three native models. Submit
all adapters together in one P100 kernel (never three separate quota jobs):

```bash
# First validate dependencies and exact parameter counts without GPU quota.
python -m scripts.kaggle_matched_benchmarks \
  --revision "$(git rev-parse HEAD)" --preflight --wait \
  --output-dir artifacts/kaggle-matched-preflight

# Submit the single bundled GPU measurement only after preflight succeeds.
python -m scripts.kaggle_matched_benchmarks \
  --revision "$(git rev-parse HEAD)" --wait \
  --output-dir artifacts/kaggle-matched
```

The launcher checks out all three full revisions, retains each adapter's raw
log and durable checkpoint, and runs `benchmarks.compare` only if every native
trainer succeeds. A failed or disappointing run is retained rather than
silently replaced.

The command fails closed if the protocol hash, dataset revision, optimizer,
seed, warmup/measurement counts, hardware, device count, precision, parameter
target/tolerance, sequence length, global batch size, or validation metric differs.
Each actual trainable parameter count must fall within the declared tolerance.
The CLI recomputes the protocol file's SHA-256 and rejects records bound to any
other protocol. Exact
40-character source revisions and finite measurements are mandatory. It never
silently presents unmatched historical numbers as an apples-to-apples result.
Records must include tokens/sec, model FLOPs utilization, compile time, peak
memory, checkpoint time, scaling efficiency, validation quality, and an
explicit limitations statement.
