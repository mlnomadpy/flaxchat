# Verified accelerator results

## Current acceptance record (`97ba1333bf46b51f85ca3d9099c8c2717438ce91`)

The latest single-host TPU v5e-8 run is archived as three immutable records:

- [suite summary](../benchmarks/results/kaggle-tpu-v5e-8-97ba133-summary.json)
- [TinyStories stage metrics and manifest](../benchmarks/results/kaggle-tpu-v5e-8-97ba133-pipeline.json)
- [1/2/4/8-device strong-scaling record](../benchmarks/results/kaggle-tpu-v5e-8-97ba133-scaling.json)

The suite passed all nine stages, including 273 tests with two skips, the
complete tokenizer → pretrain → SFT → preference → evaluation → inference
pipeline, checkpointing, attention/speculative benchmarks, and device scaling.
This is one physical host with eight devices; it is not multi-host evidence.
The scaling model has only 589,902 parameters and is intentionally retained as
an overhead regression case, not representative accelerator efficiency.

## Earlier characterization records

The raw structured record for the successful acceptance bundle is
[`benchmarks/results/kaggle-tpu-v5e-8-cde72cd.json`](../benchmarks/results/kaggle-tpu-v5e-8-cde72cd.json).
It identifies the exact source and TinyStories revisions, TPU kind/count,
stage-loss series, evaluation loss, compile times, attention throughput,
compiled memory estimates, speculative correctness/speedup, and limitations.

At `cde72cd` the bundled Kaggle TPU v5e-8 run passed every check. The pinned
TinyStories path completed tokenizer → pretrain → SFT → preference training →
evaluation → inference and published its complete Orbax/tokenizer/manifest
bundle. Splash attention measured 2.81M, 4.00M, 4.08M, and 2.46M tokens/sec at
1K, 2K, 4K, and 8K respectively. Greedy speculative output matched cached
generation exactly and measured 1.77× speedup for that aligned synthetic
upper-bound pair. That revision still shared initialization internally; the
result is mechanism evidence, not a claim about an independently trained draft.

The independent GPU bundle in
[`benchmarks/results/kaggle-gpu-p100-d33726e.json`](../benchmarks/results/kaggle-gpu-p100-d33726e.json)
also passed: 236 tests on a Tesla P100, plus the complete pinned TinyStories
pipeline, pretraining smoke test, XLA attention benchmark, and speculative
correctness check. It measured 2.78M and 2.50M attention tokens/sec at 1K and
2K. Its randomly initialized draft model accepted no proposals, correctly
making speculative decoding slower; the record preserves that negative result
rather than presenting it as a speedup.

These numbers are acceptance and systems measurements, not quality or scaling
claims. In particular, compiled XLA memory is not a whole-device HBM watermark,
and no nanochat/MaxText result is presented until the controlled fields in
`benchmarks/baselines/` can be matched on the same hardware.
