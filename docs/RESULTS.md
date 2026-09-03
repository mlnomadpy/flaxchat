# Verified accelerator results

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
generation exactly and measured 1.77× speedup for that synthetic model pair.

These numbers are acceptance and systems measurements, not quality or scaling
claims. In particular, compiled XLA memory is not a whole-device HBM watermark,
and no nanochat/MaxText result is presented until the controlled fields in
`benchmarks/baselines/` can be matched on the same hardware.
