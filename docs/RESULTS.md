# Verified accelerator results

## Current acceptance record (`12bfd8522f9a4dff46f05157108eb63159240882`)

Kaggle kernel version 12 checked out the exact revision above and passed every
stage on one TPU v5 Lite host with eight devices. The compact evidence is
archived as five immutable records:

- [suite summary](../benchmarks/results/kaggle-tpu-v5e-8-12bfd85-summary.json)
- [TinyStories stage metrics and manifest](../benchmarks/results/kaggle-tpu-v5e-8-12bfd85-pipeline.json)
- [small-model overhead scaling](../benchmarks/results/kaggle-tpu-v5e-8-12bfd85-scaling-overhead.json)
- [representative strong scaling](../benchmarks/results/kaggle-tpu-v5e-8-12bfd85-scaling-strong.json)
- [representative weak scaling](../benchmarks/results/kaggle-tpu-v5e-8-12bfd85-scaling-weak.json)

The suite passed all eleven stages, including the complete test suite and
tokenizer → pretrain → SFT → preference → evaluation → inference
pipeline, checkpointing, attention/speculative benchmarks, and device scaling.
This is one physical host with eight devices; it is not multi-host evidence.
The 589,902-parameter case remains an overhead regression test. The
29,360,454-parameter representative case reached 1.170M tokens/sec at eight
devices under fixed global work and 1.158M tokens/sec under fixed per-device
work. Against a predeclared 0.50 floor, eight-device strong-scaling efficiency
was 0.5138 and weak-scaling efficiency was 0.7412; both passed. Timing
dispersion, compile time, checkpoint
latency, MFU estimate, and compiled-memory semantics are retained in the
records. These are device-scaling measurements, not multi-host claims.

## Previous acceptance record (`7df9fe85817acf27fa61f5feb46e7f2a0774a3b1`)

The prior five-file kernel-version-10 record remains available:

- [suite summary](../benchmarks/results/kaggle-tpu-v5e-8-7df9fe8-summary.json)
- [TinyStories stage metrics and manifest](../benchmarks/results/kaggle-tpu-v5e-8-7df9fe8-pipeline.json)
- [small-model overhead scaling](../benchmarks/results/kaggle-tpu-v5e-8-7df9fe8-scaling-overhead.json)
- [representative strong scaling](../benchmarks/results/kaggle-tpu-v5e-8-7df9fe8-scaling-strong.json)
- [representative weak scaling](../benchmarks/results/kaggle-tpu-v5e-8-7df9fe8-scaling-weak.json)

## Earlier acceptance record (`97ba1333bf46b51f85ca3d9099c8c2717438ce91`)

The previous three-file record remains available for historical comparison:

- [suite summary](../benchmarks/results/kaggle-tpu-v5e-8-97ba133-summary.json)
- [TinyStories stage metrics and manifest](../benchmarks/results/kaggle-tpu-v5e-8-97ba133-pipeline.json)
- [1/2/4/8-device strong-scaling record](../benchmarks/results/kaggle-tpu-v5e-8-97ba133-scaling.json)

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

## Matched trainer comparison

Before spending GPU quota, Kaggle CPU kernel version 3 checked out all three
pinned sources, installed the isolated adapter dependencies, constructed each
native model, and verified trainable counts of 589,902 (flaxchat), 589,902
(nanochat), and 606,848 (MaxText). The [machine-readable preflight
record](../benchmarks/results/kaggle-cpu-matched-preflight-248b1fa.json) passed
the 600,000 ±5% gate without enabling a GPU or TPU. Comparative performance is
still withheld: bundled P100 versions 2 and 3 each completed flaxchat and
MaxText but exposed successive nanochat runtime-compatibility and adapter-dtype
failures. Both [negative](../benchmarks/results/kaggle-gpu-p100-matched-248b1fa-failure.json)
[records](../benchmarks/results/kaggle-gpu-p100-matched-32351d4-failure.json) are
retained rather than selectively publishing the two successful framework
measurements. The corrected free CPU preflight now includes a native nanochat
forward/backward optimizer update; another GPU measurement should only be run
after that gate passes and with an explicit quota decision.
