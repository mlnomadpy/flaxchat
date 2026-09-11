---
layout: post
title: "An Early TinyStories Training Experiment on 2xT4 GPUs"
date: 2026-04-01
author: Taha Bouhsine
tags: [jax, flax, training, tinystories, experiment]
---

> Historical note: the raw run record for this early experiment was not
> committed, so its quantitative tables have been withdrawn rather than treated
> as verified results. Current reproducible measurements live in the
> [machine-readable provenance index](../../benchmarks/results/provenance-index.json)
> and [verified-results summary](../RESULTS.md).

We trained a GPT language model from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset using **flaxchat**, our JAX/Flax NNX port of [nanochat](https://github.com/karpathy/nanochat). The experiment used **2x NVIDIA T4 GPUs** via Kaggle with data-parallel training.

## Experimental Setup

### Hardware & Software
- **2x NVIDIA T4 GPUs** (16GB each) on Kaggle
- JAX 0.7.2, Flax 0.11.2
- Data-parallel training via JAX SPMD mesh

### Model Architecture

| Feature | Detail |
|---------|--------|
| Layers | 8 |
| Dimensions | 256 |
| Heads | 8 (GQA-ready) |
| Sequence length | 512 |
| Vocab | 8,192 (BPE) |
| Attention | `jax.nn.dot_product_attention` (hardware-adaptive) |

All nanochat features ported: RoPE, QK-norm (1.2x), ReLU^2, value embeddings (alternating), residual/x0 lambdas, smear, backout, logit softcap.

### Dataset

- **500,000 stories** from TinyStories (~106M tokens)
- BPE tokenizer trained on 50K stories (3.8x compression)
- BOS-aligned packing

## Data Parallelism

The experiment used this SPMD placement pattern:

```python
mesh = Mesh(create_device_mesh((2,)), ('data',))
state = jax.device_put(state, NamedSharding(mesh, P()))
inputs = with_sharding_constraint(inputs, NamedSharding(mesh, P('data')))
```

### Generated Samples

> **Once upon a time**, there was a little girl named Lily. She loved to play with her toys and her favorite was her favorite toy, a blue teddy bear. One day, Lily's mom asked her to help with the phone. Lily didn't want to, so she said no and tried to fix her teddy bear.

> **The little dog** was very happy. She found a big tree and wanted to play with the birds inside. She took it to her friend Sally and said, "Look, Sally! It's all fun!" Sally smiled and said, "Yes, it's very nice. Let's go get it!"

> **A girl named Lily**. She was a very good girl and loved her mom very much. Every day, she tried to be brave. She would sing all day and make her own faces.

> **One day, the sun** was shining brightly. It was a sunny day and the park was happy. Then, a little boy saw the slide and ran over to it. "What is that?" he asked. "I don't know. What is inside?" the boy asked.

The KV-cache implementation used static-shape dynamic updates to avoid
token-by-token recompilation. No speed number from this unarchived run is
published as a verified result.

## Lessons Learned

**1. Device placement is everything.** JAX follows the data — if you create a `jnp.array` on CPU and pass it to a model on GPU, computation runs on CPU. Always use `jax.device_put(arr, sharding)`.

**2. Flax NNX version compat matters.** `nnx.List`/`nnx.Dict` exist in 0.12+ but not 0.11. We use `getattr(nnx, 'List', list)` as a shim. The Muon optimizer's NamedTuple state crashes `nnx.Optimizer` on 0.11 — we fall back to AdamW.

**3. Generation needs static shapes.** Token-by-token generation with changing array shapes causes JIT recompilation at every step. Fix: `jax.lax.dynamic_update_slice` for KV cache (static shape, dynamic index) or pad to max length.

**4. `jax.nn.dot_product_attention` is the right call.** It auto-selects the best kernel (cuDNN on GPU, XLA on TPU) and handles masking cleanly.

## What's Next

1. **Scale up**: Train on ClimbMix-400B on TPU v4-32 pod
2. **Expand RL experiments**: GRPO on GSM8K/SpellingBee with tool use
3. **`jax.lax.while_loop` sampler**: Eliminate Python loop overhead for generation

---

*Part of the 2026 Q1 TPU Research Sprint, supported by [Google AI Developer Programs](https://developers.google.com/programs).*

*Full code: [github.com/tahabsn/flaxchat](https://github.com/tahabsn/flaxchat)*
