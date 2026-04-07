---
layout: page
title: Contributing
permalink: /contributing/
---

# Contributing

## Setup

```bash
git clone https://github.com/tahabsn/flaxchat.git
cd flaxchat
pixi install
pixi run test  # 81 tests, ~20s
```

## Project Structure

```
flaxchat/
├── flaxchat/          # Core library (import flaxchat)
│   ├── gpt.py         # THE model — start here
│   ├── optim.py       # Muon + AdamW
│   ├── engine.py      # Inference (KV-cache)
│   ├── common.py      # Mesh, sharding, distributed
│   ├── config.py      # All config dataclasses
│   ├── eval.py        # CORE metric, BPB
│   ├── dataloader.py  # BOS best-fit packing
│   ├── tokenizer.py   # BPE (rustbpe + tiktoken)
│   ├── checkpoint.py  # Orbax save/load
│   ├── report.py      # Training reports
│   ├── remote/        # KaggleRunner
│   └── cloud/         # TPULauncher
├── scripts/           # Runnable scripts (python -m scripts.XXX)
├── tasks/             # Eval tasks (MMLU, GSM8K, etc.)
├── tests/             # pytest tests
├── docs/              # This documentation
└── configs/           # YAML configs
```

## Key Design Decisions

1. **No engine class** — parallelism is baked into train steps, not a wrapper
2. **Frozen GPTConfig** — registered as `jax.tree_util.register_static` for JIT compatibility
3. **`_NNX_LIST`/`_NNX_DICT`** — compat shim for Flax 0.11 vs 0.12
4. **`compute_init()`** — single function that does distributed init + mesh creation
5. **Replicated sharding** for inference — `_to_device(arr, replicated)` matches model params

## Running Tests

```bash
pixi run test              # All 81 tests
pixi run test-quick        # Skip slow tests
pixi run -- pytest tests/test_model.py -v  # Specific file
```

## Code Style

- No docstrings on obvious functions
- Type hints where they help (not everywhere)
- Prefer `jnp` operations over numpy inside JIT
- Use `nnx.data()` for non-trainable module fields
- Use `jax.lax.dynamic_slice` / `dynamic_update_slice` for traced indexing
- `print0()` for distributed-safe printing (only process 0)

## Test Coverage

Well tested:
- Model forward/backward, loss, gradients, softcap, causal masking
- Muon optimizer standalone + setup_optimizer integration
- Config creation/serialization (YAML, JSON, dict, roundtrip)
- KV-cache vs padded generation consistency
- Calculator sandboxing

Needs tests:
- Dataloader (best-fit packing logic)
- Checkpoint save/load roundtrip
- Tokenizer encode/decode
- Training scripts (1-step integration test)
- Sharding functions
