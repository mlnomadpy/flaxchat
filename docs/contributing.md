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
pixi run test-quick
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
pixi run test              # Complete local suite
pixi run test-quick        # Skip slow and accelerator tests
pixi run test-coverage     # Enforce the branch-coverage floor
pixi run test-multidevice  # Exercise an eight-device CPU mesh
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
- Calculator and guarded-execution behavior

Risk-based per-module coverage floors are enforced by
`scripts/check_coverage.py`. The coverage job prints each protected module's
signed distance from its floor, and a low protected module fails even when the
global percentage passes, so unrelated covered code cannot mask a regression.
Accelerator-only behavior is covered by the on-demand bundled Kaggle workflow.

Ruff enables Pyflakes, fatal pycodestyle checks, and Bugbear correctness rules.
The documented per-file Bugbear exceptions in `pyproject.toml` are the deferred
legacy baseline; new files receive every selected rule. Pyright checks the full
`flaxchat` and `scripts` trees, with its remaining JAX-heavy modules listed in
the adjacent explicit migration baseline rather than silently excluded by a
narrow include list.

## Dependency and CI policy

- Commit `pixi.lock` whenever dependency metadata changes and review JAX,
  accelerator-plugin, Orbax, and Flax compatibility together.
- GitHub Actions are pinned to reviewed commit SHAs. Dependabot proposes
  monthly updates; reviewers verify the upstream tag before merging.
- Routine changes use the single Linux validation job. Run the manual macOS,
  Kaggle GPU, or Kaggle TPU workflows only when the affected platform or a
  release candidate needs fresh evidence.
- Release tags alone run the supported-Python install matrix, checksum/SBOM
  generation, full gates, and artifact attestation.
