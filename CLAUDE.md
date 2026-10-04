# flaxchat — Claude Code Integration

## What This Is

flaxchat is an end-to-end decoder and bidirectional encoder training harness for TPU pods and GPUs, built on JAX/Flax NNX. It adapts nanochat and includes continued YAT embedding training and a PyTorch export.

## Current representation-training workflow

Use `skills/flaxchat-training/SKILL.md` and `docs/REPRESENTATION_TRAINING_RUNBOOK.md`
when continuing the trained YAT embedding model. The current audit and tool gaps
are in `docs/SYSTEM_AUDIT_2026-09-30.md` and `docs/TRAINING_TOOLS.md`.

Continue the existing trained weights. Changing the data, objective, or learning
rate horizon creates a new stage with explicit optimizer policy; exact resume
means restoring the same immutable stage and its optimizer/cursor. Do not use
CPU model runs or simulated devices as TPU qualification. Data preparation,
static checks, and evidence parsing can run locally or on a preparation VM.
GitHub checks remain under the user's existing cost preference; do not dispatch
paid workflows or cloud jobs as a side effect of editing documentation.

## Key Modules

| Module | What |
|--------|------|
| `gpt.py` | GPT model: RoPE, GQA, QK-norm, ReLU^2, value embeddings, smear, backout, softcap, sliding window, remat |
| `optim.py` | Mixed Muon+AdamW via optax.multi_transform. Polar Express + NorMuon |
| `engine.py` | Inference: padded, KV-cached, fully-JIT (while_loop), speculative decoding, streaming with tool use |
| `execution.py` | Sandboxed Python code execution (multiprocessing isolation) |
| `eval.py` | CORE metric (DCLM), BPB evaluation, multiple-choice + generative |
| `dataloader.py` | BOS-aligned best-fit packing for distributed pretraining |
| `tokenizer.py` | BPE tokenizer (HuggingFace Tokenizer + rustbpe/tiktoken) |
| `config.py` | Depth-based auto-config (single dial scales all hyperparams) |
| `common.py` | Mesh creation, distributed init, dtype detection, logging |
| `checkpoint.py` | Orbax async checkpointing with save/load/restore |
| `encoder.py`, `yat.py` | Bidirectional encoder, YAT attention/FFN, trainable alpha |
| `contrastive.py` | Global paired and mined-negative embedding objectives |
| `scripts/train_yat_embedding_finetune.py` | TPU-only embedding stage; see audit for missing guards and capabilities |
| `report.py` | Training reports and cost estimation |
| `dataset.py` | Parquet file listing for ClimbMix-400B |

## Quick Usage

```python
from flaxchat import GPT, GPTConfig, Engine, compute_init

mesh = compute_init()
config = GPTConfig(n_layer=12, n_head=6, n_embd=768, vocab_size=32768)
model = GPT(config, rngs=nnx.Rngs(0))

# Generation (4 modes)
from flaxchat.engine import generate, generate_with_cache, generate_fast, generate_speculative
tokens = generate_with_cache(model, prompt_ids, max_tokens=256, temperature=0.8)
tokens = generate_fast(model, prompt_ids, max_tokens=256)  # fully JIT, fastest
tokens = generate_speculative(model, draft_model, prompt_ids)  # speculative decoding

# Engine with tool use (calculator + Python REPL)
engine = Engine(model, tokenizer)
for token_column, masks in engine.generate(prompt_ids, num_samples=3, max_tokens=256):
    # streaming generation with automatic tool use
    pass
```

## Generation Modes

| Mode | Function | Speed | Use Case |
|------|----------|-------|----------|
| Padded | `generate()` | Measure per run | Testing |
| KV-cached | `generate_with_cache()` | Measure per run | Cached inference |
| Fully JIT | `generate_fast()` | Measure per run | TPU inference |
| Speculative | `generate_speculative()` | Measure per model pairing | Large model + small draft |

## Tool Use

Engine automatically handles `<|python_start|>...<|python_end|>` blocks:
1. First tries `use_calculator()` (safe math/string.count)
2. Leaves generated Python disabled by default; reviewed code may opt into the
   best-effort `execute_code(..., trusted=True)` reliability guard
3. Injects `<|output_start|>result<|output_end|>` tokens

## Guarded Execution

```python
from flaxchat.execution import execute_code
result = execute_code("print(2 + 2)", timeout=5.0, trusted=True)
# ExecutionResult(success=True, stdout="4\n", ...)
```

## Parallelism

```python
mesh = compute_init()  # configure the intended device mesh
# Data parallel: P('data') on batch dimension
# FSDP: shard_model_fsdp() for large models
# Multi-host: jax.distributed.initialize() automatic
```

Capabilities are trainer-specific. The embedding fine-tuner currently uses a
one-dimensional data mesh and replicated model/optimizer state; shared FSDP
helpers and MLM accumulation do not imply embedding support. Check the actual
entry point and physical evidence before making a scaling claim.

## Config (depth-based)

```python
from flaxchat.config import FlaxChatConfig
config = FlaxChatConfig.from_depth(depth=12)
# -> 12 layers, 768 dims, 6 heads, ~79M params
```

## Scripts

```bash
python -m scripts.pretrain --depth=12           # Pretrain on ClimbMix-400B
python -m scripts.sft --base-model=d12          # SFT on conversations
python -m scripts.rl --model=d12                # RL/GRPO on GSM8K
python -m scripts.eval --model=d12 --tasks=all  # Evaluate
python -m scripts.chat_web --model=d12          # Web chat
python -m scripts.run_tinystories --depth=4     # Full pipeline locally
python scripts/train_gpt2.py --depth=16 --tie-embeddings --tokens=10B  # GPT-2 medium
```

## Tied Embeddings

```python
config = GPTConfig(n_layer=16, n_embd=1024, tie_embeddings=True)
# lm_head shares weights with wte (saves ~33M params for vocab=32K)
```

## Verified Results

Use `docs/RESULTS.md`, the machine-readable provenance index, and dated model
cards. A historical "running" status or unlinked throughput number is not current
execution evidence. The embedding-v1 card reports completed training and the
cross-language regression that the continuation must address.

## Tests

```bash
pixi run test-quick        # deterministic CPU suite, no cloud needed
pixi run test-multidevice  # force eight virtual JAX devices
```
