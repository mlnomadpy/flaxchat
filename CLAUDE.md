# flaxchat — Claude Code Integration

## What This Is

flaxchat is a minimal end-to-end LLM training harness for TPU pods and GPUs, built on JAX/Flax NNX. Faithful port of nanochat (Karpathy) with full feature parity plus speculative decoding.

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
| Padded | `generate()` | ~1-2 tok/s | Testing |
| KV-cached | `generate_with_cache()` | ~10-50 tok/s | Production |
| Fully JIT | `generate_fast()` | ~200+ tok/s | TPU inference |
| Speculative | `generate_speculative()` | ~2-4x cached | Large model + small draft |

## Tool Use

Engine automatically handles `<|python_start|>...<|python_end|>` blocks:
1. First tries `use_calculator()` (safe math/string.count)
2. Falls back to `execute_code()` (sandboxed Python subprocess)
3. Injects `<|output_start|>result<|output_end|>` tokens

## Sandboxed Execution

```python
from flaxchat.execution import execute_code
result = execute_code("print(2 + 2)", timeout=5.0)
# ExecutionResult(success=True, stdout="4\n", ...)
```

## Parallelism (default, not optional)

```python
mesh = compute_init()  # auto mesh over ALL devices
# Data parallel: P('data') on batch dimension
# FSDP: shard_model_fsdp() for large models
# Multi-host: jax.distributed.initialize() automatic
```

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
```

## Tests

```bash
pixi run pytest tests/ -v  # 148 tests, no GCP needed
```
