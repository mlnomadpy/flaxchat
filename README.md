# flaxchat

A minimal, end-to-end LLM training harness for **Google Cloud TPU pods**, built on **JAX/Flax NNX**.

Faithful port of [nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy's PyTorch GPU trainer) to the JAX ecosystem, with full feature parity plus speculative decoding.

```bash
pixi install
pixi run test                     # 148 tests
python -m scripts.run_tinystories # full pipeline on TinyStories
```

---

## What is this?

flaxchat is the complete LLM pipeline running natively on TPUs and GPUs with automatic data parallelism:

| Stage | Script | Description |
|-------|--------|-------------|
| Tokenizer | `scripts/tok_train.py` | Train BPE tokenizer (rustbpe + tiktoken) |
| Pretrain | `scripts/pretrain.py` | Pretrain GPT on ClimbMix-400B or TinyStories |
| SFT | `scripts/sft.py` | Supervised fine-tuning on conversations |
| RL | `scripts/rl.py` | GRPO/REINFORCE on GSM8K with tool use |
| Eval | `scripts/eval.py` | CORE metric, MMLU, ARC, GSM8K, HumanEval |
| Chat | `scripts/chat_web.py` | FastAPI WebSocket chat UI |
| Local | `scripts/run_tinystories.py` | Full pipeline on TinyStories (laptop or GPU) |
| Export | `scripts/convert_to_tflite.py` | LiteRT/TFLite export for edge deployment |

~7,500 lines of readable, hackable JAX/Flax NNX code across 45 Python files.

## Architecture

The GPT model faithfully replicates every feature from nanochat:

- **Rotary Embeddings (RoPE)** with 100K base theta
- **Group-Query Attention (GQA)** via `jax.nn.dot_product_attention` (hardware-adaptive)
- **QK Normalization** with 1.2x scaling for sharper attention
- **ReLU^2 MLP** (squared ReLU activation)
- **Value Embeddings** (ResFormer-style, alternating layers with gating)
- **Sliding Window Attention** per-layer configurable (SSSL pattern)
- **Per-layer Residual Scaling** (`resid_lambdas` + `x0_lambdas`)
- **Smear** — cheap bigram-like token mixing from previous position
- **Backout** — subtract mid-layer residual to remove low-level features
- **Logit Soft-capping** via `tanh(x/15)*15`
- **Gradient Checkpointing** (`nnx.remat`, `dots_saveable` policy)

## Optimizer

Mixed **Muon + AdamW** (ported to optax):

| Group | Optimizer | Notes |
|---|---|---|
| Attention/MLP matrices | **Muon** | Polar Express orthogonalization + NorMuon variance reduction |
| Embeddings (wte) | AdamW | b1=0.8, b2=0.995 |
| LM head | AdamW | Lower LR for stability |
| Value embeddings | AdamW | Half embedding LR |
| Per-layer scalars | AdamW | Separate groups for resid_lambdas and x0_lambdas |
| Smear/Backout | AdamW | No weight decay |

LR schedules: warmup (40 steps) -> constant -> warmdown (65% of total, final 5% of peak).
Falls back to pure AdamW on Flax 0.11 where NamedTuple state has issues.

## Inference Engine

Four generation modes with increasing performance:

| Mode | Function | Speed | Use Case |
|------|----------|-------|----------|
| Padded | `generate()` | ~1-2 tok/s | Testing, debugging |
| KV-cached | `generate_with_cache()` | ~10-50 tok/s | Production, Python loop |
| Fully JIT | `generate_fast()` | ~200+ tok/s | TPU inference via `jax.lax.while_loop` |
| Speculative | `generate_speculative()` | ~2-4x KV-cached | Large model + small draft model |

### Tool Use

The `Engine` class provides streaming generation with automatic tool execution:

```python
engine = Engine(model, tokenizer)
for token_column, masks in engine.generate(prompt_ids, num_samples=3, max_tokens=256):
    print(tokenizer.decode([token_column[0]]), end="")
```

When the model outputs `<|python_start|>2+2<|python_end|>`, the engine:
1. Tries the safe calculator (`use_calculator`) for math and `string.count()`
2. Falls back to sandboxed Python execution (`execute_code`) for general code
3. Injects `<|output_start|>4<|output_end|>` tokens back into the stream

### Speculative Decoding

Use a smaller draft model to propose tokens, verified in batch by the main model:

```python
from flaxchat.engine import generate_speculative

# draft_model: 2-layer, model: 12-layer (same vocab)
tokens = generate_speculative(model, draft_model, prompt_ids, draft_steps=4)
```

### Sandboxed Code Execution

For HumanEval evaluation and RL tool use:

```python
from flaxchat.execution import execute_code

result = execute_code("print(sum(range(10)))", timeout=5.0)
# ExecutionResult(success=True, stdout="45\n", stderr="", error=None)
```

Process isolation, signal-based timeouts, memory limits (Linux), and dangerous function blocking.

## Parallelism (built-in, not optional)

- **`compute_init()`** creates a mesh over ALL available devices automatically
- **Data parallelism**: `with_sharding_constraint(data, P('data'))` in every train step
- **FSDP**: `shard_model_fsdp()` for models exceeding single-device memory
- **Multi-host**: `jax.distributed.initialize()` + `jax.make_array_from_process_local_data()`
- **No manual all-reduce** — JAX SPMD compiler handles gradient synchronization

## Configuration

Single-dial depth-based auto-config — all hyperparameters derive from depth:

```python
from flaxchat.config import FlaxChatConfig

config = FlaxChatConfig.from_depth(
    depth=12,            # 12 layers
    aspect_ratio=64,     # base_dim = 12 * 64 = 768
    head_dim=128,        # n_heads = 768 / 128 = 6
    max_seq_len=2048,
    window_pattern="SSSL",
)
# -> 12 layers, 768 dims, 6 heads, ~79M params
```

## Evaluation Tasks

| Task | Type | Source |
|------|------|--------|
| MMLU | Categorical (4-choice) | `cais/mmlu` |
| ARC-Challenge | Categorical | `allenai/ai2_arc` |
| GSM8K | Generative (math + calculator) | `openai/gsm8k` |
| HumanEval | Generative (code + sandbox) | `openai/humaneval` |
| SpellingBee | Generative (tool use) | Built-in (30+ templates) |
| SmolTalk | Conversation quality | `HuggingFaceTB/smol-smoltalk` |
| CORE | ICL benchmark (DCLM paper) | Hellaswag, ARC, PIQA, Winogrande |

## Quick Start

### Install

```bash
pixi install    # or: pip install -e ".[dev]"
```

### Train locally on TinyStories

```bash
python -m scripts.run_tinystories --depth=4 --steps=1000
```

### Full pipeline on TPU pod

```bash
python -m scripts.pretrain --depth=24 --num-iterations=50000
python -m scripts.sft --base-model=d24
python -m scripts.rl --model=d24
python -m scripts.eval --model=d24 --tasks=all
python -m scripts.chat_web --model=d24
```

### Remote execution

```python
# Kaggle GPU (via kgz)
from flaxchat.remote import KaggleRunner
runner = KaggleRunner("https://...")
runner.run_pipeline(depth=8, steps=5000)

# GCP TPU (via tpuz)
from tpuz import TPU
tpu = TPU("my-tpu", accelerator="v6e-8")
tpu.up()
tpu.setup(extra_pip="flaxchat")
tpu.run("python -m scripts.pretrain --depth=12", sync=".")
```

## Project Structure

```
flaxchat/
├── flaxchat/                  # Core library (~3,500 LOC)
│   ├── gpt.py                 # GPT model (all nanochat features)
│   ├── optim.py               # Mixed Muon+AdamW optimizer (optax)
│   ├── engine.py              # Inference: padded, cached, JIT, speculative, tool use
│   ├── execution.py           # Sandboxed Python code execution
│   ├── eval.py                # CORE metric + BPB evaluation
│   ├── dataloader.py          # BOS-aligned best-fit packing
│   ├── tokenizer.py           # BPE tokenizer (rustbpe + tiktoken + HF)
│   ├── config.py              # Depth-based auto-config
│   ├── common.py              # Mesh, distributed, logging
│   ├── checkpoint.py          # Orbax checkpoint manager
│   ├── report.py              # Training reports
│   └── dataset.py             # Parquet file listing
├── scripts/                   # Executable scripts (~2,500 LOC)
├── tasks/                     # Evaluation tasks (MMLU, ARC, GSM8K, HumanEval, ...)
├── tests/                     # 148 unit tests
├── docs/                      # GitHub Pages documentation
├── configs/                   # YAML configuration templates
└── runs/                      # Launch scripts
```

## Test Suite

**148 tests** across 10 test files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_model.py` | 23 | GPT architecture, forward pass, loss, gradients, masking, JIT |
| `test_engine.py` | 17 | All 4 gen modes, speculative decoding, tool use |
| `test_optim.py` | 17 | Muon, LR/WD/momentum schedules |
| `test_execution.py` | 19 | Sandbox, timeout, safety guards |
| `test_tokenizer.py` | 15 | BPE train/encode/decode/save/load |
| `test_checkpoint.py` | 10 | Orbax save/load round-trip |
| `test_eval.py` | 9 | CORE, multiple-choice, generative |
| `test_dataloader.py` | 8 | BOS packing, sharding |
| `test_config.py` | 8 | Depth scaling, YAML/JSON |
| `test_common.py` | 13 | Mesh, dtype, distributed |

## Verified Results

### GPT-2 Base on FineWeb-Edu (Kaggle TPU v5e-8)

| Metric | Value |
|--------|-------|
| Model | 12L/768d/6h (GQA: 3kv) = 203.7M params |
| Training data | FineWeb-Edu 10BT (2B tokens used) |
| Hardware | Kaggle TPU v5e-8 (8 chips, bf16) |
| Throughput | **379,000 tok/s** |
| Final loss | **2.94** |
| Training time | ~1.5h |
| W&B | [irf-sic/flaxchat](https://wandb.ai/irf-sic/flaxchat) |

### TinyStories (Kaggle 2xT4 GPU)

| Metric | Value |
|--------|-------|
| Model | 8L/256d/8h = 18.9M params |
| Training data | 500K stories, 106M tokens |
| Throughput | 55,000 tok/s (data-parallel) |
| Best val loss | 2.20 |
| Pretrain time | 50 min |

### TinyStories (Kaggle TPU v5e-8)

| Metric | Value |
|--------|-------|
| Model | 8L/512d/8h = 90.2M params |
| Training data | TinyStories (50K stories) |
| Throughput | **149,000 tok/s** |
| Final loss | 2.79 |
| Training time | 109s (500 steps) |

**148 tests passing** on CPU (local), GPU (Kaggle 2xT4), and TPU (v5e-8).

## Comparison with nanochat

| | nanochat | flaxchat |
|---|---|---|
| Framework | PyTorch | JAX/Flax NNX |
| Hardware | NVIDIA GPU (8xH100) | TPU pods + GPUs |
| Distributed | DDP + torch.distributed | JAX SPMD mesh (automatic) |
| Compile | `torch.compile` | `jax.jit` / `nnx.jit` |
| Attention | Flash Attention 3 | `jax.nn.dot_product_attention` |
| Precision | bf16/fp16/fp8 | bf16 (TPU native) |
| Optimizer | Custom MuonAdamW | Custom optax Muon+AdamW |
| Checkpointing | Pickle-based | Orbax (async, cloud-friendly) |
| Generation | KV-cache + Python loop | 4 modes: padded, cached, JIT, speculative |
| Tool use | Calculator + Python REPL | Calculator + sandboxed REPL |
| Remote execution | N/A | Kaggle (kgz) + TPU (tpuz) |
| Config | Manual | Depth-based auto-scaling |

## Acknowledgments

This project is part of the **2026 Q1 TPU Sprint**, supported by the [Google AI Developer Programs](https://developers.google.com/programs) team.

We gratefully acknowledge:
- **[Google AI Developer Programs](https://developers.google.com/programs)** for issuing GCP credits that made large-scale training experiments possible
- **[TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)** for providing free access to Cloud TPU v4, v5e, and v6e accelerators
- **Kaggle** for providing free TPU v5e access for prototyping and validation

Built on:
- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax)
- [tpuz](https://github.com/mlnomadpy/tpuz) for TPU VM management
- [kgz](https://github.com/mlnomadpy/kgz) for Kaggle kernel execution

## License

MIT
