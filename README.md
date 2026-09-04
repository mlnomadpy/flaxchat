# flaxchat

Reproducibility and checkpoint contracts are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[docs/CHECKPOINT_FORMAT.md](docs/CHECKPOINT_FORMAT.md).
Release gates and compatibility are documented in
[docs/RELEASES.md](docs/RELEASES.md).
Exact accelerator measurements and their limitations are in
[docs/RESULTS.md](docs/RESULTS.md).
The current engineering risks, refactoring target, and prioritized backlog are
in [docs/SYSTEM_AUDIT_2026-09-03.md](docs/SYSTEM_AUDIT_2026-09-03.md).
Supported Python and accelerator combinations are in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

A minimal, end-to-end LLM training harness for **Google Cloud TPU pods**, built on **JAX/Flax NNX**.

JAX/Flax NNX adaptation of [nanochat](https://github.com/karpathy/nanochat),
with TPU-oriented execution and speculative decoding.

```bash
pixi install
pixi run test
pixi run test-e2e                 # offline end-to-end acceptance smoke
python -m scripts.run_tinystories # pinned TinyStories end-to-end pipeline
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
| Padded | `generate()` | Measured per run | Testing, debugging |
| KV-cached | `generate_with_cache()` | Measured per run | Production, Python loop |
| Fully JIT | `generate_fast()` | Measured per run | TPU inference via `jax.lax.while_loop` |
| Speculative | `generate_speculative()` | Measured per pairing | Large model + small draft model |

### Tool Use

The `Engine` class provides streaming generation with automatic tool execution:

```python
engine = Engine(model, tokenizer)
for token_column, masks in engine.generate(prompt_ids, num_samples=3, max_tokens=256):
    print(tokenizer.decode([token_column[0]]), end="")
```

When the model outputs `<|python_start|>2+2<|python_end|>`, the engine:
1. Tries the safe calculator (`use_calculator`) for math and `string.count()`
2. Keeps generated Python disabled by default; reviewed code can opt into the
   best-effort reliability guard (`execute_code(..., trusted=True)`)
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

result = execute_code("print(sum(range(10)))", timeout=5.0, trusted=True)
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
| HumanEval | Generative (code execution disabled by default) | `openai/humaneval` |
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
python -m scripts.run_tinystories \
  --layers=4 --embedding-dim=256 \
  --pretrain-steps=1000 --sft-steps=100 --rl-steps=100
```

The dataset is streamed from an immutable revision. The output directory
contains the trained tokenizer, complete Orbax state, generated sample, stage
metrics, source/config/data hashes, licensing metadata, and `run_manifest.json`.
Use `--smoke` for the committed offline corpus.

### Full pipeline on TPU pod

```bash
python -m scripts.pretrain --depth=24 --num-iterations=50000
python -m scripts.sft --base-model=d24
python -m scripts.rl --model=d24
python -m scripts.eval --model=d24 --tasks=all
python -m scripts.chat_web --model=d24
```

### Remote execution

```bash
# Bundle the full suite into one Kaggle v5e-8 allocation and wait for artifacts.
python -m scripts.kaggle_tpu_tests \
  --kernel-id OWNER/flaxchat-tpu-tests --wait

# GCP TPU (via tpuz)
python - <<'PY'
from tpuz import TPU
tpu = TPU("my-tpu", accelerator="v6e-8")
tpu.up()
tpu.setup(extra_pip="flaxchat")
tpu.run("python -m scripts.pretrain --depth=12", sync=".")
PY
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
├── tests/                     # unit, integration, sharding, and accelerator tests
├── docs/                      # GitHub Pages documentation
├── configs/                   # YAML configuration templates
└── runs/                      # Launch scripts
```

## Test Suite

The test suite covers model semantics, four
generation modes, optimizer schedules and numerical safety, guarded execution,
tokenizers, exact dataloader resume, checkpoint integrity, configuration,
evaluation protocols, reports, datasets, sharding, and TPU attention parity.

```bash
pixi run test-quick       # deterministic CPU developer loop
pixi run test-coverage    # branch coverage with a 65% floor
pixi run test-multidevice # eight virtual CPU devices
pixi run lint             # fatal Python correctness/static errors
pixi run typecheck        # typed configuration and pipeline contracts
pixi run audit            # declared dependency vulnerability audit
```

## Verified Results

The canonical, machine-readable acceptance records are indexed in
[`docs/RESULTS.md`](docs/RESULTS.md). The latest verified TPU bundle is tied to
the full source SHA and distinguishes acceptance tests from scaling or quality claims.
Matched nanochat/MaxText measurements remain explicitly pending; no comparison is
presented as apples-to-apples until every protocol record validates.

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
| Tool use | Calculator + Python REPL | Calculator + opt-in guarded REPL |
| Remote execution | N/A | Kaggle CLI + TPU (tpuz) |
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
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) for headless accelerator execution

## License

MIT
