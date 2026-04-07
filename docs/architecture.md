---
layout: page
title: Architecture
permalink: /architecture/
---

# Architecture

## System Overview

flaxchat is a complete LLM training pipeline that runs on TPU pods and GPUs with automatic data parallelism.

```mermaid
graph TD
    subgraph "User Interface"
        CLI[CLI Scripts]
        WEB[Web Chat UI]
        REMOTE[Remote Runners]
    end

    subgraph "Core Library"
        GPT[gpt.py<br/>GPT Model]
        OPT[optim.py<br/>Muon + AdamW]
        DL[dataloader.py<br/>BOS Best-Fit Packing]
        TOK[tokenizer.py<br/>BPE rustbpe+tiktoken]
        ENG[engine.py<br/>KV-Cache Inference]
        EVAL[eval.py<br/>CORE + BPB]
        CKPT[checkpoint.py<br/>Orbax]
        CFG[config.py<br/>Depth Auto-Config]
        COMMON[common.py<br/>Mesh + Distributed]
        REPORT[report.py<br/>Training Reports]
    end

    subgraph "Training Scripts"
        PT[pretrain.py]
        SFT[sft.py]
        RL[rl.py]
        EV[eval script]
    end

    subgraph "Infrastructure"
        TPU[cloud/launcher.py<br/>GCP TPU Pods]
        KAGGLE[remote/kaggle_runner.py<br/>Kaggle GPUs]
    end

    CLI --> PT & SFT & RL & EV
    WEB --> ENG
    REMOTE --> TPU & KAGGLE

    PT --> GPT & OPT & DL & COMMON & CKPT
    SFT --> GPT & OPT & COMMON & CKPT
    RL --> GPT & ENG & COMMON

    GPT --> CFG
    OPT --> CFG
    DL --> TOK
    ENG --> GPT
    EV --> ENG & EVAL

    COMMON --> |"Mesh + Sharding"| GPT & DL & PT & SFT
```

## Module Dependency Graph

```
flaxchat/__init__.py
    ├── config.py          (no internal deps)
    ├── common.py          (no internal deps)
    ├── gpt.py             ← config.py, common.py
    ├── optim.py           ← config.py (via setup_optimizer)
    ├── tokenizer.py       (no internal deps)
    ├── dataloader.py      ← dataset.py, common.py
    ├── dataset.py         ← common.py
    ├── engine.py          ← gpt.py, common.py
    ├── eval.py            ← common.py
    ├── checkpoint.py      (no internal deps, uses orbax)
    ├── report.py          ← common.py
    ├── remote/
    │   ├── base.py        (no deps — abstract interface)
    │   └── kaggle_runner.py ← base.py
    └── cloud/
        ├── tpu_vm.py      (no internal deps — uses gcloud CLI)
        └── launcher.py    ← tpu_vm.py, remote/base.py
```

No circular dependencies. `config.py` and `common.py` are leaf modules.

## GPT Model Architecture

```mermaid
graph TD
    INPUT[Input Tokens<br/>B x T] --> WTE[Token Embedding<br/>wte]
    WTE --> NORM1[RMS Norm]
    NORM1 --> SMEAR[Smear<br/>bigram mixing]

    SMEAR --> BLOCK1[Block 0]
    BLOCK1 --> BLOCK2[Block 1]
    BLOCK2 --> BLOCKN[Block N-1]

    subgraph "Transformer Block"
        direction TB
        RL[resid_lambda * x<br/>+ x0_lambda * x0]
        RL --> ATTN_NORM[RMS Norm]
        ATTN_NORM --> ATTN[Causal Self-Attention<br/>RoPE + QK Norm + GQA]
        VE[Value Embedding] -.-> ATTN
        ATTN --> ADD1[Residual Add]
        ADD1 --> MLP_NORM[RMS Norm]
        MLP_NORM --> MLP[MLP<br/>ReLU^2]
        MLP --> ADD2[Residual Add]
    end

    BLOCKN --> BACKOUT[Backout<br/>subtract mid-layer]
    BACKOUT --> NORM2[RMS Norm]
    NORM2 --> LM_HEAD[LM Head<br/>untied weights]
    LM_HEAD --> SOFTCAP[Logit Softcap<br/>15 * tanh x/15]
    SOFTCAP --> OUTPUT[Logits<br/>B x T x V]
```

### Attention Detail

```mermaid
graph LR
    X[Input x] --> Q[Linear → Q]
    X --> K[Linear → K]
    X --> V[Linear → V]

    Q --> ROPE_Q[RoPE]
    K --> ROPE_K[RoPE]

    ROPE_Q --> QKN_Q[QK Norm * 1.2]
    ROPE_K --> QKN_K[QK Norm * 1.2]

    QKN_Q --> DOT[dot_product_attention<br/>+ causal mask<br/>+ sliding window]
    QKN_K --> DOT
    V --> DOT

    VE[Value Embedding] -.-> |"gate * ve"| V

    DOT --> PROJ[Linear → output]
```

## Training Pipeline

```mermaid
graph LR
    subgraph "Stage 1: Data"
        RAW[Raw Text<br/>ClimbMix-400B] --> TOK_TRAIN[Train BPE<br/>vocab=32K]
        TOK_TRAIN --> TOKENIZE[Tokenize<br/>→ parquet shards]
    end

    subgraph "Stage 2: Pretrain"
        TOKENIZE --> DL[BOS Best-Fit<br/>Dataloader]
        DL --> TRAIN[Pretrain GPT<br/>Muon + AdamW]
        TRAIN --> |"Chinchilla<br/>scaling laws"| TRAIN
        TRAIN --> BASE[Base Model<br/>checkpoint]
    end

    subgraph "Stage 3: SFT"
        BASE --> SFT_TRAIN[SFT on<br/>SmolTalk]
        SFT_TRAIN --> SFT_MODEL[SFT Model]
    end

    subgraph "Stage 4: RL"
        SFT_MODEL --> RL_TRAIN[GRPO on<br/>GSM8K]
        RL_TRAIN --> RL_MODEL[RL Model]
    end

    subgraph "Stage 5: Eval"
        RL_MODEL --> EVAL[MMLU, ARC<br/>GSM8K, HumanEval<br/>SpellingBee]
        EVAL --> REPORT[Training<br/>Report]
    end

    subgraph "Stage 6: Serve"
        RL_MODEL --> CHAT[Web Chat UI<br/>FastAPI + WS]
    end
```

## Data Parallelism

```mermaid
graph TD
    subgraph "Host 0"
        D0[Local Data Shard 0]
        GPU0[GPU/TPU 0]
        GPU1[GPU/TPU 1]
    end

    subgraph "Host 1 (multi-host)"
        D1[Local Data Shard 1]
        GPU2[GPU/TPU 2]
        GPU3[GPU/TPU 3]
    end

    MESH[JAX Mesh<br/>axis='data'] --> GPU0 & GPU1 & GPU2 & GPU3

    D0 --> |"P('data')"| GPU0 & GPU1
    D1 --> |"P('data')"| GPU2 & GPU3

    GPU0 & GPU1 & GPU2 & GPU3 --> |"Auto all-reduce<br/>(XLA SPMD)"| GRADS[Synchronized Gradients]
    GRADS --> UPDATE[Optimizer Update<br/>Replicated Params]
```

### Sharding Strategy

| Component | Sharding | Mesh Axis |
|-----------|----------|-----------|
| Input data (batch dim) | `P('data')` | Split across devices |
| Model params | `P()` | Replicated on all devices |
| Gradients | Auto all-reduce | XLA handles it |
| Optimizer state | `P()` or `P('fsdp')` | Replicated or sharded |

For models too large for one device, use `shard_model_fsdp()` which shards
the first dimension of 2D+ params across the `fsdp` mesh axis.
