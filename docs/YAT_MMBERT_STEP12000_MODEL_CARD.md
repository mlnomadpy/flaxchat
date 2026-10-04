---
license: mit
language: multilingual
tags:
  - yat
  - flax
  - jax
  - sentence-embeddings
  - contrastive-learning
  - research
library_name: flax
base_model: mlnomad/yat-mmbert-base-contrastive-6000
model-index: []
---

# YAT mmBERT base — multilingual contrastive checkpoint, step 12,000

This is a research multilingual sentence-embedding model trained with [FlaxChat](https://github.com/mlnomadpy/flaxchat). It continues the [step-6,000 YAT contrastive checkpoint](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-6000) to step 12,000 on the expanded Global Voices corpus. Its 307.8M-parameter bidirectional encoder uses YAT attention and a YAT GLU feed-forward network. The YAT bias is fixed at 1, epsilon at 0.01, and alpha is trainable. The model is not qualified for production or code search.

The release contains `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, and an `export.json` receipt. Weights use Flax NNX state paths and require [FlaxChat's loader](https://github.com/mlnomadpy/flaxchat/blob/main/flaxchat/public_encoder.py); `transformers.AutoModel` cannot load them directly.

## Use

```python
from pathlib import Path
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from flaxchat.public_encoder import load_public_encoder

folder = Path(snapshot_download("mlnomad/yat-mmbert-base-contrastive-12000"))
tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
model = load_public_encoder(folder)
ids = jnp.asarray([tokenizer.encode("Hello world").ids], dtype=jnp.int32)
embeddings = model.pool(ids).astype(jnp.float32)
embeddings = embeddings / jnp.maximum(
    jnp.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12
)
```

`model.pool` averages non-padding token states, including special tokens. Pad unequal-length batches with tokenizer ID 0. Contrastive training used at most 128 tokens; the complete MTEB/CoIR evaluation used at most 512. Longer inputs have not been qualified for this use.

## Training and data

The step-12,000 continuation used 631,332 English–other-language training pairs from the official train split of 28 shards in [Sentence Transformers' Global Voices package](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices/tree/4cc20add371f246bb1559b543f8b0dea178a1803), a global batch of 64, and sequence length 128. Global Voices content is CC-BY-3.0; credit Global Voices contributors, OPUS alignment, and Sentence Transformers packaging. The model also inherits mmBERT and earlier YAT MLM training; see the [MLM model card](https://huggingface.co/mlnomad/yat-mmbert-base-mlm-48236). The exact prepared training manifest SHA-256 is `0eb6c456c95f51feb8ac5d1bf6d99841c792d8ebe8c037eddf65e9b52372c923`.

## Evaluation

The published checkpoint was evaluated on physical v5e Spot TPUs with **MTEB 2.21.8**: all 41 English-v2 tasks, all 131 Multilingual-v2 tasks, and all 10 CoIR tasks completed with zero failures. Values below are points on a 0–100 scale (except that a category can be negative when its underlying metric permits it). The mmBERT-base column reproduces Tables 4–6 of the [mmBERT paper](https://arxiv.org/html/2509.06888v1); it is **not** a paired run under the same conditions.

| Suite | YAT step 12,000, as published | Paper mmBERT-base | YAT coverage |
| --- | ---: | ---: | ---: |
| English MTEB, mean of seven categories | **45.86** | 53.9 | 41/41 tasks |
| Multilingual MTEB, mean of eight paper-listed categories | **46.63** | 54.1 | 131/131 tasks |
| Multilingual MTEB, native mean including InstructionReranking | **41.41** | Not reported | 131/131 tasks |
| CoIR, mean nDCG@10 | **21.03** | 42.2 | 10/10 tasks |

| English MTEB category | YAT | Paper mmBERT-base |
| --- | ---: | ---: |
| Pair classification | 66.17 | 80.2 |
| Classification | 65.91 | 64.8 |
| STS | 66.71 | 74.8 |
| Retrieval | 15.54 | 44.9 |
| Clustering | 40.88 | 41.7 |
| Reranking | 39.12 | 44.9 |
| Summarization | 26.71 | 26.0 |

| Multilingual MTEB category | YAT | Paper mmBERT-base |
| --- | ---: | ---: |
| Bitext mining | 51.85 | 59.2 |
| Pair classification | 71.11 | 79.2 |
| Classification | 51.52 | 53.6 |
| STS | 59.46 | 67.1 |
| Retrieval | 31.58 | 45.8 |
| Multilabel classification | 19.16 | 17.5 |
| Clustering | 42.81 | 40.2 |
| Reranking | 45.54 | 69.9 |
| Instruction reranking | -0.37 | Not reported |

| CoIR task, nDCG@10 points | YAT | Paper mmBERT-base |
| --- | ---: | ---: |
| Apps | 2.21 | 6.1 |
| CosQA | 3.45 | 24.6 |
| Synthetic Text2SQL | 37.17 | 48.1 |
| CodeSearchNet | 8.44 | 63.8 |
| CodeSearchNet-CCR | 21.58 | 41.7 |
| CodeTrans-Contest | 24.76 | 54.9 |
| CodeTrans-DL | 25.71 | 33.0 |
| StackOverflow QA | 36.70 | 62.1 |
| CodeFeedback-ST | 25.34 | 55.5 |
| CodeFeedback-MT | 24.96 | 32.4 |

The paper fine-tuned mmBERT-base for one epoch on **1.25 million MS MARCO hard triplets** and selected its learning rate from a sweep before MTEB and CoIR evaluation. This YAT release instead continued contrastive training on English-pivot Global Voices pairs and inherited mmBERT weights. The paper does not pin every MTEB task and dataset revision; our current registry can differ from its 2025 inventory. The displayed gaps therefore describe these *released checkpoints and protocols*, not the effect of the YAT architecture. GLUE and XTREME paper results required task-specific supervised fine-tuning and have **not** been measured for this YAT checkpoint. The weakest areas here are retrieval and code search; the model remains research-only.

The [compact benchmark receipt](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-12000/blob/84bb4b0cdf5388a59b95b5503fb0378dc5442cf3/evaluation/results-summary.json) contains every task score, category aggregate, model-file hashes, MTEB version, dataset revisions, and failure count. Evaluation used BF16 inference, mean pooling of non-padding tokens including special tokens, FP32 L2 normalization, no model prompt, and a 512-token maximum. MindSmall used a larger batch on a high-memory TPU; WebLINX's six official splits and MIRACL's 18 language subsets were merged only after exact coverage and identity checks.

### Paired diagnostics against the released base checkpoint

We also compared this checkpoint with the released, **non-MS-MARCO-fine-tuned** `jhu-clsp/mmBERT-base` using the same tokenizer, mean pooling, FP32 L2 normalization, and BF16 inference on a physical v5e-8 TPU. These selected diagnostics are distinct from the paper-table comparison above.

| Diagnostic | mmBERT-base | YAT step 12,000 |
| --- | ---: | ---: |
| Tatoeba, 112-subset macro F1 | 0.3016 | 0.5269 |
| Tatoeba, 88,877-pair accuracy | 0.3390 | 0.5683 |
| STSBenchmark.v2 test Spearman | 0.5393 | 0.6733 |
| STS17 test, mean of 11 subset Spearman scores | 0.4889 | 0.6383 |
| SciFact test nDCG@10 | 0.0365 | 0.2991 |
| NFCorpus test nDCG@10 | 0.0259 | 0.1124 |

These results do **not** isolate the YAT architecture: the YAT checkpoint received contrastive training, whereas the released mmBERT-base baseline did not. This comparison is not against mmBERT's MS MARCO-fine-tuned result. Tatoeba had no exact normalized-text overlap with the continuation corpus, but earlier pretraining exposure and paraphrase overlap cannot be excluded. The four other tasks lack a training-overlap audit. Native Tatoeba scoring has unverified official MTEB tie parity. The training data is English-pivot translated news, so broad-domain, long-document, code, and production retrieval quality remain unproven. See the [full comparison](https://github.com/mlnomadpy/flaxchat/blob/main/docs/YAT_MMBERT_PAIRED_BENCHMARKS.md).

## Provenance

Source checkpoint: step 12,000 of `encoder-contrastive-global-voices-50k-carry-0928`. Checkpoint metadata SHA-256: `ccf7788280d39c414627c45dcf071ecdda34974fe8f80d1d485b5b30102f1d18`. Tokenizer SHA-256: `197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356`. The exported weights were verified against the checkpoint's per-tensor manifest; their SHA-256 and tensor count are recorded in `export.json`.
