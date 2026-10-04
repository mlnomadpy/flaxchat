---
license: mit
language: multilingual
tags:
  - pytorch
  - yat
  - sentence-embeddings
  - information-retrieval
  - code-search
base_model: mlnomad/yat-mmbert-base-embedding-v1
model-index: []
---

# YAT mmBERT base embedding v1 — PyTorch conversion

Read the [release article](https://huggingface.co/spaces/mlnomad/yat-mmbert-embedding-blog) for the YAT block equations, training recipe, results, and limitations.

This is the PyTorch weight conversion of [YAT mmBERT base embedding v1](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1), not a separately trained model. The 307.8M-parameter encoder was fine-tuned for English and multilingual retrieval and code search. Its custom `yat_encoder.py` implements bidirectional YAT attention and YAT GLU feed-forward layers with fixed bias 1, fixed epsilon 0.01, and trained alpha parameters. It retains BF16 feature/distance arithmetic and FP32 residuals and attention scores.

## Use

Install `torch`, `safetensors`, `tokenizers`, and `huggingface_hub`. The repository includes a custom PyTorch loader; it is **not** a `transformers.AutoModel` checkpoint.

```python
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

folder = Path(snapshot_download("mlnomad/yat-mmbert-base-embedding-v1-pytorch"))
sys.path.insert(0, str(folder))
from yat_encoder import YatTorchEncoder

tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YatTorchEncoder.from_pretrained(folder, device=device)
ids = torch.tensor([tokenizer.encode("Find the matching document").ids], device=device)
with torch.no_grad():
    vectors = torch.nn.functional.normalize(model.pool(ids).float(), dim=-1)
```

For unequal-length batches, pad with token ID 0. Pooling averages non-padding token states, including special tokens. Training truncated queries at 128 tokens and documents at 256; quality on longer inputs needs separate validation. The dense BF16 distance fallback favors parity over serving speed, so benchmark latency before production use.

## Evaluation

These are the [source checkpoint's physical-TPU results](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1), not a separate PyTorch benchmark. Conversion parity is recorded in `parity.json` and the exact source/converted weight hashes in `conversion.json`.

On one physical TPU v5e chip, eight English, multilingual, and code inputs at 128 tokens gave a minimum pooled-vector cosine of **0.999960**, 100% top-1 and top-3 set agreement within that eight-item batch, and a maximum pairwise cosine-score drift of **0.003854** versus the JAX source. These are conversion checks, not full benchmark equivalence or a latency measurement; individual pooled coordinates differed by as much as 0.06483.

Updated October 4, 2026. Completed source-checkpoint evaluations used MTEB 2.21.8, mean pooling, FP32 L2 normalization and a 512-token limit. Scores are 0–100 points; MTEB means average categories, while CoIR and MIRACL average their ten tasks and 18 language subsets, respectively.

| Benchmark | JAX source checkpoint | Earlier YAT parent | mmBERT-base paper |
| --- | ---: | ---: | ---: |
| English MTEB v2, 41 tasks, seven-category mean | 52.78 | 45.86 | 53.9 |
| Multilingual MTEB v2, 131 tasks, eight-category mean | 48.21 | 46.63 | 54.1 |
| CoIR, ten-task mean | 43.45 | 21.03 | 42.2 |
| MIRACL hard-negative, 18-language mean | 46.37 | 8.06 | Not reported |

The paper scores come from [Marone et al., Tables 4–6](https://arxiv.org/html/2509.06888v1), after a different MS MARCO embedding fine-tune, with benchmark revisions not pinned to this evaluation. They provide context, not a controlled architecture comparison. The native nine-category multilingual mean, including InstructionReranking omitted from the paper's table, is **42.40** (parent **41.41**). The [source model card](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1) contains complete category, code-task and MIRACL-language tables, plus authenticated evaluation receipt links. GLUE and XTREME supervised fine-tunes have not been evaluated for this YAT checkpoint.

The source model improved retrieval over its earlier contrastive checkpoint but regressed on cross-language Tatoeba from 52.72 to 35.27. CodeSearchNet pairs were used for fine-tuning, so CodeSearchNet-family CoIR results need a train/evaluation overlap audit. The model inherits earlier mmBERT and YAT weights; it is not a from-scratch replacement for mmBERT. See the [source model card](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1) for data provenance, complete per-task results, and limitations.


The historical eight-input conversion check above does not qualify the expanded context/batch parity matrix now required by the harness. Complete PyTorch benchmark equivalence, long-context numerical qualification and serving speed remain unvalidated.


## JAX source inference speed on TPU — October 3, 2026

The JAX source weights and `jhu-clsp/mmBERT-base` were measured in the same FlaxChat JAX harness on **one physical TPU v5e chip**, with identical deterministic non-padding input IDs. Each shape used three warmups and ten synchronized timed repetitions. Medians include encoder forward, mean pooling and FP32 L2 normalization; download, tensor conversion/loading, tokenization, compilation and host output copies are excluded.

| Batch | Sequence length | YAT median latency, ms | mmBERT median latency, ms | YAT / mmBERT latency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 14.88 | 13.02 | 1.14× |
| 8 | 128 | 23.10 | 14.53 | 1.59× |
| 1 | 512 | 19.94 | 14.02 | 1.42× |
| 8 | 512 | 59.85 | 32.81 | 1.82× |

At batch 8 and length 512, YAT produced **133.7 sequences/s**, versus **243.8 sequences/s** for mmBERT. All eight model/shape cases produced finite normalized vectors. This is a narrow JAX implementation comparison on fully occupied synthetic shapes, not a PyTorch/GPU serving benchmark or the GPU efficiency measurement from the mmBERT paper. YAT was slower in all four measured shapes. [Protocol and raw-receipt locations](https://github.com/mlnomadpy/flaxchat/blob/master/docs/YAT_MMBERT_LIGHT_SPEED.md).

## JAX source native-vector diagnostics — October 4, 2026

A separate physical-TPU run of the JAX source analyzed **88 handcrafted examples**, using batch 8, padded length 128, mean pooling and L2 normalization. Examples cover eight English concepts and their translations into seven languages, eight English code queries, and matching Python/JavaScript functions. All native vectors were finite and nonzero. These simple probes are **not official benchmark datasets or a representative production corpus**.

| Diagnostic | YAT embedding v1 | mmBERT MLM base |
| --- | ---: | ---: |
| Mean distinct-pair cosine | 0.1573 | 0.7224 |
| Uncentered leading-direction energy | 17.85% | 72.71% |
| Centered energy-entropy effective rank, maximum 87 | 32.52 | 26.70 |
| Same semantic-group mean cosine | 0.6273 | 0.8406 |
| Different semantic-group mean cosine | 0.1248 | 0.7142 |
| Same-minus-different group cosine margin | 0.5025 | 0.1264 |
| Translation → English, R@1; 56 queries / 8 candidates | 100% | 100% |
| Query → code, R@1; 8 queries / 16 candidates | 100% | 50% |
| Query → code, MRR | 1.000 | 0.633 |

Effective rank is `exp(-sum(p * log(p)))`, with `p` the normalized squared singular-value energy of centered unit vectors. YAT's native `[88, 768]` output array was bit-identical across two TPU allocations. Lower common-direction concentration and a larger semantic cosine margin are diagnostic observations, not proof of better generalization. In particular, **this baseline is the released mmBERT MLM checkpoint, not the MS MARCO-fine-tuned embedding model behind the paper scores above**.

Both studies pinned YAT revision `571d0ef3d8e1381055845fce334a48be253c3fea` and mmBERT revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`; full weight hashes are retained in the reports. The native-vector fixture SHA256 is `a63db9a4496e27bd23355c58234b86967cb0f88b4bf725204cedcc148fd1357d`.

The attempted instrumented layer traces failed their original numerical agreement limits. Consequently, **attention entropy, per-layer attention/FFN contributions and adaptive-distance repair frequency remain unvalidated** and are excluded from these findings. No tolerance was relaxed. [Diagnostic report, failures and raw-receipt locations](https://github.com/mlnomadpy/flaxchat/blob/master/docs/YAT_ACTIVATION_ANALYSIS_2026-10-04.md).
