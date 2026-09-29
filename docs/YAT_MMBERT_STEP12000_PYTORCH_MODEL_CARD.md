---
license: mit
language: multilingual
tags:
  - pytorch
  - yat
  - sentence-embeddings
  - contrastive-learning
  - research
base_model: mlnomad/yat-mmbert-base-contrastive-12000
model-index: []
---

# YAT mmBERT base — PyTorch conversion, contrastive step 12,000

This is a **weight conversion**, not a separately trained model. It is the PyTorch implementation of [YAT mmBERT base contrastive step 12,000](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-12000), a 307.8M-parameter multilingual bidirectional encoder. The custom implementation in `yat_encoder.py` preserves YAT attention and YAT GLU FFN with fixed bias 1, fixed epsilon 0.01, and trained alpha parameters. It uses BF16 feature/distance arithmetic with FP32 residuals and attention scores, matching the source configuration.

## Use

Install `torch`, `safetensors`, `tokenizers`, and `huggingface_hub`. Download the repository and import its bundled `yat_encoder.py` directly. The same source is maintained in [FlaxChat](https://github.com/mlnomadpy/flaxchat/blob/master/torch_port/yat_encoder.py).

```python
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

folder = Path(snapshot_download("mlnomad/yat-mmbert-base-contrastive-12000-pytorch"))
sys.path.insert(0, str(folder))
from yat_encoder import YatTorchEncoder

tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
model = YatTorchEncoder.from_pretrained(folder, device="cuda" if torch.cuda.is_available() else "cpu")
ids = torch.tensor([tokenizer.encode("Find the matching document").ids], device=next(model.parameters()).device)
with torch.no_grad():
    vectors = torch.nn.functional.normalize(model.pool(ids).float(), dim=-1)
```

`model.pool` averages non-padding token states, including special tokens. Pad unequal-length rows with tokenizer ID 0. Training and parity validation used at most 128 tokens; longer sequences have not been qualified for retrieval. The release uses a custom PyTorch loader and is **not** a drop-in `transformers.AutoModel` checkpoint. The implementation currently uses fixed-shape dense BF16 distance computation, which favors correctness over serving speed.

## TPU conversion and parity

All 181 Flax state tensors were mapped to PyTorch names, with linear kernels transposed into PyTorch layout. Conversion ran on a GCP TPU VM from the public Flax release, without moving model weights through a local machine. The source weight SHA-256 is `91908ea06c933966b954b0948a34042798395fdf865e1564c71cf82db5c37956`; the converted PyTorch weight SHA-256 is `21d8a9c3a4b80ba32960f3d7b171991a1c9b1e892fee2e4c577743beffa40f7c`.

JAX and PyTorch/XLA each ran forward passes on a physical v5e TPU for the same eight 128-token inputs spanning several languages, code-like text, and varying padding. Measured parity:

| Output | Minimum vector cosine | Mean absolute difference |
| --- | ---: | ---: |
| Embedding layer | 0.9999999 | 0.00000001 |
| Layer 0 | 0.9999928 | 0.000400 |
| Layer 10 | 0.9998915 | 0.002316 |
| Final hidden states | 0.9998041 | 0.003447 |
| Pooled embeddings | **0.9998502** | **0.011219** |

The maximum pairwise cosine-score difference was 0.00503. Top-1 neighbors and top-3 neighbor sets agreed for all eight inputs. These are numerical parity checks on a small sample, not a full cross-platform or production retrieval benchmark. Differences result from BF16 rounding across the two XLA implementations; outputs are not bitwise identical. The exact report is included as `parity.json`.

## Training data and limitations

The model inherits mmBERT and YAT MLM pretraining, then multilingual contrastive training on 631,332 Global Voices English–other-language news pairs. Source content is by [Global Voices](https://globalvoices.org/about/) contributors under CC-BY-3.0; OPUS supplied alignment and Sentence Transformers packaged the parallel-sentence data. See the [Flax model card](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-12000) for the pinned data recipe, benchmark results, and caveats. This conversion does not establish production quality, code-search quality, or equivalence to an mmBERT model with matching contrastive training.
