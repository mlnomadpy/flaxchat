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

`model.pool` averages non-padding token states, including special tokens. Pad unequal-length batches with tokenizer ID 0. Training and reported retrieval evaluation used at most 128 tokens; longer sequences have not been qualified for this use.

## Training and data

The step-12,000 continuation used 631,332 English–other-language training pairs from the official train split of 28 shards in [Sentence Transformers' Global Voices package](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices/tree/4cc20add371f246bb1559b543f8b0dea178a1803), a global batch of 64, and sequence length 128. Global Voices content is CC-BY-3.0; credit Global Voices contributors, OPUS alignment, and Sentence Transformers packaging. The model also inherits mmBERT and earlier YAT MLM training; see the [MLM model card](https://huggingface.co/mlnomad/yat-mmbert-base-mlm-48236). The exact prepared training manifest SHA-256 is `0eb6c456c95f51feb8ac5d1bf6d99841c792d8ebe8c037eddf65e9b52372c923`.

## Evaluation

We compared this checkpoint with the released `jhu-clsp/mmBERT-base` using the same tokenizer, mean pooling, FP32 L2 normalization, and BF16 inference on a physical v5e-8 TPU.

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
