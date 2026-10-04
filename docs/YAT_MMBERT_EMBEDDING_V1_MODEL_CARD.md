---
license: mit
language: multilingual
tags:
  - yat
  - flax
  - jax
  - sentence-embeddings
  - contrastive-learning
  - information-retrieval
  - code-search
  - research
library_name: flax
base_model: mlnomad/yat-mmbert-base-contrastive-12000
model-index: []
---

# YAT mmBERT base — embedding fine-tune, step 14,000

Read the [release article](https://huggingface.co/spaces/mlnomad/yat-mmbert-embedding-blog) for the YAT block equations, training recipe, results, and limitations.

This is a 307.8M-parameter research encoder for text and code retrieval. It fine-tunes [the YAT step-12,000 multilingual contrastive model](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-12000) on MS MARCO hard triplets, multilingual MIRACL triplets, and CodeSearchNet query–code pairs. The encoder uses bidirectional YAT attention and a YAT GLU feed-forward network; YAT bias is fixed at 1, epsilon at 0.01, and alpha is trainable. It inherits earlier mmBERT and YAT training, so it is not an architecture-isolated comparison with mmBERT.

The weights are Flax NNX state tensors. Load them with [FlaxChat](https://github.com/mlnomadpy/flaxchat)'s `load_public_encoder`; `transformers.AutoModel` does not load this format directly. The [PyTorch conversion](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1-pytorch) is a separate artifact with a narrow physical-TPU conversion check; the complete MTEB results below were measured using the JAX source, not rerun with PyTorch.

## Use

```python
from pathlib import Path
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from flaxchat.public_encoder import load_public_encoder

folder = Path(snapshot_download("mlnomad/yat-mmbert-base-embedding-v1"))
tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
model = load_public_encoder(folder)
ids = jnp.asarray([tokenizer.encode("Example search query").ids], dtype=jnp.int32)
embedding = model.pool(ids).astype(jnp.float32)
embedding = embedding / jnp.maximum(jnp.linalg.norm(embedding, axis=-1, keepdims=True), 1e-12)
```

For batches of unequal-length texts, pad with token ID 0. Pooling averages non-padding token states, including special tokens. Retrieval uses FP32 L2-normalized vectors and cosine similarity. Training truncated queries at 128 tokens and documents at 256; longer inputs need separate quality evaluation.

## Training

The stage ran **14,000 optimizer updates** at a global batch of **128 query–positive pairs** on **16 v5e Spot TPU chips** (four hosts). The total is 1,792,000 sampled pair presentations, including repeats: 89 MS MARCO, 13 MIRACL, and 26 CodeSearchNet pairs per batch. The prepared train sets contained 1,250,000, 144,512, and 300,000 rows, respectively. Prepared dev sets were held out by source group with exact normalized text overlap quarantined from training.

| Source | Pinned training dataset revision | Prepared train rows | Sampled presentations |
| --- | --- | ---: | ---: |
| MS MARCO hard triplets | [`sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1`](https://huggingface.co/datasets/sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1/tree/84ed2d35626f617d890bd493b4d6db69a741e0e2) | 1,250,000 | 1,246,000 |
| MIRACL multilingual triplets, 51 subsets | [`nlpai-lab/miracl-multilingual-triplets`](https://huggingface.co/datasets/nlpai-lab/miracl-multilingual-triplets/tree/71bc9f8e7d86b55203ed3104362b0661789a6c31) | 144,512 | 182,000 |
| CodeSearchNet query–code pairs | [`sentence-transformers/codesearchnet`](https://huggingface.co/datasets/sentence-transformers/codesearchnet/tree/079a958b01dc87cf07b66a68414c4b4196d889cc) | 300,000 | 364,000 |

The objective was symmetric global in-batch InfoNCE with mined negatives when supplied and duplicate masking. The stage used temperature 0.05, AdamW with FP32 optimizer state, peak learning rate 3e-5, 200 warmup steps, cosine decay to 10% of peak, weight decay 0.01, and global-norm clipping at 1. The per-source sampler is replayable across exact checkpoint resumes. Checkpoints were saved every 500 steps; the final step-14,000 checkpoint completed successfully.

## Evaluation status and limitations

Updated October 4, 2026. The completed benchmark scores below remain the September evaluation of these released weights. The October speed and native-vector diagnostics are reported separately; they are not additional MTEB tasks or a new leaderboard evaluation.

The physical-TPU evaluation completed the full English and multilingual MTEB v2 suites, CoIR, and all 18 MIRACL hard-negative language subsets. It used MTEB 2.21.8 and exactly the same task inventory, dataset revisions, tokenizer, model configuration, 512-token limit, mean pooling, and FP32 normalization as the [parent checkpoint's completed evaluation](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-12000). Scores are each task's MTEB main-score points (0–100), except the CoIR and MIRACL rows, which are means across all ten tasks and all 18 language subsets, respectively.

| Benchmark | This fine-tune | Parent checkpoint | mmBERT-base paper |
| --- | ---: | ---: | ---: |
| English MTEB v2, 41 tasks, seven-category mean | 52.78 | 45.86 | 53.9 |
| Multilingual MTEB v2, 131 tasks, eight-category mean | 48.21 | 46.63 | 54.1 |
| CoIR, ten-task mean | 43.45 | 21.03 | 42.2 |
| MIRACL hard-negative, 18-language mean | 46.37 | 8.06 | Not reported |

The paper columns use [Marone et al., Tables 4–6](https://arxiv.org/html/2509.06888v1). Different fine-tuning data and unpinned paper benchmark revisions prevent an architecture-isolated comparison. The CoIR code data also needs a train/evaluation overlap audit.

| Completed task | This fine-tune | Parent checkpoint | Change |
| --- | ---: | ---: | ---: |
| English STSBenchmark | 74.06 | 66.91 | +7.15 |
| English ArguAna | 45.12 | 41.25 | +3.87 |
| English FiQA2018 | 25.32 | 6.01 | +19.30 |
| English SCIDOCS | 10.59 | 4.02 | +6.57 |
| English AskUbuntuDupQuestions | 56.52 | 49.07 | +7.46 |
| English TRECCOVID | 69.23 | 20.19 | +49.04 |
| CoIR retrieval, all 10 tasks | **43.45** | **21.03** | **+22.42** |
| MIRACL hard-negative retrieval, all 18 languages | **46.37** | **8.06** | **+38.31** |
| Multilingual STS17 | 67.11 | 63.79 | +3.32 |
| Multilingual Tatoeba | **35.27** | **52.72** | **−17.45** |

The complete CoIR breakdown is shown alongside **paper-reported mmBERT-base** from [Marone et al., Table 6](https://arxiv.org/html/2509.06888v1). The paper's model had a different embedding fine-tuning recipe and does not pin the exact MTEB/CoIR package and dataset revisions used here. These columns are useful context, **not a controlled architecture comparison**. Our fine-tune inherited mmBERT weights and additionally trained on CodeSearchNet pairs, so its CodeSearchNet-family scores require a train/evaluation overlap audit.

| CoIR task | This fine-tune | Parent checkpoint | mmBERT-base paper |
| --- | ---: | ---: | ---: |
| AppsRetrieval | 5.00 | 2.21 | 6.1 |
| COIRCodeSearchNetRetrieval | 71.47 | 8.44 | 63.8 |
| CodeFeedbackMT | 34.99 | 24.96 | 32.4 |
| CodeFeedbackST | 57.92 | 25.34 | 55.5 |
| CodeSearchNetCCRetrieval | 42.70 | 21.58 | 41.7 |
| CodeTransOceanContest | 45.38 | 24.76 | 54.9 |
| CodeTransOceanDL | 29.87 | 25.71 | 33.0 |
| CosQA | 28.69 | 3.45 | 24.6 |
| StackOverflowQA | 65.53 | 36.70 | 62.1 |
| SyntheticText2SQL | 52.99 | 37.17 | 48.1 |
| **Mean, 10 tasks** | **43.45** | **21.03** | **42.2** |

The complete MIRACL hard-negative retrieval breakdown is:

| Language | This fine-tune | Parent checkpoint |
| --- | ---: | ---: |
| Arabic | 58.10 | 3.18 |
| Bengali | 45.44 | 0.89 |
| German | 41.62 | 17.15 |
| English | 48.12 | 6.88 |
| Spanish | 42.59 | 15.76 |
| Persian | 39.71 | 4.88 |
| Finnish | 51.16 | 10.93 |
| French | 44.06 | 14.44 |
| Hindi | 32.80 | 4.24 |
| Indonesian | 40.80 | 7.79 |
| Japanese | 52.62 | 3.62 |
| Korean | 45.93 | 11.19 |
| Russian | 49.19 | 7.59 |
| Swahili | 44.86 | 12.91 |
| Telugu | 45.93 | 0.55 |
| Thai | 48.19 | 7.02 |
| Yoruba | 57.31 | 10.62 |
| Chinese | 46.24 | 5.43 |

The Tatoeba decline spans **108 of 112** language-pair subsets. This checkpoint is therefore **not a general multilingual improvement**: within-language MIRACL retrieval improved in all 18 languages, while cross-language Tatoeba bitext retrieval worsened. The complete 41-task English and 131-task multilingual MTEB suites are reported below. The training mixture devoted 10% of each batch to MIRACL and did not replay the earlier Global Voices bitext data, a plausible cause of lost cross-language alignment, though the current evaluation does not isolate the cause.

### MTEB v2 suite comparison and remaining gaps

The **complete 41-task English MTEB 2.21.8 suite** finished on physical TPUs with zero failures and the same model and protocol hashes as the selected-task evaluation. Its seven-category mean is **52.78**, versus **45.86** for the parent and **53.9** reported for mmBERT-base in [Table 4](https://arxiv.org/html/2509.06888v1). The paper selected the best of four learning rates on MTEB after one MS MARCO epoch, while this YAT checkpoint used a fixed fine-tuning recipe with additional MIRACL and code data; the paper also does not pin identical benchmark revisions. These columns are informative but are not a controlled architecture comparison.

| English MTEB v2 category | This fine-tune | Parent checkpoint | mmBERT-base paper |
| --- | ---: | ---: | ---: |
| Pair classification | 78.33 | 66.17 | 80.2 |
| Classification | 66.32 | 65.91 | 64.8 |
| STS | 74.70 | 66.71 | 74.8 |
| Retrieval | 42.96 | 15.54 | 44.9 |
| Clustering | 40.95 | 40.88 | 41.7 |
| Reranking | 43.63 | 39.12 | 44.9 |
| Summarization | **22.58** | **26.71** | 26.0 |
| **Category mean** | **52.78** | **45.86** | **53.9** |

The fine-tune improves the English mean, but summarization regresses from the parent. The **complete 131-task multilingual MTEB 2.21.8 suite** finished on physical TPUs, including all six WebLINX splits and a clean rerun of a benchmark dataset-loading failure. Its eight-category mean is **48.21**, versus **46.63** for the parent and **54.1** reported for mmBERT-base in [Table 5](https://arxiv.org/html/2509.06888v1). The paper did not pin identical task/dataset revisions and used a different embedding fine-tuning recipe, so this is context rather than a controlled architecture comparison. The paper did not publish a separate MIRACL hard-negative 18-language mean, nor individual scores for the selected STSBenchmark, STS17, or Tatoeba tasks above.

The underlying [English MTEB receipt](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1/blob/571d0ef3d8e1381055845fce334a48be253c3fea/evaluation/english-complete.json), [multilingual MTEB receipt](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1/blob/571d0ef3d8e1381055845fce334a48be253c3fea/evaluation/multilingual-complete.json), [CoIR receipt](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1/blob/571d0ef3d8e1381055845fce334a48be253c3fea/evaluation/coir-complete.json), and [MIRACL receipt](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1/blob/571d0ef3d8e1381055845fce334a48be253c3fea/evaluation/miracl-complete.json) include task results, dataset revisions, and evaluation identity hashes. The [full-suite summary](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1/blob/571d0ef3d8e1381055845fce334a48be253c3fea/evaluation/full-mteb-summary.json) includes both multilingual means.

| Multilingual MTEB v2 category | This fine-tune | Parent checkpoint | mmBERT-base paper |
| --- | ---: | ---: | ---: |
| Bitext mining | 43.99 | 51.85 | 59.2 |
| Pair classification | 73.63 | 71.11 | 79.2 |
| Classification | 50.57 | 51.52 | 53.6 |
| STS | 64.45 | 59.46 | 67.1 |
| Retrieval | 44.33 | 31.58 | 45.8 |
| Multilabel classification | 15.07 | 19.16 | 17.5 |
| Clustering | 37.96 | 42.81 | 40.2 |
| Reranking | 55.66 | 45.54 | 69.9 |
| **Eight-category mean** | **48.21** | **46.63** | **54.1** |

The parent and fine-tune use the same pinned 131-task inventory. MTEB 2.21.8 also contains an InstructionReranking category absent from the paper's eight-category table; it is excluded from the comparison mean. Including it, the native nine-category means are **42.40** for this fine-tune and **41.41** for the parent. The bitext and clustering regressions mean this fine-tune is not a general improvement over its parent despite the higher mean; Tatoeba in particular declined sharply.

The paper also reports supervised GLUE and XTREME for mmBERT-base ([Tables 2 and 3](https://arxiv.org/html/2509.06888v1)). These are the paper's task-specific fine-tuning results; this frozen embedding checkpoint has no corresponding supervised fine-tunes. The differing task metrics should not be averaged across this table to produce an embedding score.

| Supervised benchmark | mmBERT-base paper | YAT embedding-v1 |
| --- | ---: | --- |
| GLUE CoLA | 61.9 | Not evaluated |
| GLUE SST-2 | 94.0 | Not evaluated |
| GLUE MRPC | 91.9 | Not evaluated |
| GLUE STS-B | 91.0 | Not evaluated |
| GLUE QQP | 89.4 | Not evaluated |
| GLUE MNLI | 87.7 | Not evaluated |
| GLUE QNLI | 93.3 | Not evaluated |
| GLUE RTE | 85.6 | Not evaluated |
| **GLUE mean** | **86.3** | **Not evaluated** |
| XTREME XNLI | 77.1 | Not evaluated |
| XTREME PAWS-X | 87.7 | Not evaluated |
| XTREME XCOPA | 67.5 | Not evaluated |
| XTREME XQuAD | 77.6 | Not evaluated |
| XTREME MLQA | 66.0 | Not evaluated |
| XTREME TyDiQA | 74.5 | Not evaluated |
| XTREME WikiANN | 58.2 | Not evaluated |
| XTREME UDPOS | 74.0 | Not evaluated |
| **XTREME mean** | **72.8** | **Not evaluated** |

The paper also reports an **in-distribution language subset** comparison with EuroBERT in [Table 7](https://arxiv.org/html/2509.06888v1). These subset means have different language coverage from the XTREME rows above and require supervised task fine-tuning; they are not embedding scores.

| Supervised language subset | mmBERT-base paper | YAT embedding-v1 |
| --- | ---: | --- |
| XNLI, EuroBERT languages | 77.7 | Not evaluated |
| PAWS-X, EuroBERT languages | 89.0 | Not evaluated |

The paper discusses low-resource TiQuAD and FoQA in §4.4 without a full numeric table for the final base model. Its efficiency figure uses GPU throughput rather than a directly comparable TPU setup. Neither should be assigned an invented score here.

Fixed held-out training-diagnostic batches at step 14,000 had contrastive losses of 0.155 (MS MARCO), 0.122 (MIRACL), and 0.175 (code). Those values are not retrieval metrics. CodeSearchNet was also used for fine-tuning, so any CoIR CodeSearchNet score needs a train/evaluation overlap audit before being treated as independent generalization evidence.

The model may reflect biases, errors, and licensing conditions of the inherited pretraining data and the three fine-tuning sources. Text/code pairs are not proof of production code-search quality. Check data rights, privacy, safety, and task-specific accuracy before deploying. The public release contains model weights and tokenizer, not the training examples or optimizer checkpoint.


## Inference speed on TPU — October 3, 2026

The released YAT weights and `jhu-clsp/mmBERT-base` were measured in the same FlaxChat JAX harness on **one physical TPU v5e chip**, with identical deterministic non-padding input IDs. Each shape used three warmups and ten synchronized timed repetitions. Medians include encoder forward, mean pooling and FP32 L2 normalization; download, tensor conversion/loading, tokenization, compilation and host output copies are excluded.

| Batch | Sequence length | YAT median latency, ms | mmBERT median latency, ms | YAT / mmBERT latency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 14.88 | 13.02 | 1.14× |
| 8 | 128 | 23.10 | 14.53 | 1.59× |
| 1 | 512 | 19.94 | 14.02 | 1.42× |
| 8 | 512 | 59.85 | 32.81 | 1.82× |

At batch 8 and length 512, YAT produced **133.7 sequences/s**, versus **243.8 sequences/s** for mmBERT. All eight model/shape cases produced finite normalized vectors. This is a narrow JAX implementation comparison on fully occupied synthetic shapes, not a PyTorch/GPU serving benchmark or the GPU efficiency measurement from the mmBERT paper. YAT was slower in all four measured shapes. [Protocol and raw-receipt locations](https://github.com/mlnomadpy/flaxchat/blob/master/docs/YAT_MMBERT_LIGHT_SPEED.md).

## Native-vector diagnostics — October 4, 2026

A separate physical-TPU run analyzed **88 handcrafted examples**, using batch 8, padded length 128, mean pooling and L2 normalization. Examples cover eight English concepts and their translations into seven languages, eight English code queries, and matching Python/JavaScript functions. All native vectors were finite and nonzero. These simple probes are **not official benchmark datasets or a representative production corpus**.

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
