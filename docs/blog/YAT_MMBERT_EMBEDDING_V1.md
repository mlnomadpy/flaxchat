# YAT mmBERT Embedding v1: a multilingual encoder for retrieval and code search

*September 2026*

We have released [YAT mmBERT base embedding v1](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1), a 307.8M-parameter bidirectional encoder for multilingual search and code retrieval. The [PyTorch conversion](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1-pytorch) is public alongside the original Flax/JAX weights. It is an embedding fine-tune of our earlier YAT contrastive checkpoint, which itself inherits mmBERT weights. This release is not a from-scratch pretraining result or an architecture-isolated comparison with mmBERT.

The model combines YAT attention with a YAT GLU feed-forward network. The YAT bias is fixed at 1, epsilon at 0.01, and alpha is trained. We fine-tuned on MS MARCO hard triplets, multilingual MIRACL triplets, and CodeSearchNet query–code pairs. The final stage ran 14,000 updates with a global batch of 128 query–positive pairs on 16 v5e Spot TPU chips. This amounted to 1.792 million sampled pair presentations, including repeats. The [model card](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1) records the data revisions, batch mix, optimizer, and evaluation receipts.

## Inside the YAT blocks

For vectors $u,v$ of the same width, the response used by both blocks is

$$
K_{\alpha}(u,v) = \alpha\,\frac{(u^\top v + 1)^2}{\lVert u-v\rVert_2^2 + 0.01}.
$$

The numerator bias **1** and denominator epsilon **0.01** are fixed. The scalar $\alpha$ is learned. The attention and feed-forward blocks each have their own trainable alpha in every encoder layer; the equation describes their common form, not shared parameters. The 0.01 here belongs to the YAT response and should not be confused with the layer-normalization epsilon.

In an attention head, $q_i$ and $k_j$ are projected token vectors after rotary position encoding. Instead of the usual scaled dot-product logit, the model computes $s_{ij}=K_{\alpha_{\mathrm{attn}}}(q_i,k_j)$. It masks padding and, on local-attention layers, tokens outside the window. It then applies ordinary softmax and uses the resulting probabilities to combine values:

$$
p_{ij}=\operatorname{softmax}_{j}(s_{ij}),\qquad
o_i=\sum_{j}p_{ij}v_j.
$$

The YAT response is therefore the **score supplied to softmax**, not a replacement for softmax itself. The encoder alternates local and global bidirectional attention according to its configuration.

The feed-forward block has the two linear branches of a gated FFN. Write $a_j=x^\top w_j$ for the first branch and $g_j=x^\top u_j$ for the gate branch, where $w_j$ and $u_j$ are columns of the respective input projection weights. Its hidden output is

$$
h_j=K_{\alpha_{\mathrm{ffn}}}(x,w_j)\,g_j
=\alpha_{\mathrm{ffn}}\,\frac{(x^\top w_j+1)^2}{\lVert x-w_j\rVert_2^2+0.01}\,(x^\top u_j),
\qquad \operatorname{FFN}(x)=W_o h.
$$

This is the YAT analogue of $\operatorname{GeGLU}(x)=\operatorname{GELU}(W_a x)\odot(W_g x)$: the second branch remains a **linear gate**, while the first uses the YAT response. The block adds the attention and FFN projections through FP32 residual paths. Feature vectors and distance calculations use BF16. The feed-forward distance code reuses dot products when safe and falls back to direct coordinate differences for cancellation-sensitive cases; attention forms scores in FP32 before softmax. These are the choices in the released checkpoint, not an all-BF16 or fused-attention claim. The [published PyTorch implementation](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1-pytorch/blob/main/yat_encoder.py) provides an executable definition.

## What improved

We evaluated the new checkpoint and its step-12,000 parent using the same tokenizer, pooling, truncation, task inventory, dataset revisions, and MTEB 2.21.8 harness on physical TPUs. Scores below are points on a 0–100 scale; the MTEB figures average task categories, while CoIR and MIRACL average their constituent tasks or languages.

| Evaluation | YAT embedding v1 | Parent checkpoint | Change |
| --- | ---: | ---: | ---: |
| English MTEB v2, 41 tasks | 52.78 | 45.86 | +6.92 |
| Multilingual MTEB v2, 131 tasks | 48.21 | 46.63 | +1.58 |
| CoIR, ten tasks | 43.45 | 21.03 | +22.42 |
| MIRACL hard-negative retrieval, 18 languages | 46.37 | 8.06 | +38.31 |

The same-protocol gains are substantial for retrieval, especially the MIRACL diagnostic. They are not uniform. Tatoeba bitext mining fell from **52.72 to 35.27**, with declines in 108 of 112 language-pair subsets. Multilingual clustering and bitext category means also declined. The fine-tune included MIRACL data but did not replay the earlier Global Voices bitext pairs; that is a plausible explanation for the lost cross-language alignment, not a proven cause. CodeSearchNet pairs were used in training, so CodeSearchNet-family CoIR scores need a train/evaluation overlap audit before they are treated as independent generalization evidence. The [full breakdown and receipts](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1) let readers inspect the result beyond a single mean.

## Where it stands against mmBERT and EmbeddingGemma

The [mmBERT release post](https://huggingface.co/blog/mmbert) describes a much broader multilingual pretraining program: over 3 trillion tokens across a staged language and masking schedule, followed by an MS MARCO embedding fine-tune for its retrieval tables. Its paper reports **53.9** on English MTEB v2, **54.1** on multilingual MTEB v2, and **42.2** on CoIR for mmBERT-base. Our reported figures are 52.78, 48.21, and 43.45 respectively. Different fine-tuning data and unpinned paper benchmark revisions mean these numbers give context, not a controlled test of YAT versus mmBERT. Our model also inherits mmBERT weights. See [Marone et al.](https://arxiv.org/html/2509.06888v1) for the paper tables and methodology.

[EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) is the more relevant product target: it is roughly the same size, produces 768-dimensional embeddings, supports a 2,048-token input window, and offers trained 512-, 256-, and 128-dimensional outputs. Its official card reports **65.11** for English MTEB v2 and **54.31** for multilingual MTEB v2 as *Mean (TaskType)* at 768 dimensions. Our seven-category English and eight-category multilingual means are **52.78** and **48.21**. The nominal gaps are **12.33** and **6.10** points in EmbeddingGemma's favor. These are cross-report gaps, not a head-to-head result: the exact task sets, revisions, prompts, length handling, and inference implementations have not been harmonized. EmbeddingGemma also reports **68.76** on MTEB Code v1; our **43.45 CoIR** score is from a different suite and should not be subtracted from it. Google's [model card](https://huggingface.co/google/embeddinggemma-300m) and [release post](https://huggingface.co/blog/embeddinggemma) document those scores and its query/document prompts.

For now, EmbeddingGemma has the stronger published general embedding results and a more mature deployment path. Our current case is narrower: the YAT fine-tune greatly improves retrieval over its own parent and makes the architecture available in both JAX and PyTorch. We have not measured a matched EmbeddingGemma run on our pinned MTEB inventory, CoIR, and MIRACL setup, so we do not claim to outperform it. We also have not measured production latency against it.

## Using the release

The [PyTorch repository](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1-pytorch) contains `model.safetensors`, `tokenizer.json`, a custom `yat_encoder.py`, conversion hashes, and TPU parity results. It is not a `transformers.AutoModel` checkpoint. Its model card shows how to load it with PyTorch, mean-pool non-padding token states, L2-normalize the result, and use cosine similarity. The Flax/JAX checkpoint is available separately on the Hub.

We checked the conversion on a physical v5e TPU using eight English, multilingual, and code inputs. The minimum pooled-vector cosine similarity between JAX and PyTorch was **0.999960**; top-1 and top-3 neighbor sets matched in that batch, with a maximum pairwise cosine-score drift of **0.003854**. This is a conversion parity check, not an independent PyTorch benchmark or proof of identical outputs on every input.

The fine-tune used 128-token queries and 256-token documents. We evaluated with a 512-token cap; quality and serving behavior on longer inputs remain open. The PyTorch implementation favors parity over serving speed. Before putting this model into a production search stack, we would measure latency, recall on the actual corpus, cross-language retrieval, and code-search quality separately.

## What we will improve next

Our next model should target the demonstrated weaknesses instead of optimizing a single aggregate score. We plan to restore cross-language alignment with explicit bitext replay and hard negatives, audit code-data overlap, train and evaluate longer-document retrieval, and test an asymmetric query/document recipe with task prompts. We also want matched runs of YAT, mmBERT, and EmbeddingGemma on the same pinned tasks and hardware, plus serving work for the PyTorch implementation. Those experiments will tell us whether YAT's architecture offers a practical advantage after data and inference differences are controlled.

The model is available now: [Flax/JAX weights](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1) · [PyTorch weights](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1-pytorch).
