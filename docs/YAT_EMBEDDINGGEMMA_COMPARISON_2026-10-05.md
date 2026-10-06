# YAT and EmbeddingGemma: published quality context

This comparison uses our existing unquantized YAT embedding-v1 measurements and independently published Google results. It is not a matched rerun. Task revisions, truncation, prompting and scoring must be reconciled before interpreting differences as controlled measurements. The [October 5 paired TPU results](YAT_INT8_QUALITY_2026-10-05.md) now measure native INT8 weight perturbation quality on STSBenchmark and SciFact; full quantized MTEB and compressed PyTorch serving remain unqualified.

| Benchmark aggregation | YAT v1 measured | EmbeddingGemma published / reaggregated |
| --- | ---: | ---: |
| English MTEB v2: 41 tasks, seven category means | 52.78 | 65.11 |
| Multilingual MTEB v2: 131 tasks, nine category means | 42.40 | 54.31 |
| Multilingual: eight common categories, excluding instruction retrieval | 48.21 | 60.395 (calculated) |
| Ten CoIR task names in our suite | 43.45 | 67.283 (calculated) |

Google reports 68.76 for its full twelve-task code suite. The ten-task calculation omits `CodeEditSearchRetrieval` and `CodeSearchNetRetrieval`; it averages the other ten rounded task scores from Appendix Table 11. The eight-category calculation averages the eight rounded category scores from Table 5 excluding instruction retrieval. Matching task names or counts alone does not establish identical evaluation revisions or protocols. [EmbeddingGemma paper, Tables 5 and 11](https://arxiv.org/html/2509.20354v1).

EmbeddingGemma is 308M parameters, produces 768-dimensional embeddings, and uses a 2,048-token context. It applies mean pooling followed by two linear projections. Its recipe includes teacher distillation, diverse task mixtures, a spread-out regularizer, Matryoshka objectives and checkpoint averaging. [Google model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card), [paper](https://arxiv.org/html/2509.20354v1).

For retrieval, Google prescribes `task: search result | query: {content}` for queries and `title: none | text: {content}` for documents without titles. Code queries use `task: code retrieval | query: {content}`. A future matched comparison should preserve each model's recommended prompts and record them explicitly. [Official Hugging Face model card](https://huggingface.co/google/embeddinggemma-300m).

## Google's measured quantization degradation

| Metric | BF16 | QAT INT8 | INT8 change | QAT INT4 | INT4 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| English category mean | 65.11 | 64.84 | -0.27 | 64.65 | -0.46 |
| Multilingual category mean | 54.31 | 53.95 | -0.36 | 53.61 | -0.70 |
| Code, twelve-task mean | 68.76 | 68.70 | -0.06 | 67.99 | -0.77 |

These are quantization-aware-trained, per-block checkpoints. Google applied QAT during fine-tuning; these figures do not predict quality loss from our post-training per-row INT8 export. [Google model card](https://huggingface.co/google/embeddinggemma-300m), [paper Table 1](https://arxiv.org/html/2509.20354v1#S2.SS3).

The mmBERT paper reports 53.9 English, 54.1 multilingual and 42.2 CoIR after MS MARCO embedding fine-tuning. Those numbers do not describe raw `jhu-clsp/mmBERT-base` pooled embeddings, which are the checkpoint available to our quantization sensitivity comparison. No official before/after quantized retrieval results were found in the mmBERT paper, authors' blog or base-model card during this review. [mmBERT paper](https://arxiv.org/html/2509.06888v1), [authors' blog](https://huggingface.co/blog/mmbert).

The new quantization experiment measures each candidate against its own original weights using fixed inputs, pooling and length. It reports vector drift, full STSBenchmark test correlation, and full SciFact test-query retrieval against the full corpus at length 256. This measures INT8 weight perturbations reconstructed in native JAX; it does not qualify compressed PyTorch serving, fused INT8 execution or a complete MTEB leaderboard result.
