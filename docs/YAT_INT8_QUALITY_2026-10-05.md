# INT8 embedding quality: YAT and mmBERT-base, October 5, 2026

YAT’s post-training INT8 candidate retained essentially the same measured quality on STSBenchmark and SciFact. mmBERT-base showed a 0.61-point STS correlation decrease and small retrieval changes. These are native JAX weight-perturbation measurements on a physical TPU; they do not qualify the compressed PyTorch loader or full quantized MTEB performance.

Both models encoded exactly 8,329 texts: all 1,379 STSBenchmark test pairs, all 300 SciFact test queries and the complete 5,183-document corpus, plus 88 handcrafted multilingual/code diagnostic texts. Batch size was 8, sequence length 256, prompts were absent, pooling included all nonpadding special-token states, and similarity used FP32 L2 normalization. Both runs truncated 3,142 texts. Original and quantized variants had identical token IDs and masks.

| Distance from each original model | YAT | mmBERT-base |
| --- | ---: | ---: |
| Mean embedding cosine | 0.998204 | 0.999253 |
| Worst embedding cosine | 0.996102 | 0.949082 |
| Median embedding cosine | 0.998160 | 0.999288 |
| Mean raw relative L2 change | 5.96% | 3.83% |
| SciFact top-1 ranking agreement | 97.33% | 91.00% |
| Mean top-10 set overlap | 96.13% | 94.33% |

| Model / test metric (×100) | Original | INT8 | Change in points |
| --- | ---: | ---: | ---: |
| YAT STS Spearman | 74.00 | 74.02 | +0.012 |
| YAT SciFact nDCG@10 | 54.66 | 54.63 | -0.031 |
| YAT SciFact Recall@10 | 68.24 | 68.24 | +0.000 |
| YAT SciFact MRR@10 | 51.36 | 51.39 | +0.029 |
| mmBERT-base STS Spearman | 53.34 | 52.73 | -0.611 |
| mmBERT-base SciFact nDCG@10 | 4.39 | 4.58 | +0.187 |
| mmBERT-base SciFact Recall@10 | 6.97 | 7.31 | +0.333 |
| mmBERT-base SciFact MRR@10 | 3.83 | 3.98 | +0.150 |

The paired query-bootstrap 95% interval for YAT’s SciFact nDCG change is **−0.577 to +0.482 points**; for mmBERT-base it is **−0.173 to +0.602 points**. Both include zero. These intervals assume independent queries and condition on this corpus, judgments and scoring protocol. They do not demonstrate equivalence across all tasks, languages, training seeds or document lengths. No interval was computed for the STS change.

mmBERT’s average cosine was closer to its original, but its worst case was substantially less stable: `Someone is drawing.` had cosine 0.949082 and the largest raw relative L2 change reached 44.38%. YAT’s worst cosine was 0.996102 on a SciFact abstract. High average cosine alone does not guarantee stable rankings.

The handcrafted probe retained 56/56 translation matches for both models. Code-query top-1 remained 8/8 for YAT and 4/8 for mmBERT-base. These tiny diagnostic counts are not multilingual or code benchmark acceptance.

The selected mmBERT checkpoint is the same pretrained **mmBERT-base** used earlier, not the paper’s MS MARCO-fine-tuned embedding model. YAT is contrastively fine-tuned. Absolute score differences here do not isolate the effect of YAT architecture.

## Artifact and execution evidence

The YAT candidate is the actual INT8 artifact created October 5: 310,252,696 bytes with SHA256 `237c907448a99de44c98ec6919ec02d900226115b17c9e894aa28c9106fec4a9`. INT8 values and FP32 per-row scales were authenticated and reconstructed into native JAX masters; norm vectors, decoder bias and all 44 alpha scalars were bitwise unchanged. Bias 1 and epsilon 0.01 remained fixed. mmBERT matrices received the identical quantization algorithm in Torch output-row layout. No quantization-aware training occurred.

Both original first batches repeated bit-for-bit, every pooled vector was finite/nonzero, full tensor coverage passed, and quantized outputs were confirmed to differ from originals. Model forwards ran only on physical TPU v5e, JAX 0.11.1, at BF16 feature/distance precision and FP32 residual/score precision. One chip performed inference within an allocated eight-chip single-host slice. Model-free scalar replay independently verified STS correlations and all per-query retrieval metrics from the retained rank receipts.

Frozen deployment manifest SHA256: `d28a44a4fcad3e01be7d8cde0cc84ad6a3f24b6103034a4fece3dbbb42e7b1ed`. Fixture SHA256: `47a1b8764f83f790b7f278833a38954e639d84f72dd949eba3e2f681590ce1c2`. Plan SHA256: `19f4116b3945258b602dbac81f50324e84f10a34b99b178906935d13d3c454c1`.

Actual immutable dataset revisions: STSBenchmark `b0fddb56ed78048fa8b90373c8a3cfc37b684831`; SciFact `d56462d0e63a25450459c4f213e49ffdb866f7f9`. Official test splits were evaluated; training-exposure or contamination cleanliness is not established by their split names.

Full raw vectors, per-query scores, runtime/source/model identities and partial/terminal receipts are retained at `gs://azettaai-yat-eval-0929/quant-quality-1005-v1/yat-embed-torch-quant-quality-1005/paired-native-int8-quality/results/`. Raw arrays stayed near GCP; only small JSON receipts were copied locally. The guarded campaign passed. Independent exact node and queued-resource reads returned NOT_FOUND after cleanup.

PyTorch compressed-loader numerical parity, compressed HBM use, inference speed and full quantized MTEB remain unqualified. No quantized public model or updated full benchmark card was published on this evidence. See [EmbeddingGemma comparison](YAT_EMBEDDINGGEMMA_COMPARISON_2026-10-05.md) for independently published context.
