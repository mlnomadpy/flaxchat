# YAT step-12,000 evaluation against the mmBERT paper

Source: Marone et al., [*mmBERT: A Modern Multilingual Encoder with Annealed Language Learning*](https://arxiv.org/html/2509.06888v1), Tables 2–7 and Appendix B. Candidate: [`mlnomad/yat-mmbert-base-contrastive-12000`](https://huggingface.co/mlnomad/yat-mmbert-base-contrastive-12000), 307.8M parameters. Scores in the paper tables below are percentages/points as printed. They are **paper-reported mmBERT-base scores, not YAT measurements**.

## Paper benchmark inventory and mmBERT-base numbers

| Suite | Tasks and mmBERT-base scores | Aggregate |
| --- | --- | ---: |
| GLUE, Table 2 | CoLA 61.9; SST-2 94.0; MRPC 91.9; STS-B 91.0; QQP 89.4; MNLI 87.7; QNLI 93.3; RTE 85.6 | 86.3 |
| XTREME, Table 3 | XNLI 77.1; PAWS-X 87.7; XCOPA 67.5; XQuAD 77.6; MLQA 66.0; TyDiQA 74.5; WikiANN 58.2; UDPOS 74.0 | 72.8 |
| MTEB v2 English, Table 4 | Pair classification 80.2; classification 64.8; STS 74.8; retrieval 44.9; clustering 41.7; reranking 44.9; summarization 26.0 | 53.9 |
| MTEB v2 multilingual, Table 5 | Bitext mining 59.2; pair classification 79.2; classification 53.6; STS 67.1; retrieval 45.8; multilabel classification 17.5; clustering 40.2; reranking 69.9 | 54.1 |
| CoIR code retrieval, Table 6 | Apps 6.1; CosQA 24.6; Synthetic Text2SQL 48.1; CodeSearchNet 63.8; CodeSearchNet-CCR 41.7; CodeTrans-Contest 54.9; CodeTrans-DL 33.0; StackOverflow QA 62.1; CodeFeedback-ST 55.5; CodeFeedback-MT 32.4 | 42.2 |
| Low-resource QA, §4.4 | TiQuAD (Tigrinya), FoQA (Faroese): discussed via figure/relative gains; no complete numeric table for the final mmBERT-base model in the HTML paper | — |
| Efficiency, §4.5 | Variable-length and long-context throughput in Figure 3 on GPU; no directly comparable TPU table | — |

The paper's XNLI/PAWS-X language-restricted EuroBERT comparison (Table 7) gives mmBERT-base 77.7 and 89.0 respectively over the selected in-distribution languages. These are **different aggregates** from XTREME Table 3's XNLI 77.1 and PAWS-X 87.7.

## Evaluation protocol that controls comparability

- GLUE/XTREME are supervised downstream evaluations. Appendix B reports a batch size of 32, 0.06 warmup ratio, seven learning rates from 2e-5 through 8e-5, and epoch choices 1/2/3/5/10. The best setting was selected per model and task in what the authors call an *oracle* sweep. The paper does not supply the exact selected configuration for each table cell. A frozen-encoder linear probe or zero-shot embedding test cannot be compared to these scores.
- MTEB v2 English, MTEB v2 multilingual, and CoIR were run **after** one epoch of SentenceTransformers training on 1.25M MS MARCO hard triplets from `sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1`. Four learning rates were swept (1e-4, 3e-4, 5e-4, 7e-4); 1e-4 won for mmBERT-base. The paper averages MTEB category scores, not every task or language row equally. A released mmBERT-base MLM checkpoint is not the fine-tuned paper embedding model.
- The YAT checkpoint's final contrastive stage instead used 631,332 English-pivot Global Voices pairs, global batch 64, 12,000 updates, and maximum training length 128. Its data, training objective/negatives, pooling/truncation and fine-tuning budget are not matched to the paper's MS MARCO stage. The YAT model also inherits mmBERT weights. Neither cross-protocol scores nor a comparison with released mmBERT-base isolates the YAT architecture.
- The locally installed MTEB 2.21.8 registry currently lists 41 English-v2 tasks (10 retrieval, 9 STS, 8 each clustering/classification, 3 pair classification, 2 reranking, 1 summarization) and 131 Multilingual-v2 tasks. This is a **current registry inventory, not a claim that the September 2025 paper used these precise task revisions**. The paper reports categories but does not pin the MTEB package version or all dataset revisions. A paper-score replication therefore needs an archived benchmark manifest in addition to a trained checkpoint.

## YAT measurements currently complete

The **full current CoIR suite is complete** on physical v5e Spot TPUs: 10/10 tasks, zero failures, model-file hashes matching the published step-12,000 checkpoint. The merged raw result is [`coir-complete.json`](../artifacts/yat-full-eval-0929/coir-complete.json). It gives **21.03 mean nDCG@10 points** for YAT as-is. The paper's mmBERT-base result is 42.2 points after a different MS MARCO embedding fine-tuning stage, so the 21.17-point gap is not an architecture-controlled comparison.

| CoIR task | YAT as-is | Paper mmBERT-base |
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
| **Mean** | **21.03** | **42.2** |

The English suite is **complete at 41/41 tasks with zero failures**, with a native seven-category mean of **45.86 points** versus the paper's 53.9. The multilingual suite is **complete at 131/131 tasks with zero failures**. Its native nine-category mean is **41.41 points**. Excluding the current registry's InstructionReranking category, which the paper did not report, gives a **46.63-point common-category mean** versus the paper's 54.1. These are not same-task, same-data, same-training comparisons. The YAT category values below use MTEB 2.21.8, whereas the paper does not pin its benchmark inventory and used a different embedding fine-tuning recipe.

| English MTEB category | YAT completed category | Paper mmBERT-base |
| --- | ---: | ---: |
| Pair classification | 66.17 | 80.2 |
| Classification | 65.91 | 64.8 |
| STS | 66.71 | 74.8 |
| Retrieval | 15.54 | 44.9 |
| Clustering | 40.88 | 41.7 |
| Summarization | 26.71 | 26.0 |
| Reranking | 39.12 | 44.9 |

| Multilingual MTEB category | YAT completed category | Paper mmBERT-base |
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

The verified complete receipts are [`english-complete.json`](../artifacts/yat-full-eval-0929/english-complete.json) and [`multilingual-complete.json`](../artifacts/yat-full-eval-0929/multilingual-complete.json). MindSmall completed on a high-memory physical v5e TPU after the original single-chip VM stalled on host paging; its result is in [`mindsmall-backup.json`](../artifacts/yat-full-eval-0929/mindsmall-backup.json). WebLINX was assembled from all six official split receipts, under the same model hash and dataset revision, in [`weblinx-complete.json`](../artifacts/yat-full-eval-0929/weblinx-complete.json).

These were paired on a physical v5e TPU against **released, non-MS-MARCO-fine-tuned** mmBERT-base with identical tokenizer, mean pooling, FP32 L2 normalization, and BF16 inference. The complete procedure and report paths are in [YAT_MMBERT_PAIRED_BENCHMARKS.md](YAT_MMBERT_PAIRED_BENCHMARKS.md).

| Benchmark and metric | Released mmBERT-base | YAT step 12,000 | Paper cell comparable? |
| --- | ---: | ---: | --- |
| Tatoeba, 112-subset macro F1 | 30.16 | 52.69 | No; not full multilingual MTEB, and training differs |
| Tatoeba, 88,877-pair accuracy | 33.90 | 56.83 | No |
| STSBenchmark.v2, Spearman | 53.93 | 67.33 | No; only one English STS task |
| STS17, 11-subset mean Spearman | 48.89 | 63.83 | No; only one multilingual STS task |
| SciFact, nDCG@10 | 3.65 | 29.91 | No; only one English retrieval task |
| NFCorpus, nDCG@10 | 2.59 | 11.24 | No; only one English retrieval task |

The Tatoeba continuation-corpus audit found no exact normalized text overlap, but cannot rule out semantic overlap or exposure in inherited pretraining. The four MTEB task samples have no completed overlap audit. Tatoeba native search tie-breaking has not been proven identical to official MTEB. The PyTorch conversion has physical TPU JAX/TorchXLA output parity on eight inputs, but this establishes implementation agreement only; it is not a quality benchmark.

I re-read the local raw reports rather than relying on the summary table: [`tatoeba-paired-r5.json`](../artifacts/encoder-contrastive-50k-benchmark-0928/tatoeba-paired-r5.json) records all 112 subsets and 88,877 pairs; [`mteb-paired.json`](../artifacts/encoder-contrastive-50k-benchmark-0928/mteb-v2-reports-r2/mteb-paired.json) records four official MTEB tasks under MTEB 2.21.8. The candidate MTEB run used a physical eight-device TPU, encoded 22,855 texts, and truncated 651 at the frozen 512-token evaluation limit; contrastive training itself used length 128.

The aggregate conceals regressions. On Tatoeba, YAT loses to the released mmBERT baseline on 2/112 language-pair F1 scores: Estonian–English 23.3 versus 29.5, and Tagalog–English 30.8 versus 36.5. On the 11 STS17 subsets, Korean–Korean declines to 61.2 from 62.0 Spearman. Several low-resource Tatoeba absolute F1 scores remain poor even though they improve over baseline: Uyghur–English 1.4, Kabyle–English 2.1, and Yiddish–English 3.9. These values come from the same physical TPU report and are exploratory for the same contamination/protocol reasons.

## Coverage and decision

**The as-is full-suite TPU run is complete for English MTEB, multilingual MTEB, and CoIR.** It evaluates the published step-12,000 checkpoint without any further training on the current MTEB 2.21.8 inventories: 41 English tasks, 131 multilingual tasks, and 10 CoIR tasks. The evaluation uses 512-token maximum length, per-batch length buckets, BF16 model inference, mean pooling of nonpadding tokens (including special tokens), FP32 L2 normalization, and default batch size 16. The result receipts record the exact model file hashes, dataset revisions, per-task scores, failures, and task-level coverage. Each worker wrote its receipt atomically after a task and synced it to `gs://azettaai-yat-eval-0929/`. The runner is [`evaluate_yat_public_mteb.py`](../scripts/evaluate_yat_public_mteb.py); independent TPU shards were merged with [`merge_yat_mteb_shards.py`](../scripts/merge_yat_mteb_shards.py).

The 18-language MIRACL task was evaluated as eight disjoint language-subset shards with [`evaluate_yat_mteb_subsets.py`](../scripts/evaluate_yat_mteb_subsets.py). Its 18 rows were merged only after exact subset coverage, identical model hashes, and the same dataset revision were verified by [`merge_yat_mteb_subsets.py`](../scripts/merge_yat_mteb_subsets.py). This preserves the task's per-language scores while allowing the large task to finish within affordable Spot leases.

For the large MindSmall task, the evaluator replaces MTEB's Python-list copy of the identical query text column with a zero-copy Arrow column and writes embeddings into one preallocated array. A physical-TPU ArguAna rerun at the original batch size 16 reproduced the prior 0.41253 score exactly after the Arrow change. MindSmall runs at batch size 64 to finish within the Spot lease. WebLINX's six official splits run independently on six otherwise idle chips at batch size 32, with strict split-coverage merging via [`evaluate_yat_mteb_splits.py`](../scripts/evaluate_yat_mteb_splits.py) and [`merge_yat_mteb_splits.py`](../scripts/merge_yat_mteb_splits.py). The other tasks run at batch size 16. No custom model prompt is applied; any instruction text supplied by the benchmark data is retained.

The MIRACL task is now complete: **18/18 language subsets**, zero missing or duplicate rows, **8.06 mean nDCG@10 points**. The merged receipt is [`miracl-complete.json`](../artifacts/yat-full-eval-0929/miracl-complete.json). This is one task within the multilingual suite, not its full retrieval-category score.

GLUE and XTREME still have no as-is YAT numbers. The paper's figures result from task-specific supervised fine-tuning and oracle hyperparameter selection, which would change the checkpoint and answer a different question. The current evaluation can characterize this published checkpoint's embedding quality; it cannot establish that its architecture exceeds mmBERT under the paper's training protocol.

A paper-comparable architecture comparison would require matched downstream training and benchmark revisions for both models. That is separate from the as-is evaluation requested here.
