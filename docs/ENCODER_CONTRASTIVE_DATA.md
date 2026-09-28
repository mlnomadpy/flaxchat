# Contrastive continuation data

The current expanded stage uses the **official train split** of all 28 English–other-language shards in [Sentence Transformers' Global Voices parallel-sentence package](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices/tree/4cc20add371f246bb1559b543f8b0dea178a1803). The revision, cached dataset API digest, and all 28 raw Parquet SHA-256 values are pinned in [`configs/contrastive_global_voices_28.json`](../configs/contrastive_global_voices_28.json). Global Voices [licenses its content CC-BY-3.0](https://globalvoices.org/about/); attribute Global Voices and its contributors, OPUS alignment, and Sentence Transformers packaging. The package does not retain article URLs, so this pipeline does not reconstruct article-level attribution. Do not republish its raw text as if it were project-authored data.

The data-only builder, [`scripts/expand_global_voices_contrastive.py`](../scripts/expand_global_voices_contrastive.py), selects the 10,000 smallest SHA-256 hashes of eligible paired text per language. Eligibility requires 8–128 tokens on both sides under the unchanged mmBERT tokenizer. It never truncates. It preserves original Parquet row coordinates and every source checksum. The preparation stage excludes exact normalized texts from all 112 pinned Tatoeba test shards and **both** train and dev files from the earlier 500-step pilot, then assigns connected text components to train or dev by a seeded hash. The previous-stage exclusion is essential: its train examples have already influenced the starting weights and cannot become new validation examples.

The verified local build of 2026-09-28 yielded 1,102,846 raw rows, 1,042,352 length-eligible rows, and 214,520 hash-selected rows. After quarantine and deduplication, it contains **197,827 training pairs and 4,048 development pairs** across 28 language pairs. Exact normalized-text overlap is zero between new train and dev and between either new split and all previous-stage rows. The prepared manifest SHA-256 is `9083c1ce5fdd6f6489e545e73e40543caa6439547467bf3c2d2f928dc56d4ab0`. The prior-stage text exclusion removed 11,112 selected rows; the sealed Tatoeba text exclusion removed 32. Per-language counts, source hashes, and token array hashes are in the generated receipts under `artifacts/encoder-contrastive-global-voices-28-0928/` (ignored by Git; copy the receipts and arrays with the training job).

This source is English-pivot translated news, so its size and language count do not establish retrieval quality on other domains or genuinely low-resource languages. Hash selection is a diversity-balanced cap, not translation-quality scoring. Exact-text decontamination cannot rule out paraphrase, substring, or earlier base-model exposure. Treat the new dev set as an internal diagnostic; retain a separate sealed external benchmark for model selection and claims. Additional data sources such as WikiMatrix need their own license, provenance, alignment-quality, and benchmark-overlap audit before integration.

## Bounded TPU continuation

The 2026-09-28 v5e8 run initialized **model weights only** from the committed 500-step contrastive pilot, itself descended from MLM step 48,236. It started a fresh AdamW optimizer and fixed 6,000-step warmup/cosine schedule on the new manifest. The global batch was 64, sequence length 128, peak learning rate `1e-5`, and warmup 100 steps. The planned exposure was 1.94 passes over the new training pairs. Checkpoints at steps 3,000, 4,000, 5,000, and 6,000 are under `gs://tpubuilders-flaxchat-validation-0921/encoder-contrastive-global-voices-28-0928/checkpoints/`; the final metadata SHA-256 is `73b45e275ed625cc6276eb102f3f85ace369dbcb3f32ada60551116de13ceafe`.

The same 4,048-pair development split was scored on a physical TPU. The cross-dataset baseline explicitly used `--allow-external-data`, and its report records that its training manifest differs from the evaluated manifest. The continued-stage checkpoints match the evaluated manifest.

| Checkpoint | English → translation recall@1 | Translation → English recall@1 | Equal-language macro recall@1, both directions |
|---|---:|---:|---:|
| Parent step 500 | 0.9074 | 0.8943 | 0.8589 |
| New stage step 1,000 | 0.9523 | 0.9432 | 0.9233 |
| New stage step 3,000 | 0.9642 | 0.9489 | 0.9387 |
| New stage step 6,000 | **0.9654** | **0.9503** | **0.9430** |

These are within-source, sentence-translation retrieval results, not MTEB or an independent multilingual retrieval benchmark. The 112-subset Tatoeba test remains sealed. For the next quality stage, add licensed, pinned non-news data and real multilingual query–document relevance pairs; retain disjoint training and evaluation identities and measure low-resource languages separately.
