# Paired YAT/mmBERT embedding evaluation

Freeze the model choice and training data before opening held-out benchmarks.
The released comparison is `jhu-clsp/mmBERT-base` versus a YAT contrastive
checkpoint, both with the unchanged mmBERT tokenizer, identical BF16/FP32
runtime choices, mean pooling over nonpadding tokens including special tokens,
and FP32 L2-normalized cosine embeddings. The architectures intentionally
differ in attention and FFN; the comparison script records both complete
configurations instead of pretending they are identical.

## Sealed Tatoeba bitext

The checked-in source plan covers all 112 pinned official MTEB Tatoeba shards
and 88,877 aligned test pairs. On a single-host TPU worker with access to the
checkpoint and the prepared files, freeze the selected step:

```bash
python -m scripts.audit_contrastive_tatoeba_overlap \
  --prepared artifacts/encoder-contrastive-global-voices-50k-carry-0928/prepared \
  --sealed-manifest artifacts/encoder-contrastive-data-0928/sealed-manifest.json \
  --output /tmp/tatoeba-overlap-audit.json
```

The September 28 expanded-corpus audit found zero exact normalized-text
overlap: 631,332 training pairs and 1,033,271 unique training texts against
all 112 shards, 88,877 test pairs and 158,068 unique test texts. The prepared
manifest SHA-256 is `0eb6c456c95f51feb8ac5d1bf6d99841c792d8ebe8c037eddf65e9b52372c923`.
This is an exact-text check, not a semantic-overlap clearance.

Then freeze the selected checkpoint step:

```bash
python -m scripts.plan_yat_mmbert_bitext_pair \
  --source-plan artifacts/encoder-bitext-matched-0923/plan.json \
  --checkpoint gs://BUCKET/CONTRASTIVE_RUN/checkpoints --step 12000 \
  --overlap-audit /tmp/tatoeba-overlap-audit.json \
  --output /tmp/yat-tatoeba-plan.json
python -m scripts.export_encoder_bitext_campaign \
  --plan /tmp/yat-tatoeba-plan.json --role baseline \
  --output /tmp/tatoeba-baseline --batch-size 32
python -m scripts.export_encoder_bitext_campaign \
  --plan /tmp/yat-tatoeba-plan.json --role candidate \
  --output /tmp/tatoeba-candidate --batch-size 32
python -m scripts.compare_encoder_bitext_campaign \
  --plan /tmp/yat-tatoeba-plan.json \
  --baseline /tmp/tatoeba-baseline --candidate /tmp/tatoeba-candidate \
  --output /tmp/tatoeba-paired.json
```

The plan pins checkpoint metadata, released weights, tokenizer, task inventory,
and every prepared input hash. Both campaigns require complete subset coverage.
The output gives per-subset and macro support-weighted F1 and pair accuracy.
Search ties use ascending document IDs, so it is a native matched benchmark,
not yet certified as MTEB's exact `torch.topk` behavior. Keep Tatoeba sealed:
do not tune the checkpoint, pooling, or hyperparameters on its results.
Repeat the overlap audit if the training manifest changes. The old FineWeb
pilot audit does not cover the expanded contrastive corpus.

## Exploratory MTEB v2 slice

The optional MTEB adapter uses MTEB `2.21.8` to score `STSBenchmark`, `STS17`,
`SciFact`, and `NFCorpus` on a physical single-host TPU. Both roles share a
512-token cap (unless frozen otherwise), tokenizer, mean pooling, L2
normalization, no prompts, and MTEB's task/metric code. It records any
truncated texts. Install `mteb==2.21.8` on the TPU worker, then:

```bash
python -m scripts.evaluate_encoder_mteb_v2 --build-plan \
  --checkpoint gs://BUCKET/CONTRASTIVE_RUN/checkpoints --step STEP \
  --snapshot artifacts/mmbert-base --output /tmp/mteb-plan.json
python -m scripts.evaluate_encoder_mteb_v2 --role baseline \
  --plan /tmp/mteb-plan.json --output /tmp/mteb-baseline.json
python -m scripts.evaluate_encoder_mteb_v2 --role candidate \
  --plan /tmp/mteb-plan.json --output /tmp/mteb-candidate.json
python -m scripts.evaluate_encoder_mteb_v2 --compare \
  --baseline-report /tmp/mteb-baseline.json \
  --candidate-report /tmp/mteb-candidate.json \
  --output /tmp/mteb-paired.json
```

The adapter preserves each task's dataset revision, split, subset, and raw
metrics. This four-task slice is **not** the mmBERT paper suite or a leaderboard
aggregate. Its datasets are not sealed under the current training-data audit;
until that audit is done, use the scores for diagnosis, not a contamination-free
quality claim. Pin a new plan for every selected continuation checkpoint.
