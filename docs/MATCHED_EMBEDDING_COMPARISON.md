# Matched embedding comparison from complete receipts

`scripts.compare_embedding_receipts` compares two already measured, complete
MTEB-style receipts without importing a model or numerical backend. It computes
row, task and category scores from the retained results. Paper-reported columns,
partial shards and old summaries are not accepted as matched measurements.

Before execution, freeze a `matched-embedding-comparison-plan-v1` JSON document
with these fields:

- `aggregation_policy`: the exact policy in `scripts.evaluation_contract.POLICY`.
- `inventory`: the **independently frozen complete benchmark task inventory**,
  including each category, immutable dataset revision, split, subset and expected
  row product. Generate this from pinned benchmark metadata before evaluation;
  do not infer the intended suite from whichever results finished.
- `common`: `benchmark`, `mteb_version`, `sequence_length`, `padding_policy`,
  `truncation_policy` and `score_scale`. The accepted scale is explicitly
  `native-mteb-main-score/no-rescaling`. No multiplying fractions by 100 or
  treating native scores as percentage points is inferred.
- `models`: exactly `baseline` and `candidate`. Each declares `model`, complete
  `files_sha256` including weights, `prompts`, `pooling`, `normalization`, and the
  independently pinned `receipt_sha256`. Use each competitor's intended prompt
  and pooling configuration; model-specific choices remain visible in output.
- `scoring_source_sha256`: pinned shared scoring files, including
  `scripts/evaluation_contract.py`. Both receipts must identify the same hashes.

The receipt format is the complete `evaluate_yat_public_mteb`/suite-merge shape:
`identity`, `inventory`, `selected_tasks`, `results`, `failures`, `aggregate`.
The utility requires explicit length, truncation, scale, pooling, normalization
and prompt fields in each identity. It verifies scoring package/interpreter and
physical device provenance, recomputes aggregates and rechecks all input content
pins before returning. Runtime and model identity differences are retained rather
than hidden. Declared backend must agree with recorded TPU/GPU devices.

The current YAT evaluator emits explicit truncation, normalization and native
score-scale fields. Historical receipts that omit them remain unchanged and
cannot pass this stricter comparison. Do not add guessed execution policy to old
results. Run a newly frozen physical evaluation with the intended model protocol
when missing evidence matters. A differently configured competitor requires its
own faithful evaluation adapter and complete receipt; this comparison utility
neither implements that adapter nor verifies its model-specific recommendations.

```bash
python -m scripts.compare_embedding_receipts \
  --plan /FROZEN/comparison-plan.json \
  --plan-sha256 EXACT_INDEPENDENTLY_PINNED_PLAN_SHA256 \
  --baseline /EVIDENCE/mmbert-complete.json \
  --candidate /EVIDENCE/yat-complete.json \
  --output /FRESH/EVIDENCE/matched-comparison.json
```

The output contains per-task/split/subset scores and deltas, recomputed category
means, model-specific differences, full recorded identities and input hashes.
It explicitly leaves `architecture_effect_isolated=false`: same benchmarks do
not isolate architecture from different training data or histories. Authentication
here means bytes match supplied independent pins and their internal contracts;
it is not a fresh benchmark execution or cryptographic attestation of the remote
measurement. The independent plan and receipts must come from the actual frozen
campaign evidence.

Adversarial metadata tests cover pinned-byte changes, forged aggregates,
missing/duplicate rows, revisions, splits, subsets, narrowed inventories on both
sides, incomplete selections, failures, nonfinite scores, undeclared prompts,
scoring source/runtime changes and backend/device contradictions. No actual
matched mmBERT or EmbeddingGemma result was produced by implementing this tool.
