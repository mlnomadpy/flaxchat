# Exact benchmark exposure audit

Use [audit_embedding_overlap.py](../scripts/audit_embedding_overlap.py) before treating CodeSearchNet-family CoIR results as independent generalization. The tool reads actual supplied prepared train/dev raw JSONL and supplied benchmark query/corpus JSONL. It does not download data, execute a model, change quarantine or decide which checkpoint to publish.

```bash
python -m scripts.audit_embedding_overlap \
  --prepared code=/ACTUAL/PREPARED/CODE \
  --benchmark-manifest /ACTUAL/PINNED/benchmark-overlap.json \
  --output /ACTUAL/EVIDENCE/code-overlap.json \
  --max-rows 1000000 --timeout-seconds 600 --max-samples 100
```

Prepared directories must contain their real manifest, `train.jsonl` and `dev.jsonl`, exact raw-file hashes, declared row counts, immutable source revisions, and the current `modality-exact-v2` identity policy. The script imports the existing `flaxchat.embedding_data.text_identity` helper. It preserves code case, indentation and literals, normalizes line endings, and applies NFC only for natural-language fields. It adds no second normalizer. Older prepared identity policies require a separately reviewed preparation/migration; they are not silently reinterpreted.

The benchmark input manifest explicitly declares the intended task/split/subset/query-or-corpus inventory. Example structure (replace every example value with authenticated actual data):

```json
{
  "format": "embedding-benchmark-overlap-input-v1",
  "expected_units": [
    {"task": "COIRCodeSearchNetRetrieval", "split": "test", "subset": "python", "role": "corpus", "rows": 123}
  ],
  "files": [
    {
      "task": "COIRCodeSearchNetRetrieval", "split": "test", "subset": "python", "role": "corpus",
      "dataset_repo": "ACTUAL/PINNED-DATASET", "dataset_revision": "40-character-immutable-commit",
      "path": "python-corpus.jsonl", "sha256": "64-character-actual-file-hash", "rows": 123,
      "text_field": "code", "modality": "code"
    }
  ]
}
```

Use distinct query units with `role: query` and the appropriate actual field/modality. Include all benchmark subsets and queries/corpora used in the intended comparison. Paths resolve relative to the manifest. Sharded files may share a unit if their total rows do not exceed its inventory. Unexpected units, duplicate paths, floating revisions, hash changes and contradictory row counts fail. An incomplete supplied inventory cannot certify unsupplied benchmark tasks: `complete` means only the declared input scope.

The report hashes manifests, raw inputs, shared identity implementation and audit implementation, records source/benchmark revisions, and separates training versus development overlaps and query/positive/negative fields. `overlap_counts` counts overlapping benchmark rows for each unit/source/split/field combination; it does not sum training repetitions into apparent benchmark coverage. Bounded samples contain only content identities, row indexes and matching prepared presentation counts, not raw code. Null negatives are absent rather than aliases of positives. Empty strings are explicitly excluded and counted; whitespace remains content.

SQLite stores the prepared identity index on temporary disk with an8MiB cache. Default limits scan at most one million rows per file, cap the audit at600 seconds, and retain100 match samples. Raw files are hash-checked before and after scanning; changed files cannot receive a complete receipt. Row-limit exhaustion, absent declared units or a deadline produces explicit partial coverage, which never establishes zero overlap. Dataset provenance beyond supplied revision/file hashes still needs independent source authentication.

Exit0 means complete declared-scope coverage and zero exact overlap; exit1 means complete coverage with exact overlap; exit2 means partial coverage. Deterministic schema/hash errors abort with an explicit failed receipt. A running/failed receipt replaces any old output before validation, preventing stale successful output from surviving a failed rerun. Even exit0 leaves semantic/near-duplicate/translation overlap, parent pretraining and earlier-stage exposure unresolved. No receipt here proves unbiased benchmark generalization or physical TPU acceptance. Report original and clean evaluation protocols separately if overlap-driven filtering is adopted.

Seven model-free fixture tests cover case-sensitive code and line endings, train/dev/query exposure, partial coverage, missing units, tampering, floating revisions, concealed extra rows, bounded timeout, failure-time database closure, invalid text containers and explicit unresolved exposure. Actual prepared/benchmark corpora were not available in this implementation task, so no production contamination conclusion has been made.
