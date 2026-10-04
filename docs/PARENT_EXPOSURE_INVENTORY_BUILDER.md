# Build the retained parent exposure inventory

[build_parent_exposure_inventory.py](../scripts/build_parent_exposure_inventory.py) generates the input for the independent-development exposure check from retained artifacts. It avoids manually inventing stage hashes, source mappings or checked row counts. No model, network or cloud allocation is involved.

```bash
python -m scripts.build_parent_exposure_inventory \
  --parent-directory /ACTUAL/AUTHENTICATED/PARENT \
  --stage-metadata /ACTUAL/STAGE-12000/checkpoint-metadata.json \
  --stage-metadata /ACTUAL/STAGE-14000/checkpoint-metadata.json \
  --prepared-directory /ACTUAL/STAGE-12000/prepared/bitext \
  --prepared-directory /ACTUAL/STAGE-14000/prepared/msmarco \
  --prepared-directory /ACTUAL/STAGE-14000/prepared/miracl \
  --prepared-directory /ACTUAL/STAGE-14000/prepared/code \
  --producer-policies /ACTUAL/verified-historical-producer-policies.json \
  --output /ACTUAL/FRESH/EVIDENCE/parent-exposure-input.json \
  --timeout-seconds 600 --require-parent-lineage
```

These paths/stages are illustrative, not a claim that those artifacts are locally available or capacity is running. Supply every known relevant stage and its actual prepared-source directories. Sources match each stage's committed `resolved_config.data_manifests` by manifest SHA256 and source name. Different stages may use distinct versions of the same named source; list each unique directory once. Missing, ambiguous or contradictory artifacts fail instead of being reconstructed from model-card row counts.

The builder hashes actual parent weights/config/tokenizer. If the parent contains its authenticated `export.json`, checkpoint metadata and checkpoint manifest, it validates that export lineage and requires its current checkpoint metadata among supplied stages. `--require-parent-lineage` rejects absent export lineage. Without that flag, a legacy weights-only parent can produce an inventory, but `parent_checkpoint_metadata_bound` remains false and historical lineage is explicitly unresolved. Such output must not be called a complete production-parent lineage proof.

Every supplied prepared manifest must pin its upstream dataset repository/revision and declare real positive training row counts/raw hashes. The builder scans the entire historical train JSONL, validates shared raw text identities, checks declared row counts and verifies raw/manifest/metadata stability. Default bounds are600 seconds,5million rows per file,64 stages and256 source directories. Metadata is capped at16MiB and each raw row at4MiB. Unused supplied source manifests are reported. Duplicate stage/directories, changed files, absent committed source identities and exceeded bounds leave a failed receipt with `known_contrastive_inventory_complete: false`.

A complete builder receipt means only that **the supplied known-stage inventory was authenticated**. It does not check candidate overlap, certify undisclosed history, prove parent model restoration or establish representation quality. Feed its output to [independent development preparation](INDEPENDENT_REPRESENTATION_DEVELOPMENT.md); preparation/admission then recompute exact/aligned candidate exposure against those actual historical files. A parent-exposed candidate still fails. Broader inherited MLM, semantic exposure and completeness of undisclosed contrastive lineage remain unresolved.

Initial targeted local inspection found only three `/private/tmp/yat-embedding-pilot-0929` source manifests declaring `synthetic_pilot: true`,128 train/dev rows each, and no immutable upstream provenance. Those cannot substitute for production exposure evidence.

The coordinator subsequently recovered the actual original step14,000 checkpoint metadata and three source manifests at `/private/tmp/flaxchat-fixes-20260930/production-exposure-1001`. Its metadata byte SHA256 is `ff894fb65f2dcea80ae94c19d5d665d3dbcc84672db084c1e7452fc063963587`; the original metadata is distinct from the earlier committed-reader view that adds receipt and step fields. Independently inspected source manifests match the exact committed `data_manifests` values:

| Source | Committed manifest SHA256 | Declared train rows |
| --- | --- | ---: |
| code | `f9e1e8a2424264867dec4b8c8d13a9ece60a2b526f3673381f1fcc5fa985cb56` |300000 |
| miracl | `4fa334d7b1edded82fa3b8b2420ea38cac72a41ffd5bd6c9dff2b02278f4b2fe` |144512 |
| msmarco | `185cf096e242d1fe456b715eafcdf4619b23d9eff7e1aea19e8eea82d84d1204` |1250000 |

The [metadata-only inspection receipt](audit-2026-09-30/production-parent-exposure-metadata-2026-10-01.json) preserves these real identities and records that the actual train JSONL files have not been downloaded locally. The coordinator's separate [manifest verification receipt](audit-2026-09-30/production-exposure-manifest-receipt-2026-10-01.json) and [retained object inventory](audit-2026-09-30/production-exposure-objects-2026-10-01.json) document durable storage metadata. Their object presence/size/generation is not an exact exposure scan. The next concrete input step is to materialize and authenticate those actual raw inputs alongside the intended parent file inventory, then run this builder and the candidate exposure check under bounded execution. No production complete inventory or clean-parent result has been generated.

Four model-free tests cover hash-based source discovery feeding the real exposure checker; missing/tampered/duplicate/bounded/strict-lineage rejection; authenticated parent export requiring its current metadata; and a deadline leaving failed admission state. Test weights and rows are clearly synthetic metadata fixtures, never loaded as a model. No production inventory or clean-parent exposure conclusion was fabricated.

For generation-pinned cloud materialization, use [the GCS scan tool](GCS_PARENT_EXPOSURE_SCAN.md) and pass the same `--producer-policies` file to this builder. That optional JSON maps source names to independently authenticated historical producer policy objects. The builder requires the exact registered producer fingerprints and pinned MIRACL source identity, records the policy file's absolute path/SHA256, rechecks its bytes at completion, validates every actual historical row through the shared read-only adapter, and retains `historical_row_policy` in each relevant source entry. Thus independent preparation/admission can reconstruct the same aligned identifiers without changing original historical manifests or raw bytes. Missing legacy policy, altered fingerprints, unknown source mapping or incompatible upstream pins fail closed.

The builder test suite now has five methods. The added model-free integration test streams legacy MIRACL grammar through a fake CLI, materializes byte-identical raw/manifest files, passes the verified policy through this builder, and feeds its output to the real candidate exposure checker. Missing and forged producer evidence are rejected. This is synthetic fixture coverage of the authenticated production-producer policy path, not a claim that production raw exposure is clean.
