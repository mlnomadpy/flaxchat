# Authenticate historical raw inputs near GCS

[scan_gcs_parent_exposure.py](../scripts/scan_gcs_parent_exposure.py) streams generation-pinned objects through `gcloud storage cat`, hashes their original bytes, validates every row with the shared text identity policy, and checks exact declared counts and byte sizes. Run it on an existing authenticated cloud host near the bucket to avoid transferring the corpus through the user's local connection. The tool does not allocate compute or execute a model. It requires the repository's metadata dependencies (including NumPy), `gcloud`, and read access to the specified object generations.

The [actual retained object receipt](audit-2026-09-30/production-exposure-objects-2026-10-01.json) declares three train objects totaling 2,075,961,764 bytes: 300,000 code rows, 144,512 MIRACL rows and 1,250,000 MS MARCO rows. The October 1 [actual scan receipt](audit-2026-09-30/cloud-parent-scan-01/scan.json) authenticates every supplied row, byte count and raw SHA. The [controller receipt](audit-2026-09-30/cloud-parent-scan-01/controller.json) records completion in about 761 seconds; [independent cleanup](audit-2026-09-30/cloud-parent-scan-01/cleanup-verification.json) found the owned scratch and processes absent. No candidate exclusions were supplied, so overlap is unknown and admission remains false. This scan covers only the declared known contrastive inputs.

```bash
python -m scripts.scan_gcs_parent_exposure \
  --stage-metadata /CLOUD/production-exposure/checkpoint-metadata.json \
  --prepared-directory /CLOUD/production-exposure/code \
  --prepared-directory /CLOUD/production-exposure/miracl \
  --prepared-directory /CLOUD/production-exposure/msmarco \
  --object-receipt docs/audit-2026-09-30/production-exposure-objects-2026-10-01.json \
  --development-exclusions /CLOUD/actual-candidate-bundle \
  --producer-policies /CLOUD/verified-historical-producer-policies.json \
  --materialize-directory /CLOUD/FRESH/verified-historical-raw \
  --output /CLOUD/FRESH/gcs-parent-raw-scan.json \
  --timeout-seconds 1800 --max-total-bytes 4294967296
```

The cloud paths are illustrative. Supply original metadata and byte-identical manifests; each source must match the checkpoint's committed `resolved_config.data_manifests` SHA256. The supplied stage/source/object inventory must match exactly. Distinct historical versions of the same source require separate campaigns. Duplicate, missing or mutable object identities fail. Provider stderr is discarded, and no row text or credentials enter the report.

`--producer-policies` is optional except where the shared historical adapter requires authenticated legacy producer evidence. Its JSON maps source names to policy objects validated by `flaxchat.embedding_historical_rows`; the policy file's hash enters the scan receipt and is checked again at completion. Arbitrary policy strings cannot authorize inference. The adapter constructs an ephemeral semantic view for known immutable source identities, including CodeSearchNet code fields and explicitly verified legacy MIRACL aligned group identifiers. Original raw bytes and historical manifests remain unchanged. Missing or conflicting known-source semantics fail closed. Unsupported upstream semantics, inherited MLM exposure, translations and semantic near-duplicates remain unresolved.

Without `--development-exclusions`, `candidate_exposure_checked` is false and overlap counts are null. With it, the scanner authenticates the actual candidate spec/receipt/raw bundle, checks exact text and aligned group exposure against every historical row, and rechecks candidate identity at the end. Nonzero overlap is recorded in a complete scan; it never grants clean development admission. The CLI success status means authenticated coverage, not absence of exposure. A failed or incomplete scan remains `coverage_complete: false`, with already completed source evidence retained.

Default bounds are1800 seconds,4GiB total declared bytes,5million rows per source,64 stages and256 sources. The hard ceilings are3600 seconds,64GiB and10million rows per source. Metadata is capped at16MiB, each line at4MiB, and streaming reads at64KiB. The deadline covers cooperative processing and blocked stream reads. Every process is reaped; its process group is killed during cleanup, with direct-child termination fallback if host sandbox policy forbids group signals. Partial output files are removed on failure. Previously authenticated complete files may remain in a failed campaign directory, but the failed receipt cannot qualify the campaign.

Optional materialization writes the byte-identical source manifest and a temporary train file on that cloud host, then renames the train file only after raw SHA256, bytes, row count and stream exit status all pass. The fresh destination must not exist. No storage cleanup of complete materialized files is automatic: retain them for admission, then remove them under the ordinary storage-retention policy.

The scanner receipt alone is deliberately **not an admission input** (`admission_ready: false`). Feed the verified directories to [the existing parent inventory builder](PARENT_EXPOSURE_INVENTORY_BUILDER.md), together with every known stage and the actual authenticated parent export. Then independent-development preparation and training admission rescan those actual files and reject exposed candidates. These unchanged checks keep an operator-authored success JSON from replacing genuine raw evidence. The complete parent inventory, full lineage scope, physical restoration and model quality remain separate requirements.

Five model-free tests exercise a fake CLI stream: verified materialization feeding the real builder and exposure checker; eight malformed/truncated/tampered/failed/budget/identity cases; a stalled subprocess deadline and reaping with ResourceWarnings treated as errors; authentic candidate overlap; and a scan without candidate evidence never claiming cleanliness. No network, paid allocation or CPU model execution occurs in these tests.
