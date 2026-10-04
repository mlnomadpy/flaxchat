# Native MIRACL candidate preparation gate

The executable input is
`configs/data/representation-development-native-miracl-candidate-v1.json`.
It prepares retrieval candidates only. It is not a full production v2 data
recipe, a training registry, or an admitted independent development suite.
Code-source independence and authenticated bitext/STS coverage remain separate
requirements. No actual rows have been prepared by adding this specification.

The [pinned source card](https://huggingface.co/datasets/sentence-transformers/miracl/blob/07e2b629250bf4185f4c87f640fac15949b8aa73/README.md)
and prior metadata review in `docs/REPRESENTATION_DEVELOPMENT_V2.md` define the
source identity. Eight native non-English train configurations map explicitly
to `ar`, `bn`, `fr`, `hi`, `ja`, `ru`, `sw`, and `zh`. English is excluded from
this replacement candidate. Raw `anchor`, `positive`, and `negative` text is
preserved. Query-text hashes define groups; this flattened source does not
provide an original aligned query ID, so none is fabricated.

## Frozen cloud-side input plan

Use a new, owned, finite-lived data-only cloud job; do not execute the model.
Verify its account/project, runtime, current compute-price basis, ancillary
reservation, scratch capacity, independent deadline and cleanup before starting.
The previous scan's terminal job cannot be reattached as a live process. Do not
allocate a TPU for data preparation or transfer the 2 GB historical corpus
through the local connection.

Freeze and hash the current exporter, quarantine loader, scanner and their
transitive imports together with the candidate spec. The old scanner archive
contains older exporter/loader implementations and must not be used as the
current code bundle. Its authenticated metadata inputs remain useful:

- Historical source/input archive:
  `gs://azettaai-yat-eval-0929/system-fixes-1001/cloud-parent-scan-01/source-inputs.tar.gz#1790861018454344`.
- Archive SHA-256:
  `07e9df50ee5c5dbf1e4adb0544396e7ef391472c703d9f30f3b316f2a32cdc08`.
- Reuse verified `inputs/checkpoint-metadata.json`, `inputs/objects.json`,
  `inputs/producer-policies.json` and the code/MIRACL/MS MARCO manifests.
  Authenticate them against the archived inventory and
  `docs/audit-2026-09-30/cloud-parent-scan-01/admission.json` before use.
- Freeze the data-only Python environment, including `datasets` and its
  streaming dependencies; record exact installed versions. No floating
  dependency installation is part of this plan.

The following commands assume execution from that verified source bundle on
the cloud host. `WORK` denotes a fresh owned scratch directory with at least
6 GiB available disk; `timeout` is an independently verified process timeout
utility. The complete job needs a separate deadline that also bounds setup,
result upload, and cleanup. Do not treat these commands alone as a supervisor.

```sh
timeout --signal=TERM --kill-after=30s 360s python -m scripts.prepare_representation_development \
  --spec configs/data/representation-development-native-miracl-candidate-v1.json \
  --output "$WORK/native-miracl-candidate" \
  --timeout-seconds 300 \
  --max-input-bytes 268435456

timeout --signal=TERM --kill-after=30s 1260s python -m scripts.scan_gcs_parent_exposure \
  --stage-metadata inputs/checkpoint-metadata.json \
  --prepared-directory inputs/code \
  --prepared-directory inputs/miracl \
  --prepared-directory inputs/msmarco \
  --object-receipt inputs/objects.json \
  --producer-policies inputs/producer-policies.json \
  --development-exclusions "$WORK/native-miracl-candidate" \
  --materialize-directory "$WORK/historical-raw" \
  --output "$WORK/native-miracl-parent-scan.json" \
  --timeout-seconds 1200 \
  --max-total-bytes 3000000000 \
  --max-rows-per-file 5000000
```

Preparation scans at most 5,000 rows per configuration and selects at most 128
per language by the declared hash reservation. The 256 MiB budget measures
serialized input records, not downloaded Parquet bytes. Neither selected-row
count nor distinct-query/document count is guaranteed before execution. Fail
the gate if actual coverage misses the declared production plan; do not pad
with duplicates. Materializing raw history avoids another remote transfer when
the authenticated parent inventory and admission loader later need actual rows.

## Result that changes the next action

The preceding historical integrity scan completed for 1,694,512 supplied rows
and 2,075,961,764 bytes, but had no candidate identity and no overlap counts.
Its `scan.json` therefore cannot establish cleanliness for this candidate.
The new scan must authenticate the candidate directory and every pinned raw
object, complete its full declared inventory, and report actual exact-text/group
overlap counts. Retain candidate spec/raw/exclusions hashes, runtime, source
hashes, stdout/stderr and the terminal controller/cleanup receipt.

Any observed overlap requires an explicit replacement candidate identity and
re-preparation; do not edit an admitted suite in place. Zero exact overlap
would cover only the supplied known contrastive rows. Undisclosed lineage,
MLM/pretraining exposure, translations and semantic near-duplicates remain
unresolved. A retrieval-only success still cannot satisfy the production
retrieval/bitext/code/STS coverage gate.

After acceptable raw exposure results, authenticate the actual parent export,
build its known-stage inventory from retained raw files, quarantine every future
training source with the same candidate identity, tokenize the independent
suite with the parent tokenizer, then evaluate the restored parent on physical
TPU. The scanner receipt does not replace those admission or model-quality steps.
Keep large raw history cloud-side, upload only bounded receipts and candidate
artifacts under a fresh generation-guarded prefix, and independently verify
owned-process termination and scratch cleanup after completion.
