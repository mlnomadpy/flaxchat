# Next-stage development preparation

The `representation-development-next-stage-v1.json` specification is a candidate
pool, not admitted training data. It covers native MIRACL retrieval in eight
languages, Global Voices bitext in eight large language pairs, CodeSearchNet
code retrieval, and STS-B validation. It scans at most 180,000 rows and selects
at most 9,216 rows. Code programming-language provenance remains unknown in the
pinned stripped source. No test split is used for selection.

The larger 512-row candidate cap allows explicit rejection of historical
training collisions. It does not establish how many rows survive. The bounded
`scripts.prepare_next_stage_development` runner exports the pool, scans all
supplied historical training files, filters collisions, then rechecks the
filtered identity. Fewer than 64 surviving rows in any configuration fails.
Run this data-only process near GCS under the independent data-job supervisor;
its default total timeout is 1,800 seconds. Retain its raw files, logs, original
scan, collision details, filtered scan and terminal receipt.

Historical admission now interprets the original
`modernbert_contrastive_encoder` checkpoint's `data_manifest_sha256` and
`data_manifest_identity`, requiring agreement. Original pair manifests and raw
rows stay unchanged on disk. A read-only semantic view exposes sentence1 and
sentence2 to the same exact-text quarantine checks as later query/positive
training. Every original row must retain the pinned Global Voices training
provenance. Distinct original recipes receive source keys
`contrastive_<full manifest SHA256>`; GCS object receipts may specify these keys
explicitly because their original paths all end in `prepared/train.jsonl`.

Include all three earlier pair recipes (pilot, 28-language continuation and
50k-per-language continuation), plus the final embedding stage. An inventory
covering only the final MS MARCO/MIRACL/code stage cannot establish bitext
independence. Original development rows retain their held-out status; never
fold them into the next training mixture. Scan completeness concerns declared
known history and exact text only, not semantic or inherited MLM exposure.

Metadata acceptance covers original-schema mapping, source byte preservation,
GCS stream coverage, exposed-pair rejection, tampered raw rejection and
mismatched committed identity rejection. It runs no model. A successful data
runner still requires tokenized retrieval/STS preparation, mixture-wide
quarantine, shared production admission and actual physical TPU baseline and
continuation checks before a training or publication claim.

## Completed candidate screening — 2026-10-05

The original v1 candidate failed the unchanged minimum after complete history
screening: Arabic retained 54 rows and Dutch 45, below 64 per configuration.
The separate `representation-development-next-stage-v2.json` keeps the same
18 configurations, immutable source revisions, seed 29 and 10,000 scanned rows
per configuration. Only its selected-row cap increases from 512 to 2,048.
This expands the development pool without choosing rows using model scores.

The expanded candidate passed full screening of all six authenticated raw
sources across the four declared contrastive stages. The original scan found
23,213 overlapping historical rows; the filtered candidate's independent
full-history recheck found zero exact/aligned overlapping rows.

| Development task | Retained rows | Coverage |
| --- | ---: | --- |
| Native MIRACL retrieval | 3,776 | ar, bn, fr, hi, ja, ru, sw, zh |
| Global Voices bitext | 4,077 | ar, de, es, fr, it, nl, pt, ru |
| CodeSearchNet retrieval | 1,419 | Programming language unknown in upstream stripped data |
| STS-B | 1,500 | Validation split; no test split |

Arabic and Dutch bitext retain 199 and 169 rows respectively. These are
independent development candidates, not new training examples or benchmark
scores. Remaining limitations include undisclosed lineage, inherited MLM
exposure and semantic/near-duplicate translations.

Durable artifacts use the prefix
`gs://azettaai-yat-eval-0929/representation-v2-1005/gce-data-expanded02/`:

- `03-filtered-candidate/filtered.tar.gz#1791217025629055` (5,202,851 bytes).
- `04-filtered-recheck/filtered-scan.json#1791217149600503`, SHA256
  `75ffcdcc5d569d7b45608819eaa9429c8b2436ee27f1c7f78df853efa20763a7`.
- `terminal-data.json#1791217153418357`, SHA256
  `e3b9b3afc100bd17b5bb65b7db2463028d2baa565922445ad583972b86ba4f88`.

The filtered exclusion receipt SHA256 is
`c0321faab3b5ff653a6e584fa32330648c609010a8e640c784e22419b2c42e7a`.
The bounded job finished successfully with independently observed process
cleanup and scratch removal. This screening alone does not qualify model
quality, TPU execution, or continued training. Tokenized portable preparation
is a separate subsequent gate.

## Portable tokenized development — verified 2026-10-05

All four development tasks above are now tokenized against the authenticated
embedding-v1 tokenizer/config. Retrieval, bitext and code use query length 128
and document length 256; STS uses length 128. This is preprocessing only and
executes no model. The three task-specific known-history proofs were
independently recomputed and exactly matched their stored proofs, each with
complete declared-stage coverage and zero overlapping rows.

The complete portable archive is:

`gs://azettaai-yat-eval-0929/representation-v2-1005/gce-portable-final04/portable-development.tar.gz#1791218219777261`

- Size: 1,265,421,119 bytes.
- SHA256: `4447de1376a17cd0c5c98cf082d16b37391469d34fa5a6225e317ebff8df2559`.
- Verification: the full generation-pinned object was downloaded again near
  GCS and its complete SHA256 and byte count matched. No range-only or CRC-only
  check substitutes for this verification.
- The archive contains the six original raw training histories once, four
  original stage metadata inputs and producer policies, filtered candidate,
  token arrays and task manifests, relative exposure references, frozen data
  preparation source, and parent-artifact identities. It contains no parent
  model weights.

Extract the archive into one common directory and preserve this layout:

```text
candidate/
historical/<six original source directories>/
metadata-inputs/
independent/retrieval/
independent/bitext/
independent/code/
sts/
parent-files.json
parent-artifacts.json
data-preparation-source.tar.gz
```

Keep `historical/` and `metadata-inputs/` beside `independent/`; copying only
its token arrays breaks recomputable parent-exposure admission. Materialize the
parent separately from `parent-artifacts.json`, checking every full SHA256;
the weights and tokenizer already have independent immutable GCS generations.
Do not treat the original absolute-path inventory retained for lineage as a
replacement for the sealed relative task-specific exposure inputs.

The final successful receipts under `gce-portable-final04/` are:

- `terminal-data.json#1791218519228490`: full proof replay, remote-byte
  verification and all task manifests.
- `controller-terminal.json#1791218569827161`: success and verified process
  cleanup/scratch removal.
- `job-spec.json#1791218571806548`: externally pinned 950-second job.
- `finalization-source.tar.gz#1791218573896008`: the exact bounded finalizer
  and controller.

An earlier additional aggregate proof replay hit a 180-second subprocess
limit after successful tokenization. Its failed receipt remains under
`gce-portable-dev03/terminal-data.json`; finalization reused the unchanged
prepared bytes, allowed 600 seconds for the same three proofs, and completed
without retokenizing or changing validation. The final artifact supersedes the
pending qualification recorded in its earlier `archive-receipt.json`.

Next data work is the *training* mixture, not another development export:
quarantine all six original held-out inventories and the complete independent
candidate before preparing replay, native multilingual retrieval and code
rows; then run mixture-wide quarantine and shared production admission.
The initial fixed proposal is 45% MS MARCO replay, 20% native MIRACL, 15%
code retrieval and 20% bitext replay. This mixture is not yet materialized or
qualified. Physical TPU model validation and measured quality improvements
remain separate prerequisites for training/publication claims.
