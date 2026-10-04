# Pinned candidate development recipe — 2026-10-01

`configs/data/representation-development-v1.json` is an executable candidate
specification for a development-only probe. Public repository metadata and
immutable dataset cards were read; no upstream dataset rows, prepared arrays,
or model outputs were downloaded or generated in this verification. The
metadata receipt is
`docs/audit-2026-09-30/development-source-metadata-2026-10-01.json`.

| Purpose | Pinned source / revision | Verified fields and split |
|---|---|---|
| Multilingual retrieval | `nlpai-lab/miracl-multilingual-triplets` / `71bc9f8e7d86b55203ed3104362b0661789a6c31` | `id`, `query`, `positive`, `negative`; train, 2,863 aligned examples per language |
| Bitext retention | `sentence-transformers/parallel-sentences-global-voices` / `4cc20add371f246bb1559b543f8b0dea178a1803` | `english`, `non_english`; individual `en-*` train subsets |
| Code retrieval | `sentence-transformers/codesearchnet` / `079a958b01dc87cf07b66a68414c4b4196d889cc` | `comment`, `code`; pair/train, 1,375,067 examples; no programming-language column |
| Semantic similarity | `sentence-transformers/stsb` / `ab7a5ac0e35aa22088bdcf23e7fd99b220e53308` | `sentence1`, `sentence2`, `score`; validation, 1,500 examples; scores normalized to 0–1 |

The MIRACL-derived source consists of translated English **training**
triplets, rather than the official MIRACL benchmark development set. Its
language alignment is declared upstream in the
[immutable card](https://huggingface.co/datasets/nlpai-lab/miracl-multilingual-triplets/blob/71bc9f8e7d86b55203ed3104362b0661789a6c31/README.md).
The [Global Voices card](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices/blob/4cc20add371f246bb1559b543f8b0dea178a1803/README.md)
confirms the two text columns and per-language training configurations; the
combined `all` configuration drops language provenance and is excluded.
The stripped [CodeSearchNet card](https://huggingface.co/datasets/sentence-transformers/codesearchnet/blob/079a958b01dc87cf07b66a68414c4b4196d889cc/README.md)
confirms only comment/code columns, so programming language stays `unknown`.
The [STS-B card](https://huggingface.co/datasets/sentence-transformers/stsb/blob/ab7a5ac0e35aa22088bdcf23e7fd99b220e53308/README.md)
separates validation from test. Only validation is eligible here.

## Bounded preparation

```sh
python -m scripts.prepare_representation_development \
  --spec configs/data/representation-development-v1.json \
  --output /path/to/new-candidate-development
```

This uses immutable revisions and streaming reads, bounded at 10,000 scanned
rows and 128 selected probe rows per declared configuration. These are small
candidate probes, not a full benchmark inventory. STS currently selects a
bounded prefix of validation; train-derived probes use a stable group-hash
reservation. The spec chooses eight retrieval languages and eight bitext
language pairs; it does not assert all-language coverage. Increase budgets or
change coverage only as an explicit newly pinned probe identity.

The exporter writes raw JSONL probes, `spec.json`, and `exclusions.json`.
MIRACL reservation keys share source alignment IDs across language subsets;
all scanned translations matching a held-out ID are reserved together.
Global Voices reservation keys use English text across language subsets.
Code reservation keys use comment text. Text identities preserve code syntax.
Capped-out reserved groups are also recorded, avoiding a silent change in
group admission at the probe cap.

Training preparation now accepts `--development-exclusions /path/to/candidate`.
It authenticates the spec, complete source/config inventory, raw probe hashes
and exclusion receipt before selecting either train or internal dev rows. Any
reserved query/document/negative text is excluded across source identities.
For aligned upstream datasets, the original `upstream_group` is retained and
reserved IDs are excluded across all language subsets of the same pinned
repository. Missing original group provenance fails rather than allowing an
unverifiable translation through. The prepared manifest binds this quarantine
identity and records rejection counts. Every prospective training source must
use the same candidate identity; preparing one source does not qualify the
whole mixture. These exclusions do not establish parent cleanliness.

The current training registry intentionally accepts only upstream training
splits. This development specification is a separate interface and **must not
be passed directly to `--registry`**. It does not generate fake training rows
to satisfy the trainer's paired train/dev schema. The existing STS preparation
interface can consume the exported STS JSONL with its recorded SHA256,
`--source-split validation`, and pinned source repository/revision.

## Required adoption gates

1. Check selected probe text and aligned groups against the actual parent
   training inventories. Data already seen by the existing model cannot
   become honest held-out development merely because a later stage reserves
   it. Existing model contamination status is currently unverified.
2. Apply the exclusion receipt across all prospective training sources,
   including matching translated MIRACL IDs and English Global Voices groups;
   verify complete known positives and exact-text quarantine. The exporter
   intentionally records `quarantine_applied: false` and
   `parent_exposure_checked: false`; generation alone does not meet admission.
3. Use the implemented `scripts.prepare_embedding_retrieval_dev` and
   `--retrieval-dev NAME=DIR` stage admission/evaluation integration for independent
   retrieval/bitext/code probes. Actual authenticated preparation, quarantine and
   production coverage remain required. Pin tokenizer/array manifests, selected
   rows, candidate corpus, metric and quality policy before tuning. Generic
   preflight currently allows empty independent suites; it is not a production
   coverage gate.
4. Execute actual TPU baseline/continuation comparisons. Bitext here is
   directional retrieval of aligned pairs; it is not Tatoeba final-test
   reporting. Code language coverage remains unknown for this upstream release.
   Broader multilingual STS coverage is also still missing.

Official final benchmark splits, MIRACL official benchmark dev, and Tatoeba
final test are excluded from tuning. Keep final test inventories unchanged for
release reporting. Synthetic exporter fixtures establish split, pinning,
alignment reservation, bounded scans, and honest receipt status; they do not
establish availability/quality of all actual prepared rows.

## Bounded exporter follow-up

The exporter now caps aggregate scanned configurations/rows, selected rows,
serialized input bytes, row size and admitted raw/exclusion file size. It reads
exactly the admitted number of records without fetching one extra record,
authenticates the specification before and after preparation, and removes
staged output on failure. CLI limits include `--timeout-seconds` (default 1800,
maximum 3600) and `--max-input-bytes` (default 512 MiB, maximum 4 GiB).
The byte count measures serialized records, not upstream Parquet download bytes.
The cooperative deadline checks processing boundaries; run remote streaming
preparation under an independent process timeout to bound a blocked provider
read. These limits do not certify candidate independence or model quality.
