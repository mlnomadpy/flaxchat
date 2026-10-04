# Replacement development-source review — October 1, 2026

This is a metadata review, not an admitted or prepared v2 recipe. The existing
v1 specification remains unchanged. No new v2 configuration is supplied because
code-source independence is unresolved. No model was executed and no cloud
allocation was made. Actual candidate text must pass the retained parent scan
before any source becomes independent development.

The bounded retrieval-only [native MIRACL candidate gate](NATIVE_MIRACL_CANDIDATE_GATE.md) now supplies an executable specification and cloud-side preparation plan. It does not replace the missing full production recipe or actual exposure checks.

## Retrieval candidate

The pinned [translated MIRACL card](https://huggingface.co/datasets/nlpai-lab/miracl-multilingual-triplets/blob/71bc9f8e7d86b55203ed3104362b0661789a6c31/README.md)
states that its 2,863 English examples come from **English MIRACL training**, with
translations into 50 languages. They do not come from MS MARCO. Card SHA-256:
`8659acdeb3640b802aa66acf00db29f5a52e211a0b3e205c690f22c94be2e238`.

[Sentence Transformers MIRACL](https://huggingface.co/datasets/sentence-transformers/miracl/blob/07e2b629250bf4185f4c87f640fac15949b8aa73/README.md)
is pinned at `07e2b629250bf4185f4c87f640fac15949b8aa73`. Its card describes
reformatted BGE-M3 MIRACL data, retaining the first positive and first negative.
Card SHA-256: `dc2423545fc9d07a4dc730ce199d3ce58f7b5d09140e6bc2d0edd8c29422369b`.
Public HF API metadata was read alongside the pinned card.

| Candidate configuration | Declared train rows | Fields |
|---|---:|---|
| ar-triplet | 3,495 | anchor, positive, negative |
| bn-triplet | 1,631 | anchor, positive, negative |
| fr-triplet | 1,143 | anchor, positive, negative |
| hi-triplet | 1,169 | anchor, positive, negative |
| ja-triplet | 3,477 | anchor, positive, negative |
| ru-triplet | 4,683 | anchor, positive, negative |
| sw-triplet | 1,901 | anchor, positive, negative |
| zh-triplet | 1,312 | anchor, positive, negative |

Exclude `en-triplet` from this replacement proposal: its 2,863-row count and
shared English MIRACL origin provide no metadata basis for independence from
the retained English-derived parent data. Native non-English train configurations
are candidates, not proven clean alternatives. No official benchmark dev/test
split should be used for checkpoint selection.

The exporter now requires an explicit pinned configuration-to-language map for
native retrieval configurations, for example `ar-triplet` → `ar`. Admission
also verifies the selected row language and receipt provenance against that
map. This adapter is implemented and checked with injected rows; actual native
MIRACL rows remain unprepared and their parent exposure remains unverified. Preserve raw
anchor/positive/negative text and use query-text grouping: the flattened schema
has no original query-ID field and does not establish aligned translation IDs.
Unmapped native configuration names are rejected before loading.
Do not fabricate alignment or silently reinterpret a previously prepared source.

## Code-source decision

[Google CodeXGLUE code-to-text](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text/blob/2678080e78a4b20ad79d1f0a514f04d815912564/README.md)
has train configurations for Go, Java, JavaScript, PHP, Python and Ruby, and
authenticates `docstring`, `code` and `language` fields. Its card explicitly
describes CodeSearchNet-derived data. It can recover genuine language provenance,
but a new repository name cannot establish new historical exposure. Card SHA:
`d0d9ac9154a125b5d1d5eda1feb3c492b6fb5c3b00310f5704dc7d6913cf2412`.

Scanning a later portion of the original CodeSearchNet train release is another
candidate-selection option, but stream order and retained-parent overlap must be
verified from actual raw rows. An offset is not proof of independence. Either
option still requires a bounded parent-overlap scan and future-source quarantine.

[Jina code exercises](https://huggingface.co/datasets/jinaai/code_exercises/blob/998698fc883c9885d966481e4a2dc7eaf7a85738/README.md)
is a different-origin synthetic Python problem/solution source, but its declared
noncommercial license and mismatch between card narrative and API row inventory
make it unsuitable as the default commercial product recipe without further
review. It is not included. No code source is declared independently clean by
this metadata review.
