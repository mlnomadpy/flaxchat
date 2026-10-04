# Programming-language provenance

Preparation preserves declared CodeSearchNet programming-language columns
(`programming_language`, `language`, or `lang`, in that order). A stripped
query/code source without those columns is labeled `unknown`; comments or code
syntax are never used to invent provenance. Human query language remains its
own `language` field (`und` for the built-in code source). A programming-language
label is not evidence that natural text is multilingual.

Pinned registry sources can map `fields.programming_language` or declare a
`programming_language` constant separately from `fields.language`/`language`.
Declare code modalities, for example `modalities.positive = "code"`, so the
mixture recognizes these rows as code. The raw row stores the label and its
column/constant/unavailable provenance, covered by the raw file checksum.
Tokenization manifests summarize declared programming-language counts for
each split. Prepared mixtures derive host-only programming-language IDs and
labels from authenticated raw rows; they do not add a model input or alter
natural-language replay probabilities.

Cumulative exposure reports/checkpoints include a separate
`programming_languages` distribution. Missing code provenance appears as
`unknown`; non-code rows appear as `not-applicable`. Resume validates total
counts and available labels. Source and mixture receipt changes are immutable
stage-identity changes: migrate deliberately to a new stage from trained
weights rather than silently changing an existing exact resume.

This improves provenance and coverage reporting. It does not implement
programming-language balancing or per-programming-language development gates,
and does not establish code retrieval quality. Those require explicitly pinned
data coverage, representative held-out tasks, and physical TPU continuation
evidence. Existing raw artifacts lacking provenance cannot recover labels that
their upstream release omitted; reprepare from an authenticated labeled source
when that coverage is required.

Model-free acceptance is in `tests/test_code_provenance_metadata.py`: declared
versus unavailable labels, independent registry human/code fields, authenticated
mixture exposure and exact metadata restore, and invalid code exposure rejection.
