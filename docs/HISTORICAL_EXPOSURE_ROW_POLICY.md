# Historical exposure row semantics

Exposure checks authenticate historical stage metadata, prepared manifests and
raw training bytes. Those originals must remain unchanged. A semantic migration
is an ephemeral row view with its own versioned policy and source fingerprints;
it is not permission to rewrite historical JSONL or regenerate a manifest to
make checks pass.

`flaxchat.embedding_historical_rows.historical_row_view` is shared by local
known-stage exposure checks and generation-pinned GCS scanning. It recognizes
exact upstream repository/revision pairs rather than dataset nicknames.

For `sentence-transformers/codesearchnet` at
`079a958b01dc87cf07b66a68414c4b4196d889cc`, the authenticated upstream schema is
comment/code. The view assigns natural-text query and exact-code
positive/negative modalities, independently of a historical source alias.
Conflicting declared modalities fail. No programming language is inferred:
the stripped source did not preserve that column. This prevents a false-clean
result when missing code modalities caused natural-text NFC normalization of
decomposed Unicode, while the candidate used exact code identity. Other
revisions of the known source require a separately authenticated adapter.

For `nlpai-lab/miracl-multilingual-triplets` at
`71bc9f8e7d86b55203ed3104362b0661789a6c31`, retained `upstream_group` is the
alignment key. Missing alignment cannot be replaced by an arbitrary local
group or guessed from its final colon component. Legacy inference is accepted
only under an explicit producer policy whose original source archive and
preparation script fingerprints have been independently verified and entered
into the helper's allowlist. The exact group grammar must come from that
verified producer implementation, not from a newer version or documentation.
Unverified policy or ambiguous/malformed groups fail closed.

A producer policy can be supplied separately as
`sources[].historical_row_policy` in the authenticated exposure input, avoiding
any change to the historical manifest. Exposure proof records this policy and
continues hashing/rechecking the original rows and their exact inventory.
Generation-pinned scans must additionally bind original GCS generation, size
and digest; applying a row view does not relax those transport checks.

The original September 29 producer has now been independently read and
fingerprinted from the generation-pinned archive recorded in
`docs/audit-2026-09-30/legacy-producer-verification-2026-10-01.json`.
Archive SHA256 is
`0565ad45ada5b95801be607bdd366a19f606a7c79e51c0093d85b4bafd2cbb2d`;
the preparation member SHA256 is
`8e084b084582eb5fa1192393cdd50add77f8d13396a03930730314f49b3250dc`.
Both hashes and equality to the standalone source were independently rechecked
without executing the producer. Its lines 72–74 preserve
`group = miracl:{language}:{id}` and `coordinate = {language}:{position}`;
it does not retain a separate human-language field. Policy
`yat-embedding-src-0929-v1` permits this precise read-only reconstruction.
It requires exact archive URI/generation and both fingerprints, a recognized
pinned language subset, canonical nonnegative integer ID, and matching
language/position coordinate. A separately supplied language field must agree.
Malformed/ambiguous encodings and wrong producer/source pins fail closed.

The legacy producer's lines 55–57 normalize text for stored identity hashes
using NFKC/casefold/whitespace, but lines 211–219 write the original text fields
unchanged. Exposure checks therefore rederive exact text/code identities from
authenticated raw text rather than trusting those legacy hash fields. This
does not change which inputs the historical tokenizer saw or claim semantic
non-overlap.

Successful checks establish exact text/aligned-group non-overlap only over the
declared, authenticated contrastive-stage inventory. They do not establish that
the inventory is the entire model history, that inherited MLM data was clean,
or that semantic/near-duplicate/translation exposure is absent. Preserve those
unresolved scopes in the proof and model documentation. A historical source
alias, current data recipe, or successful scan cannot supply missing lineage.
