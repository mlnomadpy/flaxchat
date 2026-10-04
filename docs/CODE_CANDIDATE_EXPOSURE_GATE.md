# Next development data gates: code, bitext and STS

The retained CodeSearchNet training set is **not the first 300,000 rows of the
upstream stream**. Skipping 300,000 raw rows cannot establish independence.
The authenticated historical producer loaded the pinned `pair/train` dataset,
checked its 1,375,067-row inventory, shuffled with seed 29, and selected the
first 350,000 shuffled positions. It deduplicated candidate triplets, assigned
internal development groups, rejected development-text collisions, and capped
the resulting training set at 300,000 rows. The internal dev file has 3,542
rows. A raw-stream offset and a shuffled-position offset are different domains.

This finding comes from the verified legacy producer in archive
`gs://azettaai-yat-eval-0929/training-input/yat-embedding-src-0929.tar.gz#1790708780709885`:
archive SHA-256
`0565ad45ada5b95801be607bdd366a19f606a7c79e51c0093d85b4bafd2cbb2d`,
producer SHA-256
`8e084b084582eb5fa1192393cdd50add77f8d13396a03930730314f49b3250dc`.
The actual prepared code manifest hash is
`f9e1e8a2424264867dec4b8c8d13a9ece60a2b526f3673381f1fcc5fa985cb56`.
Its raw training hash is
`261f09495697aa5cf8976d24973deafc60904a5412e3c9257931ee17c5956d12`.
The completed historical raw scan verifies the 300,000 training rows and their
bytes; it does not verify the separate historical dev file or complete lineage.

## Code candidate decision

Keep the existing upstream identity:
`sentence-transformers/codesearchnet@079a958b01dc87cf07b66a68414c4b4196d889cc`,
configuration `pair`, split `train`, fields `comment` and `code`. It has no
programming-language column, so labels remain unknown. A different repository
containing the same CodeSearchNet source cannot establish new exposure.

The next useful implementation is **parent-text rejection before candidate
reservation**, rather than a guessed offset. The scanner now supports bounded
candidate collision evidence through `--overlap-details-output FILE` and
`--max-overlap-text-identities N`. It emits only matched SHA identities, binds
the authenticated candidate and known-parent raw inputs, and records whether
details were truncated. An incomplete scan or exceeded cap cannot claim complete
collision details. The existing live job retains its frozen earlier scanner.

A bounded candidate-pool scan can therefore replace a large complete parent
index: freeze the candidate pool, scan actual parent rows against it, and use
complete collision identities to reject candidate text before creating a new
final candidate identity. The explicit adapter is now
`scripts.filter_representation_development`; it does not edit the original
candidate or reinterpret its receipt.
Exact-text details alone also cannot remove an aligned-group collision.

For either approach, perform one bounded data-only cloud join:

1. Authenticate the supplied parent raw objects, manifests and historical
   producer policies, preserving original bytes. Build an exact text membership
   index from their query, positive and real negative fields using the existing
   modality identity policy. A bounded disk-backed index can avoid storing
   millions of Python string objects in host RAM. Retain the complete index
   input hashes and counts, and the identity-policy/source hashes.
2. Stream at most 10,000 pinned CodeSearchNet train records without shuffling or
   a claimed clean offset. Bound serialized input bytes at 256 MiB, per-row size
   at 4 MiB, and the independently supervised processing lifetime. Reject any
   query or code text present in the authenticated parent index **before**
   group-hash reservation and probe selection. This prevents capped-out exposed
   rows from entering the exclusion receipt as supposedly independent probes.
3. Select at most 128 real candidate pairs with query-text grouping. Require
   actual minimum selected rows and distinct query/document coverage. Persist
   scanned, exposed, reserved and selected counts plus actual raw/spec/receipt
   hashes. If insufficient, report failed coverage; do not pad or move to
   benchmark test rows.
4. Recompute candidate exclusions and scan all supplied historical raw files
   against the resulting candidate. Apply the same candidate identity across
   every future training source and recompute admission before TPU evaluation.

The exporter has no full parent-index rejection flag. The implemented bounded
candidate-pool alternative authenticates complete SHA collision evidence before
creating a separate filtered candidate:

```sh
timeout --signal=TERM --kill-after=30s 180s python -m scripts.filter_representation_development \
  --candidate "$WORK/original-code-candidate-pool" \
  --scan "$WORK/code-pool-parent-scan.json" \
  --scan-sha256 "$VERIFIED_SCAN_SHA256" \
  --overlap-details "$WORK/code-pool-overlap-details.json" \
  --overlap-details-sha256 "$VERIFIED_DETAILS_SHA256" \
  --output "$WORK/new-filtered-code-candidate" \
  --min-rows-per-config 64 \
  --timeout-seconds 120
```

Obtain both expected hashes from the verified scanner execution. The tool checks
the complete candidate identity, source/row/hash coverage and scan/detail
agreement; rejects truncated, failed, foreign or aligned-group collision
evidence; and rejects selected rows sharing an exposed group across
configurations. It does not resample or relabel. It retains the original
candidate and receipts inside bounded `filter-evidence`, and admission replays
the rejection to reject rehashed output edits. The original directory remains
unchanged. The new specification records a new identity and explicitly reserves
only retained selected rows; old capped-out reservations are not silently carried
forward as clean probes. The candidate still reports
`parent_exposure_checked: false`: new-identity exposure admission and production
coverage remain required. No raw offset provides a cleanliness guarantee.
The frozen live native-MIRACL data job is unaffected.
If reproducing shuffled positions beyond 350,000 instead, authenticate upstream
row order and the historical shuffle implementation/runtime first. Even then,
different positions can repeat text, so actual parent overlap rejection remains
necessary.

Where available, authenticate the historical code `dev.jsonl` separately
(declared SHA-256
`b0927a294d44bab3f0d4a03e058c774eafef271b22f3bf0b24d153c83cc17afa`)
and reject those texts too: prior checkpoint selection may have used them.
Do not silently label a train-only scan as covering historical validation.
Undisclosed stages, MLM exposure, translations and semantic near-duplicates
remain unresolved after zero exact overlap.

## Independent bitext and STS candidate preparation

`configs/data/representation-development-bitext-sts-candidate-v1.json` is an
executable **partial** candidate input derived from the existing verified v1
source pins. It contains eight individually labeled Global Voices `en-*`
training configurations and STS-B validation. It excludes final test and official
benchmark dev splits. Global Voices groups share English text across language
subsets; STS uses exact pair-text grouping. This input does not include retrieval
or code, does not establish historical cleanliness, and cannot alone meet the
production quality plan.

On a separately admitted, frozen data-only cloud job, the existing exporter can
prepare that input directly:

```sh
timeout --signal=TERM --kill-after=30s 660s python -m scripts.prepare_representation_development \
  --spec configs/data/representation-development-bitext-sts-candidate-v1.json \
  --output "$WORK/bitext-sts-candidate" \
  --timeout-seconds 600 \
  --max-input-bytes 268435456
```

No such job was launched by this document. Preparation still requires actual
coverage and candidate-bound parent exposure checks. The existing native-MIRACL
candidate and these probes must eventually form **one newly frozen complete
candidate specification and exclusion identity**. Production STS must match
that same candidate used by the independent suites; separate partial bundle
identities cannot be presented as a complete production plan. Preserve old
receipts and make the merged plan a new identity. Full parent restoration,
quarantine of all future sources, independent tokenization, and physical TPU
baseline/continuation quality gates remain necessary.
