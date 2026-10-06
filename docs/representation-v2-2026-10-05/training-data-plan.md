# Representation-v2 training data preparation

Status: full data preparation, final quarantine and shared production preflight
passed on 2026-10-06 UTC, using source freeze `303b5058…`. The prepared archive
is generation-pinned and fully hash-verified; see `repair-validation.json`.
The live training controller still requires physical TPU qualification before training. This is continued contrastive representation training from embedding-v1,
not a new MLM run and not training from random initialization.

## One cloud-side command

Run from the frozen repository root on an already admitted, provider-limited
Linux data VM, using a cloud-side durable controller. The laptop is not the
controller. Install the campaign's pinned versions of `numpy`, `tokenizers`,
and `datasets` with their dependencies; no JAX/model execution is required.

```sh
python -m scripts.prepare_representation_v2_training \
  --portable-archive /inputs/portable-development.tar.gz \
  --heldout-archive /inputs/complete-heldout.tar.gz \
  --heldout-sha256 2310e8cb0607e928412a34deafc211e9242225702c3012b2b6f2335f88208ee4 \
  --tokenizer /parent/tokenizer.json \
  --output /work/training-v2 \
  --total-seconds 7200
```

The output must not exist. This is one finite execution: full raw preparation,
11 source tokenizations, mixture-wide qualification, and independent final raw
quarantine audit. All subprocess groups are killed at their timeout/exit. The
shared deadline covers the subprocess phases; reserve separate bounded time
for input transfer, environment setup, durable upload/readback, and VM deletion.
The provider-enforced deletion deadline and cloud controller remain external
requirements. This runner never provisions a VM/TPU or launches training.

The 7,200-second duration is a maximum, not a measured runtime guarantee.
Allow roughly 60 GiB working disk headroom and preferably 32 GiB host RAM for
full replay, repeated tokenization, identity sets and mixture grouping. These
are planning allowances, not measured peaks. Do not occupy paid TPU capacity
while preparing raw data.

## Immutable inputs

- Portable development archive:
  `gs://azettaai-yat-eval-0929/representation-v2-1005/gce-portable-final04/portable-development.tar.gz#1791218219777261`.
  Full SHA256 `4447de1376a17cd0c5c98cf082d16b37391469d34fa5a6225e317ebff8df2559`.
- Complete six-source held-out supplement:
  `gs://azettaai-yat-eval-0929/representation-v2-1005/repair-1005/complete-heldout.tar.gz#1791242883233725`.
  13,902,096 bytes, full SHA256
  `2310e8cb0607e928412a34deafc211e9242225702c3012b2b6f2335f88208ee4`.
  Pass this exact hash with `--heldout-sha256`. The old three-pair-only
  supplement is insufficient and must not be used on its own.
- Parent tokenizer: full hash must match the archive's `parent-files.json`;
  expected current parent SHA256
  `197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356`.

Archives are fully hashed before extraction. Links and special members are
rejected. The original archives are not altered. The portable bundle extracts
to `development/`; the supplement extracts to `original-heldout/`.

The six held-out inputs are `original-heldout/{msmarco,miracl,code}/dev.jsonl`
and `original-heldout/contrastive_<manifest SHA>/dev.jsonl` for all three pair
stages. Portable history has no MS MARCO/MIRACL/code dev files; never assume
those files are supplied by the development archive. Every held-out file must match its original manifest checksum. Missing
inputs fail closed. `heldout-inventory.json` records exact relative paths,
checksums and counts. None of these rows becomes training data.

## Mixture and coverage

| Source | Weight | Available candidate scope before quarantine |
| --- | ---: | --- |
| MS MARCO replay | 45% | All 1,250,000 authenticated old training triplets |
| CodeSearchNet replay | 15% | All 300,000 authenticated old training pairs |
| Global Voices bitext replay | 20% | All 631,332 authenticated old training pairs across 28 language pairs |
| Native MIRACL | 20% total | Full pinned training configurations ar, bn, fr, hi, ja, ru, sw, zh; 2.5% each |

The replay subtotal is 2,181,332 candidate rows. Full preparation admitted
2,165,175 training rows across all eleven sources after splitting and quarantine. The full replay
pool replaces the earlier proposed caps of 250k/100k/200k. Replay reuses old
training data; it must not be reported as entirely new exposure. Native MIRACL
uses `sentence-transformers/miracl` at revision
`07e2b629250bf4185f4c87f640fac15949b8aa73`, `<language>-triplet`, `train`.

CodeSearchNet's stripped source has unknown programming-language labels. This
mixture includes code but cannot establish per-programming-language coverage,
repository-scale file search performance, or that the model has seen everything.
Twenty-eight-language bitext coverage does not establish equally strong native
retrieval in all those languages. The independent native retrieval development
set covers only the eight declared MIRACL languages.

All raw candidate fields are checked against all six original held-out inputs
under text and code normalization and against the entire independent candidate.
Preparation uses new development split seed **41**, separately from native
source shuffle seed **29**. Original MS MARCO/code replay training was already
filtered by split seed 29; reusing that seed would deterministically leave
those replay sources with no new development rows. The new seed creates a new
1% group-based development split within the remaining historical training pool,
exact-text quarantine, and mixture-wide shared-query/group equivalence checks.
These new replay development slices were exposed to the parent during earlier
training: use them only as training diagnostics. Quality decisions use the
authenticated independent retrieval, bitext, code and STS development sets.
A model-free fixture running the actual raw preparation split (tokenization
mocked) reproduced the seed-29 empty-dev failure on 9,900 old training groups;
seed 41 produced 9,796 train and 104 development rows. This verifies the fix on
a fixture, not the final production row counts.
The final audit independently checks every final train/dev row against the
historical and independent exclusions. Token lengths remain query 128/document
256 to match the admitted representation path. Longer code context requires a
separate data/TPU qualification; it is not silently enabled here.

## Outputs and launch handoff

Retain `registry.json`, `weights.json`, `heldout-inventory.json`, all command
logs, `terminal-data.json`, `final-quarantine.json`, `mixture/mixture.json`, and
each `mixture/<source>` directory. Keep the portable development directory with
its original relative layout and historical references intact. The prepared
raw inputs must remain immutable; manifests preserve their identities.

Successful `terminal-data.json` and `final-quarantine.json` establish only
model-free preparation. Before training, authenticate the final arrays and
parent through shared stage preflight; run bounded physical TPU parent/forward/
backward/recovery checks and parent baselines. A new stage uses parent weights
with a fresh optimizer, new output prefix, fixed YAT bias 1/epsilon 0.01 and
trainable alpha. Continue only under the campaign's declared quality, cost and
lease bounds. No representation quality or EmbeddingGemma competitiveness is
inferred from row volume or time spent training.


## Complete original-heldout repair

The earlier three-pair supplement was incomplete: the portable development
archive contains history manifests and training rows, but no historical
MS MARCO/MIRACL/code dev rows. The exact original cloud code failed with
`FileNotFoundError: historical/code/dev.jsonl`; the traceback is retained in
`raw-failure-reproduction.json`.

Use the complete six-source supplement, verified against original manifests and
full GCS readback:
`gs://azettaai-yat-eval-0929/representation-v2-1005/repair-1005/complete-heldout.tar.gz#1791242883233725`.
It is13,902,096bytes, SHA256
`2310e8cb0607e928412a34deafc211e9242225702c3012b2b6f2335f88208ee4`.
Do not use the earlier three-pair archive alone. All six heldout files are now
validated before scanning training rows. Changed archive SHA must be explicitly
passed and bound in downstream receipts.

48 tests passed on the actual Linux VM, including the miniature eleven-source
raw/tokenization/mixture/audit pipeline, actual historical schemas, lifecycle
checks and subprocess failures. The separate full production preparation and
shared stage preflight also passed. These remain model-free evidence; physical
TPU acceptance is separate. Durable results are linked in `repair-validation.json`.
