# Coverage sampler transition with preserved training state

Design review of current `train_encoder.py`, checkpoint APIs and sampler on
October15,2026. This is source/metadata review, not physical qualification. The
existing frozen cloud campaign remains unchanged while this transition is built.
No allocation, restart, cleanup, IAM change or CPU model execution is authorized
by this document.

## Migration contract

An explicit coverage migration is a new checkpoint lineage with **the same model,
optimizer, learning-rate schedule and completed training progress**. It is neither
model-only initialization nor an exact resume under the old sampler identity.
After the first migrated commit, ordinary strict exact resume applies to that new
source/sampler identity.

The supported old state is `coverage_sampling=pool_epoch_v1`; the new state is
`remaining_pool_epoch_v1` with one declared transition step and original batch
size. Authenticate the selected committed metadata and full manifest with
externally pinned SHA256 values and the old model-source SHA. Require the
metadata/receipt selected step and restored completed-step cursor to equal that
transition. Select and pin the actual final checkpoint **after** the current cloud
campaign terminates and owned resources are independently absent. Do not use a
checkpoint number observed while a trainer may still advance or prune it.

Allow only the sampler field replacement and explicit transition step/batch
identity fields. Require exact, type-sensitive equality for all other resolved
configuration: architecture and precision, runtime/numerical settings, tokenizer,
corpus manifest, batch/accumulation/mesh, seed, masking, optimizer hyperparameters,
learning-rate schedule, warmup and500000-update horizon. Do not whitelist arbitrary
new metadata fields or grant a generic source compatibility bypass. The externally
pinned old source is accepted only by this named migration; the new source is
independently frozen and qualified.

Call `restore_model_from_checkpoint` with an optimizer and
`load_training_state=True`, supplying the **old** full expected identity. This
path authenticates actual model, optimizer and training-state manifests before
mutating live state and rejects optimizer restores from unsupported v2 schemas.
Do not call `--initialize-from-checkpoint`, which deliberately creates a fresh
optimizer/schedule/cursor. Do not rewrite checkpoint metadata to make a new source
look like the old one.

Preserve the stored optimizer moments, Optax schedule counter, NNX optimizer step
and completed/update cursors. Require consistent counters rather than resetting
one to make it fit. Reconstruct exactly the same500000-step schedule and prove its
next learning rate agrees at the restored update count. Preserve existing trained
initialization lineage and append the migration parent manifest/metadata/source,
old/new sampler policies, transition step and actual new source SHA.

## Remaining-data semantics

Reconstruct old exposure using the exact old sampler algorithm, seed, global batch
and completed-step cursor. Admission must bind the unchanged actual token arrays,
document provenance, pool order and manifest. Preserve a deterministic digest of
the legacy exposure state and remaining-row inventory. A replay from changed code
is insufficient unless its old-policy implementation is verified against the
frozen legacy sampler. All hosts must agree on that exposure identity.

Select previously unseen corpus row identities until each remaining row has been
used; retain deterministic checkpoint-based seeking and replay. Do not silently
start each pool at epoch0 or use a newly seeded legacy exposure approximation.
The target is corpus-row coverage; it does not prove unique semantic/text content,
full vocabulary prediction, or every token masked exactly once. Completing the
remaining pool inventory changes the realized source/language allocation relative
to indefinite temperature sampling; report actual exposures and pool counts.

The final incomplete global batch must contribute each remaining row once and
mask unused batch positions out of both loss and exposure. Never pad by repeating
valid training rows, and never pass negative example IDs into the current masking
API. Physical acceptance must cover zero-target microbatches and correct global
masked-token weighting under accumulation. If the completion step exceeds the
preserved schedule horizon, fail admission; extending that horizon is a separate
decision with a different optimization contract.

## Required gates

1. Model-free rejection tests for changed pins/source, malformed metadata, another
   tokenizer/data/config/runtime/schedule/batch, mismatched transition cursor,
   repeated migration, initializer/resume conflicts and aliasing output paths.
   Assert rejected cases do not create output managers or mutate a parent.
2. Metadata/sampler tests proving legacy-exposure equivalence, exclusion of all
   legacy-seen row identities, complete remaining-row coverage, deterministic
   seek/resume and final partial-batch accounting. Check sum of exposures against
   actual rows/tokens rather than inferring full coverage from step count.
3. On physical TPU, restore the actual protected final checkpoint and compare
   model, optimizer and training-state manifest identities immediately before any
   update. Verify exact counters and unchanged next scheduled learning rate.
4. In an isolated qualification namespace, perform a few real migrated updates,
   save, restore with strict new identity and continue. Compare with an
   uninterrupted migrated reference under existing bounds and verify exact
   disjoint row/exposure progression. Evaluate the same fixed-mask heldout rows
   against the final pre-transition model. Qualify incomplete-batch weighting.
5. Freeze the accepted migration source/runtime/receipt and deploy one bounded
   guarded campaign with a fresh namespace, retained protected parent, shared
   ledger, checkpoint cadence and independently verified cleanup. Publish only
   observed committed progress and measured coverage; migration qualification is
   not evidence of a completed10B corpus pass or embedding quality.

## API observations and pitfalls

`load_checkpoint_metadata` is a metadata-only view; it cannot qualify actual tensor
or optimizer restoration. `read_committed_metadata` supplies externally bound
JSON identity and selected step, while full `restore_model_from_checkpoint`
checks tensor manifests and conforms to the intended live topology. Keep both
checks, including revalidation if storage changes between them.

`validate_resume_cursor` permits rejected legacy updates, so completed steps and
optimizer updates need not be interchangeable in a generic checkpoint. Preserve
their real values and compare the stored optimizer counter. Historical finite
all-applied update evidence may support equality for this specific production
parent; do not infer it from the checkpoint directory number alone.

The existing sampler can recreate exposure from step and batch. A transition
requires both the fixed legacy cutoff and the current new-state cursor; resuming
must not move the legacy cutoff forward. Logs must name the new remaining-coverage
policy, actual legacy/new exposure and completion state; a generic
`with_replacement` label or full-horizon expected-row estimate would be misleading.

Distinct cloud controller names do not change training lineage. Preserve active
owned handles while the old campaign runs, and obtain fresh terminal checkpoint,
credit/budget, scope, quota and cleanup admission before a separately authorized
physical migration campaign.

## Implemented CLI and final source review

The explicit migration invocation adds these flags to the unchanged authenticated
training recipe:

| Flag | Meaning |
| --- | --- |
| `--migrate-coverage-from-checkpoint` | Existing protected old-policy checkpoint namespace; must differ from output |
| `--remaining-coverage-from-step` | Exact committed legacy cutoff, retained unchanged for every subsequent resume |
| `--coverage-parent-manifest-sha256` | Independently pinned canonical full parent manifest hash |
| `--coverage-parent-metadata-sha256` | Independently pinned canonical committed metadata hash |
| `--coverage-parent-source-sha256` | Actual old committed model-source identity |
| `--coverage-sampling` | Required alongside the remaining-coverage policy |

Use a distinct `--output`. Migration excludes `--resume`, pretrained/public or
model-only initialization flags. Keep every original training option explicit;
the parser defaults are not the production recipe. In particular preserve the
500000-step schedule, original seed, global batch, accumulation, learning rate,
warmup, masking, language exponent and precision/kernel configuration.

After the zero-update migrated checkpoint is committed, use `--resume` with the
same new output, source/recipe and `--remaining-coverage-from-step` cutoff. Omit
`--migrate-coverage-from-checkpoint` and all three parent-pin CLI flags. The cutoff
is the original legacy transition, not the current new-stage checkpoint number.
`--stop-after` may bound an invocation while preserving the full schedule horizon;
the trainer additionally clamps completion to the finite remaining-row horizon.

The latest source authenticates the canonical **actually loaded** checkpoint
manifest via `load_checkpoint(expected_manifest_sha256=...)` before state
validation, conformance or mutation. The migration helper passes its external
manifest pin through `restore_model_from_checkpoint`'s expected identity. This
closes the admission/restore gap without rereading mutable storage. Calls that
omit the optional pin preserve the existing API behavior.

Migration checks the restored NNX optimizer step against the restored update
cursor. Remaining-coverage exact resumes also check equality before the existing
counter assignment; they cannot silently repair a mismatch. A durable initial
commit at the transition step saves full model/optimizer/training state with zero
new updates, allowing physical comparison to parent leaf hashes. An all-padding
tail does not trigger an AdamW-only update; zero selected targets retain the
actual optimizer-update count while a completed input cursor can still advance.
Coverage, completed steps and optimizer updates must remain distinct in reports.

Independent read-only interface review found no further deterministic activation
blocker in these changes. Python AST checks passed without model imports. This is
**go for the bounded physical qualification gate only**; physical migration,
interrupted-resume equality, tail weighting and production activation remain
unqualified, and the existing frozen cloud run remains untouched.
