---
name: flaxchat-training
description: Continue, audit and qualify existing YAT multilingual embedding training with FlaxChat on physical GCP TPUs, including representation quality, recovery, profiling and release evidence.
---

# Continue YAT representation training

Improve the existing trained YAT weights for multilingual text and code representations. Locate the active FlaxChat checkout and read the relevant section of `docs/REPRESENTATION_TRAINING_RUNBOOK.md`. Installed skill paths do not resolve repository-relative documents. Use the checkout's current source and durable receipts, not a historical launch script or remembered cloud state.

## Choose the requested gate

| Request | Read in the checkout | Main tools / acceptance |
|---|---|---|
| Audit / GitHub issues | Latest system audit, `docs/ISSUE_READINESS_CHECKLIST.md`, `docs/TRAINING_TOOLS.md` | Refresh every open issue; reconcile findings against existing acceptance; preserve original issue bodies and avoid duplicates. When parallel agents are requested, give training/data, infrastructure and evaluation/release separate ownership. |
| Prepare independent development data | `docs/REPRESENTATION_DEVELOPMENT_RECIPE.md`, `docs/INDEPENDENT_REPRESENTATION_DEVELOPMENT.md`, `docs/GCS_PARENT_EXPOSURE_SCAN.md` | Pinned actual rows, authenticated exposure, cross-source quarantine; metadata fixtures do not qualify production data. |
| Continue training | `docs/REPRESENTATION_TRAINING_RUNBOOK.md`, `docs/INFRASTRUCTURE_RUNS.md` | Shared stage preflight, frozen manifest, bounded physical single-host qualification, declared quality decision. |
| Scale / recover | `docs/EMBEDDING_MULTIHOST_ADMISSION.md` | Physical topology, sustained global negatives, disjoint local rows, exact recovery and owned cleanup; one host cannot qualify multiple hosts. |
| Evaluate / compare / publish | `docs/EVALUATION_RELEASE_CONTRACT.md`, `docs/MATCHED_EMBEDDING_COMPARISON.md` | Complete pinned inventory and measured protocol, current physical parity, immutable remote artifact verification. |

For data-only executor work, read `docs/BOUNDED_DATA_JOBS.md` and use `scripts.bounded_data_job` after its Linux lifecycle qualification; it never provisions hardware. Keep existing temporary executions on their original handles. Observation requires the independently retained canonical spec SHA via `--expected-spec-sha256` (CLI) or the explicit API argument; remote self-consistency alone cannot identify the intended job. Changed controller source needs fresh lifecycle qualification; earlier Linux passes remain bound to their recorded freeze. Explicitly authorize a new Cloud Shell session and verify actual GCS read access before dispatch; an ACTIVE account listing alone is insufficient. For measured scalar uncertainty read `docs/EMBEDDING_UNCERTAINTY.md`; paired intervals require identical authenticated queries and scoring protocol, and are reporting-only. For global useful-exposure definitions and collective qualification read `docs/EMBEDDING_TELEMETRY.md`.

For a full audit, read [audit-and-tools.md](references/audit-and-tools.md). It defines issue reconciliation, reviewer ownership and tool acceptance. For a failed artifact copy, read `docs/ARTIFACT_TRANSFER_DIAGNOSTIC.md`: measure bounded received-byte/time progress before retrying, and retain independent full SHA authentication. A range hash never authenticates the entire parent.

For fragile procedures and diagnosis, read only the relevant sections of [qualification-procedures.md](references/qualification-procedures.md): bounded execution; independent development and numerical diagnosis; production quality; parent materialization; collision filtering; matched comparisons. Keep dated run outcomes in repository receipts rather than appending each attempt to this entrypoint.

## Invariants

- Preserve the trained parent and the user's YAT choices: fixed bias **1**, fixed epsilon **0.01**, trainable alpha. Do not initialize randomly or change precision based on CPU measurements.
- Exact resume restores the same weights, optimizer, schedule, seed, sampler cursor and immutable stage identity. A changed recipe/objective/incompatible schedule is a **new stage**, initialized from prior weights, with a new output prefix and explicit optimizer policy.
- Model forward/backward, numerical, performance and topology acceptance run on **physical TPU**. Local metadata, manifests, byte integrity, docs and orchestration checks are permitted. A skip, tiny synthetic fixture or old source freeze is not current production acceptance.
- Keep implemented, model-free verified, physical passed/failed/unrun, and measured quality states separate. Retain original failed assertions and tolerances. A failed cache gate blocks cache-enabled training; a separately qualified direct path can proceed without claiming cache acceptance.
- Training loss does not select a production representation model. Require authenticated retrieval, bitext, code and STS development coverage, parent baselines, per-language/task regression bounds, and independent best/recovery retention. Operational row minima are not statistical confidence. Final benchmark tests must stay out of mixture selection. Current bounded retrieval probes rank selected positive passages; name that scope explicitly and require separately pinned full-corpus/mixed-language evaluation for product retrieval claims.
- Apply gates to the attempted scope: multi-host before scaling; release parity before publication. Neither blocks a bounded single-host training-only qualification that does not use those paths.

## One concrete execution, not repeated loops

Maintain one next gate with frozen input/source/runtime identity, required checks, finite deadline, maximum exposure, result and next action. Run deterministic admission before allocation. Give one campaign owner the shared ledger; carry prior reservations across retries. Do not assume a local ledger coordinates independent controllers.

Before allocation, verify the current account/project, resource ownership, quota eligibility, regional pricing, credit observation, cleanup principal's exact zone/prefix permissions and independent cleanup deadline. Quota eligibility is not physical capacity. Reserve cleanup and ancillary exposure as well as the workload. Preserve existing authorization; this skill does not authorize unrelated spend, account/token transfer, publication or deletion.

Capacity, startup, setup and workload waits must fit the original global lease. Use separate capacity/startup budgets. A deterministic failure requires a changed implementation/admission before retry. A transport error requires observation of the **same execution**: inspect durable worker status, ownership identity and live processes before reattaching or restarting. Successful SSH exit is not proof of worker success; SSH timeout is not proof of worker termination.

Report current training state and last durable checkpoint honestly. Verify exact owned queue/node/process/scratch cleanup independently; never turn scoped absence into an all-cloud-clean or empty-storage claim. Retain evidence and protected weights. Storage versions and soft-deleted objects need separate inventory.

## Data and parent authentication

Authenticate all required loaded arrays, original raw rows, negative-valid flags, tokenizer, full committed checkpoint manifest and lineage. `checkpoint_metadata.read_committed_metadata` authenticates recorded JSON identity without restoring tensors; actual parent restoration needs `scripts.validate_production_parent_tpu`.

Use `scripts.prepare_enriched_parent_export` near the data with an outer timeout. It streams Safetensors headers/leaf byte hashes without model execution. File-stability checks must ignore read-induced access time while retaining device, inode, size, modification/change time and symlink checks. A successful public-weight download or metadata test is not enriched materialization or TPU import. The validator admits tokenizer.json up to32MiB while other JSON artifacts remain8MiB; retain bounded reads and artifact SHA rather than raising all JSON limits for a large tokenizer.

Prepare development exclusions before future training sources; pass `--development-exclusions` to every preparation. Preserve code identity and programming-language provenance separately from human language. Missing labels remain unknown. MIRACL aligned groups span languages. A policy change requires new prepared data and a new stage.

Use complete authenticated scan/collision detail receipts with `scripts.filter_representation_development` to create a **separate** candidate; preserve the original and replayable proof. Recheck retained rows against actual authenticated historical bytes. Exact zero overlap only establishes the declared history/text/group scope, not undisclosed MLM, semantic or translation cleanliness or a complete four-domain quality plan.

## Numerical, performance and release decisions

When cache Adam states fail, inspect named gradient/moment/model leaves, shape-matched ordinary autodiff and each moment's own gradient. Keep original gates. Gradient accumulation does not enlarge the simultaneous contrastive negative pool. Current cache/FSDP availability must be read from the implementation, not inferred from other libraries.

After correctness passes, collect short warmed JAX/XProf traces and measure useful tokens/pairs, input time, compute, collectives, optimizer, checkpoint overhead, HBM and host RSS. Aggregate multi-host metrics before reporting whole-slice throughput. Separate warmed unprofiled steps from end-to-end useful-exposure rates that include compile, evaluation and checkpoint overhead. Adopt an optimization for better full-step cost/quality, not a CPU kernel timing.

Pin competitor revisions, dataset revisions, task/split/subset rows, prompts, pooling, truncation, normalization, score scale and aggregation. `scripts.compare_embedding_receipts` validates measured receipts; it does not implement a competitor evaluator. Paper numbers remain context.

Use separate sequential JAX and Torch/XLA environments for parity. Current publication requires complete v4 row/shape/backend scopes, bounded host RAM/scratch/evidence and retained raw numerical replay bound to actual weights/config/tokenizer/loader/input/runtime. Old eight-input parity does not qualify a new release.

Separate **reservations**, provider **posted charges before promotional credits**, applied credits and net due. A credit-balance change or month-wide project cost is not per-model spend. If export is disabled, say so; do not imply enabling it reconstructs all earlier runs.

## Handoff

Deliver the issue inventory, concrete findings with acceptance evidence, changes, validation scope, current training/checkpoint state, cost attribution limits, independently observed cleanup and one next gate. Update the runbook/tool inventory when behavior changes. Do not close issues on documentation or metadata checks alone. No expensive workflow dispatch is implied by an audit.
