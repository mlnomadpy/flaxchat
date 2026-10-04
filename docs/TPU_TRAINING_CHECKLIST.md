# TPU training checklist

> Historical execution log through September 26, 2026. Session handles, budget
> observations, and statements that jobs are active below are dated observations;
> do not use them to infer current cloud state or resume an old controller.
> The current execution policy is in the
> [representation training runbook](REPRESENTATION_TRAINING_RUNBOOK.md) and
> [September 30 audit](SYSTEM_AUDIT_2026-09-30.md).

Updated September 26, 2026. This is the execution queue for the current task.
Immediate milestone: launch and evaluate a bounded real-data run of the 307M YAT multilingual encoder. Broader optimization, scaling and model-quality goals remain open; they must not create an endless prerequisite loop before the pilot.

## Current evidence

- [x] Physical four-device v5e foundation checks: exact initialization, distinct batch rows, retained state layouts and empty-mask no-update behavior for FSDP factors 1/2/4. Evidence: `artifacts/encoder-fsdp-foundation-tpu-v5e4-0925/RESULTS.md`.
- [x] Implement opt-in FSDP in the real encoder trainer, with direct sharded initialization, batch placement, checkpoint identity and layout checks. Replication remains the default; unsupported local accumulation/kernel and partial-JIT combinations are rejected.
- [x] Run the 307,786,284-parameter model on four physical v5e devices: replicated baseline, uninterrupted FSDP, interrupted FSDP and resumed FSDP. The final report records finite updates, retained layouts and exact model/optimizer/training-state/identity parity between uninterrupted and resumed FSDP. Evidence: `artifacts/encoder-fsdp-trainer-tpu-v5e4-0925/final-report.json`.
- [x] Capture paired physical TPU JAX profiles. Baseline summary is saved locally; candidate summary and raw profiles are in the run's GCS wrapper. These cover two profiled steps, not sustained throughput.
- [x] Stop the obsolete direct-profile-download retry loop. Preserve any partial file as incomplete, not evidence.
- [x] Independently verified saved manifests and frozen source hashes; analysis passed. Controller exited 0 and both resources independently returned NOT_FOUND. See `artifacts/encoder-fsdp-trainer-tpu-v5e4-0925/RESULTS.md`.

The latest run used synthetic data and random initialization. It does not establish pretrained YAT conversion quality, numerical equivalence to replicated training, sustained real-data behavior, or leaderboard competitiveness. Baseline/FSDP model and optimizer hashes differ. That is a finding to quantify, not a reason to call either model qualified.
## Ordered launch checklist

Work on the first unfinished item. Each item has a finite output and pass condition.

| ID | Status | Work | Done when |
|---|---|---|---|
| L1 | Complete | Close the existing validation run | Download final manifests, logs and compact profile summaries once; verify frozen source identity and resume parity; report aggregate warm throughput, memory, checkpoint time and loss differences. Controller is terminal and independent VM plus queue queries both return NOT_FOUND. No new allocation before this. |
| L2 | Audit ready; freeze pending | Freeze the pilot specification | Record the exact model/config/source revision, pretrained-versus-scratch initialization, tokenizer, sequence length, global batch, accumulation, sharding, precision, optimizer, seed, checkpoint interval and stopping horizon. Preserve fixed YAT bias 1, epsilon 0.01 and trainable alpha. Resolve conflicting older recipe documents against the latest agreed plan. No silent architectural or precision change. |
| L3 | Local hashes verified; corpus uploaded | Verify actual training inputs | Verify the pinned pretrained snapshot if used, dataset/source revisions and licenses, prepared-file hashes, tokenizer identity, train/validation contamination checks, realized language/source token shares and repetition exposure. Identify explicitly whether the staged corpus implements the agreed mixture or is only a smaller pilot. Produce one input receipt; do not substitute synthetic data. |
| L4 | Replicated recipe qualified for bounded first stage; FSDP4 remains failed | Close the remaining targeted correctness checks | R4 passed 3 physical TPU import tests and both loss comparisons, but 34/181 and 53/181 gradient leaves failed the preset per-element bound on the two real batches. The separate replicated stage completed real-data finite updates, direct GCS checkpointing and TPU resume. Do not use FSDP4 until diagnosed and requalified. |
| L5 | Complete for first model; refresh for next stage | Confirm the budget and launch controls | Chrome shows $770.46 of the expiring credit on Sep 26 after the first model; billing posts with a delay. Both bounded allocations had independent cleanup and separate safety caps. Re-admit the next allocation with fresh exposure and a hard lease. |
| L6 | Complete | Launch the real-data pilot | The replicated four-chip v5e model completed 3,910 finite updates and 50,003,785 useful tokens with exact step16→256→320→3910 resume. Final checkpoint3910 has a GCS commit marker. Both controller attempts exited0 and VM plus queue independently returned NOT_FOUND. |
| L7 | Complete for engineering gate; model quality open | Evaluate the pilot and choose the next training stage | Fixed 432-row/27-language heldout YAT NLL improved 17.5868→2.7771; conventional mmBERT initial was1.0122. The preset continuation gate passed, but production quality/parity did not. Evidence hashes, all3,910 ordered updates and cleanup passed `artifacts/encoder-first-model-continuation-v5e4-0926/analysis.json`. |
| L8 | Segment 1 active under two-hour guard | Start the longer training stage | User requested 300M more useful-token exposures from the existing 27-language corpus. The new stage imports model weights from parent3910 with fresh optimizer/schedule, seed18, 23,467 fixed steps, checkpoint interval1024 and retention4. Independent two-hour cleanup limits this to segments1–8192, 8193–16384 and 16385–23467; only segment1 is allocated. Its step1024 and step8192 quality gates must pass before later allocations. |

L2–L5 may be prepared while waiting for L1, but cannot launch a second TPU allocation. Quality improvement is an outcome of L6–L7, not a circular requirement to prove before the first real-data pilot.

## Remaining work after the pilot starts

These remain part of the broader goal, but are not automatic pilot blockers.

- [ ] Measure sustained throughput and checkpoint overhead at the actual training sequence lengths and batch sizes; distinguish compiler buffer estimates, allocator peaks and total HBM.
- [ ] Use paired JAX profiles to address the dominant measured bottleneck. Current candidates include vocabulary projection/embedding communication, host dispatch and checkpoint transfers. Change one variable per experiment and include the real training step.
- [ ] Investigate fused/Flash-style YAT attention with correct global softmax statistics, centered derivatives, masks and alpha gradients. Require physical TPU forward/backward edge cases, numerical bounds, memory and full-step improvement before adoption.
- [ ] Improve MLM projection only when the measured loss/gradient behavior and end-to-end cost justify it. Do not promote the prior capacity/tile/compensation experiments solely on isolated speed.
- [ ] Integrate FSDP with partial JIT and local accumulation/kernels only if profiling justifies the complexity; qualify their combined layouts and resume behavior.
- [ ] Validate checkpoint import/restore and cursor replay across topology changes and GCS, including controlled interruption. Current full-model exact-resume evidence is one host and local scratch.
- [ ] Complete the requested physical scaling coverage: v6e, sustained multi-host training, and the intended 4/8/16/32-chip configurations. Assess v4 availability if still useful. Publish a matrix with actual pass/fail/unavailable/not-run evidence; extrapolation is not validation.
- [ ] Complete held-out multilingual MLM and downstream quality evaluation. Add the agreed embedding supervision, hard-negative/distillation recipe and retrieval evaluations before making embedding leaderboard claims.
- [ ] Reconcile documentation/configuration, review the final diff, run relevant TPU regressions, and commit/merge/push the validated work under the user's existing authorization. Preserve unrelated worktree changes.
- [ ] Audit actual final spend and cloud resources. Keep intentionally retained datasets/checkpoints under an explicit storage/retention plan; report them instead of claiming storage is empty.

## Rules that prevent loops

1. Every work turn must produce a code change, an evidence-backed decision, a completed checklist item, or a verified wait on one specific live handle. A rewritten plan or repeated status is not progress.
2. Keep one active TPU workload. Never restart because SSH, a poll, or a local observation times out. Verify authoritative terminal state and cleanup before another allocation.
3. Poll the existing controller at meaningful phase boundaries; use bounded waits and backoff. Do not alternate several near-identical status commands. A long compile is not a new investigation.
4. Do not rerun a passing check without a relevant source/configuration/topology change or an identified evidence gap. Link the existing evidence.
5. Before an experiment, write its hypothesis, exact change, acceptance criteria, maximum runtime/cost and the decision its result will enable. If it cannot change the next action, do not run it.
6. After a failed experiment, diagnose the concrete failure. Retry only with a relevant fix; keep the failed evidence. Do not lower tolerances or narrow the workload merely to get a pass.
7. Separate required correctness/stability fixes from optional speed work. Once L1–L5 pass, launch L6; do not insert another optimization campaign as a new prerequisite.
8. Run model computations and benchmarks on physical TPUs. Local source inspection, artifact preparation and parsing of saved TPU evidence are allowed; CPU model results do not qualify a change.
9. Record exact remaining work and handles here and in the continuation record. Do not add competing parallel to-do lists.
10. Report actual findings and scope. Do not call synthetic tests real training, a short run sustained validation, lower-level speed end-to-end savings, or a reservation actual spend.

## Current handles

- First-model continuation controller `48509` exited0; resource `flaxchat-validation-yat-mmbert-resume-0-4f0129ae5b7945eb` in `us-west4-a` and its queue both independently returned NOT_FOUND. Final checkpoint3910 and local compact evidence are verified. No TPU is running for this stage.
- Second-stage segment1 controller session `40383` owns `flaxchat-validation-yat-mmbert-s2s1-0-80b74bd013c94156` in `us-west4-a`; guard execution `24713d6f-a3be-4115-a5b1-b14b52cc733a` was verified before provisioning. Launch root `artifacts/encoder-second-stage-r2-v5e4-0926/segment-1`. Physical model training has begun. Do not submit segment2 until segment1 controller exits and VM plus queue independently return NOT_FOUND.
- First-model intermediate GCS checkpoints were pruned by an exact 62-prefix allowlist. Only the complete 3910 checkpoint remains live under its prefix (3,447,186,067 bytes), down from 217,540,940,251 live bytes. The data and compact evidence remain. Bucket-wide lifecycle deletes after30days; soft-deleted bytes may bill for7days.
- A first second-stage attempt with a proposed five-hour guard failed before TPU provisioning. Auto-review rejected modifying the shared cleanup workflow to allow that lease. The local workflow/controller limits were reverted to7200seconds; segment1 uses that unchanged guard. Failed r1 VM and queue independently returned NOT_FOUND, and its unused uploaded overlay was removed.
- The earlier first-stage controller exited 0. Its VM and queue both independently returned NOT_FOUND. Its checkpoints remain under `gs://tpubuilders-flaxchat-validation-0921/encoder-first-model-replicated-v5e4-0926/checkpoints/`.
- R4 evidence: `artifacts/encoder-real-calibration-r4-tpu-v5e4-0926/gradient-analysis.json`, physical compact report under `tpu-evidence/evidence/report.json`, and `cleanup-verification.json`. Do not rerun R4 unchanged or loosen the gate.
- The explicit replicated first stage completed; evidence and limitations are in `artifacts/encoder-first-model-replicated-v5e4-0926/RESULTS.md`. Its intermediate policy allows only a newly budgeted resume of the existing 3,910-step schedule. FSDP investigation remains open separately.
- Refreshed Chrome credit receipt: `artifacts/encoder-pilot-readiness-0925/credits-observation-0926.json`.

## Active swarm ownership

The user requested a swarm. `pilot_config` owns the L2 configuration audit, `pilot_data` owns the L3 input audit, and `fsdp_results` independently reviews the completed TPU evidence. The parent integrates reports, updates this checklist, checks billing and exclusively controls cloud launches. Reports are under `artifacts/encoder-pilot-readiness-0925/`. No agent runs CPU model tests or independently allocates TPUs.

Swarm findings: `config-audit.md` identifies the stale specialized wrapper and recommends an explicit recipe using the generic trainer; `fsdp-review.md` independently confirms resume and the measured throughput/checkpoint tradeoff. Chrome showed $774.86 remaining in the September 28 credit; see `credits-observation.json`. Data audit is verifying the newer 358,973,214-token, 27-language, sequence-512 corpus. These audit findings do not themselves freeze or launch the pilot.

Swarm implementation assignments: `pilot_data` packages only the verified prepared scale inputs with a hash inventory; `pilot_config` writes explicit calibration/pilot config artifacts and useful-token horizon planning; `fsdp_results` implements the bounded TPU-only calibration worker. The parent reviews these, stages one frozen payload and owns the single guarded cloud launch. No parallel TPU allocation is authorized to an agent.

Latest swarm review: all three calibration source-review findings are addressed (finite applied updates, matched input replay, and actual batch coverage). `calibration/spec.json` pins the draft, weights, data manifest and selected rows. Corpus upload completed; remote generation `1790374397110607`, size 652,676,046 bytes. TPU-side archive SHA256 verification remains required. No calibration allocation has started. Next: freeze source and compact-evidence wrapper, admit one bounded calibration launch, then run L4.

Active calibration: controller `69329`, resource `flaxchat-validation-yat-mmbert-realcal-0-6fafae6e2af942ca`, independent cleanup execution `fe241eaf-3836-4d92-8239-f452620f23fd`. Frozen source SHA256 `921049588d494f57426f7fdefbf5afe09f593748c1c7b231ac0e726f03170467`; run root `artifacts/encoder-real-calibration-tpu-v5e4-0925`. One attempt, 2400-second lease, up to 1800 seconds model work, $6 safety reservation (not actual spend). Poll this controller; never duplicate it on observation timeout. Final wrapper evidence goes to its matching GCS prefix `/wrapper`; checkpoint outputs under `/checkpoints` are intentionally retained. Next allocation requires terminal controller and independent VM/queue absence.

Calibration attempt failed before tests: eager checkpoint-analysis import acquired the TPU in its supervisor. Failure logs are preserved. Imports are now lazy and supervisor has a backend-import guard. Controller `69329` is still being observed for cleanup; do not duplicate or relaunch until terminal plus independent absence. See run-root `FAILURE.md`.

First real-calibration controller69329 is terminal exit1. Independent cleanup now confirms VM and queue NOT_FOUND; fresh r2 inventories are empty. Corrected source is frozen at SHA256 `0f817c6b4ab169f422930ea49e4989827f8db19c8f75ea6b923799ca7e3911e1` under `encoder-real-calibration-r2-tpu-v5e4-0925`. Retry fixes only supervisor backend ownership and missing locked pytest; no tolerance/scope changes. Six-dollar safety admission passed; actual billing remains unobserved for this attempt.

Corrected calibration submitted: active controller `22677`, run root `artifacts/encoder-real-calibration-r2-tpu-v5e4-0925`. Poll this exact handle; no additional allocation until terminal and independent cleanup.

Pilot quality preparation: frozen `quality/selection.json` SHA256 `492ba0546004113aeb270dc29bdcb2990a03e3fdaee930d479279c01014fead6` covers432 rows/27 languages/16 rows each/24,785 masked targets, minimum694 per language. Data-only coverage passes; no model quality result yet. Three-mode TPU evaluator and independent frozen-training identity checks are implemented, with an evidence comparator being completed. Optional vocab-parallel exact-loss prototype and TPU-only tests exist under `artifacts/vocab-parallel-prototype-0925`, unqualified and not integrated into training.

R2 physical result:3/3 pretrained import/layout/atomic-rejection tests passed on the four-device TPU for FSDP factors1/2/4; no skips (3.10s). Read-only live report lists tests completed; gradient/full-model and durable-recovery gates remain pending. Controller22677 remains active. Step256 quality comparator now distinguishes resume of admitted pilot from expansion beyond it.

R2 gradient phase stopped after both factor1 cases at a diagnostic NNX optimizer in_shardings prefix error (Param-wrapped sharding). All137 shared imported tensors verified and181 gradient leaves recorded percase; factor4 comparison and durable training not reached. Fix is confined to diagnostic gradient prefixes. Preserve3passed import tests; controller22677 cleanup remains active.
