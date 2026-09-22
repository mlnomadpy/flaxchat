# Harness audit after physical TPU validation

Audited September 21, 2026 (Pacific), September 22 UTC. Scope: current working tree, training/update paths, data preparation, checkpoint/recovery, evaluation and post-training, GCP operations, CI, and recorded physical evidence. Kaggle execution is excluded. This is an audit, not a claim of exhaustive verification or a new physical acceptance run. No cloud compute was allocated and no implementation was changed.

Python source SHA-256: `14c688f0d5077d960d7c50620eab05bd08dc264660e0cd78f7cd5f728f4a4148`, matching the accepted physical campaign. Base commit: `c311b63a7ff035e7539155ae0955b60d8c9de29e`. The base commit alone does **not** identify the tested changes; use the Python digest and archived campaign artifacts. The working tree contains substantial unpublished changes.

## Verdict

The harness has credible small-scale training and exact distributed recovery evidence. It is suitable for bounded development experiments, but it still needs correctness fixes and production qualification before a substantial training spend or a paper-reproduction claim. Passing unit tests does not demonstrate convergence, sustained scaling, benchmark equivalence, or cost efficiency.

Existing evidence remains valid: 503 tests passed and four intentionally skipped on one TPU host/four chips; nine recovery stages passed on two hosts/eight chips; a 12-layer, sequence-128 GPT smoke completed four updates with FSDP=8. These are previously recorded results, not rerun in this audit. See [accepted evidence](HARNESS_TPU_TEST_RESULTS_2026-09-21.md).

## Correctness findings

P1 means fix before relying on the affected training/evaluation path. P2 means an important reliability, efficiency, or reproducibility improvement. Reproduced means an executable local probe; static means the conclusion follows from the inspected code and was not reproduced on TPU.

### A1 — P1: evaluation defaults silently select no examples (reproduced)

`flaxchat/stages/eval.py:94`, `:115`, and `:132` pass `max_per_task=0` directly as `stop=0` to MMLU/GSM8K/ARC. The CLI describes zero as full evaluation, but `tasks/common.py:30` interprets it as an empty slice. The stage reports zero accuracy/zero examples and returns the default successful `StageResult`.

Probe: a Task with 100 examples has length zero with `stop=0`, versus 100 with `stop=None`. This uses the same inherited slicing implementation as those tasks. CORE has separate limit handling and is not affected by this empty-slice bug.

Fix: normalize unlimited limits to `None`, clamp finite limits to dataset size, reject empty evaluations, validate task names, and return a nonzero stage exit code for incomplete required tasks. Add offline CLI-to-stage tests for default, finite, excessive, and unknown-task inputs. CORE currently suppresses its aggregate on failure, but the outer stage still returns success (`stages/eval.py:152`).

### A2 — P1: invalid multiple-choice scores can become correct answers (reproduced)

`flaxchat/eval.py:65–83` uses `nanmean` and then selects the minimum without checking finiteness or a nonempty scored span. A model returning all-NaN logits produced `(True, {'prediction': 0, 'gold': 0, 'scores': [nan, nan]})` in a local probe. Partial NaNs can also be hidden by `nanmean`.

Fix: require valid continuation spans and finite losses throughout every scored span; fail the task instead of assigning a prediction. Add all-NaN, partial-NaN, infinity, identical-choice, and empty-continuation cases. Validate prompt/token boundaries and scores against a pinned external evaluation reference before publishing comparable benchmark numbers.

### A3 — P1: finite-input checks do not protect resulting optimizer state (reproduced)

`flaxchat/training.py:30–43` checks loss and gradients before `optimizer.update`, then unconditionally returns true. A finite FP32 gradient of `1e20` with ordinary AdamW at LR `1e-3` was accepted while the optimizer state became nonfinite. A separate SGD stress probe with finite LR/gradient produced an infinite parameter and was also accepted. These are adversarial probes, not observed failures of the prior TPU run. The prepared AdamW recipe clips gradients, so its exposure differs from the unclipped SFT/RL AdamW paths.

Fix: validate candidate updates and optimizer state before committing, or fail immediately with recovery from the last verified checkpoint. Decide and document whether atomic rollback is required, and profile the memory cost. Add overflow and already-corrupt-state regressions, not only NaN-input tests.

The legacy pretrainer also counts rejected updates but continues consuming data until its attempted-step horizon (`stages/pretrain.py:576–625`), with no rejection threshold; prepared training instead aborts. Unify the policy so persistent failures cannot exhaust a budget while producing almost no useful training.

### A4 — P1: checkpoint-to-evaluation/post-training identity is incomplete (static)

Evaluation reconstructs architecture from the model tag and defaults (`stages/eval.py:60–71`) instead of checkpoint metadata. Restore checks tensor schema but architecture identity is optional. Same-shaped semantic changes, such as an attention window change, can therefore escape a schema check. Other configurations fail unnecessarily because defaults differ from the saved model.

SFT and RL now load architecture metadata, but validate the initial tokenizer only by vocabulary size (`stages/sft.py:100–110`, `stages/rl.py:128–136`). Two tokenizers can have equal vocabulary sizes and different token-ID meanings. Their later resume identity does not fix an incorrect initial checkpoint/tokenizer pairing. Evaluation records a mutable directory path rather than a concrete checkpoint step/content identity.

Fix: one checkpoint-loading contract for eval, SFT, RL and inference: exact model config, concrete step, model digest, serialized tokenizer hash and artifact path. Reject equal-size/different-tokenizer inputs. Add a prepared-token-pool train → evaluate → SFT smoke using the saved tokenizer.

### A5 — P1 for configured precision/reproduction: declared precision can diverge from compute (static)

`TPUConfig.precision` is accepted and validated in `flaxchat/config.py`, but runtime dtype is selected at module import from `FLAXCHAT_DTYPE` or backend defaults (`common.py:65–94`). The inspected training paths do not connect the configuration field to this selection. A requested `f32` setting alone therefore does not establish FP32 execution on TPU.

The prepared trainer's identity (`scripts/train_gpt2.py:110–119`) omits effective compute dtype, JAX/XLA flags, and installed dependency versions. FP32 master parameter dtypes alone cannot detect a change in BF16 versus FP32 compute.

Fix: resolve effective precision before model/runtime initialization; reject conflicts; persist the effective compute/parameter/accumulation policy and environment identity. Test actual projection dtypes for each supported user configuration. Distinguish explicit cross-precision migration from exact continuation.

### A6 — P1 for release protection: CI skips relevant distributed and infrastructure checks (reproduced)

`select_scope([], force_full=True)` returns full unit tests but false for multi-device, end-to-end, build and dependency-audit jobs. Thus a manual full dispatch does not run the full acceptance set. Changes to `flaxchat/common.py` or `flaxchat/training.py` also omit multi-device jobs. Changes to `scripts/gcp_cleanup_guard.py` select only pipeline/stage tests, not the guard tests; changes to `infra/tpu/spot_watchdog.py` select only quality-policy tests.

The two-process tests are opt-in via `FLAXCHAT_RUN_DISTRIBUTED_CPU=1`; the Linux workflow does not set it. Its eight-device task primarily runs sharding tests and a topology subprocess, not the full validated checkpoint/accumulation matrix.

Fix: make forced-full unconditional, map safety-critical files to their tests, and add explicit two-process, eight-device checkpoint/accumulation and BF16 jobs. Keep paid physical acceptance a separately budgeted gate; CPU CI should be sufficient to catch these routing regressions.

## Scale, reliability, and cost gaps

| ID | Priority / evidence | Missing capability or qualification | Acceptance criterion |
|---|---|---|---|
| A7 | P1 before billion-token preparation; static | Preparation accumulates every token in Python lists, retains both splits, then converts to arrays (`scripts/prepare_token_pool.py:38–55`, `token_pool.py:19–30`). Runtime requires one complete local file per split and scans it for checksum and maximum ID on every worker/start (`token_pool.py:45–54`). Memory and restart I/O scale with total dataset size. | Incrementally write immutable shards with bounded RAM, atomic manifests, per-shard verification and deterministic global cursors. Measure preparation RSS and restart latency; prove next-token equality across sharding and resume. Preserve integrity rather than simply removing checksums. |
| A8 | P2; static/evidence gap | Checkpoint hashing is memory-bounded but synchronously reads the full model and optimizer before Orbax save. Sharded hashing issues repeated chunk operations. Async storage does not make that pre-save work asynchronous. The physical fault injection kills after a committed checkpoint, not mid-write. | Profile save/restore pause, host/HBM peaks, storage requests and restart latency at representative model size. Inject failures during serialization, finalization, and retention; verify last-good recovery or an explicit fail-closed outcome. |
| A9 | P1 before a long Spot run; static/evidence gap | The GCP runner is a bounded validation command runner (maximum 2,400 seconds), and the durable cleanup guard is a validation lease (maximum 7,200 seconds). Together they do not implement a sustained training supervisor with bounded reprovisioning, replay accounting, and total retry budget. | Define production lifecycle and lease renewal; test whole-slice loss/replacement, coordinator loss, failed deletion, stale receipts and exhausted retry budget. Require verified final resource absence and a durable recovery checkpoint. Do not merely lengthen timeouts. |
| A10 | P1 before cost claims; evidence gap | Actual Spot charges are unreconciled. Input hourly rates and short smoke throughput are insufficient to predict full-run cost. Per-step JSONL is written before checkpoint duration is added (`train_gpt2.py:222–234`), so checkpoint timing survives only in a successful final summary. | Per-allocation ledger with resource IDs, SKU/region, ready/deleted times, actual gross usage, credits separately, storage/network, failure/replay costs, and a budget reserve. Persist checkpoint timing events as they happen; measure dollars per committed billion tokens and per quality target. |
| A11 | P2; evidence gap | The accepted production-like FSDP smoke has four updates at context 128. It does not establish sustained 2K/4K context performance, HBM margin, convergence, or efficiency beyond two hosts/eight chips. | At least 100 post-warmup measured updates on a representative configuration, plus a bounded quality-learning run. Profile compile, input stalls, collectives, HBM, evaluation and checkpoints. Validate additional host counts only when a chosen workload needs them. |
| A12 | P2; static | Multi-device training deliberately falls back from Splash to XLA; tensor parallelism is not implemented. The scan path uses per-layer branches rather than a stacked homogeneous layer representation. Current FSDP placement is a dimensional heuristic, not demonstrated optimal communication placement. | Profile first. If attention/compile/collectives dominate, add shard-local attention or stacked scanning behind parity tests and benchmark DP/FSDP layouts. Require numerical equivalence and measured end-to-end savings before defaulting to them. |
| A13 | P2; static | Only four core TPU dependencies are pinned in the setup file; extra dependencies and installer resolution remain floating. A pip freeze is recorded, but it is not an immutable fully reproducible build. Validation changes remain unpublished in the working tree. | Locked tested runtime/image, host environment agreement, immutable code/config/data/tokenizer artifacts and an install-from-wheel smoke. Publish the accepted source and raw evidence together. |
| A14 | P2; evidence gap | Data has pinned source identity and disjoint source-document boundaries, but this alone does not establish deduplication, benchmark decontamination, mixture quality, or held-out generalization. SFT has no held-out quality gate; multi-host SFT/RL is explicitly unsupported. | Fixed held-out corpus/domain losses and downstream evaluation with contamination checks. Add SFT quality/regression tests before optional packing or preference methods. Extend distributed post-training only if it is actually needed. |

## Paper-reproduction readiness

The current `standard_gpt` is not a Qwen3/Puro implementation. Depth currently determines heads and width in the prepared trainer; the model has different normalization/MLP/logit behavior. The existing optimizer is a named NorMuon/AdamW variant, not MuonH. A faithful port requires configurable architecture, exact tokenizer, optimizer reference-update parity, data curriculum, schedule, checkpoint averaging, and matching evaluation. FP8 versus BF16 must be an explicit experimental variable.

Use the smaller paper experiment as a separate optional milestone after the correctness gates. Neither the historical TPU tests nor this audit establish the paper's quality or savings in this harness. Do not spend on additional optimizers or low-precision kernels before the baseline produces trustworthy quality and cost measurements.

## Validation performed in this audit

- Confirmed the current Python digest matches the accepted TPU source.
- Ran 70 targeted CPU tests successfully, one skipped, in 25.53 seconds: training safety, harness regressions, CI routing, token pools and fine-tuning recovery. The skip is a device-dependent case. These do not cover the new adversarial cases above.
- Executed independent probes for CI routing, finite-gradient optimizer overflow, zero-length task slicing, and NaN multiple-choice scoring. Findings above distinguish these from static observations.
- The initial `.venv` test command could not run because pytest was absent; the test suite was then run successfully with `.pixi/envs/default/bin/python`. No dependency changes were made.
- Reviewed the recorded physical unit/recovery reports. No new TPU validation, full CPU-suite rerun, external evaluation parity, full billing reconciliation, or sustained training was performed.
- Read current GitHub issue status. No issues were opened, closed, or edited.

## Recommended work order

1. Fix A1–A6 and add the missing adversarial tests. These are local correctness/CI work and require no TPU allocation.
2. Unify the production artifact contract and implement bounded sharded preparation (A4/A5/A7). Validate a complete train/eval/SFT handoff locally.
3. Establish production Spot lifecycle and durable cost records (A8–A10). Reconcile existing charges before deciding a new run cap.
4. Run one explicitly bounded representative TPU qualification with sustained profiling, fault injection and a held-out loss target (A11). Optimize only measured bottlenecks (A12).
5. Publish the tested source/environment/evidence (A13), then run a small quality comparison or the paper-specific port. Keep dataset/evaluation quality gates (A14) attached to every comparison.

Current open GitHub work: [#1 readiness](https://github.com/mlnomadpy/flaxchat/issues/1), [#12 physical acceptance publication](https://github.com/mlnomadpy/flaxchat/issues/12), [#23 matched benchmarks](https://github.com/mlnomadpy/flaxchat/issues/23), [#11 release](https://github.com/mlnomadpy/flaxchat/issues/11), [#19 checkpoint/demo](https://github.com/mlnomadpy/flaxchat/issues/19). [#31 Kaggle](https://github.com/mlnomadpy/flaxchat/issues/31) remains excluded. A1–A14 are audit identifiers, not newly filed GitHub issues.
