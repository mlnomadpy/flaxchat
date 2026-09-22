# Follow-up audit fixes and qualification

This follows [audit A1–A14](HARNESS_AUDIT_FOLLOWUP_2026-09-21.md). Kaggle remains excluded. Implementation changes are not evidence of convergence, pod-scale performance, or paper reproduction.

## Correctness and release gates

- **A1:** Unlimited evaluation uses the full task; finite limits are clamped. Empty tasks and unknown names fail. Incomplete required CORE evaluation returns a nonzero exit code.
- **A2:** Multiple-choice scoring rejects nonfinite likelihoods and empty continuation spans. It no longer uses `nanmean` to hide invalid tokens.
- **A3:** Updates commit only if both the candidate model and optimizer state are finite. Rejected updates restore all previous state, including optimizer counters. Both pretraining paths stop instead of consuming more data after a rejected update. Regression tests cover finite gradients overflowing Adam moments and SGD parameters.
- **A4:** Evaluation, SFT, RL and chat reconstruct the saved architecture, verify tokenizer identity, and restore the concrete metadata step. Restore rejects same-shaped architecture mismatches. Explicit checkpoint/tokenizer paths allow prepared training artifacts to be consumed. The loader accepts both strong serialized-tokenizer identities and verified persisted-file hashes; old metadata that omits the step is bound to the actual selected checkpoint directory after integrity verification. Weak vocabulary-only identities fail and need explicit migration. A real prepared-pool train → evaluation → SFT test passes. Evaluation uses the tokenizer's actual BOS ID, including GPT-2's EOS/BOS convention.
- **A5:** Model compute dtype is explicit, including embeddings, projections, rotary constants and inference caches. Prepared training stores effective dtype and runtime package/XLA identity; exact resume rejects an environment mismatch. FP32 master parameters are retained.
- **A6:** Forced-full CI now enables distributed checks, end-to-end tests, build and dependency audit. Common/training/model changes trigger distributed checks. The workflow explicitly runs real two-process CPU, eight-device checkpoint/update tests and BF16 regressions. Infrastructure changes select the relevant guard/runner tests.

## Data, recovery and cost controls

- **A7:** Version-2 token pools stream into bounded immutable shards and publish through a staging-directory rename. Each shard has a checksum, count and range validation. Runtime verifies content when a shard is first accessed, reads across boundaries, and retains deterministic global batch coordinates. Version-1 pools remain readable. Preparing the new qualification pool used pinned dataset/tokenizer revisions and saved the tokenizer artifact. Billion-token RSS and remote-shard caching are not yet benchmarked.
- **A8:** A subprocess is killed during an unfinished Orbax save, and recovery verifies the previous committed checkpoint. This filesystem/Orbax fault test deliberately uses a CPU child even inside the TPU test suite. Serialization/finalization/retention failure coverage and large-model profiling remain incomplete. Integrity hashing stays synchronous and bounded; no unmeasured asynchronous rewrite was enabled.
- **A9:** `scripts.gcp_spot_supervisor` reserves the entire attempt/deletion allowance before allocation, requires a server-verified cloud expiry guard, and verifies deletion before another attempt. Confirmed preemption may retry only with explicit resume arguments. Setup failures, failures on live hardware, exhausted budgets, existing resource names and failed cleanup stop the run. Each attempt is bounded; this does not implement indefinite lease renewal or unattended multi-day training.
- **A10:** A locked, atomically written run ledger persists reservations, resource identity, lifecycle timestamps, gross posted usage and promotional credits separately. Checkpoint-duration events are appended immediately after a committed save. Posted charges remain unknown until billing evidence is available. A reserve is not an actual charge or a guarantee against cloud deletion failure.
- **A11:** Scale plans include 110-update 1K and 4K context profiles. The 1K profile completed physically; 4K and larger pod shapes remain unvalidated. A profile's existence does not validate that shape.
- **A12:** Unsupported tensor parallelism remains rejected and multi-device attention retains its explicit XLA fallback. New attention kernels, stacked scanning and alternative FSDP layouts remain profiling-led work; they are not silently enabled or claimed faster.
- **A13:** TPU setup now applies a full version-constraint file derived from the recorded Linux environment, and workers agree on runtime identity. Archives carry a checksum and base commit. Python provenance now also covers evaluation tasks and infrastructure Python. This is stronger than four pinned packages, but is not a hash-locked immutable OS image. Source changes remain in the working tree until published.
- **A14:** A new `scripts.validate_training_quality` command requires finite loss, the full committed horizon, improvement, and loss below both the uniform baseline and an optional stricter predeclared threshold. It fails on this campaign's result. This is a basic loss sanity gate, not a quality certification. Corpus deduplication/decontamination, held-out SFT quality gates and distributed post-training remain separate work. The train/eval/SFT handoff test uses a chat-capable ByteTokenizer; an unmodified GPT-2 tokenizer does not supply the harness's chat role tokens.

## Local verification

- **530 tests passed** on final local source; five skipped and four accelerator tests deselected. Coverage was **78.37%**; every required per-module floor passed.
- 94 initial tests passed with eight virtual CPU devices; 51 artifact/checkpoint/handoff tests passed on final source with eight devices.
- 66 BF16 regression tests passed on final source.
- Two real CPU-process distributed tests passed.
- Ruff and Pyright passed; isolated wheel/sdist build and dependency audit passed. An installed-wheel smoke, run outside the checkout, imported the installed package and executed a finite model forward pass.
- TinyStories training, checkpoint, tokenizer packaging and inference smoke passed.

These suites overlap; do not sum their counts as unique tests. Raw logs, JUnit and coverage are under `artifacts/audit-fixes/`.

The artifact-loader follow-up passed the full CPU suite and targeted eight-device/BF16 suites, including published-artifact compatibility and the new train/eval/SFT handoff. The standalone quality gate and its CI routing were added afterward and passed five new regressions plus the final full suite. The gate is also packaged as `python -m flaxchat.training_quality` and its installed-wheel invocation correctly rejects the physical result. Final local Python SHA-256: `2536925ff4c3e22c9270d2d908c6fc5e3416f6f8b4b906d7fabd469800af551c`. The physical suite tested the earlier frozen source below; the follow-up loader and quality-gate changes have CPU evidence, not a new physical acceptance claim.

The synthetic writer memory check increased input from 1M to 32M tokens while peak process RSS increased from 163.4 MB to 175.5 MB. Manifest-open latency was 0.19/0.75 ms and first-shard verification/read 7.43/8.34 ms. These are macOS, cached-filesystem microbenchmarks, not cloud throughput or billion-token qualification.

## Physical qualification

Completed on one Spot v5p-16 slice in `tpubuilders`, two hosts/eight chips. Both queue and TPU VM were deleted, and independent final listings are empty.

- The full single-host suite passed **526 tests, four skips, in 255.69 seconds**; [every test is listed here](HARNESS_AUDIT_TPU_TESTS_2026-09-22.md).
- **All nine multi-host recovery stages passed.** Model, optimizer and training-cursor hashes match exactly for DP coordinator-kill recovery, FSDP stop/resume, and the CPU/BF16 bridge back to two TPU hosts.
- A 12-layer GPT-style model completed **110 updates / 901,120 tokens**, including 100 updates after warmup, at context 1,024 with global batch 8, FSDP=8, BF16, rematerialization and chunked loss. This is not an exact GPT-2 architecture or paper reproduction.
- Validation loss improved from **23.7768 to 13.5806**, but remains worse than uniform-token loss **10.8249**. The retrospectively applied loss sanity gate fails. The run is numerical/recovery evidence, not a useful-model quality pass.

The allocation had a 2,200-second cloud expiry and a $39.33 conservative reservation within a $40 cap, including 30 minutes of deletion allowance and $2 ancillary reserve. Creation to verified absence was 1,198.56 seconds: **$11.19 at the conservative on-demand compute rate**. This is not the actual Spot bill and excludes ancillary charges. Gross posted usage and promotional credits remain null. A read-only BigQuery dataset listing returned no visible datasets in `tpubuilders`; exports elsewhere are not ruled out.

Frozen Python SHA-256: `5f14eae3afaa8a029dfe2118a87a4f599500dff4064a9a2ed786ce48c9871609`.

Archive SHA-256: `8cdd194e09e98e4ffc8b6e0fe0c0cecb4a5b6275c844be5acbe6b1878d2e8d35`.

The conservative rate is eight chips at the published $4.20/chip-hour on-demand price, rather than assuming a particular Spot discount. [Google Cloud TPU pricing](https://cloud.google.com/tpu/pricing). Actual posted Spot charges must be reconciled separately.

## Measured cost bottleneck

The training invocation took **192.62 seconds**. First-step compile/execution took **40.81 seconds**, and its three checkpoint saves took **41.02, 40.81 and 40.10 seconds**—**121.93 seconds total, 63.3% of invocation time**. The 100 post-warmup updates had a median step time of **41.71 ms**. Steady step throughput was 173,662 tokens/s, while invocation throughput including startup, evaluation and checkpoints was only 4,678 tokens/s. These figures describe this small workload; neither is a full-run cost prediction.

Allocator-reported peak usage was approximately **557 MB per chip on worker zero**, versus a roughly 102.8 GB reported limit. These are allocator statistics, not a complete compiler/HBM trace. This shape gives no evidence that eight chips are necessary for capacity.

Prioritize a larger checkpoint interval, checkpoint/hash profiling, and a smaller-slice or larger-batch comparison before custom attention kernels. The frequent saves here were deliberate recovery/measurement work. Keep a preemption/replay bound when increasing the interval; do not remove checkpoint integrity to improve a throughput number.

A CPU initialization-only diagnostic on 32 held-out tokens also points to the tied embedding/output scale as a recipe candidate: reducing the embedding initialization scale from 0.8 to 0.02 reduced loss from 21.29 to 11.87 in a one-block, width-768 model. This is not the 12-layer learning experiment, still exceeds uniform loss, and does not justify silently changing initialization defaults. Compare tied/untied initialization and learning curves under a fixed quality target next.

Full source identities, every TPU test, lifecycle ledger, recovery comparisons and all 110 training rows are in the [machine-readable report](../benchmarks/results/harness-audit-fixes-20260922.json). The quality gate can be repeated with:

```bash
python -m scripts.validate_training_quality \
  --summary artifacts/audit-fixes/training_summary.json \
  --output artifacts/audit-fixes/quality-gate.json
```

Exit code 1 is the expected result for this campaign. Select stricter experiment-specific targets and a fixed held-out corpus before future comparisons.

## Remaining acceptance work

Before a substantial training spend: fix the recipe's loss sanity failure through a controlled initialization/learning-curve comparison; reduce the measured checkpoint overhead; reconcile posted billing; and establish decontaminated held-out evaluation. Multi-day replacement, 4K context, additional host counts, immutable image/source publication, and distributed post-training remain open acceptance work. Paper reproduction still needs a faithful architecture/optimizer/data/schedule/evaluation port; these harness fixes do not supply it.
