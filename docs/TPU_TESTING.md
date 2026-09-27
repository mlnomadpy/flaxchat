# TPU test-suite campaigns

Run `python -m scripts.validate_test_suite` only under the bounded Spot supervisor.
The runner asserts a single physical TPU host and an exact device count, then
executes every `tests/test_*.py` module in a separate process. This bounds the
lifetime of compiled programs and device buffers without changing test semantics.

Each module has a maximum 240-second timeout inside the overall deadline. A timed
out process group is killed; its timeout is a failure, never a passing skip.
JUnit reports preserve individual passes, failures, errors and skip reasons.
Missing, empty or entirely skipped JUnit reports cannot qualify a module. Skips
are reported even when other cases in that module pass. The campaign does not
claim that skipped tests executed.

Evidence uploads have a 60-second timeout and happen after every module. The
controller's exit status and the separate TPU cleanup receipt are required in
addition to the suite report. An expired campaign reports remaining modules as
`not_run`. A single-host campaign does not qualify physical multi-host behavior,
all TPU generations, sustained throughput or downstream quality.

## Test matrix

| Layer | Purpose | Execution |
| --- | --- | --- |
| CPU suite | Deterministic unit, failure-injection and orchestration contracts | Local and Linux CI |
| CPU multi-process | Actual process loss and exact checkpoint continuation | Opt-in distributed tests |
| Physical TPU suite | Device execution, kernels and TPU-specific regressions | This runner; isolated modules |
| Released encoder qualification | Released-weight parity, update/checkpoint/recovery and long contexts | `scripts.validate_encoder_tpu` |
| Physical multi-host | Worker agreement and collective recovery | Multi-host orchestrator and aggregate evidence |
| Quality evaluation | Held-out multilingual and downstream acceptance | Separate from execution/performance gates |

The suite uses FP32 as the generic harness compute default and highest matmul
precision, except the BF16 GPT Splash module, which requires native/default
matmul precision on the pinned TPU runtime. Per-module precision is recorded.
Tests that explicitly exercise BF16 retain their own configuration.
This setting is recorded; it does not certify a full BF16 training campaign.
Frozen source hashes include uncommitted Python changes. Save a source archive
manifest alongside launch arguments so a later checkout can reproduce the run.

## Storage discipline

This test runner uploads reports, logs and JUnit XML only. Test checkpoint fixtures
remain on the ephemeral TPU filesystem. Source bundles and evidence still consume
GCS storage. Download them locally, verify them, and remove the campaign prefix
when no longer needed, respecting the bucket's existing soft-delete retention.
Do not disable recovery protection as an implicit part of a test run.


`--test-files` selects an explicit retry subset and labels the report as partial.
`--include-distributed-cpu` enables the localhost multiprocess recovery tests on
the VM; those are CPU checks and do not establish physical multi-host TPU coverage.
Keep initial failure evidence, retry evidence and both frozen source manifests.
Only combine them after checking which source files and runtime settings changed.

The supervisor submits queued-resource creation asynchronously, with a maximum
60-second submission timeout, before polling under `--capacity-wait-seconds`.
This prevents a blocking `gcloud create` from consuming the compute lease before
the capacity deadline starts. An ambiguous submission is never replayed; cleanup
must verify that both the queue and TPU node are absent before another attempt.
