# GCP multi-host validation — 2026-09-21

Status: all nine physical multi-host acceptance stages passed on Spot v5p-16
(two hosts, eight TPU chips). Both the long-context and GPT-2-sized scale smokes also passed. Kaggle work is excluded from
this campaign at the user's request.

## Harness changes

- Explicit coordinator, rank and process count are supplied before JAX backend
  discovery. Host initialization is tested with two actual OS processes.
- Prepared-token training supports a fixed global batch across topologies.
  Hosts read the row intervals assigned by actual device sharding, including
  noncontiguous host ownership, and verify that the actual global array exactly
  reconstructs the canonical batch. Code/data/tokenizer/schedule digests must
  agree across workers before training.
- Data parallel and FSDP meshes place model, optimizer and cursor arrays on the
  global mesh. Local arrays use a multi-host-safe callback for initial placement.
- Shared `gs://` checkpoint URIs are preserved. Integrity hashing handles
  non-addressable sharded arrays; restores use the current topology. One leaf
  is gathered at a time, so a very large individual leaf must still fit in host
  memory. Pod-scale memory behavior remains unvalidated.
- The physical probe checks the expected loss, gradient and parameter update,
  distinct hostnames, complete workers, device totals, data order, source and
  environment digests. CPU records cannot pass physical acceptance.
- The GCP runner bounds every remote process independently of SSH, records
  per-worker logs, runtime version and provisioning model, and launches workers
  concurrently. The campaign stops after a failed prerequisite.
- Recovery acceptance injects SIGKILL on rank zero after a committed checkpoint.
  Model, optimizer and data-cursor hashes must exactly match an uninterrupted
  baseline. FSDP resume has a separate baseline. A single-host bridge transfers
  a checkpoint without an optimizer update and verifies all state hashes.
- Scale plans cover configurable devices/hosts/FSDP, correctness, 1,024-token
  context, GPT-2-sized models and an optional larger smoke. Plans declare token
  requirements and refuse invalid meshes, nonfinite prices and over-budget
  runs. A plan is not a successful hardware result.
- Quota preflight distinguishes a zone's zero override from its positive project
  default. Positive quota does not establish capacity or permission to submit to
  a particular provisioning queue.
- The deletion watchdog retries control-plane timeouts and deletes the owned
  node when queue deletion is rejected. VM shutdown and process timeout are
  secondary guards; neither alone proves cloud-resource deletion.

## Physical acceptance

The slice used two distinct physical hosts with four local devices each. The
probe verified matching source and installed-environment hashes, canonical
host data partitioning, finite synchronized loss, and the expected gradient
and parameter update.

All nine stages passed: probe, DP baseline, committed-checkpoint rank-zero
SIGKILL, DP resume, FSDP baseline, FSDP checkpoint stop, FSDP resume,
single-host checkpoint bridge, and multi-host restore. The bridge used **CPU**
on one host with bfloat16 constants; this is not a single-host TPU claim.

| Comparison at update 8 | Model | Optimizer | Data cursor |
|---|---|---|---|
| DP uninterrupted vs forced-kill recovery | Exact | Exact | Exact |
| DP uninterrupted vs topology round trip | Exact | Exact | Exact |
| FSDP uninterrupted vs checkpoint resume | Exact | Exact | Exact |

Recovery used global batch 16, sequence length 32, depth 2 and a fixed eight
update schedule. FSDP recovery used a 2×4 data/FSDP mesh. The deliberate kill
occurred only after checkpoint 4 was committed; it does not establish safety
for every possible failure inside an in-flight checkpoint write.

## Scale evidence

Both profiles passed four updates, finite evaluation, and GCS checkpoint save
on eight chips/two hosts with **FSDP=8**, so parameter sharding crossed host
boundaries. Global batch was eight:

| Profile | Layers | Sequence length | Training tokens |
|---|---|---|---|
| Context | 2 | 1,024 | 32,768 |
| GPT-2-sized | 12 | 128 | 4,096 |

These are bounded engineering smokes. No convergence, large-model quality,
more-than-two-host result, tensor-parallel result, or v4 result is claimed.
The scale planner is regression-tested for 1 through 256 devices; those are
command/validation tests, not hardware measurements at those scales.

## Current TPU unit suite

**458 passed, 4 skipped** in 184.41 seconds on one isolated physical host with
four TPU chips. The runner first asserted TPU backend, four devices and one
process. This used the same Python-source digest as the multi-host campaign.

The four skips are intentional: two opt-in localhost multiprocess tests, one
CPU/GPU fallback test, and the dedicated eight-virtual-device assertion. These
have complementary coverage in the CPU runs below. Two dependency deprecation
warnings (Starlette/httpx and AnyIO) remain; neither is a test failure.

The [machine-readable campaign record](../benchmarks/results/gcp-multihost-20260921.json)
contains the commands, topology, source identity, exact-state comparisons,
scale summaries, and all individual CPU/TPU test results.

## Local evidence

- Full CPU suite: **454 passed, 8 skipped**.
- Eight virtual CPU devices: **30 passed**, including exact FSDP resume and
  checkpoint transfer.
- Two-process localhost suite: **2 passed**, covering canonical global data,
  numeric collectives, sharded checkpoint integrity and restore.
- Ruff and Pyright pass. Documentation/link checks and `git diff --check` pass.
- A Starlette/httpx deprecation warning remains in the local run; it is not a test failure.

See the [complete test inventory](GCP_MULTIHOST_TEST_INVENTORY_2026-09-21.md).
The results in this subsection are CPU evidence, including the actual OS-process
tests. The physical TPU measurements are reported separately above.

## Provisioning evidence and operational findings

| Attempt | Result |
|---|---|
| Spot v4-16, us-central2-b | Type not exposed by the project's accelerator list; create returned NOT_FOUND. |
| Spot v5e-16, us-west4-a | Repeated VM stockout (`ZONE_RESOURCE_POOL_EXHAUSTED`); failed queue and nodes deleted. |
| Spot v5e-16, us-west4-b | GCP recommended the zone for capacity, but project Spot quota there is zero; no resources created. |
| Spot v5e-16, us-central1-a | Positive quota but VM stockout; failed queue and nodes deleted. |
| First Spot v5p-16, us-east5-a | Allocation began during cancellation; nodes and queue deleted without executing validation. This cancellation was premature and cost time. |
| On-demand v5p-16, us-east5-a | Queue permission denied despite positive quota and a positive global accelerator gate; no resources created. |
| Second Spot v5p-16, us-east5-a | READY; all nine multi-host acceptance stages passed. |

A queued request can remain in `PROVISIONING` beyond its requested allocation
window. GCP can reject deletion in that state. Inspect the underlying Compute
instance insertion errors, not just the TPU queue state, and verify both node
and queue absence after cleanup. Keep failed attempts in the report.

The reusable commands and setup instructions are in the
[TPU operator runbook](../infra/tpu/README.md). For v5p, accelerator suffixes count
cores: `v5p-16` has eight chips on two hosts. Scale plans use actual JAX device
and host counts, not the suffix. See [Google's topology table](https://docs.cloud.google.com/tpu/docs/v5p).

## Cleanup

The slice left READY and entered DELETING after collection. Forced queue
deletion completed. At **22:54:15 UTC**, API inventories confirmed no validation
nodes or queues remained in us-east5-a, us-west4-a, us-west4-b, us-central1-a,
or us-central2-b. The independent deletion watchdog was then stopped. Evidence
and checkpoints remain in the existing lifecycle-managed bucket.

## Budget and limits

The validation reserve remains $50 within the user's $900 total. Actual billing
has not been retrieved. Google documents that TPU charges accrue while nodes
are READY; failed allocation windows must not be represented as measured billed
cost. The full on-demand reference for a v5p-16 slice is $33.60/hour (eight chips
at $4.20), or $16.80 for 30 minutes, before storage/network. For budget accounting, reserving the full 30 minutes for this slice ($16.80),
ten minutes for the prematurely canceled v5p allocation ($5.60), and the prior
single-host campaign's conservative reference ($7.60) gives **$30.00** of
on-demand reference compute allowance. This is not an invoice or a measured
Spot charge; storage/network are additional. The $50 reserve leaves room for
those extras. Spot billing should be lower. See [Cloud TPU pricing](https://cloud.google.com/tpu/pricing).

The main source archive SHA-256 is
`c1238dc2ec530fda8f1b52352405326019f6f6bde15c64910eff8d852a8c5d1d`.
Both physical workers reported Python-source digest
`e7e303fc66515bb3def748bf6022ace6a936a34e7317ebf2bc85c7e476d16c37`
and installed-environment digest
`485b70909264909daf92963c9737764a36fbb392544981457155ca754f78f67f`.
Runtime was `v2-alpha-tpuv5`, with JAX 0.11.1, Flax 0.12.9,
Optax 0.2.8 and Orbax 0.12.4. The archive includes the workspace changes;
the base commit alone does not reproduce this campaign.

Source is the workspace based on commit
`c311b63a7ff035e7539155ae0955b60d8c9de29e`, with an additional source/archive digest;
it has not been committed, released or pushed. Private evidence is retained in
`gs://tpubuilders-flaxchat-validation-0921/multihost/` under the existing 30-day
bucket lifecycle, with local copies in `artifacts/gcp-multihost/`.

## Remaining GitHub work

| Issue | Status |
|---|---|
| [#12 Physical multi-host acceptance](https://github.com/mlnomadpy/flaxchat/issues/12) | Physical acceptance passed on two hosts/eight chips. Evidence publication/review remains; issue not closed. |
| [#23 Matched nanochat/MaxText benchmarks](https://github.com/mlnomadpy/flaxchat/issues/23) | Still open. Short smoke timings are not controlled matched benchmarks. |
| [#11 Release](https://github.com/mlnomadpy/flaxchat/issues/11) | Still open. No tag, wheel publication or release was performed. |
| [#19 Public checkpoint/demo](https://github.com/mlnomadpy/flaxchat/issues/19) | Still open. A validation checkpoint is not a useful public model. |
| [#1 Readiness umbrella](https://github.com/mlnomadpy/flaxchat/issues/1) | Still open; sustained training, matched measurements and release artifacts remain. |
| [#31 Kaggle](https://github.com/mlnomadpy/flaxchat/issues/31) | Deliberately excluded from this campaign. |
