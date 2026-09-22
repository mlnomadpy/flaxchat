# Cloud TPU operator runbook

## Bounded Spot supervisor

For the newer guarded validation lifecycle, use
`python -m scripts.gcp_spot_supervisor --help`. This integrates a persistent
budget ledger, a server-verified `gcp_cleanup_guard` lease, worker setup,
optional single-worker checks and multi-host acceptance, and verified teardown.
The deployed cleanup workflow must cover the chosen project, zone and dedicated
`flaxchat-validation-*` resource names before this command is usable.

`--setup`, `--checks`, `--workload` and `--resume-workload` take JSON arrays of
argv strings. Setup runs on every worker; checks run on worker zero without
distributed JAX variables; workload runs on every worker with explicit JAX
coordination. TPU unit checks on a multi-host slice must additionally set the
appropriate single-host TPU topology environment, as recorded in the accepted
campaign. Do not run independent TPU tests concurrently with distributed training.

Use `--acceptance-prefix gs://BUCKET/FRESH_PREFIX` to run the nine-stage recovery
campaign before the workload. The prepared source archive must include its
small pinned token pool. The source archive checksum, base commit and the
version constraints in `validation-lock.txt` are inputs to `setup_validation.sh`.

Every attempt reserves `whole_slice_hourly_rate * (attempt_seconds + 1800) / 3600`
plus the ancillary reserve. The budget includes **all** retry reservations.
An existing resource name or ledger attempt cannot be reused. More than one
attempt requires explicit durable-checkpoint resume arguments; only confirmed
preemption is retryable. A failed deletion stops the controller and leaves a
visible failed ledger record. Cloud API failure can exceed the reserved window;
the allowance is not an absolute billing cap.

The ledger's `posted_usage_usd` remains null until reconciled against actual
billing evidence. Record gross usage and promotional credits separately using
`RunLedger.reconcile`; do not infer usage from a credit balance. Save output
metrics to durable storage before the workload exits, because teardown follows
immediately. See the [follow-up report](../../docs/HARNESS_AUDIT_FIXES_2026-09-22.md)
for verified scope and outstanding production/quality work.

## Manual adapter

This directory is the non-interactive Cloud TPU adapter. Kaggle remains the
free accelerator CI path; this launcher is for controlled multi-host and
longer training runs. The acceptance default is a `v5litepod-16`, which must
still prove `jax.process_count() >= 2` at runtime. Cost guardrails are a one-hour
maximum run, a one-hour queue expiry, and an explicit billing acknowledgement.

Prerequisites are `gcloud auth login`, an explicitly selected project with
billing, the Compute/TPU APIs, a same-region GCS bucket, and the TPU service
identity. Never place durable checkpoints only on the TPU VM boot disk.

```bash
export PROJECT_ID=your-project
export FLAXCHAT_APPROVE_PAID_RUN=I_UNDERSTAND_TPU_BILLING
infra/tpu/flexstart.sh create
infra/tpu/flexstart.sh status
```

After the resource becomes `ACTIVE`, run setup on every worker. Pin both the
repository revision and environment; do not install from a moving branch:

```bash
REVISION=$(git rev-parse HEAD)
gcloud compute tpus tpu-vm ssh flaxchat-tpu --zone=us-west4-a --worker=all \
  --command="git clone https://github.com/mlnomadpy/flaxchat.git && cd flaxchat && git checkout $REVISION && python3 -m venv .venv && .venv/bin/pip install -r infra/tpu/environment.txt -e ."
```

Use a shared `gs://` output path for checkpoints and per-process logs. Preserve
the resolved config, run manifest, `jax.process_count()`, source revision, and
environment file alongside results. For a multi-host run, launch the same
command with `--worker=all`; JAX discovers worker coordination from the TPU VM
runtime. Copy logs before teardown:

Start with the fail-fast probe before any longer training. It exits unless at
least two physical JAX processes and a TPU backend are present, proves disjoint
host-local batches reconstruct global order, and performs one synchronized
finite gradient update over a data-sharded global array:

```bash
gcloud compute tpus tpu-vm ssh flaxchat-tpu --zone=us-west4-a --worker=all \
  --command="cd flaxchat && .venv/bin/python -m scripts.multihost_acceptance run --project=$PROJECT_ID --zone=us-west4-a --slice=v5litepod-16"
```

Collect every worker record, then provide the billed cost from the Cloud
Billing report to the fail-closed summarizer:

```bash
gcloud compute tpus tpu-vm scp --zone=us-west4-a --worker=all \
  --recurse flaxchat-tpu:~/flaxchat/artifacts/multihost ./artifacts/tpu-workers
python -m scripts.multihost_acceptance summarize \
  --input-dir artifacts/tpu-workers \
  --output artifacts/multihost-summary.json \
  --cost-usd ACTUAL_BILLED_COST
```

Only continue to cross-topology checkpoint and interrupted-resume phases when
the summary status is `probe_passed`. The probe summary explicitly does not
claim those later phases; issue #12 remains incomplete until their physical
records are attached.

Copy logs before teardown:

```bash
gcloud compute tpus tpu-vm scp --zone=us-west4-a --worker=all \
  --recurse flaxchat-tpu:~/flaxchat/artifacts ./artifacts/tpu-workers
infra/tpu/flexstart.sh delete
```

Always delete the queued resource, including when it reaches `FAILED`: queued
resources can retain quota and a provisioned VM incurs cost. The `delete`
operation is intentionally explicit and is never run by a test or import.

## GCP multi-host and scale validation

The prepared-data trainer accepts `--global-batch-size` and `--fsdp`. Keep the
global batch fixed when moving checkpoints between topologies: it is part of
the checkpoint identity and learning-rate schedule. Every host verifies the
same source, data and configuration digest, reads a distinct token interval,
and checks that the actual global array reconstructs the canonical batch.
Multi-host training requires a shared `gs://` checkpoint directory. Install
`gcsfs` alongside `infra/tpu/validation-environment.txt` and archive `pip freeze`
on every worker. Keep token pools local and immutable on each worker.

Prepared training defaults to a 300-second checkpoint interval measured after
the last completed save. Rank zero broadcasts the save decision, and the final
or requested clean-stop step always saves. Use `--save-every N` for an explicit
update cadence or `--checkpoint-interval-seconds S` for a time override; these
options are mutually exclusive. Standard tied GPT uses a 0.02 shared-table
initialization scale; saved checkpoint tensors are restored unchanged.

On an already prepared slice, the local orchestrator launches every worker
concurrently, injects explicit coordinator/rank/count variables before JAX
imports, and installs a remote process timeout. Arm `spot_watchdog.py` separately;
a process timeout or VM shutdown alone does not delete the cloud resource.

TPU runtime process indices may differ from gcloud worker indices. Select the
fault-injection coordinator using the launcher's coordination-service rank,
and discover result files by their actual runtime index. The SSH runner claims
a unique command identity on each worker and reports completion through a
marker, preventing gcloud from replaying a failed command on that worker.
Missing/duplicate markers and ambiguous transport failures fail validation.

```bash
python -m scripts.validate_gcp_multihost \
  --project YOUR_PROJECT --zone YOUR_ZONE --node YOUR_EXISTING_NODE \
  --checkpoint-prefix gs://YOUR_BUCKET/UNIQUE_RUN/checkpoints \
  --output artifacts/multihost-campaign --global-batch 16 --fsdp 4
```

This bounded campaign covers the physical topology/collective probe, GPT data
parallel training, rank-zero SIGKILL after a committed checkpoint, exact resume,
FSDP training/resume, and multi-host → single-host CPU → multi-host checkpoint
transfer. All three model/optimizer/cursor digests must match their uninterrupted
baseline. The CPU bridge is explicitly identified; it is not single-host TPU
acceptance. A failed prerequisite stops the campaign. Raw logs and return codes
are preserved per worker. Use a fresh GCS prefix for each campaign; retain failed
attempts as well as successful ones. Setup must put the checkout and environment
at `/tmp/flaxchat-validation` on every worker, including the prepared token pool
at `artifacts/gcp-validation/fineweb-pool/manifest.json`.

Generate smoke commands for any available topology, rather than assuming that
eight devices means one host:

```bash
python -m scripts.tpu_scale_plan --devices 16 --hosts 4 --fsdp 4 \
  --profiles correctness context gpt2 --hourly-usd CURRENT_WHOLE_SLICE_RATE \
  --budget-usd 10 --token-manifest /data/manifest.json \
  --checkpoint-prefix gs://YOUR_BUCKET/UNIQUE_SCALE_RUN
```

The planner supports topology-dependent global batch, token requirements,
DP/FSDP, long context, GPT-2-sized models and an optional larger smoke. It refuses
invalid meshes, nonfinite prices and plans above budget. These are **plans**, not
claims that 1–256 devices have passed. A smoke pass establishes execution only;
it does not establish convergence, throughput scaling efficiency or model quality.
For each physical scale, run recovery acceptance as well as the selected model
profiles and record compilation, steady step time, checkpoint time, memory,
source digest and actual topology. Parameter/optimizer integrity hashing gathers
one sharded leaf at a time; very large embeddings still require that leaf to fit
in host memory, and this implementation has not been validated at pod scale.

Cheap local regressions:

```bash
pixi run python -m pytest tests/test_token_pool.py tests/test_checkpoint.py
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  pixi run python -m pytest tests/test_token_pool.py tests/test_sharding.py
FLAXCHAT_RUN_DISTRIBUTED_CPU=1 pixi run python -m pytest tests/test_distributed_cpu.py
```

The last test requires localhost socket access and runs two real OS processes.
The physical summarizer rejects its CPU records, repeated hostnames, incorrect
numeric gradients, missing workers, overlapping data and mismatched code digests.

### Provisioning failure cleanup

Queue expiry does not necessarily interrupt a request already in `PROVISIONING`.
GCP can reject queued-resource deletion in that state. Check both the queue and
node lists; if the owned node exists, delete it explicitly, then delete the
suspended queue. Retry control-plane timeouts and verify absence before declaring
cleanup complete or allocating replacement hardware. Retain failure records and
never report a provisioning attempt as a passed TPU test.

Before creating a resource, run the read-only quota/type preflight:

```bash
python -m scripts.gcp_tpu_preflight --project YOUR_PROJECT --zone YOUR_ZONE \
  --accelerator-type v5litepod-16
```

An explicit zone quota bucket with an omitted effective limit means **zero**,
not the project default. A positive quota limit does not guarantee free capacity
or account for current usage. Check the Compute instance insertion errors if a
queued resource repeatedly creates and removes TPU node records; those errors
can distinguish a stockout from permissions or quota failures.

For repeatable worker setup, upload the source archive and
`infra/tpu/setup_validation.sh` to your private validation bucket, then run the
setup script on each worker with the archive URI, SHA-256 and base commit:

```bash
bash setup_validation.sh gs://YOUR_BUCKET/source.tar.gz ARCHIVE_SHA256 SOURCE_COMMIT
```

The script verifies the archive, installs the pinned JAX stack, GCS support and
CPU PyTorch reference-test dependency, and writes the actual environment freeze
to `/tmp/flaxchat-env.txt`. It expects an already scheduled VM shutdown and fails
if that guard is absent. Run it through `gcp_tpu_run --setup --directory /tmp`
with a timeout. The runtime image and provisioning model are recorded by the
launcher, while the probe compares environment-freeze digests across workers.

## Durable cleanup for validation runs

The validation campaign requires a live Cloud Workflows cleanup receipt. The
workflow survives loss of the launcher and verifies that the exact queued
resource disappears after `force=true` deletion. Deployment sources are
`cleanup_workflow.yaml`, `cleanup_role.yaml` and the project-specific
`cleanup_condition.json`. The service account cannot provision compute.

Arm the workflow **before** creating a uniquely named `flaxchat-validation-*`
queue. Do not reuse queue names, and set the queue's `valid-until-time` no later
than the guard deadline. Verify the receipt again immediately before allocation.

```bash
python -m scripts.gcp_cleanup_guard --project tpubuilders --zone us-east5-a \
  --queue flaxchat-validation-UNIQUE --seconds 3600 --receipt artifacts/guard.json
python -m scripts.gcp_cleanup_guard --project tpubuilders --zone us-east5-a \
  --queue flaxchat-validation-UNIQUE --receipt artifacts/guard.json --verify
```

Pass `--cleanup-receipt artifacts/guard.json` to `validate_gcp_multihost`. Its
campaign timeout must leave at least 120 seconds before expiry. The runner now
accepts `--optimizer muon --accumulation-steps 2 --loss-chunk-size 16 --remat`
for validation of the production optimizer and memory-saving path. Keep a local
watchdog as a second control and always delete resources promptly on completion.
Cloud API cleanup can be delayed; neither watchdog is an exact billing cap.

For training-only machines, the optional fourth setup argument `training`
installs the pinned JAX stack and data dependencies without the validation web,
Torch and development packages. GCP worker commands set a local persistent JAX
compilation cache. Record the actual installed environment for every campaign.
