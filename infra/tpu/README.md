# Cloud TPU operator runbook

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
