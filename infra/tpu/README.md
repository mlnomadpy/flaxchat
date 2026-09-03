# Cloud TPU operator runbook

This directory is the non-interactive Cloud TPU adapter. Kaggle remains the
free accelerator CI path; this launcher is for controlled multi-host and
longer training runs. It uses flex-start queued resources with two cost
guardrails: a four-hour maximum run and a four-hour queue expiry.

Prerequisites are `gcloud auth login`, an explicitly selected project with
billing, the Compute/TPU APIs, a same-region GCS bucket, and the TPU service
identity. Never place durable checkpoints only on the TPU VM boot disk.

```bash
export PROJECT_ID=your-project
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

```bash
gcloud compute tpus tpu-vm scp --zone=us-west4-a --worker=all \
  --recurse flaxchat-tpu:~/flaxchat/artifacts ./artifacts/tpu-workers
infra/tpu/flexstart.sh delete
```

Always delete the queued resource, including when it reaches `FAILED`: queued
resources can retain quota and a provisioned VM incurs cost. The `delete`
operation is intentionally explicit and is never run by a test or import.
