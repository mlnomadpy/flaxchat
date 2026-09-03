---
layout: page
title: Deployment
permalink: /deployment/
---

# Deployment Guide

## Remote Execution Backends

flaxchat supports local execution plus two CLI-driven accelerator paths:

```mermaid
graph TD
    USER[Your Laptop] --> |"pixi run"| LOCAL[Local CPU/GPU]
    USER --> |"kaggle CLI"| KAGGLE[Kaggle TPU v5e-8]
    USER --> |"tpuz CLI/API"| GCP[GCP TPU Pod]
```

All backends implement the same `RemoteRunner` interface.

## 1. Local (laptop/workstation)

```bash
pixi install
python -m scripts.run_tinystories --layers=4 --pretrain-steps=1000
```

## 2. Kaggle TPU test bundle

```bash
python -m pip install kaggle
python -m scripts.kaggle_tpu_tests \
  --kernel-id OWNER/flaxchat-tpu-tests --wait
```

The generated private kernel checks out the exact current Git revision, installs
TPU JAX, verifies eight devices, runs the entire suite once, executes the pinned
TinyStories end-to-end pipeline, and records attention/speculative benchmarks.
Logs, artifacts, JUnit XML, and a JSON summary are downloaded together.

## 3. GCP TPU Pod

The production flex-start launcher, pinned environment, worker coordination,
artifact collection, and teardown guardrails live in [`infra/tpu/`](../infra/tpu/README.md).

### Python API

```python
from tpuz import TPU

tpu = TPU("flaxchat-d24", accelerator="v6e-8")
tpu.up()
tpu.setup(extra_pip="flaxchat")
tpu.run("python -m scripts.pretrain --depth=24", sync=".")
```

### TPU Types

| Accelerator | Chips | Workers | Free Tier | Zones |
|-------------|-------|---------|-----------|-------|
| `v4-8` | 4 | 1 | On-demand | us-central2-b |
| `v4-32` | 16 | 4 | No | us-central2-b |
| `v5litepod-8` | 8 | 1 | TRC | us-central1-a |
| `v5litepod-64` | 64 | 8 | TRC | us-central1-a |
| `v6e-8` | 8 | 1 | TRC | europe-west4-a |
| `v6e-64` | 64 | 8 | No | europe-west4-a |

### Multi-Host Training

For pods with multiple processes, JAX distributed initialization coordinates
hosts and FlaxChat builds one global SPMD mesh. Each host consumes alternating
parquet row groups and reconstructs global batches with
`jax.make_array_from_process_local_data`.

### Preemption Recovery

```mermaid
graph TD
    RUNNING[Training Running] --> |"Poll every 60s"| CHECK{VM State?}
    CHECK --> |READY| ALIVE{Process alive?}
    ALIVE --> |Yes| RUNNING
    ALIVE --> |No, completed| DONE[Done]
    ALIVE --> |No, crashed| RESTART[Restart training]
    CHECK --> |PREEMPTED| RECOVER[Delete → Recreate → Setup → Resume]
    RECOVER --> RUNNING
    RESTART --> RUNNING
```

Relies on Orbax checkpoints for resume — training picks up from last saved step.

## Export

### Checkpoint Formats

| Format | File | Use Case |
|--------|------|----------|
| JAX checkpoint | `model.pkl` | Resume training, load in flaxchat |
| NumPy weights | `weights.npz` | Portable, load anywhere |
| StableHLO | `model.stablehlo` | LiteRT/TFLite conversion input |

### LiteRT/TFLite Conversion

```bash
# On Linux with TensorFlow:
python -m scripts.convert_to_tflite \
    --checkpoint=exports/model.pkl \
    --output=exports/model.tflite
```
