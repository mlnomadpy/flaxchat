# Reproducibility

## CPU verification

From a clean checkout:

```bash
pixi run test-quick
pixi run test-coverage
pixi run test-multidevice
FLAXCHAT_BASE_DIR=/tmp/flaxchat-smoke pixi run python -m scripts.pretrain --cpu-smoke
```

The smoke command uses a fixed synthetic-data seed, runs two optimizer updates,
and publishes a versioned checkpoint. It requires no dataset or accelerator.

The four stage modules under `flaxchat.stages` expose typed `PretrainRequest`,
`SFTRequest`, `RLRequest`, and `EvalRequest` inputs and return a common
`StageResult` containing the resolved configuration, metrics, artifact paths,
and exit status. Supplying `resolved_config` makes every stage consume the same
already-validated `FlaxChatConfig`; CLI defaults resolve one when omitted. The
modules in `scripts/` only parse CLI arguments, construct the request, invoke
the service, and translate its result to a process exit code.

## Training identity

Pretraining checkpoints record the resolved model/training configuration,
effective global token batch, update horizon, schedule parameters, tokenizer
identity, source revision, dataloader state and data-manifest identity. Exact
dataloader resume rejects changes to the ordered file manifest, tokenizer,
packing configuration, sequence length, or process topology.

Resume restores model variables, the complete optimizer state, update and
microbatch counters, and the exact dataloader cursor before the next batch is
constructed. The integration suite compares ten uninterrupted updates against
five updates plus checkpoint/restore plus five updates, including optimizer
state.

## Accelerator verification

The Kaggle CLI runner submits one script containing device validation, all
tests (including XLA/Splash forward-and-gradient parity), and the two-update
pretraining smoke test:

```bash
python -m pip install kaggle
python -m scripts.kaggle_tpu_tests \
  --kernel-id OWNER/flaxchat-tpu-tests --wait
```

The generated kernel pins the exact Git revision and downloads JUnit, summary,
and command logs into `artifacts/kaggle`. GitHub's `Kaggle TPU tests` workflow
exposes the same path when `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets are set.
Local, GCP, and Kaggle launchers serialize the same `LaunchSpec`. GCP transports
a fixed wrapper command; the remote wrapper reads the manifest and executes its
argv directly without interpolating training arguments into a shell string.

## CORE evaluation

CORE datasets are pinned to immutable Hugging Face revisions. By default the
full declared split is evaluated. A bounded run uses a seeded sample rather
than the first rows:

```bash
python -m scripts.eval --tasks=core --max-per-task=100 \
  --manifest-path=artifacts/core-manifest.json
```

The output includes exact scored and few-shot indices, protocol hash, sample
counts, confidence intervals, and per-example predictions/scores. If any task
fails, the aggregate is marked incomplete instead of averaging a synthetic
zero.
