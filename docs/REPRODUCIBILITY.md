# Reproducibility

## CPU verification

From a clean checkout:

```bash
pixi run test-quick
FLAXCHAT_BASE_DIR=/tmp/flaxchat-smoke pixi run python -m scripts.pretrain --cpu-smoke
```

The smoke command uses a fixed synthetic-data seed, runs two optimizer updates,
and publishes a versioned checkpoint. It requires no dataset or accelerator.

## Training identity

Pretraining checkpoints record the resolved model/training configuration,
effective global token batch, update horizon, schedule parameters, tokenizer
identity, source revision, dataloader state and data-manifest identity. Exact
dataloader resume rejects changes to the ordered file manifest, tokenizer,
packing configuration, sequence length, or process topology.

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
