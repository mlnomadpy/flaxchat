# Looped GOAT encoder and LoopCD

Implemented October 9, 2026. **Experimental; physical TPU validation and quality
measurement are not yet run for this source.** Existing training deployments keep
their frozen source and configuration. This change does not launch training.

## Architecture

`ModernBert` supports a prelude, shared core, and coda. `num_hidden_layers` is the
number of **unique** blocks, including all three parts. `loop_core_start` counts
prelude blocks, `loop_core_layers` counts shared blocks, and `loop_steps` counts
core applications. Effective block depth is

```
num_hidden_layers + (loop_steps - 1) * loop_core_layers
```

The example `configs/encoder/looped_goat_input.json` has one prelude block, six
shared blocks repeated four times, and one coda block: eight unique blocks and
26 block applications. Its size is not equivalent to the previous 22-block model.
The large vocabulary embedding table remains unchanged; do not infer a total
parameter or cost reduction from the block-count ratio.

The same NNX block objects and their trainable YAT alphas are reused. An NNX scan
broadcasts their parameter state across recurrent iterations; its body contains
one core application and returns only the final carry, not a stack of hidden
states. Each block can still rematerialize its activations for backward. This
limits forward graph growth; backward memory and throughput need measurement.
There is no detached gradient, early-exit policy, or learned halting mechanism.

A prelude of at least one block is required: ModernBERT block zero omits its
attention normalization, so repeatedly applying that block would silently change
normalization semantics. Core blocks retain their physical block indices,
local/global attention windows and RoPE bases on every repetition. Choosing a
core size divisible by `global_attn_every_n_layers` preserves the periodic pattern
across recurrent boundaries, as the example does. Other core sizes deliberately
repeat their own physical-index pattern.

GOAT-input still calculates YAT scores from normalized input geometry, with no
Q/K projections, diagonal masking, and a separately projected V. YAT bias is
fixed at 1, epsilon at 0.01, and alpha is trainable. Recurrence does not change the
attention kernel or FFN. Existing dot-product and YAT attention/FFN choices can
also use recurrence.

## Training and recovery

The regular MLM call and contrastive encoder pooling use the configured loop
count and **zero LoopCD guidance**. The training objective is unchanged. Config
JSON is accepted by the existing encoder trainers, native public loader, and
exporter. MLM CLI overrides are `--loop-core-start`, `--loop-core-layers`, and
`--loop-steps`; architecture and effective depth appear in run-config telemetry.

All recurrence fields are included in serialized configuration and stage
identity. Direct checkpoint restoration additionally rejects mismatched loop
settings, even when parameter shapes agree. Native weight export stores each
unique block once. Plain encoder configs default to no recurrence and retain the
same forward path, block names and shapes. Historical checkpoint identities and
source hashes are not rewritten or relaxed.

An ordinary 22-block checkpoint is **not** silently converted to an eight-block
looped model. HF import is rejected for looped models. Changing recurrence requires
a separate stage with an explicitly designed migration policy or fresh training;
a migration policy is not included here. The current PyTorch release loader
rejects looped configs until recurrence and parity are implemented there.

Example new-stage command, **only after physical qualification**, using prepared
training rows and a fresh output namespace:

```sh
python -m scripts.train_encoder \
  --config configs/encoder/looped_goat_input.json \
  --data /path/to/prepared-training-rows \
  --output gs://YOUR_BUCKET/NEW_LOOPED_STAGE/checkpoints \
  --dtype bfloat16 --residual-dtype float32 \
  --mlm-projection masked --steps 100000 --batch-size 64
```

This is a template, not an authorized cloud allocation or a claimed production
recipe. Existing budget, source/runtime/data authentication and cleanup admission
still apply. Default MLM trainer CLI execution options can override their JSON
counterparts; pin the complete resolved config in the stage manifest.

## LoopCD inference

[Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/pdf/2610.02185)
compares early and final recurrent representations. Our early state is the output
of the **first complete core application**; the final state follows the requested
number of core applications. Both are before the coda and final normalization.

Fixed-strength hidden guidance is

```
h_guided = h_final + strength * (h_final - h_first)
```

followed by **one** coda, final normalization and optional MLM head. It is available
through `model.encode(..., loop_guidance_strength=0.5)` and
`model.pool(..., loop_guidance_strength=0.5)`. Both accept a static inference
`loop_steps` override without mutating the training configuration. Treat those
settings as evaluation protocol and record them with results. Different loop
counts compile separate scans. Guidance defaults to zero. At one loop the
correction is zero.

`model.loopcd_logits(ids, strength=0.5)` instead runs both boundary states through
the coda, final normalization and MLM head, then computes

```
z_guided = z_final + strength * (z_final - z_first)
```

With `adaptive=True`, strength is multiplied per token by
`1 - (p_top1 - p_top2)` from the final logits' softmax. The implementation uses
FP32 head outputs and log-sum-exp for the probability margin. Strength must be a
finite nonnegative static scalar; runtime options must be static under JIT.

Logits guidance materializes dense vocabulary logits and executes an extra
coda/head pass. It is an explicit inference API, not part of memory-bounded MLM
training. Do not use it to infer training throughput. The paper evaluates
**autoregressive decoding**; our bidirectional MLM/embedding adaptation has no
measured quality benefit yet. It is unrelated to the YAT InfoNCE objective.

## Validation and next gate

Local model-free admission:

```sh
python -m pytest -q tests/test_looped_encoder_config.py
```

Observed locally on October 9: **35 passed** across
`test_looped_encoder_config.py` and `test_goat_metadata.py`; changed Python files
passed Ruff and whitespace checks. The seven TPU acceptance cases collected
successfully but were **not executed**. No CPU model tests were run.

On an independently admitted physical TPU worker with the current frozen source:

```sh
JAX_PLATFORMS=tpu python -m pytest -q tests/test_looped_encoder_tpu.py
```

The TPU suite fails explicitly on any other backend. It covers compiled recurrence
against an independently unrolled reference, every parameter gradient, remat
on/off, mixed/BF16 geometry, packed positions/masks, padding, unchanged parameter
count, trainable alpha, hidden/logits/adaptive guidance, one-loop equivalence,
MLM update, optimizer/cursor recovery, loop-identity rejection, and native public
export/reload. Its tolerances are declared assertions, not measured accuracy.

Passing this bounded suite is necessary, not production acceptance. Next compare
full warmed steps, compilation, HBM, checkpoint overhead and multilingual/code
held-out quality at matched tokens **and compute**. Retain failed assertions and
receipts. Full-size, sustained training, multi-host and PyTorch parity remain
separate gates. No scaling or quality claim follows from model-free checks.
