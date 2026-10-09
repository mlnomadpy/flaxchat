# GOAT attention: score input, project values

The requested architecture is `EncoderConfig(attention_score='goat_input')`. Attention scores use the attention input directly. Only the values have a learned V projection; there are no learned Q or K projections.

Let `h` be the block's existing attention input (after its existing pre-norm; the first block keeps its original normalization policy). Split its channels into heads and apply the existing positional rotation:

\[
z_i=\operatorname{RoPE}(h_i),\qquad v_i=h_iW_V,
\]
\[
s_{ij}=\alpha\frac{(z_i^\top z_j+1)^2}{\|z_i-z_j\|^2+0.01},\qquad
p_{ij}=\operatorname{softmax}_{j\in A_i}(s_{ij}),\qquad
o_i=\operatorname{concat}_{h}\Bigl(\sum_{j\in A_i}p_{ij}v_j\Bigr)W_O.
\]

`A_i` excludes the diagonal, padding, other packed documents and (for local layers) tokens outside the local window. Softmax supplies exp(YAT). Empty rows return zero attention output; the residual carries each token's own state. Bias1 and epsilon0.01 are fixed; alpha remains trainable. Norms are calculated from the actual head vectors; LayerNorm does not imply unit head-vector norms.

Changing W_V changes values and aggregation but does not change this block's score inputs or attention weights for a fixed h. Gradients through scores flow into h and earlier layers directly. Gradients through V also train W_V. An output projection and the existing YAT FFN remain.

## Versioned behavior

`attention_score='goat'` records the earlier implementation, where RoPE(V) supplied the score geometry. Its frozen checkpoints retain that behavior. The user stopped that training run on October7; the input-scored architecture has a distinct config identity and must use a new stage/checkpoint namespace. Old source/runtime/test receipts do not qualify the new mode.

Use `scripts.train_encoder --attention-score goat_input` for a fresh encoder run. The bounded scratch wrapper selects it with `--random-init --goat-score-source input`. The wrapper rejects input-scored migration or implicit continuation of the old stage. It propagates the selection to preflight, training, evaluation, seeded baseline, and continuation identity. The authenticated architecture/tokenizer reference supplies configuration only; no trained tensors are loaded for random initialization.

Both modes have only W_V and W_O attention projections. Parameter counts match: removing Q/K saves25,952,256parameters for22layers/width768. Tiled XLA local/global attention, BF16 geometry and existing FP32 score/residual policy remain. Global attention still has quadratic arithmetic. QAT/Splash/manual local sharding and PyTorch release are not qualified for this mode.

## Current acceptance

Metadata admission and static checks pass. The suite contains30 physical TPU cases: both variants' forward/gradient oracle checks, window/document/diagonal/empty-row handling, projection migration, input/value independence in blocks0/1, and tiny MLM update/checkpoint/exact resume for both variants. The new input-scored numerical cases are **not yet executed on TPU**. No replacement training was launched.

The bounded wrapper requires a passing physical receipt with all30cases and the explicit input/value separation and input-scored resume cases before training this architecture. A current full-model random update/save/resume and heldout sanity check against the same random initializer must also pass.

```
FLAXCHAT_PHYSICAL_TPU=1 pytest -q tests/test_goat_physical_tpu.py
```

No model numerics or performance checks run on CPU.
