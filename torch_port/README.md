# flaxchat → PyTorch port

A standalone PyTorch port of the flaxchat GPT model (GELU MLP variant) plus
a conversion script that loads an Orbax checkpoint produced by Flax NNX
training and emits a `.pt` file.

Target checkpoint: `mlnomad/gelu-d12-chinchilla-261M`
(d=12, n_embd=768, n_head=12, n_kv_head=12, vocab=32768, seq_len=1024,
`tie_embeddings=True`, `window_pattern="SSSL"`, GELU MLP).

## Files

| File | What |
|------|------|
| `torch_gpt.py`             | Pure PyTorch `nn.Module` (`GELU_GPT`, `GPTConfig`) |
| `convert_flax_to_torch.py` | One-off Orbax → `.pt` converter (needs JAX+Flax+Orbax) |
| `validate_parity.py`       | Runs Flax + Torch on identical inputs, prints max logit diff |
| `requirements.txt`         | Inference deps (`torch`, `numpy`) |
| `REPORT.md`                | Final parity numbers and architectural notes |

## Install

Inference only (no JAX needed):

```bash
pip install torch numpy
```

Conversion (JAX/Flax/Orbax already available in the repo’s `pixi` env):

```bash
pixi install   # from the flaxchat repo root, not torch_port/
```

## Convert the checkpoint

```bash
pixi run python torch_port/convert_flax_to_torch.py \
    --flax-ckpt models/gelu-d12-chinchilla-seed0/19920 \
    --out gelu_d12.pt
```

The output file is a single `.pt` containing both the config and the
converted state dict. It is ~1 GB (fp32).

## Use the ported model

```python
import torch
from torch_port.torch_gpt import GELU_GPT

model = GELU_GPT.from_pretrained("gelu_d12.pt")   # pure torch, no JAX
model.eval()

ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
with torch.no_grad():
    logits = model(ids)   # (1, 3, 32768)
print(logits.shape)
```

The model exposes only the forward pass; text generation (temperature
sampling, KV cache, speculative decoding) is NOT implemented here — use
flaxchat’s JAX engine (`flaxchat.engine.generate_with_cache`) for that, or
wrap this module with HuggingFace `generate` semantics yourself.

## Validate parity

```bash
pixi run python torch_port/validate_parity.py \
    --flax-ckpt models/gelu-d12-chinchilla-seed0/19920 \
    --torch-ckpt gelu_d12.pt
```

This runs the same `[1, 2, 3, ..., 32]` sequence through both Flax and
Torch in fp32 on CPU and prints `max |flax_logits - torch_logits|`.
Acceptance threshold: `1e-4`.

## Architecture notes

Every flaxchat feature is ported 1:1 — see `torch_gpt.py` for the full list:

- RoPE (base 100000, head_dim 64, split-half layout not interleaved)
- Parameterless RMSNorm (no learnable gain, `eps=1e-6`)
- QK-norm with `1.2×` scaling, applied **after** RoPE
- Group-Query Attention (`n_kv_head ≤ n_head`, `repeat_interleave` on K/V)
- Additive causal + sliding-window attention mask (pattern `"SSSL"` for d12)
- Value embeddings on alternating layers, gated by `3·sigmoid(ve_gate(x[..., :12]))`
- Per-layer learnable residual scalars (`resid_lambdas`, `x0_lambdas`)
- Smear: learnable gate on first 24 dims mixes in the previous token
- Backout: subtract mid-layer residual (`layer = n_layer // 2`)
- Logit soft-cap: `15 · tanh(logits / 15)`
- Tied embeddings (`lm_head = wte.T`)
- No biases anywhere

## Weight mapping

| Flax key                                  | Torch key                              | Transpose? |
|-------------------------------------------|----------------------------------------|------------|
| `wte.embedding`                           | `wte.weight`                           | no         |
| `blocks.{i}.attn.c_{q,k,v,proj}.kernel`   | `blocks.{i}.attn.c_{q,k,v,proj}.weight`| **yes**    |
| `blocks.{i}.attn.ve_gate.kernel`          | `blocks.{i}.attn.ve_gate.weight`       | **yes**    |
| `blocks.{i}.mlp.c_{fc,proj}.kernel`       | `blocks.{i}.mlp.c_{fc,proj}.weight`    | **yes**    |
| `resid_lambdas`, `x0_lambdas`             | same                                   | no         |
| `smear_gate.kernel`                       | `smear_gate.weight`                    | **yes**    |
| `smear_lambda`, `backout_lambda`          | same                                   | no         |
| `value_embeds.{i}.embedding`              | `value_embeds.{i}.weight`              | no         |
| `lm_head.kernel` *(only if untied)*       | `lm_head.weight`                       | **yes**    |

JAX `nnx.Linear.kernel` is `(in, out)`; PyTorch `nn.Linear.weight` is `(out, in)` — hence the transposes. JAX `nnx.Embed.embedding` matches
PyTorch `nn.Embedding.weight` directly.
