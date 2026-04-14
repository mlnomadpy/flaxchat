"""
PyTorch port of the flaxchat GPT model (GELU MLP variant).

Faithful 1:1 port of flaxchat/gpt.py — every architectural feature matches:
- RoPE (base 100000, head_dim=64 for d12, split-half layout)
- Parameterless RMSNorm (no learnable gain)
- QK-norm with 1.2x scaling (applied after RoPE, before SDPA)
- Group-Query Attention (supports n_kv_head < n_head, repeats K/V heads)
- Value embeddings on alternating layers (ResFormer-style), gated by
  `3 * sigmoid(ve_gate(x[..., :12]))`
- Per-layer learnable residual scalars (`resid_lambdas`, `x0_lambdas`)
- Smear: learnable gate on first 24 dims mixes in prev token
- Backout: subtract mid-layer residual from late layers
- Logit soft-cap: `15 * tanh(logits / 15)`
- Sliding-window attention via window pattern (e.g. "SSSL")
- Tied embeddings (lm_head = wte.T)
- No biases in any Linear (attn Q/K/V/proj, MLP fc/proj, smear_gate, ve_gate)
- MLP is GELU (`Linear -> gelu -> Linear`), the GELU variant

This file is pure PyTorch (no JAX / Flax import). The conversion script
in `convert_flax_to_torch.py` loads the Orbax checkpoint once and emits a
`torch.save` state dict; `GELU_GPT.from_pretrained(path)` can then load it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768
    window_pattern: str = "SSSL"
    tie_embeddings: bool = True
    rope_base: float = 100000.0
    pad_vocab_size_to: int = 64
    mlp: str = "gelu"  # "gelu" or "relu2"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def padded_vocab_size(self) -> int:
        v = self.vocab_size
        p = self.pad_vocab_size_to
        return ((v + p - 1) // p) * p


# ---------------------------------------------------------------------------
# Parameterless RMSNorm
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: float = 100000.0, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Matches Flax `precompute_rotary_embeddings` exactly.

    Shapes: cos, sin  -> (1, seq_len, 1, head_dim // 2)
    """
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, D/2)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, D). Uses split-half (flaxchat layout):

        x1 = x[..., :D/2]; x2 = x[..., D/2:]
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        return concat([y1, y2], dim=-1)
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_window_sizes(config: GPTConfig) -> List[Tuple[int, int]]:
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"
    long_window = config.sequence_len
    # ceil(long / 4 / 128) * 128 via negative-floor trick
    short_window = -(-long_window // 4 // 128) * 128
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    window_sizes[-1] = (long_window, 0)
    return window_sizes


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self._has_ve = has_ve(layer_idx, config.n_layer)

        head_dim = config.head_dim
        self.c_q = nn.Linear(config.n_embd, config.n_head * head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if self._has_ve:
            self.ve_gate = nn.Linear(12, config.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, T, C = x.shape
        n_head = self.config.n_head
        n_kv_head = self.config.n_kv_head
        head_dim = self.config.head_dim

        q = self.c_q(x).reshape(B, T, n_head, head_dim)
        k = self.c_k(x).reshape(B, T, n_kv_head, head_dim)
        v = self.c_v(x).reshape(B, T, n_kv_head, head_dim)

        if self._has_ve and ve is not None:
            ve = ve.reshape(B, T, n_kv_head, head_dim)
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., :12]))  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = rms_norm(q) * 1.2
        k = rms_norm(k) * 1.2

        if n_kv_head < n_head:
            repeats = n_head // n_kv_head
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)

        # Build causal + sliding-window mask. Matches Flax implementation:
        #   causal[i, j] = True iff i >= j
        #   window_mask[i, j] = True iff (i - j) <= window_left
        # mask = causal & window (True = attend, False = mask out with -1e9)
        window_left = window_size[0]
        device = x.device
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        causal_mask = row_idx >= col_idx  # (T, T)
        if 0 < window_left < T:
            window_mask = (row_idx - col_idx) <= window_left
            causal_mask = causal_mask & window_mask
        # Additive bias: 0 where attend, -1e9 where mask out.
        bias = torch.where(
            causal_mask,
            torch.zeros((), dtype=x.dtype, device=device),
            torch.full((), -1e9, dtype=x.dtype, device=device),
        )
        bias = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Rearrange to (B, H, T, D) for torch SDPA.
        q_bhtd = q.transpose(1, 2)
        k_bhtd = k.transpose(1, 2)
        v_bhtd = v.transpose(1, 2)

        # Match Flax's `jax.nn.dot_product_attention(..., bias=bias, scale=1/sqrt(D))`:
        # that is a plain softmax-attention with the additive bias and explicit scale.
        # We implement it manually (rather than F.scaled_dot_product_attention with
        # attn_mask) to guarantee bit-identical numerics with Flax's path.
        scale = 1.0 / math.sqrt(head_dim)
        att = torch.matmul(q_bhtd, k_bhtd.transpose(-2, -1)) * scale  # (B,H,T,T)
        att = att + bias
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, v_bhtd)  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().reshape(B, T, -1)
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------------------------
# MLP (GELU variant — matches train_d12_chinchilla.py --mlp gelu branch)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        if self.config.mlp == "gelu":
            # jax.nn.gelu default uses approximate=True (tanh form). Match that.
            x = F.gelu(x, approximate="tanh")
        elif self.config.mlp == "relu2":
            x = F.relu(x).pow(2)
        else:
            raise ValueError(f"Unsupported mlp: {self.config.mlp}")
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), ve, cos, sin, window_size)
        x = x + self.mlp(rms_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class GELU_GPT(nn.Module):
    """PyTorch port of flaxchat.gpt.GPT with GELU MLP.

    Naming mirrors the Flax module tree so weight conversion is a direct
    key-for-key mapping (see convert_flax_to_torch.py).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.window_sizes = compute_window_sizes(config)

        padded_vocab = config.padded_vocab_size
        self.padded_vocab_size = padded_vocab

        # Token embedding
        self.wte = nn.Embedding(padded_vocab, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

        # lm_head (only materialized when NOT tied)
        self.tie_embeddings = config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        else:
            self.lm_head = None

        # Per-layer learnable scalars (stored as (n_layer,) tensors to match Flax)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Smear
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Backout
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Value embeddings — dict keyed by stringified layer index, to mirror Flax
        head_dim = config.head_dim
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )

        # Precompute RoPE tables (non-trainable, stored as buffers).
        # Flax uses rotary_seq_len = sequence_len * 10; mirror that so long
        # contexts + parity tests both work.
        rotary_seq_len = config.sequence_len * 10
        cos, sin = precompute_rotary_embeddings(
            rotary_seq_len, config.head_dim, base=config.rope_base
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        config = self.config

        cos = self.rope_cos[:, :T].to(dtype=self.wte.weight.dtype)
        sin = self.rope_sin[:, :T].to(dtype=self.wte.weight.dtype)

        x = self.wte(idx)
        x = rms_norm(x)

        # Smear: x_smeared[t] = x[t] + (smear_lambda * sigmoid(smear_gate(x[t,:24]))) * x[t-1],
        # for t >= 1; x[0] passes through.
        gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))  # (B, T-1, 1)
        x_smeared = x[:, 1:] + gate * x[:, :-1]
        x = torch.cat([x[:, :1], x_smeared], dim=1)

        x0 = x
        n_layer = config.n_layer
        backout_layer = n_layer // 2

        x_backout = None
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = str(i)
            ve = self.value_embeds[ve_key](idx).to(dtype=x.dtype) if ve_key in self.value_embeds else None
            x = block(x, ve, cos, sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        x = rms_norm(x)

        # Project to (padded) vocab, trim to real vocab, then softcap.
        softcap = 15.0
        if self.tie_embeddings:
            logits = x @ self.wte.weight.t()
        else:
            logits = self.lm_head(x)
        logits = logits[..., : config.vocab_size]
        logits = logits.to(torch.float32)
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path: str, map_location: str | torch.device = "cpu") -> "GELU_GPT":
        """Load a torch state dict + config produced by convert_flax_to_torch.py.

        The checkpoint file stores both config and state_dict as:
            {"config": {...}, "state_dict": {...}}
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(payload, dict) and "config" in payload and "state_dict" in payload:
            config = GPTConfig(**payload["config"])
            state_dict = payload["state_dict"]
        else:
            raise ValueError(
                f"{path} does not look like a converted flaxchat checkpoint; "
                "expected a dict with keys {'config', 'state_dict'}."
            )
        model = cls(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Buffers (rope_cos/sin) are not saved — they're recomputed in __init__.
        # Everything else must match.
        real_missing = [k for k in missing if not k.startswith("rope_")]
        if real_missing:
            raise RuntimeError(f"Missing keys when loading: {real_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading: {unexpected}")
        model.eval()
        return model


__all__ = ["GPTConfig", "GELU_GPT", "precompute_rotary_embeddings", "apply_rotary_emb", "rms_norm"]
