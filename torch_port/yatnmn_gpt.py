"""
PyTorch port of the flaxchat GPT model with YatNMN-Softplus MLP.

Matches nmn.nnx.layers.YatNMN forward for the config used in training:
  use_bias=True, softplus_bias=True, learnable_epsilon=True
  scalar_bias: False (per-neuron) OR True (shared (1,))
  constant_alpha: False (learnable) OR True (α=1 fixed)

YatNMN formula (see nmn/nnx/layers/nmn.py:291):
    y_dot   = x @ W                               # (..., out)
    dist²   = ||x||² + ||W_j||² - 2·y_dot         # (..., out)
    y_num   = y_dot + softplus(bias)              # if use_bias & softplus_bias
    out     = α · y_num² / (dist² + softplus(ε))

All other features (RoPE, GQA, QK-norm, RMSNorm, value embeds, smear, backout, softcap,
sliding-window, tied embeddings, no biases in Linear) match `torch_gpt.py` exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .torch_gpt import (
        rms_norm, precompute_rotary_embeddings, apply_rotary_emb,
        has_ve, compute_window_sizes,
    )
except ImportError:
    from torch_gpt import (  # type: ignore
        rms_norm, precompute_rotary_embeddings, apply_rotary_emb,
        has_ve, compute_window_sizes,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class YatGPTConfig:
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

    # YatNMN-specific
    mlp_type: str = "yatnmn-softplus"
    scalar_bias: bool = False       # False = per-neuron (ff,) bias; True = shared (1,)
    softplus_bias: bool = True
    learnable_epsilon: bool = True
    epsilon_init: float = 1e-3
    constant_alpha: bool = False    # False = learnable α; True = α fixed at 1

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def padded_vocab_size(self) -> int:
        v = self.vocab_size
        p = self.pad_vocab_size_to
        return ((v + p - 1) // p) * p


# ---------------------------------------------------------------------------
# YatNMN layer
# ---------------------------------------------------------------------------
class YatNMN(nn.Module):
    """PyTorch port of nmn.nnx.layers.YatNMN matching the flaxchat training config."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        softplus_bias: bool = True,
        scalar_bias: bool = False,
        learnable_epsilon: bool = True,
        epsilon_init: float = 1e-3,
        use_alpha: bool = True,
        constant_alpha: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.softplus_bias = softplus_bias
        self.scalar_bias = scalar_bias
        self.learnable_epsilon = learnable_epsilon
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        # kernel shape matches JAX nmn: (in_features, out_features)
        self.kernel = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.trunc_normal_(self.kernel, mean=0.0, std=1.0 / math.sqrt(in_features))

        if use_bias:
            b_shape = (1,) if scalar_bias else (out_features,)
            self.bias = nn.Parameter(torch.zeros(b_shape))
        else:
            self.register_parameter("bias", None)

        if learnable_epsilon:
            # softplus(x) = log(1+exp(x)); we want softplus(raw) = epsilon_init.
            # → raw = log(exp(epsilon_init) - 1) = log(expm1(epsilon_init))
            raw = math.log(math.expm1(epsilon_init))
            self.epsilon_param = nn.Parameter(torch.full((1,), raw))
            self._epsilon_const = None
        else:
            self.register_parameter("epsilon_param", None)
            self._epsilon_const = epsilon_init

        if use_alpha and not constant_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
            self.register_buffer("_alpha_const", torch.empty(0), persistent=False)
        elif use_alpha and constant_alpha:
            self.register_parameter("alpha", None)
            self.register_buffer("_alpha_const", torch.ones(1))  # fixed α=1
        else:
            self.register_parameter("alpha", None)
            self.register_buffer("_alpha_const", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match nmn's fp32 path: (y_num² / (dist + ε)) is unstable in bf16.
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        W = self.kernel.to(torch.float32)                            # (in, out)

        y_dot = x32 @ W                                              # (..., out)

        # ||x - W_j||² = ||x||² - 2·x·W_j + ||W_j||²
        x_sq = (x32 * x32).sum(dim=-1, keepdim=True)                 # (..., 1)
        W_sq = (W * W).sum(dim=0, keepdim=False)                     # (out,)
        distances = torch.clamp(x_sq + W_sq - 2.0 * y_dot, min=0.0)  # (..., out)

        # numerator
        if self.use_bias and self.bias is not None:
            b = self.bias.to(torch.float32)
            if self.softplus_bias:
                b = F.softplus(b)
            y_num = y_dot + b          # broadcast: b is (1,) or (out,)
        else:
            y_num = y_dot

        # epsilon
        if self.learnable_epsilon:
            eps = F.softplus(self.epsilon_param.to(torch.float32))
        else:
            eps = torch.tensor(self._epsilon_const, dtype=torch.float32, device=y_num.device)

        out = (y_num * y_num) / (distances + eps)

        if self.use_alpha:
            if self.alpha is not None:
                out = out * self.alpha.to(torch.float32)
            elif self._alpha_const.numel() > 0:
                out = out * self._alpha_const.to(torch.float32)

        return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Attention (shared with torch_gpt) — redefined here so this file is
# self-contained when loaded via trust_remote_code.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: YatGPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self._has_ve = has_ve(layer_idx, config.n_layer)

        head_dim = config.head_dim
        self.c_q = nn.Linear(config.n_embd, config.n_head * head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = nn.Linear(12, config.n_kv_head, bias=False) if self._has_ve else None

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, T, C = x.shape
        cfg = self.config
        n_head, n_kv_head, head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim

        q = self.c_q(x).reshape(B, T, n_head, head_dim)
        k = self.c_k(x).reshape(B, T, n_kv_head, head_dim)
        v = self.c_v(x).reshape(B, T, n_kv_head, head_dim)

        if self._has_ve and ve is not None:
            ve = ve.reshape(B, T, n_kv_head, head_dim)
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + gate.unsqueeze(-1) * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q) * 1.2
        k = rms_norm(k) * 1.2

        if n_kv_head < n_head:
            repeats = n_head // n_kv_head
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)

        window_left = window_size[0]
        device = x.device
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        causal_mask = row_idx >= col_idx
        if 0 < window_left < T:
            causal_mask = causal_mask & ((row_idx - col_idx) <= window_left)
        bias = torch.where(
            causal_mask,
            torch.zeros((), dtype=x.dtype, device=device),
            torch.full((), -1e9, dtype=x.dtype, device=device),
        ).unsqueeze(0).unsqueeze(0)

        q_bhtd = q.transpose(1, 2)
        k_bhtd = k.transpose(1, 2)
        v_bhtd = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        att = torch.matmul(q_bhtd, k_bhtd.transpose(-2, -1)) * scale
        att = att + bias
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, v_bhtd)
        y = y.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# MLP (YatNMN variant)
# ---------------------------------------------------------------------------
class YatMLP(nn.Module):
    def __init__(self, config: YatGPTConfig):
        super().__init__()
        n, ff = config.n_embd, 4 * config.n_embd
        self.c_fc = YatNMN(
            n, ff,
            use_bias=True,
            softplus_bias=config.softplus_bias,
            scalar_bias=config.scalar_bias,
            learnable_epsilon=config.learnable_epsilon,
            epsilon_init=config.epsilon_init,
            use_alpha=True,
            constant_alpha=config.constant_alpha,
        )
        self.c_proj = nn.Linear(ff, n, bias=False)
        # Training used zeros-init on c_proj (GPT._init_weights patched); init here matches.
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.c_fc(x))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: YatGPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = YatMLP(config)

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
class Yat_GPT(nn.Module):
    """PyTorch port of flaxchat.gpt.GPT with YatNMN-Softplus MLP.

    Layer/parameter naming mirrors the Flax module tree exactly so the
    weight converter is a direct key-for-key mapping.
    """

    def __init__(self, config: YatGPTConfig):
        super().__init__()
        self.config = config
        self.window_sizes = compute_window_sizes(_ConfigShim(config))
        padded_vocab = config.padded_vocab_size
        self.padded_vocab_size = padded_vocab

        self.wte = nn.Embedding(padded_vocab, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

        self.tie_embeddings = config.tie_embeddings
        self.lm_head = None if config.tie_embeddings else nn.Linear(config.n_embd, padded_vocab, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        head_dim = config.head_dim
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(padded_vocab, kv_dim)
             for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        )

        rotary_seq_len = config.sequence_len * 10
        cos, sin = precompute_rotary_embeddings(rotary_seq_len, config.head_dim, base=config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        cfg = self.config

        cos = self.rope_cos[:, :T].to(dtype=self.wte.weight.dtype)
        sin = self.rope_sin[:, :T].to(dtype=self.wte.weight.dtype)

        x = self.wte(idx)
        x = rms_norm(x)

        gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x_smeared = x[:, 1:] + gate * x[:, :-1]
        x = torch.cat([x[:, :1], x_smeared], dim=1)

        x0 = x
        n_layer = cfg.n_layer
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

        softcap = 15.0
        logits = x @ self.wte.weight.t() if self.tie_embeddings else self.lm_head(x)
        logits = logits[..., : cfg.vocab_size].to(torch.float32)
        return softcap * torch.tanh(logits / softcap)

    @classmethod
    def from_pretrained(cls, path: str, map_location: str | torch.device = "cpu") -> "Yat_GPT":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not (isinstance(payload, dict) and "config" in payload and "state_dict" in payload):
            raise ValueError(f"{path} must contain {{'config', 'state_dict'}}")
        config = YatGPTConfig(**payload["config"])
        model = cls(config)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        # rope_* are recomputed buffers; _alpha_const is a fixed buffer (not saved in checkpoint)
        real_missing = [k for k in missing if not k.startswith("rope_") and "_alpha_const" not in k]
        if real_missing:
            raise RuntimeError(f"Missing keys when loading: {real_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading: {unexpected}")
        model.eval()
        return model


class _ConfigShim:
    """Small shim so compute_window_sizes (which expects .sequence_len, .window_pattern,
    .n_layer on a GPTConfig) works when given a YatGPTConfig."""
    def __init__(self, cfg: YatGPTConfig):
        self.sequence_len = cfg.sequence_len
        self.window_pattern = cfg.window_pattern
        self.n_layer = cfg.n_layer


__all__ = ["YatGPTConfig", "Yat_GPT", "YatNMN"]
