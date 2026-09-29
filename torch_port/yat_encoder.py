"""PyTorch inference implementation of FlaxChat's released YAT encoder.

This module intentionally has no JAX dependency. It preserves BF16 feature and
distance arithmetic and FP32 residuals/scores from the step-12,000 checkpoint.
The dense distance fallback favors a fixed-shape TPU graph over fast inference;
it should be optimized only after accelerator parity is established.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


@dataclass(frozen=True)
class YatEncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    global_attn_every_n_layers: int
    local_attention: int
    global_rope_theta: float
    local_rope_theta: float
    norm_eps: float
    pad_token_id: int
    yat_epsilon: float
    yat_bias: float
    attention_score: str
    ffn_type: str
    compute_dtype: str
    residual_dtype: str
    yat_compute_mode: str
    yat_ffn_compute_mode: str
    yat_attention_implementation: str
    yat_attention_block_size: int
    yat_global_attention_block_size: int

    @classmethod
    def from_json(cls, path: str | Path) -> "YatEncoderConfig":
        data = json.loads(Path(path).read_text())
        config = cls(**{name: data[name] for name in cls.__dataclass_fields__})
        if (config.yat_epsilon != .01 or config.yat_bias != 1.
                or config.attention_score != "yat_softmax" or config.ffn_type != "yat_glu"
                or config.compute_dtype != "bfloat16" or config.residual_dtype != "float32"
                or config.yat_compute_mode != "bf16"
                or config.yat_ffn_compute_mode != "bf16_adaptive"
                or config.yat_attention_implementation != "centered_fp32_scores"):
            raise ValueError("This PyTorch implementation supports the released YAT configuration only")
        return config


def _linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x.to(torch.bfloat16), module.weight.to(torch.bfloat16))


def _norm(module: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), module.normalized_shape, module.weight.float(),
                        None, module.eps)


def _rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = positions.float().unsqueeze(-1) * freq
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1).unsqueeze(2).to(x.dtype)
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1).unsqueeze(2).to(x.dtype)
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def _direct_distance(x: torch.Tensor, y: torch.Tensor, block: int = 32) -> torch.Tensor:
    """BF16 pairwise squared distance with the JAX coordinate-block order."""
    shape = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]) + (x.shape[-2], y.shape[-2])
    result = torch.zeros(shape, device=x.device, dtype=torch.bfloat16)
    for start in range(0, x.shape[-1], block):
        delta = x[..., :, None, start:start + block] - y[..., None, :, start:start + block]
        result = result + (delta * delta).sum(dim=-1, dtype=torch.bfloat16)
    return result


def _adaptive_ffn_distance(x: torch.Tensor, prototypes: torch.Tensor,
                           dots: torch.Tensor) -> torch.Tensor:
    xnorm = (x * x).sum(dim=-1, dtype=torch.bfloat16)
    ynorm = (prototypes * prototypes).sum(dim=-1, dtype=torch.bfloat16)
    total = xnorm[:, None] + ynorm[None, :]
    raw = total - 2 * dots
    zero = (x == 0).all(dim=-1)[:, None] & (prototypes == 0).all(dim=-1)[None, :]
    sensitive = (raw <= total * .125) & ~zero
    fast = torch.where(zero, torch.zeros_like(raw), raw.clamp_min(0))
    # Fixed-shape fallback works on torch_xla; dynamic nonzero indexing does not.
    direct = _direct_distance(x, prototypes)
    return torch.where(sensitive, direct, fast)


class YatBlock(nn.Module):
    def __init__(self, config: YatEncoderConfig, index: int):
        super().__init__()
        width = config.hidden_size
        self.config = config
        self.index = index
        self.attn_norm = nn.LayerNorm(width, eps=config.norm_eps, elementwise_affine=True,
                                      bias=False) if index else None
        self.mlp_norm = nn.LayerNorm(width, eps=config.norm_eps, elementwise_affine=True,
                                     bias=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.wi = nn.Linear(width, 2 * config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, width, bias=False)
        self.yat_alpha = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.yat_attention_alpha = nn.Parameter(torch.ones((), dtype=torch.float32))

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   segments: torch.Tensor, radius: int | None) -> torch.Tensor:
        c = self.config
        batch, length, heads, width = q.shape
        valid = segments >= 0
        q, k, v = [torch.where(valid[:, :, None, None], a, 0) for a in (q, k, v)]
        tile = min(c.yat_attention_block_size if radius is not None
                   else c.yat_global_attention_block_size, length)
        outputs = []
        for start in range(0, length, tile):
            end = min(start + tile, length)
            low = max(0, start - radius) if radius is not None else 0
            high = min(length, end + radius) if radius is not None else length
            qq = q[:, start:end].permute(0, 2, 1, 3)
            kk = k[:, low:high].permute(0, 2, 1, 3)
            vv = v[:, low:high].permute(0, 2, 1, 3)
            dots = torch.matmul(qq, kk.transpose(-1, -2)).to(torch.bfloat16)
            distance = _direct_distance(qq, kk)
            scores = self.yat_attention_alpha * (dots.float() + 1).square() / (
                distance.float() + c.yat_epsilon)
            qs = segments[:, start:end]
            ks = segments[:, low:high]
            mask = (qs[:, :, None] == ks[:, None, :]) & (qs[:, :, None] >= 0) & (ks[:, None, :] >= 0)
            if radius is not None:
                qi = torch.arange(start, end, device=q.device)
                ki = torch.arange(low, high, device=q.device)
                mask = mask & ((qi[:, None] - ki[None, :]).abs() <= radius)[None]
            fallback = (qs < 0)[:, :, None] & (torch.arange(high-low, device=q.device) == 0)[None, None, :]
            mask = mask | fallback
            probs = torch.softmax(scores.masked_fill(~mask[:, None], -torch.inf), dim=-1)
            out = torch.matmul(probs.to(torch.bfloat16), vv).to(torch.bfloat16)
            out = out.permute(0, 2, 1, 3)
            outputs.append(torch.where((qs >= 0)[:, :, None, None], out, 0))
        return torch.cat(outputs, dim=1).reshape(batch, length, heads * width)

    def forward(self, x: torch.Tensor, segments: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        c = self.config
        h = _norm(self.attn_norm, x) if self.attn_norm is not None else x
        qkv = _linear(self.qkv, h).reshape(*x.shape[:2], 3, c.num_attention_heads, -1)
        q, k, v = qkv.unbind(dim=2)
        local = self.index % c.global_attn_every_n_layers != 0
        base = c.local_rope_theta if local else c.global_rope_theta
        q, k = _rope(q, positions, base), _rope(k, positions, base)
        radius = c.local_attention // 2 if local else None
        x = x + _linear(self.attn_out, self._attention(q, k, v, segments, radius)).float()
        h = _norm(self.mlp_norm, x).to(torch.bfloat16)
        dots, gate = _linear(self.wi, h).chunk(2, dim=-1)
        flat = h.reshape(-1, h.shape[-1])
        prototypes = self.wi.weight[:c.intermediate_size].to(torch.bfloat16)
        distance = _adaptive_ffn_distance(flat, prototypes, dots.reshape(-1, dots.shape[-1]))
        activated = self.yat_alpha.to(torch.bfloat16) * (dots + 1).square() / (
            distance.reshape(dots.shape) + c.yat_epsilon) * gate
        return x + _linear(self.wo, activated).float()


class YatTorchEncoder(nn.Module):
    def __init__(self, config: YatEncoderConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.embedding = nn.Embedding(config.vocab_size, width)
        self.embedding_norm = nn.LayerNorm(width, eps=config.norm_eps, bias=False)
        self.layers = nn.ModuleList(YatBlock(config, i) for i in range(config.num_hidden_layers))
        self.final_norm = nn.LayerNorm(width, eps=config.norm_eps, bias=False)
        self.head_dense = nn.Linear(width, width, bias=False)
        self.head_norm = nn.LayerNorm(width, eps=config.norm_eps, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, device: str | torch.device = "cpu") -> "YatTorchEncoder":
        directory = Path(directory)
        config = YatEncoderConfig.from_json(directory / "config.json")
        with torch.device("meta"):
            model = cls(config)
        state = load_file(str(directory / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    def encode(self, ids: torch.Tensor, *, segment_ids: torch.Tensor | None = None,
               position_ids: torch.Tensor | None = None) -> torch.Tensor:
        c = self.config
        if ids.ndim != 2 or ids.shape[1] > c.max_position_embeddings:
            raise ValueError("Expected [batch, sequence] within configured context length")
        segments = (ids != c.pad_token_id).long() - 1 if segment_ids is None else segment_ids
        if segments.shape != ids.shape:
            raise ValueError("segment_ids must match input shape")
        positions = (torch.arange(ids.shape[1], device=ids.device)[None, :].expand_as(ids)
                     if position_ids is None else position_ids)
        if positions.shape != ids.shape:
            raise ValueError("position_ids must match input shape")
        x = _norm(self.embedding_norm, self.embedding(ids))
        for layer in self.layers:
            x = layer(x, segments, positions)
        return torch.where((segments >= 0)[..., None], _norm(self.final_norm, x), 0)

    def pool(self, ids: torch.Tensor, *, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encode(ids, segment_ids=segment_ids)
        valid = ids != self.config.pad_token_id if segment_ids is None else segment_ids >= 0
        return hidden.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, ids: torch.Tensor, *, segment_ids: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(ids, segment_ids=segment_ids, position_ids=position_ids)
