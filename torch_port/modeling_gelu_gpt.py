"""HuggingFace-compatible CausalLM wrapper around GELU_GPT with KV cache.

Adds:
  * GeluGPTForCausalLM — subclass of PreTrainedModel (+ GenerationMixin)
  * KV cache for fast autoregressive generation
  * Compatible with AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)

The inner `GELU_GPT` module in torch_gpt.py is left untouched; this module
layers a cache-aware forward path on top by re-implementing the attention
block in generation mode. Non-generation (full-sequence) forward simply
delegates to the plain `GELU_GPT`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

try:
    from .configuration_gelu_gpt import GeluGPTConfig
    from .torch_gpt import GELU_GPT, GPTConfig, rms_norm, apply_rotary_emb, has_ve, compute_window_sizes
except ImportError:  # when loaded as flat files via trust_remote_code
    from configuration_gelu_gpt import GeluGPTConfig
    from torch_gpt import GELU_GPT, GPTConfig, rms_norm, apply_rotary_emb, has_ve, compute_window_sizes


def _kvcache_attn(
    attn_module: nn.Module,
    x_norm: torch.Tensor,
    ve: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    window_size: Tuple[int, int],
    past_k: Optional[torch.Tensor],
    past_v: Optional[torch.Tensor],
    input_raw_for_ve_gate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached-KV attention matching `CausalSelfAttention.forward` exactly.

    Returns (y, new_k, new_v). `new_k/v` are the FULL accumulated K/V after
    post-RoPE + QK-norm, shape (B, H_kv, T_total, D).
    """
    cfg = attn_module.config
    B, T_new, C = x_norm.shape
    n_head = cfg.n_head
    n_kv_head = cfg.n_kv_head
    head_dim = cfg.head_dim

    q = attn_module.c_q(x_norm).reshape(B, T_new, n_head, head_dim)
    k = attn_module.c_k(x_norm).reshape(B, T_new, n_kv_head, head_dim)
    v = attn_module.c_v(x_norm).reshape(B, T_new, n_kv_head, head_dim)

    if attn_module._has_ve and ve is not None:
        ve_r = ve.reshape(B, T_new, n_kv_head, head_dim)
        gate = 3.0 * torch.sigmoid(attn_module.ve_gate(input_raw_for_ve_gate[..., :12]))
        v = v + gate.unsqueeze(-1) * ve_r

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q = rms_norm(q) * 1.2
    k = rms_norm(k) * 1.2

    # Transpose to (B, H, T, D) so KV cat happens on time axis.
    k_bhtd = k.transpose(1, 2)  # (B, Hk, T_new, D)
    v_bhtd = v.transpose(1, 2)
    q_bhtd = q.transpose(1, 2)  # (B, Hq, T_new, D)

    if past_k is not None:
        k_bhtd = torch.cat([past_k, k_bhtd], dim=2)
        v_bhtd = torch.cat([past_v, v_bhtd], dim=2)
    new_k = k_bhtd
    new_v = v_bhtd
    T_total = new_k.shape[2]

    # GQA repeat.
    if n_kv_head < n_head:
        repeats = n_head // n_kv_head
        k_bhtd = new_k.repeat_interleave(repeats, dim=1)
        v_bhtd = new_v.repeat_interleave(repeats, dim=1)
    else:
        k_bhtd = new_k
        v_bhtd = new_v

    # Mask: for each query position q_pos (absolute = T_total - T_new + i),
    # allow keys j in [max(0, q_pos - window_left), q_pos]. Causal already
    # satisfied by j <= q_pos.
    window_left = window_size[0]
    device = x_norm.device
    q_abs = torch.arange(T_total - T_new, T_total, device=device).unsqueeze(1)  # (T_new, 1)
    k_abs = torch.arange(T_total, device=device).unsqueeze(0)                    # (1, T_total)
    causal = k_abs <= q_abs
    if 0 < window_left < T_total:
        causal = causal & ((q_abs - k_abs) <= window_left)
    bias = torch.where(
        causal,
        torch.zeros((), dtype=x_norm.dtype, device=device),
        torch.full((), -1e9, dtype=x_norm.dtype, device=device),
    )
    bias = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, T_new, T_total)

    scale = 1.0 / math.sqrt(head_dim)
    att = torch.matmul(q_bhtd, k_bhtd.transpose(-2, -1)) * scale
    att = att + bias
    att = F.softmax(att, dim=-1)
    y = torch.matmul(att, v_bhtd)
    y = y.transpose(1, 2).contiguous().reshape(B, T_new, -1)
    y = attn_module.c_proj(y)
    return y, new_k, new_v


class GeluGPTForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible wrapper around `GELU_GPT`.

    - `forward()` accepts standard HF args (`input_ids`, `labels`,
      `past_key_values`, `use_cache`, ...) and returns `CausalLMOutputWithPast`.
    - `past_key_values` is a tuple of per-layer `(k, v)` tensors, each
      shaped `(B, H_kv, T, D)` with post-RoPE + QK-norm applied.
    """

    config_class = GeluGPTConfig
    base_model_prefix = "gelu_gpt"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]
    _supports_cache_class = False
    _supports_static_cache = False

    # HF 5.x treats _supports_default_dynamic_cache as a method. Return False
    # so generate() leaves our past_key_values alone (we manage a tuple
    # (kv_list, last_embed) internally).
    def _supports_default_dynamic_cache(self):  # type: ignore[override]
        return False

    def __init__(self, config: GeluGPTConfig):
        super().__init__(config)
        inner_cfg = GPTConfig(
            sequence_len=config.sequence_len,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_embd=config.n_embd,
            window_pattern=config.window_pattern,
            tie_embeddings=config.tie_embeddings,
            rope_base=config.rope_base,
            pad_vocab_size_to=config.pad_vocab_size_to,
            mlp=config.mlp,
        )
        self.inner_config = inner_cfg
        self.model = GELU_GPT(inner_cfg)
        self.post_init()

    # HF plumbing -------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.wte

    def set_input_embeddings(self, v):
        self.model.wte = v

    def can_generate(self):  # noqa: D401
        return True

    # Forward -----------------------------------------------------------
    def _forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def _forward_with_cache(
        self,
        input_ids_new: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        prev_token_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        m = self.model
        cfg = m.config
        B, T_new = input_ids_new.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        T_total = past_len + T_new

        cos_full = m.rope_cos.to(dtype=m.wte.weight.dtype)
        sin_full = m.rope_sin.to(dtype=m.wte.weight.dtype)
        cos = cos_full[:, past_len:T_total]
        sin = sin_full[:, past_len:T_total]

        x_new = rms_norm(m.wte(input_ids_new))

        # Smear: for position p (absolute), if p>=1 we mix in embedding at p-1.
        # When past_len == 0, the first new token has no prev → skipped, as in
        # full forward. When past_len > 0, the first new token needs the
        # embedding of the token at index past_len-1 (passed in as
        # prev_token_embed).
        if past_len == 0:
            if T_new >= 2:
                gate = m.smear_lambda * torch.sigmoid(m.smear_gate(x_new[:, 1:, :24]))
                x_smeared = x_new[:, 1:] + gate * x_new[:, :-1]
                x = torch.cat([x_new[:, :1], x_smeared], dim=1)
            else:
                x = x_new
        else:
            assert prev_token_embed is not None, "prev_token_embed required for smear with past"
            # concat [prev, new] so smear applies correctly to the new tokens
            x_cat = torch.cat([prev_token_embed, x_new], dim=1)
            gate = m.smear_lambda * torch.sigmoid(m.smear_gate(x_cat[:, 1:, :24]))
            x = x_cat[:, 1:] + gate * x_cat[:, :-1]

        x0 = x
        backout_layer = cfg.n_layer // 2
        x_backout = None

        new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(m.blocks):
            x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0
            ve_key = str(i)
            ve = m.value_embeds[ve_key](input_ids_new).to(dtype=x.dtype) if ve_key in m.value_embeds else None

            past_k = past_key_values[i][0] if past_key_values is not None else None
            past_v = past_key_values[i][1] if past_key_values is not None else None
            x_norm = rms_norm(x)
            attn_out, new_k, new_v = _kvcache_attn(
                block.attn, x_norm, ve, cos, sin,
                m.window_sizes[i], past_k, past_v,
                input_raw_for_ve_gate=x_norm,
            )
            new_past.append((new_k, new_v))
            x = x + attn_out
            x = x + block.mlp(rms_norm(x))

            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - m.backout_lambda * x_backout
        x = rms_norm(x)

        softcap = 15.0
        if m.tie_embeddings:
            logits = x @ m.wte.weight.t()
        else:
            logits = m.lm_head(x)
        logits = logits[..., : cfg.vocab_size]
        logits = logits.to(torch.float32)
        logits = softcap * torch.tanh(logits / softcap)
        # Also return the rms-normed embedding of the last NEW token so the
        # next call can apply smear correctly.
        last_embed = x_new[:, -1:, :]
        return logits, new_past, last_embed

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids is required")
        use_cache = bool(use_cache) if use_cache is not None else (past_key_values is not None)

        # past_key_values is our opaque tuple (kv_list, last_embed) when present.
        kv_list = None
        prev_embed = None
        if past_key_values is not None:
            kv_list, prev_embed = past_key_values

        if use_cache:
            logits, new_past, new_last_embed = self._forward_with_cache(
                input_ids, kv_list, prev_token_embed=prev_embed,
            )
            pkv = (tuple(new_past), new_last_embed)
        else:
            logits = self._forward_full(input_ids)
            pkv = None

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=pkv,
            hidden_states=None,
            attentions=None,
        )

    # prepare_inputs_for_generation — slice to last token when a KV cache
    # exists so we only recompute one column. The smear needs one prior
    # embedding, which is stashed in past_key_values[1].
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    # HF 5.x calls this during .generate() to see cache size. Our opaque cache
    # carries (kv_list, last_embed); len of new kv cache is kv[0][0].shape[-2].
    def _get_cache_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        kv_list, _ = past_key_values
        if kv_list is None or len(kv_list) == 0:
            return 0
        return kv_list[0][0].shape[-2]


__all__ = ["GeluGPTConfig", "GeluGPTForCausalLM"]
