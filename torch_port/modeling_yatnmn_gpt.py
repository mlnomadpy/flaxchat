"""HuggingFace-compatible CausalLM wrapper for YatNMN-Softplus GPT, with KV cache.

Mirrors modeling_gelu_gpt.py one-for-one. The only difference is the inner module
(`Yat_GPT` instead of `GELU_GPT`) — KV-cache, smear handling, generation glue, and
the `(kv_list, last_embed)` past_key_values format are identical.
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
    from .configuration_yatnmn_gpt import YatGPTHfConfig
    from .yatnmn_gpt import Yat_GPT, YatGPTConfig
    from .torch_gpt import rms_norm, apply_rotary_emb
except ImportError:
    from configuration_yatnmn_gpt import YatGPTHfConfig
    from yatnmn_gpt import Yat_GPT, YatGPTConfig
    from torch_gpt import rms_norm, apply_rotary_emb


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
):
    cfg = attn_module.config
    B, T_new, _ = x_norm.shape
    n_head, n_kv_head, head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim

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

    k_bhtd = k.transpose(1, 2)
    v_bhtd = v.transpose(1, 2)
    q_bhtd = q.transpose(1, 2)

    if past_k is not None:
        k_bhtd = torch.cat([past_k, k_bhtd], dim=2)
        v_bhtd = torch.cat([past_v, v_bhtd], dim=2)
    new_k, new_v = k_bhtd, v_bhtd
    T_total = new_k.shape[2]

    if n_kv_head < n_head:
        repeats = n_head // n_kv_head
        k_bhtd = new_k.repeat_interleave(repeats, dim=1)
        v_bhtd = new_v.repeat_interleave(repeats, dim=1)
    else:
        k_bhtd, v_bhtd = new_k, new_v

    window_left = window_size[0]
    device = x_norm.device
    q_abs = torch.arange(T_total - T_new, T_total, device=device).unsqueeze(1)
    k_abs = torch.arange(T_total, device=device).unsqueeze(0)
    causal = k_abs <= q_abs
    if 0 < window_left < T_total:
        causal = causal & ((q_abs - k_abs) <= window_left)
    bias = torch.where(
        causal,
        torch.zeros((), dtype=x_norm.dtype, device=device),
        torch.full((), -1e9, dtype=x_norm.dtype, device=device),
    ).unsqueeze(0).unsqueeze(0)

    scale = 1.0 / math.sqrt(head_dim)
    att = torch.matmul(q_bhtd, k_bhtd.transpose(-2, -1)) * scale
    att = att + bias
    att = F.softmax(att, dim=-1)
    y = torch.matmul(att, v_bhtd)
    y = y.transpose(1, 2).contiguous().reshape(B, T_new, -1)
    return attn_module.c_proj(y), new_k, new_v


class YatGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = YatGPTHfConfig
    base_model_prefix = "yatnmn_gpt"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]
    _supports_cache_class = False
    _supports_static_cache = False

    def _supports_default_dynamic_cache(self):
        return False

    def __init__(self, config: YatGPTHfConfig):
        super().__init__(config)
        inner = YatGPTConfig(
            sequence_len=config.sequence_len, vocab_size=config.vocab_size,
            n_layer=config.n_layer, n_head=config.n_head, n_kv_head=config.n_kv_head,
            n_embd=config.n_embd, window_pattern=config.window_pattern,
            tie_embeddings=config.tie_embeddings, rope_base=config.rope_base,
            pad_vocab_size_to=config.pad_vocab_size_to, mlp_type=config.mlp_type,
            scalar_bias=config.scalar_bias, softplus_bias=config.softplus_bias,
            learnable_epsilon=config.learnable_epsilon,
            epsilon_init=config.epsilon_init, constant_alpha=config.constant_alpha,
        )
        self.inner_config = inner
        self.model = Yat_GPT(inner)
        self.post_init()

    def get_input_embeddings(self): return self.model.wte
    def set_input_embeddings(self, v): self.model.wte = v
    def can_generate(self): return True

    def _forward_full(self, input_ids):
        return self.model(input_ids)

    def _forward_with_cache(self, input_ids_new, past_key_values, prev_token_embed=None):
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

        if past_len == 0:
            if T_new >= 2:
                gate = m.smear_lambda * torch.sigmoid(m.smear_gate(x_new[:, 1:, :24]))
                x_smeared = x_new[:, 1:] + gate * x_new[:, :-1]
                x = torch.cat([x_new[:, :1], x_smeared], dim=1)
            else:
                x = x_new
        else:
            assert prev_token_embed is not None, "prev_token_embed required for smear with past"
            x_cat = torch.cat([prev_token_embed, x_new], dim=1)
            gate = m.smear_lambda * torch.sigmoid(m.smear_gate(x_cat[:, 1:, :24]))
            x = x_cat[:, 1:] + gate * x_cat[:, :-1]

        x0 = x
        backout_layer = cfg.n_layer // 2
        x_backout = None

        new_past = []
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
        logits = x @ m.wte.weight.t() if m.tie_embeddings else m.lm_head(x)
        logits = logits[..., : cfg.vocab_size].to(torch.float32)
        logits = softcap * torch.tanh(logits / softcap)
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
            raise ValueError("input_ids required")
        use_cache = bool(use_cache) if use_cache is not None else (past_key_values is not None)

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
            loss=loss, logits=logits, past_key_values=pkv,
            hidden_states=None, attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _get_cache_length(self, past_key_values):
        if past_key_values is None: return 0
        kv, _ = past_key_values
        if not kv: return 0
        return kv[0][0].shape[-2]


__all__ = ["YatGPTHfConfig", "YatGPTForCausalLM"]
