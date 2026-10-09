"""ModernBERT-compatible bidirectional encoder and memory-bounded MLM head.

The supported checkpoint architecture is tied embeddings, GELU GLU, bias-free
blocks and zero dropout (including jhu-clsp/mmBERT-base). Packed documents use
nonnegative segment IDs; negative IDs are padding. No causal GPT code is changed.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
import math
from typing import cast

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from flaxchat.yat import squared_distance_bf16, softmax_bf16, adaptive_squared_distance_bf16


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int = 256000
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    max_position_embeddings: int = 8192
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 160000.0
    norm_eps: float = 1e-5
    pad_token_id: int = 0
    mask_token_id: int = 4
    compute_dtype: str = 'float32'
    residual_dtype: str = 'float32'
    attention_backend: str = 'xla'
    use_remat: bool = True
    loss_chunk_size: int = 128
    mlm_projection: str = 'dense'
    mlm_projection_capacity: int | None = None
    mlm_loss_backend: str = 'xla'
    mlm_vocab_tile: int = 1024
    ffn_type: str = 'geglu'
    yat_epsilon: float = 0.01
    yat_bias: float = 1.0
    yat_alpha_trainable: bool = True
    yat_alpha: float = 1.0
    yat_attention_alpha: float | None = None
    attention_score: str = 'dot_product'
    yat_compute_mode: str = 'mixed'
    yat_ffn_compute_mode: str | None = None
    yat_ffn_backward_block: int | None = None
    yat_local_shards: bool = False
    yat_attention_block_size: int = 0
    yat_global_attention_block_size: int = 0
    yat_softmax_backward: str = 'factored'
    yat_attention_implementation: str = 'standard'

    weight_quantization: str = 'none'

    def __post_init__(self):
        if self.attention_score in ('goat', 'goat_input') and self.yat_local_shards:
            raise ValueError('GOAT uses automatic data sharding; disable yat_local_shards')
        if self.weight_quantization not in ('none', 'int8_per_channel_ste'):
            raise ValueError('Unknown weight_quantization')
        if self.weight_quantization != 'none' and (
                self.ffn_type != 'yat_glu' or self.attention_score != 'yat_softmax'):
            raise ValueError('QAT currently requires the YAT embedding architecture')
        if self.yat_ffn_compute_mode is not None:
            if self.yat_ffn_compute_mode not in ('mixed', 'bf16', 'bf16_adaptive'):
                raise ValueError('Unknown yat_ffn_compute_mode')
            if self.ffn_type != 'yat_glu':
                raise ValueError('yat_ffn_compute_mode requires YAT FFN')
            if self.yat_ffn_compute_mode != 'mixed' and self.compute_dtype != 'bfloat16':
                raise ValueError('BF16 FFN requires bfloat16 compute dtype')
        if self.mlm_projection_capacity is not None:
            if type(self.mlm_projection_capacity) is not int or self.mlm_projection_capacity < 1:
                raise ValueError('mlm_projection_capacity must be a positive integer')
            if self.mlm_projection != 'masked':
                raise ValueError('mlm_projection_capacity requires masked projection')
        if self.yat_ffn_backward_block is not None:
            if type(self.yat_ffn_backward_block) is not int or self.yat_ffn_backward_block < 1:
                raise ValueError('yat_ffn_backward_block must be a positive integer')
            if self.ffn_type != 'yat_glu' or (self.yat_ffn_compute_mode or self.yat_compute_mode) != 'bf16':
                raise ValueError('yat_ffn_backward_block requires direct BF16 YAT FFN')
        if self.yat_attention_implementation not in ('standard', 'centered_fp32_scores'):
            raise ValueError('Unknown yat_attention_implementation')
        if self.yat_attention_implementation == 'centered_fp32_scores' and (
                self.attention_score not in ('yat_softmax', 'goat', 'goat_input') or self.yat_compute_mode != 'bf16'
                or self.yat_softmax_backward != 'factored'
                or type(self.yat_attention_block_size) is not int
                or type(self.yat_global_attention_block_size) is not int
                or self.yat_attention_block_size < 1 or self.yat_global_attention_block_size < 1):
            raise ValueError('centered_fp32_scores requires direct BF16 YAT, factored softmax, and positive local/global tiles')
        if self.yat_softmax_backward not in ('factored', 'max_centered'):
            raise ValueError('Unknown yat_softmax_backward')
        if self.yat_softmax_backward == 'max_centered' and (
                self.attention_score not in ('yat_softmax', 'goat', 'goat_input') or self.yat_compute_mode == 'mixed'):
            raise ValueError('max_centered requires BF16 YAT attention')
        for name in ("yat_attention_block_size", "yat_global_attention_block_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.yat_compute_mode not in ('mixed', 'bf16', 'bf16_adaptive'):
            raise ValueError('yat_compute_mode must be mixed, bf16 or bf16_adaptive')
        if self.yat_compute_mode in ('bf16', 'bf16_adaptive') and self.compute_dtype != 'bfloat16':
            raise ValueError('BF16 YAT requires bfloat16 compute dtype')
        if self.attention_score not in ('dot_product', 'yat_softmax', 'goat', 'goat_input'):
            raise ValueError('attention_score must be dot_product, yat_softmax, goat or goat_input')
        if self.attention_score in ('yat_softmax', 'goat', 'goat_input') and self.attention_backend != 'xla':
            raise ValueError('YAT attention currently requires the exact XLA reference backend')
        if self.ffn_type not in ('geglu', 'yat_glu'):
            raise ValueError('ffn_type must be geglu or yat_glu')
        if (self.ffn_type == 'yat_glu' or self.attention_score in ('yat_softmax', 'goat', 'goat_input')) and (self.yat_epsilon != .01 or self.yat_bias != 1.0 or self.yat_alpha_trainable is not True):
            raise ValueError('YAT GLU requires fixed bias 1, fixed epsilon 0.01 and trainable alpha')
        if not all(math.isfinite(v) and v > 0 for v in (self.yat_epsilon, self.yat_alpha)):
            raise ValueError('YAT epsilon and alpha must be finite and positive')
        if self.yat_attention_alpha is not None and (
                not math.isfinite(self.yat_attention_alpha) or self.yat_attention_alpha <= 0):
            raise ValueError('YAT attention alpha must be finite and positive')
        for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'max_position_embeddings', 'global_attn_every_n_layers',
                     'local_attention', 'loss_chunk_size'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.hidden_size % self.num_attention_heads or (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError('head dimension must be an even integer')
        if self.local_attention % 2:
            raise ValueError('local_attention must be even')
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.norm_eps, self.global_rope_theta, self.local_rope_theta)):
            raise ValueError('norm epsilon and RoPE bases must be finite and positive')
        if self.compute_dtype not in ('float32', 'bfloat16'):
            raise ValueError('compute_dtype must be float32 or bfloat16')
        if self.residual_dtype not in ('float32', 'bfloat16'):
            raise ValueError('residual_dtype must be float32 or bfloat16')
        if self.compute_dtype == 'float32' and self.residual_dtype != 'float32':
            raise ValueError('BF16 residuals require BF16 compute')
        if self.attention_backend not in ('xla', 'splash'):
            raise ValueError('attention_backend must be xla or splash')
        if self.mlm_projection not in ('dense', 'masked'):
            raise ValueError('mlm_projection must be dense or masked')
        if self.mlm_loss_backend not in ('xla', 'xla_full', 'xla_local', 'pallas'):
            raise ValueError('mlm_loss_backend must be xla, xla_full, xla_local or pallas')
        if self.mlm_vocab_tile <= 0 or self.mlm_vocab_tile % 128:
            raise ValueError('mlm_vocab_tile must be a positive multiple of 128')
        if self.mlm_loss_backend == 'pallas' and self.hidden_size % 128:
            raise ValueError('Pallas projection requires hidden size divisible by 128')
        if not 0 <= self.pad_token_id < self.vocab_size or not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError('special token IDs must be inside vocabulary')
        if self.pad_token_id == self.mask_token_id:
            raise ValueError('Padding and mask token IDs must differ')

    def projection_capacity(self, length: int) -> int:
        """Static per-row target capacity; overflow always retains the dense path."""
        if self.mlm_projection_capacity is not None:
            return min(length, self.mlm_projection_capacity)
        chunk = self.loss_chunk_size
        return min(length, ((length + 4 * chunk - 1) // (4 * chunk)) * chunk)

    @classmethod
    def from_hf(cls, config, **overrides):
        if config.get('model_type') != 'modernbert':
            raise ValueError('Only ModernBERT checkpoints are supported')
        expected = dict(tie_word_embeddings=True, hidden_activation='gelu',
                        attention_bias=False, mlp_bias=False, norm_bias=False,
                        decoder_bias=True, classifier_bias=False, classifier_activation='gelu',
                        attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0)
        for name, value in expected.items():
            if config.get(name, value) != value:
                raise ValueError(f'Unsupported ModernBERT setting: {name}')
        if config.get('rope_scaling') is not None:
            raise ValueError('Scaled RoPE is not supported')
        values = {f.name: config[f.name] for f in fields(cls) if f.name in config}
        if values.get('local_rope_theta') is None:
            values['local_rope_theta'] = config.get('global_rope_theta', 160000.)
        return cls(**(values | overrides))


jax.tree_util.register_static(EncoderConfig)


@lru_cache(maxsize=32)
def _splash_kernel(heads, length, radius):
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash, splash_attention_mask as masks,
    )
    mask = (masks.FullMask((length, length)) if radius is None else
            masks.LocalMask((length, length), window_size=(radius, radius), offset=0))
    return splash.make_splash_mha_single_device(masks.MultiHeadMask([mask] * heads))


def bidirectional_attention(q, k, v, segments, *, radius=None, backend='xla', packed=True,
                            score='dot_product', alpha: float | jax.Array = 1.0, yat_compute_mode='mixed',
                            yat_local_shards=False, yat_attention_block_size=0,
                            yat_softmax_backward='factored', yat_attention_implementation='standard'):
    """Symmetric attention with document isolation and zero padded outputs."""
    if yat_attention_implementation not in ('standard', 'centered_fp32_scores'):
        raise ValueError('Unknown yat_attention_implementation')
    if score in ('goat', 'goat_input'):
        if backend != 'xla' or yat_local_shards:
            raise ValueError('GOAT requires XLA with automatic data sharding')
        if yat_attention_implementation == 'centered_fp32_scores' and (
                yat_compute_mode != 'bf16' or yat_attention_block_size < 1
                or yat_softmax_backward != 'factored'):
            raise ValueError('GOAT centered scores require tiled direct BF16 and factored softmax')
        from flaxchat.yat_attention import windowed_yat_attention
        local_segments = segments if packed else jnp.where(segments >= 0, 0, -1)
        return windowed_yat_attention(
            q, k, v, local_segments, radius=radius, alpha=alpha,
            compute_mode=yat_compute_mode, block_size=yat_attention_block_size or 64,
            softmax_backward_mode=yat_softmax_backward, exclude_diagonal=True,
            precision='fp32_scores' if yat_attention_implementation == 'centered_fp32_scores' else 'native')
    if yat_attention_implementation not in ('standard', 'centered_fp32_scores'):
        raise ValueError('Unknown yat_attention_implementation')
    if yat_attention_implementation == 'centered_fp32_scores' and (
            score != 'yat_softmax' or yat_compute_mode != 'bf16'
            or backend != 'xla' or yat_attention_block_size < 1 or yat_softmax_backward != 'factored'):
        raise ValueError('centered_fp32_scores requires tiled direct BF16 YAT with XLA and factored softmax')
    if yat_softmax_backward not in ('factored', 'max_centered'):
        raise ValueError('Unknown yat_softmax_backward')
    if yat_softmax_backward == 'max_centered' and (
            score != 'yat_softmax' or yat_compute_mode == 'mixed'):
        raise ValueError('max_centered requires BF16 YAT attention')
    if yat_local_shards and score == 'yat_softmax' and jax.device_count() > 1:
        from jax.sharding import Mesh, PartitionSpec as P
        if q.shape[0] % jax.device_count():
            raise ValueError('Local YAT attention batch must divide across data devices')
        # Each repair predicate and loop belongs to one chip. Global dynamic
        # slices otherwise force the partitioner to handle cross-chip tiles.
        local = jax.shard_map(
            lambda qq, kk, vv, ss, aa: bidirectional_attention(qq, kk, vv, ss,
                radius=radius, backend=backend, packed=packed, score=score,
                alpha=aa, yat_compute_mode=yat_compute_mode, yat_attention_block_size=yat_attention_block_size,
                yat_softmax_backward=yat_softmax_backward,
                yat_attention_implementation=yat_attention_implementation),
            mesh=Mesh(np.asarray(jax.devices()), ('data',)),
            in_specs=(P('data'), P('data'), P('data'), P('data'), P()),
            out_specs=P('data'), check_vma=False)
        return local(q, k, v, segments, jnp.asarray(alpha))
    if score == 'yat_softmax' and yat_attention_block_size:
        if yat_attention_implementation == 'centered_fp32_scores':
            from flaxchat.yat_attention_centered import centered_yat_attention
            local_segments = segments if packed else jnp.where(segments >= 0, 0, -1)
            return centered_yat_attention(q, k, v, local_segments, radius=radius, alpha=alpha,
                                          block_size=yat_attention_block_size)
        if backend != 'xla':
            raise ValueError('Windowed YAT attention requires XLA backend')
        from flaxchat.yat_attention import windowed_yat_attention
        # Unpacked callers can supply arbitrary nonnegative document labels.
        local_segments = segments if packed else jnp.where(segments >= 0, 0, -1)
        return windowed_yat_attention(q, k, v, local_segments, radius=radius, alpha=alpha,
            compute_mode=yat_compute_mode, block_size=yat_attention_block_size,
            softmax_backward_mode=yat_softmax_backward)
    valid = segments >= 0
    if score == 'yat_softmax':
        if backend != 'xla':
            raise ValueError('YAT attention requires XLA reference backend')
        if yat_compute_mode not in ('mixed', 'bf16', 'bf16_adaptive'):
            raise ValueError('Unknown YAT compute mode')
        q, k, v = [cast(jax.Array, jnp.where(valid[:, :, None, None], a, 0)) for a in (q, k, v)]
        strict_bf16 = yat_compute_mode in ('bf16', 'bf16_adaptive')
        dtype = jnp.bfloat16 if strict_bf16 else jnp.float32
        dots = jnp.einsum('bqhd,bkhd->bhqk', q, k, preferred_element_type=dtype)
        if strict_bf16:
            qx, kx = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
            distance = (adaptive_squared_distance_bf16(qx, kx, dots) if yat_compute_mode == 'bf16_adaptive'
                        else squared_distance_bf16(qx, kx))
        else:
            qnorm = jnp.sum(jnp.square(q.astype(jnp.float32)), -1).transpose(0, 2, 1)
            knorm = jnp.sum(jnp.square(k.astype(jnp.float32)), -1).transpose(0, 2, 1)
            distance = jnp.maximum(qnorm[..., :, None] + knorm[..., None, :] - 2 * dots, 0.)
        logits = jnp.asarray(alpha, dtype=dtype) * jnp.square(dots + 1.) / (distance + .01)
        mask = valid[:, :, None] & valid[:, None, :]
        if packed:
            mask &= segments[:, :, None] == segments[:, None, :]
        if radius is not None:
            indices = jnp.arange(q.shape[1])
            mask &= jnp.abs(indices[:, None] - indices[None, :])[None] <= radius
        mask |= (~valid)[:, :, None] & jnp.eye(q.shape[1], dtype=bool)[None]
        masked_logits = jnp.where(mask[:, None], logits, jnp.asarray(-jnp.inf, dtype=dtype))
        weights = (softmax_bf16(masked_logits, backward_mode=yat_softmax_backward)
                   if strict_bf16 else jax.nn.softmax(masked_logits, axis=-1))
        out = jnp.einsum('bhqk,bkhd->bqhd', weights.astype(v.dtype), v,
                         preferred_element_type=dtype).astype(v.dtype)
        return jnp.where(valid[:, :, None, None], out, 0)
    if score != 'dot_product':
        raise ValueError('Unknown encoder attention score')
    if backend == 'splash':
        if jax.default_backend() != 'tpu' or q.shape[1] % 128:
            raise ValueError('Splash requires TPU and sequence length divisible by 128')
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
        kernel = _splash_kernel(q.shape[2], q.shape[1], radius)
        query, key, value = [jnp.transpose(a, (0, 2, 1, 3)) for a in (q, k, v)]
        out = jax.vmap(lambda a, b, c, s: kernel(a, b, c, splash.SegmentIds(s, s)))(
            query / math.sqrt(q.shape[-1]), key, value, segments)
        out = jnp.transpose(cast(jax.Array, out), (0, 2, 1, 3))
    elif backend == 'xla':
        if packed:
            same = segments[:, :, None] == segments[:, None, :]
            mask = same & valid[:, None, :]
            # Padded queries attend to themselves to avoid entirely masked softmax.
            mask = mask | ((~valid)[:, :, None] & jnp.eye(q.shape[1], dtype=bool)[None])
            mask = mask[:, None]
        else:
            # Broadcast a linear-size key mask; no [batch, length, length] allocation.
            fallback = (~valid.any(axis=1))[:, None] & (jnp.arange(q.shape[1])[None] == 0)
            mask = (valid | fallback)[:, None, None, :]
        out = jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=False,
            local_window_size=None if radius is None else (radius, radius), implementation='xla')
    else:
        raise ValueError('Unknown encoder attention backend')
    return jnp.where(valid[:, :, None, None], out, 0)


def _rope(x, positions, base):
    half = x.shape[-1] // 2
    freq = base ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angle = positions[..., None] * freq
    cos = jnp.concatenate([jnp.cos(angle)] * 2, -1)[:, :, None].astype(x.dtype)
    sin = jnp.concatenate([jnp.sin(angle)] * 2, -1)[:, :, None].astype(x.dtype)
    rotated = jnp.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin


def _yat_glu_epilogue(dots, gate, input_norm, weight_norm, alpha):
    distance = jnp.maximum(input_norm + weight_norm - 2 * dots, 0.)
    return alpha * jnp.square(dots + 1.0) / (distance + 0.01) * gate


_remat_yat_glu_epilogue = jax.checkpoint(_yat_glu_epilogue, prevent_cse=False)


def yat_glu(x, kernel, *, alpha: float | jax.Array = 1.0, compute_mode='mixed', local_shards=False,
            distance_backward_block=None):
    """YAT with fixed numerator bias 1 and epsilon .01, times a linear gate.

    Kernel layout is [input, 2 * intermediate], matching GeGLU. Low precision
    operands default to FP32 accumulation/geometry. The bf16 mode uses blocked
    direct distances and BF16 array arithmetic. bf16_adaptive reuses the dot
    with selective direct fallback. All modes return the operand dtype.
    Alpha may be a trainable scalar. This changes the model function and is not
    GELU-checkpoint numerical parity.
    """
    if distance_backward_block is not None and compute_mode != 'bf16':
        raise ValueError('distance_backward_block requires direct bf16 mode')
    if kernel.ndim != 2 or kernel.shape[0] != x.shape[-1] or kernel.shape[1] % 2:
        raise ValueError('YAT GLU requires an [input, 2 * intermediate] kernel')
    if local_shards and jax.device_count() > 1:
        from jax.sharding import Mesh, PartitionSpec as P
        if x.shape[0] % jax.device_count():
            raise ValueError('Local YAT FFN batch must divide across data devices')
        local = jax.shard_map(
            lambda xx, ww, aa: yat_glu(xx, ww, alpha=aa, compute_mode=compute_mode,
                                      distance_backward_block=distance_backward_block),
            mesh=Mesh(np.asarray(jax.devices()), ('data',)),
            in_specs=(P('data'), P(), P()), out_specs=P('data'), check_vma=False)
        return local(x, kernel, jnp.asarray(alpha))
    kernel = kernel.astype(x.dtype)
    if compute_mode in ('bf16', 'bf16_adaptive'):
        if x.dtype != jnp.bfloat16:
            raise ValueError('BF16 YAT requires BF16 inputs')
        dots, gate = jnp.split(jnp.matmul(x, kernel, preferred_element_type=jnp.bfloat16), 2, axis=-1)
        prototypes = kernel[:, :kernel.shape[1] // 2].T
        flat = x.reshape(-1, x.shape[-1])
        distance = (adaptive_squared_distance_bf16(flat, prototypes, dots.reshape(-1, dots.shape[-1]))
                    if compute_mode == 'bf16_adaptive' else squared_distance_bf16(
                        flat, prototypes, backward_block=distance_backward_block)).reshape(dots.shape)
        return jnp.asarray(alpha, dtype=jnp.bfloat16) * jnp.square(dots + 1.) / (distance + .01) * gate
    if compute_mode != 'mixed':
        raise ValueError('Unknown YAT compute mode')
    dots, gate = jnp.split(jnp.matmul(x, kernel, preferred_element_type=jnp.float32), 2, axis=-1)
    x32 = x.astype(jnp.float32)
    prototypes = kernel[:, :kernel.shape[1] // 2].astype(jnp.float32)
    # Recompute cheap elementwise geometry in backward instead of retaining
    # additional [tokens, intermediate] distance/reciprocal activations. The
    # combined feature+gate GEMM is outside this rematerialization boundary.
    return _remat_yat_glu_epilogue(
        dots, gate, jnp.sum(x32 * x32, axis=-1, keepdims=True),
        jnp.sum(prototypes * prototypes, axis=0), alpha).astype(x.dtype)


class EncoderBlock(nnx.Module):
    def __init__(self, config, index, *, rngs):
        self.config = config
        self.index = index
        dtype = getattr(jnp, config.compute_dtype)
        def linear(a, b, dtype=dtype):
            return nnx.Linear(a, b, use_bias=False, dtype=dtype, rngs=rngs,
                              kernel_init=nnx.initializers.normal(.02))
        def norm():
            return nnx.LayerNorm(config.hidden_size, epsilon=config.norm_eps,
                                 use_bias=False, dtype=getattr(jnp, config.residual_dtype), rngs=rngs)
        self.attn_norm = norm() if index else None
        self.mlp_norm = norm()
        if config.attention_score in ('goat', 'goat_input'):
            self.v = linear(config.hidden_size, config.hidden_size)
        else:
            self.qkv = linear(config.hidden_size, 3 * config.hidden_size)
        self.attn_out = linear(config.hidden_size, config.hidden_size)
        self.wi = linear(config.hidden_size, 2 * config.intermediate_size)
        self.wo = linear(config.intermediate_size, config.hidden_size)
        if config.ffn_type == 'yat_glu':
            self.yat_alpha = nnx.Param(jnp.array(config.yat_alpha, dtype=jnp.float32))
        if config.attention_score in ('yat_softmax', 'goat', 'goat_input'):
            initial_alpha = (config.yat_alpha if config.yat_attention_alpha is None
                             else config.yat_attention_alpha)
            self.yat_attention_alpha = nnx.Param(jnp.array(initial_alpha, dtype=jnp.float32))

    def __call__(self, x, segments, positions, *, packed=True):
        c = self.config
        h = self.attn_norm(x) if self.attn_norm is not None else x
        from flaxchat.weight_qat import qat_linear, fake_quantize_int8
        if c.attention_score in ('goat', 'goat_input'):
            v = qat_linear(self.v, h, c).reshape(*x.shape[:2], c.num_attention_heads, -1)
            q = k = v
        else:
            qkv = qat_linear(self.qkv, h, c).reshape(*x.shape[:2], 3, c.num_attention_heads, -1)
            q, k, v = (qkv[:, :, i] for i in range(3))
        local = self.index % c.global_attn_every_n_layers != 0
        base = c.local_rope_theta if local else c.global_rope_theta
        if c.attention_score in ('goat', 'goat_input'):
            # Legacy GOAT derives geometry from V; input GOAT deliberately
            # separates token geometry from its learned value projection.
            score_input = (h.astype(getattr(jnp, c.compute_dtype)).reshape(v.shape)
                           if c.attention_score == 'goat_input' else v)
            q = k = _rope(score_input, positions, base)
        else:
            q, k = _rope(q, positions, base), _rope(k, positions, base)
        h = bidirectional_attention(q, k, v, segments,
                                    radius=c.local_attention // 2 if local else None,
                                    backend=c.attention_backend, packed=packed,
                                    score=c.attention_score, yat_compute_mode=c.yat_compute_mode,
                                    yat_local_shards=c.yat_local_shards,
                                    yat_softmax_backward=c.yat_softmax_backward,
                                    yat_attention_implementation=c.yat_attention_implementation,
                                    yat_attention_block_size=c.yat_attention_block_size if local else c.yat_global_attention_block_size,
                                    alpha=self.yat_attention_alpha[...] if c.attention_score in ('yat_softmax', 'goat', 'goat_input') else 1.)
        x = x + qat_linear(self.attn_out, h.reshape(x.shape), c)
        h = self.mlp_norm(x)
        if c.ffn_type == 'yat_glu':
            wi = self.wi.kernel[...]
            if c.weight_quantization != 'none':
                wi = fake_quantize_int8(wi, 0)
            # The same reconstructed kernel supplies both projection and distance.
            h = yat_glu(h.astype(getattr(jnp, c.compute_dtype)), wi,
                        alpha=self.yat_alpha[...], compute_mode=c.yat_ffn_compute_mode or c.yat_compute_mode,
                        local_shards=c.yat_local_shards,
                        distance_backward_block=c.yat_ffn_backward_block)
        else:
            a, gate = jnp.split(self.wi(h), 2, axis=-1)
            h = jax.nn.gelu(a, approximate=False) * gate
        return x + qat_linear(self.wo, h, c)


class ModernBert(nnx.Module):
    """Encoder outputs, masked mean pooling, and tied MLM logits/loss."""
    def __init__(self, config: EncoderConfig, *, rngs):
        self.config = config
        # Set only on a device-local copy by the accumulation wrapper.
        self._local_projection_loss = False
        dtype = getattr(jnp, config.compute_dtype)
        # Gather FP32 master rows before casting activations. Casting the whole
        # table first also makes repeated-token gradient scatter-adds BF16.
        self.embedding = nnx.Embed(config.vocab_size, config.hidden_size, dtype=jnp.float32,
                                   embedding_init=nnx.initializers.normal(.02), rngs=rngs)
        def norm():
            return nnx.LayerNorm(config.hidden_size, epsilon=config.norm_eps,
                                 use_bias=False, dtype=getattr(jnp, config.residual_dtype), rngs=rngs)
        self.embedding_norm = norm()
        self.layers = nnx.List([EncoderBlock(config, i, rngs=rngs) for i in range(config.num_hidden_layers)])
        self.final_norm = norm()
        self.head_dense = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=False,
                                     dtype=dtype, rngs=rngs)
        self.head_norm = norm()
        self.decoder_bias = nnx.Param(jnp.zeros(config.vocab_size, jnp.float32))

    def encode(self, ids, *, segment_ids=None, position_ids=None):
        if ids.ndim != 2 or ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError('Expected [batch, sequence] within configured context length')
        segments = jnp.where(ids == self.config.pad_token_id, -1, 0) if segment_ids is None else segment_ids
        if segments.shape != ids.shape:
            raise ValueError('segment_ids must match input shape')
        positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape) if position_ids is None else position_ids
        if positions.shape != ids.shape:
            raise ValueError('position_ids must match input shape')
        embedded = self.embedding(ids)
        if self.config.weight_quantization != 'none':
            from flaxchat.weight_qat import fake_quantize_int8
            # Gather before quantizing: never materialize a quantized vocabulary.
            embedded = fake_quantize_int8(embedded, -1)
        x = self.embedding_norm(embedded.astype(getattr(jnp, self.config.residual_dtype)))
        for layer in self.layers:
            if self.config.use_remat:
                x = nnx.remat(lambda block, h, s, p: block(h, s, p, packed=segment_ids is not None))(
                    layer, x, segments, positions)
            else:
                x = layer(x, segments, positions, packed=segment_ids is not None)
        return jnp.where((segments >= 0)[..., None], self.final_norm(x), 0)

    def pool(self, ids, *, segment_ids=None):
        """One embedding per row; callers should provide one document per row."""
        h = self.encode(ids, segment_ids=segment_ids)
        valid = ids != self.config.pad_token_id if segment_ids is None else segment_ids >= 0
        return h.sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1)

    def prediction_features(self, hidden):
        """Share the BF16 head transform across loss backends and chunk sizes."""
        if self.config.weight_quantization != 'none':
            raise ValueError('Weight-only QAT currently supports embedding encode/pool, not MLM')
        return self.head_norm(jax.nn.gelu(self.head_dense(hidden), approximate=False))

    def decode(self, features):
        if self.config.weight_quantization != 'none':
            raise ValueError('Weight-only QAT does not support MLM decoding')
        dtype = getattr(jnp, self.config.compute_dtype)
        return jnp.matmul(features.astype(dtype), self.embedding.embedding[...].astype(dtype).T,
                          preferred_element_type=jnp.float32, precision=jax.lax.Precision.HIGHEST) + self.decoder_bias[...]

    def project(self, hidden):
        return self.decode(self.prediction_features(hidden))

    def __call__(self, ids, targets=None, *, segment_ids=None, position_ids=None, loss_reduction='mean'):
        if loss_reduction not in ('mean', 'sum'):
            raise ValueError('loss_reduction must be mean or sum')
        hidden = self.encode(ids, segment_ids=segment_ids, position_ids=position_ids)
        if targets is None:
            return self.project(hidden)
        if targets.shape != ids.shape:
            raise ValueError('MLM targets must be unshifted and match inputs')
        valid = ids != self.config.pad_token_id if segment_ids is None else segment_ids >= 0
        targets = cast(jax.Array, jnp.where(valid, targets, -1))
        if self.config.mlm_projection == 'masked':
            # Compact independently per row to preserve the batch sharding axis.
            # Static capacity is rounded to projection chunks for TPU matmuls.
            length = ids.shape[1]
            capacity = self.config.projection_capacity(length)
            def sparse(model, h, y):
                selected = y >= 0
                indices = jax.vmap(lambda row: jnp.nonzero(row, size=capacity, fill_value=0)[0])(selected)
                features = jnp.take_along_axis(h, indices[..., None], axis=1)
                labels = jnp.take_along_axis(y, indices, axis=1)
                occupied = jnp.arange(capacity)[None, :] < selected.sum(axis=1, keepdims=True)
                return model._projection_loss(features, jnp.where(occupied, labels, -1), reduction=loss_reduction)
            # Never truncate Bernoulli masks: overflow takes the full dense path.
            return nnx.cond(jnp.any((targets >= 0).sum(axis=1) > capacity),
                            lambda model, h, y: model._projection_loss(h, y, reduction=loss_reduction),
                            sparse, self, hidden, targets)
        return self._projection_loss(hidden, targets, reduction=loss_reduction)

    def _projection_loss(self, hidden, targets, *, reduction='mean'):
        # Transform once before chunking. Repeating the BF16 transformation in
        # every scan iteration changes gradient rounding and reduction order.
        hidden = self.prediction_features(hidden)
        if self.config.mlm_loss_backend in ('pallas', 'xla_full', 'xla_local'):
            from flaxchat.fused_cross_entropy import sharded_fused_loss
            dtype = getattr(jnp, self.config.compute_dtype)
            # Local XLA casts inside each rematerialized chunk, so the FP32
            # master-weight gradients accumulate without repeated BF16 rounding.
            local_xla = self.config.mlm_loss_backend == 'xla_local'
            return sharded_fused_loss(hidden if local_xla else hidden.astype(dtype),
                                     self.embedding.embedding[...] if local_xla else self.embedding.embedding[...].astype(dtype),
                                     self.decoder_bias[...], targets, tile=self.config.mlm_vocab_tile,
                                     backend=self.config.mlm_loss_backend, chunk_size=self.config.loss_chunk_size, compute_dtype=dtype,
                                     local_only=self._local_projection_loss, reduction=reduction)
        chunk = self.config.loss_chunk_size
        n = targets.size
        pad = (-n) % chunk
        features = jnp.pad(hidden.reshape(n, -1), ((0, pad), (0, 0))).reshape(-1, chunk, hidden.shape[-1])
        labels = jnp.pad(targets.reshape(-1), (0, pad), constant_values=-1).reshape(-1, chunk)
        # Rematerialize vocab projection: never retain [batch, length, vocab].
        @nnx.remat
        def loss_chunk(model, h, y):
            logits = model.decode(h)
            losses = jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, jnp.maximum(y, 0)[:, None], -1)[:, 0]
            return jnp.where(y >= 0, losses, 0).sum()
        @nnx.scan(in_axes=(None, nnx.Carry, 0, 0), out_axes=nnx.Carry)
        def accumulate(model, total, h, y):
            return total + loss_chunk(model, h, y)
        total = accumulate(self, jnp.float32(0), features, labels)
        return total if reduction == 'sum' else total / jnp.maximum((targets >= 0).sum(), 1)


def import_hf_weights(model, tensors):
    """Validate all names/shapes before mutation. Values are NumPy-like arrays."""
    if model.config.attention_score in ('goat', 'goat_input'):
        raise ValueError('GOAT requires explicit QKV-to-V migration; HF import is not an exact restore')
    mapping = {'model.embeddings.tok_embeddings.weight': (model.embedding.embedding, False),
               'model.embeddings.norm.weight': (model.embedding_norm.scale, False),
               'model.final_norm.weight': (model.final_norm.scale, False),
               'head.dense.weight': (model.head_dense.kernel, True),
               'head.norm.weight': (model.head_norm.scale, False),
               'decoder.bias': (model.decoder_bias, False)}
    for i, layer in enumerate(model.layers):
        prefix = f'model.layers.{i}.'
        for name, var, transpose in (
            ('attn.Wqkv.weight', layer.qkv.kernel, True), ('attn.Wo.weight', layer.attn_out.kernel, True),
            ('mlp.Wi.weight', layer.wi.kernel, True), ('mlp.Wo.weight', layer.wo.kernel, True),
            ('mlp_norm.weight', layer.mlp_norm.scale, False)):
            mapping[prefix + name] = (var, transpose)
        if layer.attn_norm is not None:
            mapping[prefix + 'attn_norm.weight'] = (layer.attn_norm.scale, False)
    extra = set(tensors) - set(mapping) - {'decoder.weight'}
    missing = set(mapping) - set(tensors)
    if extra or missing:
        raise ValueError(f'Checkpoint keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    if 'decoder.weight' in tensors and not np.array_equal(tensors['decoder.weight'], tensors['model.embeddings.tok_embeddings.weight']):
        raise ValueError('Decoder weights must be tied to embeddings')
    converted = []
    for name, (var, transpose) in mapping.items():
        value = np.asarray(tensors[name])
        value = value.T if transpose else value
        if value.shape != var.shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid checkpoint tensor: {name}')
        # Import into the existing target layout without an intermediate full
        # device allocation; the host snapshot is present on each process.
        target = var[...].sharding
        host_value = value.astype(var.dtype, copy=False)
        converted.append((var, jax.make_array_from_callback(
            value.shape, target, lambda index, source=host_value: source[index])))
    for var, value in converted:
        var[...] = value
