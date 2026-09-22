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
    mlm_loss_backend: str = 'xla'
    mlm_vocab_tile: int = 1024

    def __post_init__(self):
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
        if self.mlm_loss_backend not in ('xla', 'xla_full', 'pallas'):
            raise ValueError('mlm_loss_backend must be xla, xla_full or pallas')
        if self.mlm_vocab_tile <= 0 or self.mlm_vocab_tile % 128:
            raise ValueError('mlm_vocab_tile must be a positive multiple of 128')
        if self.mlm_loss_backend == 'pallas' and self.hidden_size % 128:
            raise ValueError('Pallas projection requires hidden size divisible by 128')
        if not 0 <= self.pad_token_id < self.vocab_size or not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError('special token IDs must be inside vocabulary')
        if self.pad_token_id == self.mask_token_id:
            raise ValueError('Padding and mask token IDs must differ')

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


def bidirectional_attention(q, k, v, segments, *, radius=None, backend='xla', packed=True):
    """Symmetric attention with document isolation and zero padded outputs."""
    valid = segments >= 0
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
        self.qkv = linear(config.hidden_size, 3 * config.hidden_size)
        self.attn_out = linear(config.hidden_size, config.hidden_size)
        self.wi = linear(config.hidden_size, 2 * config.intermediate_size)
        self.wo = linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x, segments, positions, *, packed=True):
        c = self.config
        h = self.attn_norm(x) if self.attn_norm is not None else x
        qkv = self.qkv(h).reshape(*x.shape[:2], 3, c.num_attention_heads, -1)
        q, k, v = (qkv[:, :, i] for i in range(3))
        local = self.index % c.global_attn_every_n_layers != 0
        base = c.local_rope_theta if local else c.global_rope_theta
        q, k = _rope(q, positions, base), _rope(k, positions, base)
        h = bidirectional_attention(q, k, v, segments,
                                    radius=c.local_attention // 2 if local else None,
                                    backend=c.attention_backend, packed=packed)
        x = x + self.attn_out(h.reshape(x.shape))
        a, gate = jnp.split(self.wi(self.mlp_norm(x)), 2, axis=-1)
        return x + self.wo(jax.nn.gelu(a, approximate=False) * gate)


class ModernBert(nnx.Module):
    """Encoder outputs, masked mean pooling, and tied MLM logits/loss."""
    def __init__(self, config: EncoderConfig, *, rngs):
        self.config = config
        dtype = getattr(jnp, config.compute_dtype)
        self.embedding = nnx.Embed(config.vocab_size, config.hidden_size, dtype=getattr(jnp, config.residual_dtype),
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
        x = self.embedding_norm(self.embedding(ids))
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
        return self.head_norm(jax.nn.gelu(self.head_dense(hidden), approximate=False))

    def decode(self, features):
        dtype = getattr(jnp, self.config.compute_dtype)
        return (features.astype(dtype) @ self.embedding.embedding[...].astype(dtype).T).astype(jnp.float32) + self.decoder_bias[...]

    def project(self, hidden):
        return self.decode(self.prediction_features(hidden))

    def __call__(self, ids, targets=None, *, segment_ids=None, position_ids=None):
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
            chunk = self.config.loss_chunk_size
            capacity = min(length, math.ceil(length / (4 * chunk)) * chunk)
            def sparse(model, h, y):
                selected = y >= 0
                indices = jax.vmap(lambda row: jnp.nonzero(row, size=capacity, fill_value=0)[0])(selected)
                features = jnp.take_along_axis(h, indices[..., None], axis=1)
                labels = jnp.take_along_axis(y, indices, axis=1)
                occupied = jnp.arange(capacity)[None, :] < selected.sum(axis=1, keepdims=True)
                return model._projection_loss(features, jnp.where(occupied, labels, -1))
            # Never truncate Bernoulli masks: overflow takes the full dense path.
            return nnx.cond(jnp.any((targets >= 0).sum(axis=1) > capacity),
                            lambda model, h, y: model._projection_loss(h, y),
                            sparse, self, hidden, targets)
        return self._projection_loss(hidden, targets)

    def _projection_loss(self, hidden, targets):
        # Transform once before chunking. Repeating the BF16 transformation in
        # every scan iteration changes gradient rounding and reduction order.
        hidden = self.prediction_features(hidden)
        if self.config.mlm_loss_backend in ('pallas', 'xla_full'):
            from flaxchat.fused_cross_entropy import sharded_fused_loss
            dtype = getattr(jnp, self.config.compute_dtype)
            return sharded_fused_loss(hidden.astype(dtype), self.embedding.embedding[...].astype(dtype),
                                     self.decoder_bias[...], targets, tile=self.config.mlm_vocab_tile,
                                     backend=self.config.mlm_loss_backend)
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
        return total / jnp.maximum((targets >= 0).sum(), 1)


def import_hf_weights(model, tensors):
    """Validate all names/shapes before mutation. Values are NumPy-like arrays."""
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
        converted.append((var, jnp.asarray(value, dtype=var.dtype)))
    for var, value in converted:
        var[...] = value
