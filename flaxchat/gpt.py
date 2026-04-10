"""
GPT model in Flax NNX — faithful port of nanochat's architecture.

Notable features (all from nanochat):
- Rotary embeddings (RoPE), no learned positional embeddings
- QK norm with 1.2x scaling for sharper attention
- Untied weights for token embedding and lm_head
- ReLU^2 activation in MLP
- Norm after token embedding (parameterless RMSNorm)
- No bias in linear layers
- Group-Query Attention (GQA) support
- Value embeddings (ResFormer-style, alternating layers)
- Per-layer residual scaling (resid_lambdas) and input blending (x0_lambdas)
- Smear: cheap bigram-like token mixing
- Backout: subtract mid-layer residual to remove low-level features
- Logit soft-capping via tanh
- Sliding window attention pattern (SSSL)
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from jax.sharding import PartitionSpec as P

from flaxchat.config import GPTConfig
from flaxchat.common import COMPUTE_DTYPE, print0

# Compat: nnx.List/Dict exist in Flax 0.12+, plain list/dict work in 0.11
_NNX_LIST = getattr(nnx, 'List', list)
_NNX_DICT = getattr(nnx, 'Dict', dict)


def _maybe_shard(x, spec):
    """No-op — sharding is handled at the train step level."""
    return x


# ---------------------------------------------------------------------------
# RMSNorm (parameterless, like nanochat)
# ---------------------------------------------------------------------------
def rms_norm(x):
    """Parameterless RMS normalization, runs in compute dtype."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


# ---------------------------------------------------------------------------
# Rotary Embeddings
# ---------------------------------------------------------------------------
def precompute_rotary_embeddings(seq_len: int, head_dim: int, base: float = 100000.0):
    """Precompute cos/sin for RoPE. Returns (cos, sin) of shape (1, seq_len, 1, head_dim//2)."""
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(COMPUTE_DTYPE)
    sin = jnp.sin(freqs).astype(COMPUTE_DTYPE)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings. x: (B, T, H, D)."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=3)


# ---------------------------------------------------------------------------
# Helper: which layers have value embeddings
# ---------------------------------------------------------------------------
def has_ve(layer_idx: int, n_layer: int) -> bool:
    """True if layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        # Store config (registered as static) and scalar metadata
        self.config: GPTConfig = nnx.data(config)
        self.layer_idx: int = nnx.data(layer_idx)
        self._has_ve: bool = nnx.data(has_ve(layer_idx, config.n_layer))

        head_dim = config.n_embd // config.n_head
        self.c_q = nnx.Linear(config.n_embd, config.n_head * head_dim, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(config.n_embd, config.n_kv_head * head_dim, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(config.n_embd, config.n_kv_head * head_dim, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, use_bias=False, rngs=rngs)

        if self._has_ve:
            self.ve_gate = nnx.Linear(12, config.n_kv_head, use_bias=False, rngs=rngs)

    def __call__(self, x, ve, cos, sin, window_size):
        B, T, C = x.shape
        # Read from static config — these are Python ints, not traced values
        n_head = self.config.n_head
        n_kv_head = self.config.n_kv_head
        head_dim = self.config.n_embd // self.config.n_head

        q = self.c_q(x).reshape(B, T, n_head, head_dim)
        k = self.c_k(x).reshape(B, T, n_kv_head, head_dim)
        v = self.c_v(x).reshape(B, T, n_kv_head, head_dim)

        # Guard on _has_ve (Python bool) rather than `ve is not None` so the
        # scan path can always pass a (possibly zero) ve tensor without
        # triggering ve_gate lookups on layers that lack it.
        if self._has_ve and ve is not None:
            ve = ve.reshape(B, T, n_kv_head, head_dim)
            gate = 3.0 * jax.nn.sigmoid(self.ve_gate(x[..., :12]))
            v = v + gate[..., None] * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = rms_norm(q) * 1.2
        k = rms_norm(k) * 1.2

        # GQA: repeat K,V heads to match Q heads
        if n_kv_head < n_head:
            repeats = n_head // n_kv_head
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        # Build causal + sliding window mask
        window_left = window_size[0]
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        if window_left > 0 and window_left < T:
            row_idx = jnp.arange(T)[:, None]
            col_idx = jnp.arange(T)[None, :]
            window_mask = (row_idx - col_idx) <= window_left
            causal_mask = causal_mask & window_mask
        # (T, T) -> (1, 1, T, T) for broadcast
        bias = jnp.where(causal_mask, 0.0, -1e9)[None, None, :, :]

        # Use JAX's hardware-adaptive attention (auto-selects cuDNN/XLA kernels)
        # q,k,v: (B, T, H, D) — BTNH layout for dot_product_attention
        y = jax.nn.dot_product_attention(q, k, v, bias=bias, scale=1.0 / math.sqrt(head_dim))
        y = y.reshape(B, T, -1)
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------------------------
# MLP with ReLU^2
# ---------------------------------------------------------------------------
class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.relu(x) ** 2
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block (with optional gradient checkpointing)
# ---------------------------------------------------------------------------
class Block(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs,
                 use_remat: bool = False):
        self.attn = CausalSelfAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)
        # Store as plain Python bool — NOT nnx.data — to avoid JIT tracing issues
        self._use_remat = use_remat

    def _forward(self, x, ve, cos, sin, window_size):
        x = x + self.attn(rms_norm(x), ve, cos, sin, window_size)
        x = _maybe_shard(x, P('data', None, None))
        x = x + self.mlp(rms_norm(x))
        x = _maybe_shard(x, P('data', None, None))
        return x

    def __call__(self, x, ve, cos, sin, window_size):
        if self._use_remat:
            return nnx.remat(
                self._forward,
                policy=jax.checkpoint_policies.dots_saveable,
            )(self, x, ve, cos, sin, window_size)
        return self._forward(x, ve, cos, sin, window_size)


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------
class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs,
                 pad_vocab_size_to: int = 64, use_remat: bool = False):
        """
        Args:
            config: Model config
            rngs: Flax RNG streams
            pad_vocab_size_to: Pad vocab for tensor core efficiency
            use_remat: Enable gradient checkpointing (saves memory, slower)
        """
        # Store config and non-trainable data
        self.config: GPTConfig = nnx.data(config)
        window_sizes = self._compute_window_sizes(config)
        self.window_sizes: list = nnx.data(window_sizes)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size}")
        self.padded_vocab_size: int = nnx.data(padded_vocab_size)

        # Token embedding
        self.wte = nnx.Embed(padded_vocab_size, config.n_embd, rngs=rngs)

        # Transformer blocks
        self.blocks = _NNX_LIST([Block(config, i, rngs=rngs, use_remat=use_remat)
                                 for i in range(config.n_layer)])

        # Language model head (tied or untied)
        self._tie_embeddings = config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nnx.Linear(config.n_embd, padded_vocab_size, use_bias=False, rngs=rngs)
        else:
            self.lm_head = None  # use wte.embedding.T

        # Per-layer learnable scalars
        self.resid_lambdas = nnx.Param(jnp.ones(config.n_layer))
        self.x0_lambdas = nnx.Param(jnp.zeros(config.n_layer))

        # Smear
        self.smear_gate = nnx.Linear(24, 1, use_bias=False, rngs=rngs)
        self.smear_lambda = nnx.Param(jnp.zeros(1))

        # Backout
        self.backout_lambda = nnx.Param(0.2 * jnp.ones(1))

        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = _NNX_DICT({str(i): nnx.Embed(padded_vocab_size, kv_dim, rngs=rngs)
                                       for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # Precompute rotary embeddings (stored as non-node data)
        rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary_embeddings(rotary_seq_len, head_dim)
        self.rope_cos: jax.Array = nnx.data(cos)
        self.rope_sin: jax.Array = nnx.data(sin)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights matching nanochat exactly."""
        config = self.config
        n_embd = config.n_embd
        n_layer = config.n_layer

        # Embedding: normal(0, 0.8)
        self.wte.embedding[...] = jax.random.normal(
            jax.random.key(0), self.wte.embedding[...].shape
        ) * 0.8
        self.wte.embedding[...] = self.wte.embedding[...].astype(COMPUTE_DTYPE)

        # lm_head: normal(0, 0.001) — skip if tied
        if self.lm_head is not None:
            self.lm_head.kernel[...] = jax.random.normal(
                jax.random.key(1), self.lm_head.kernel[...].shape
            ) * 0.001

        # Transformer blocks: uniform init
        s = 3**0.5 * n_embd**-0.5
        for i, block in enumerate(self.blocks):
            key = jax.random.key(100 + i)
            keys = jax.random.split(key, 6)

            block.attn.c_q.kernel[...] = jax.random.uniform(keys[0], block.attn.c_q.kernel[...].shape, minval=-s, maxval=s)
            block.attn.c_k.kernel[...] = jax.random.uniform(keys[1], block.attn.c_k.kernel[...].shape, minval=-s, maxval=s)
            block.attn.c_v.kernel[...] = jax.random.uniform(keys[2], block.attn.c_v.kernel[...].shape, minval=-s, maxval=s)

            block.attn.c_proj.kernel[...] = jnp.zeros_like(block.attn.c_proj.kernel[...])
            block.mlp.c_proj.kernel[...] = jnp.zeros_like(block.mlp.c_proj.kernel[...])

            block.mlp.c_fc.kernel[...] = jax.random.uniform(keys[3], block.mlp.c_fc.kernel[...].shape, minval=-s*0.4, maxval=s*0.4)

            if block.attn._has_ve:
                block.attn.ve_gate.kernel[...] = jax.random.uniform(keys[4], block.attn.ve_gate.kernel[...].shape, minval=0.0, maxval=0.02)

        # Per-layer scalar init
        resid_vals = jnp.array([1.15 - (0.10 * i / max(n_layer - 1, 1)) for i in range(n_layer)])
        x0_vals = jnp.array([0.20 - (0.15 * i / max(n_layer - 1, 1)) for i in range(n_layer)])
        self.resid_lambdas[...] = resid_vals
        self.x0_lambdas[...] = x0_vals

        # Value embeddings: uniform like c_v
        for key_str, ve in self.value_embeds.items():
            i = int(key_str)
            ve.embedding[...] = jax.random.uniform(
                jax.random.key(200 + i), ve.embedding[...].shape, minval=-s, maxval=s
            )
            ve.embedding[...] = ve.embedding[...].astype(COMPUTE_DTYPE)

    @staticmethod
    def _compute_window_sizes(config: GPTConfig):
        """Compute per-layer (left, right) window sizes for sliding window attention."""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128

        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _forward_loop(self, x, x0, idx, cos, sin, n_layer, backout_layer):
        """Run transformer blocks in a Python loop (default path)."""
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[...][i] * x + self.x0_lambdas[...][i] * x0
            ve_key = str(i)
            ve = self.value_embeds[ve_key](idx).astype(x.dtype) if ve_key in self.value_embeds else None
            x = block(x, ve, cos, sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x
        return x, x_backout

    def _forward_scan(self, x, x0, idx, cos, sin, n_layer, backout_layer):
        """Scan-based forward pass: O(1) XLA compile time, constant memory.

        Uses jax.lax.scan over layer indices. Per-layer data (lambdas, window
        sizes, value embeddings) are pre-stacked into arrays and indexed
        dynamically. Blocks are dispatched via jax.lax.switch so XLA only
        needs to compile a single block body.
        """
        resid_lambdas = self.resid_lambdas[...]
        x0_lambdas = self.x0_lambdas[...]

        # Stack window sizes: (n_layer, 2)
        window_sizes = jnp.array(self.window_sizes, dtype=jnp.int32)

        # Precompute ALL value embeddings stacked: (n_layer, B, T, kv_dim).
        # Non-VE layers get zeros; the block's _has_ve guard skips them.
        head_dim = self.config.n_embd // self.config.n_head
        kv_dim = self.config.n_kv_head * head_dim
        B, T = idx.shape
        ve_list = []
        for i in range(n_layer):
            ve_key = str(i)
            if ve_key in self.value_embeds:
                ve_list.append(self.value_embeds[ve_key](idx).astype(x.dtype))
            else:
                ve_list.append(jnp.zeros((B, T, kv_dim), dtype=x.dtype))
        ve_all = jnp.stack(ve_list, axis=0)  # (n_layer, B, T, kv_dim)

        # Build per-block dispatch functions for jax.lax.switch.
        # Each branch closes over its own Block module and window_size.
        branches = []
        for i in range(n_layer):
            block = self.blocks[i]
            ws = self.window_sizes[i]
            # Capture block and ws by value via default arg
            branches.append(
                lambda x_, ve_, cos_=cos, sin_=sin, b=block, w=ws: b(x_, ve_, cos_, sin_, w)
            )

        def scan_body(carry, layer_idx):
            x_c, x0_c, x_backout_c = carry

            # Per-layer residual scaling
            rl = jax.lax.dynamic_index_in_dim(resid_lambdas, layer_idx, keepdims=False)
            xl = jax.lax.dynamic_index_in_dim(x0_lambdas, layer_idx, keepdims=False)
            x_c = rl * x_c + xl * x0_c

            # Value embedding for this layer
            ve_i = jax.lax.dynamic_index_in_dim(ve_all, layer_idx, axis=0, keepdims=False)

            # Dispatch to the correct block via lax.switch
            x_c = jax.lax.switch(layer_idx, branches, x_c, ve_i)

            # Capture backout at the right layer
            x_backout_c = jnp.where(
                layer_idx == backout_layer, x_c, x_backout_c
            )
            return (x_c, x0_c, x_backout_c), None

        x_backout_init = jnp.zeros_like(x)
        init_carry = (x, x0, x_backout_init)
        layer_indices = jnp.arange(n_layer, dtype=jnp.int32)

        (x, _, x_backout), _ = jax.lax.scan(scan_body, init_carry, layer_indices)
        return x, x_backout

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        config = self.config

        cos = self.rope_cos[:, :T]
        sin = self.rope_sin[:, :T]

        x = self.wte(idx)
        x = x.astype(COMPUTE_DTYPE)
        x = rms_norm(x)
        x = _maybe_shard(x, P('data', None, None))

        # Smear
        gate = self.smear_lambda[...].astype(x.dtype) * jax.nn.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x_smeared = x[:, 1:] + gate * x[:, :-1]
        x = jnp.concatenate([x[:, :1], x_smeared], axis=1)

        x0 = x
        n_layer = config.n_layer
        backout_layer = n_layer // 2

        if config.use_scan:
            x, x_backout = self._forward_scan(x, x0, idx, cos, sin, n_layer, backout_layer)
        else:
            x, x_backout = self._forward_loop(x, x0, idx, cos, sin, n_layer, backout_layer)

        if x_backout is not None:
            x = x - self.backout_lambda[...].astype(x.dtype) * x_backout
        x = rms_norm(x)

        softcap = 15.0
        if self._tie_embeddings:
            logits = x @ self.wte.embedding[...].T
        else:
            logits = self.lm_head(x)
        logits = logits[..., :config.vocab_size]
        logits = logits.astype(jnp.float32)
        logits = softcap * jnp.tanh(logits / softcap)

        if targets is not None:
            one_hot = jax.nn.one_hot(targets, config.vocab_size)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            mask = (targets >= 0).astype(jnp.float32)
            loss = -jnp.sum(one_hot * log_probs, axis=-1)
            loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            return loss
        else:
            return logits

    def estimate_flops(self) -> int:
        config = self.config
        all_params = jax.tree.leaves(nnx.state(self, nnx.Param))
        nparams = sum(p.size for p in all_params)

        nparams_exclude = (
            self.wte.embedding[...].size +
            sum(ve.embedding[...].size for ve in self.value_embeds.values()) +
            self.resid_lambdas[...].size +
            self.x0_lambdas[...].size +
            self.smear_gate.kernel[...].size +
            self.smear_lambda[...].size +
            self.backout_lambda[...].size
        )

        h = config.n_head
        q = config.n_embd // config.n_head
        t = config.sequence_len

        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_params(self) -> int:
        all_params = jax.tree.leaves(nnx.state(self, nnx.Param))
        return sum(p.size for p in all_params)
