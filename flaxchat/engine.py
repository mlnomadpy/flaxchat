"""
Inference engine for autoregressive generation.

Three modes:
1. generate() — padded forward, simple, slow
2. generate_with_cache() — KV-cached with Python loop
3. generate_fast() — fully JIT'd jax.lax.while_loop sampler (fastest)

Engine class provides:
- generate() — streaming generator yielding (token_column, token_masks) per step
- generate_batch() — non-streaming, returns final token sequences with masks
- generate_text() — convenience method returning decoded text

All arrays placed on replicated sharding for multi-device compatibility.
"""

import math
from functools import partial
from collections import deque

import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.gpt import GPT, rms_norm, apply_rotary_emb, COMPUTE_DTYPE
from flaxchat.execution import execute_code


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------
def _get_replicated_sharding():
    try:
        from flaxchat.common import get_mesh
        from jax.sharding import NamedSharding, PartitionSpec as P
        return NamedSharding(get_mesh(), P())
    except (AssertionError, ImportError):
        return None


def _to_device(arr, sharding=None):
    if sharding is not None:
        return jax.device_put(arr, sharding)
    return arr


# ---------------------------------------------------------------------------
# Single-token forward with explicit KV cache arrays
# This is a pure function (no Python state) — compatible with while_loop
# ---------------------------------------------------------------------------
def _single_step_forward(model, token_id, pos, k_cache, v_cache, prev_emb):
    """
    Forward one token through the model using KV cache.

    Args:
        model: GPT model
        token_id: (1, 1) int32 — the new token
        pos: int scalar — current position in cache
        k_cache: (n_layers, 1, max_len, n_kv_head, head_dim) — key cache
        v_cache: same shape — value cache
        prev_emb: (1, 1, n_embd) — previous token embedding for smear

    Returns:
        logits: (1, vocab_size)
        k_cache: updated
        v_cache: updated
        new_prev_emb: (1, 1, n_embd)
    """
    config = model.config
    n_layer = config.n_layer
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // config.n_head
    max_len = k_cache.shape[2]

    # RoPE for this position — use dynamic_slice for JIT compatibility
    cos = jax.lax.dynamic_slice(model.rope_cos, (0, pos, 0, 0), (1, 1, 1, head_dim // 2))
    sin = jax.lax.dynamic_slice(model.rope_sin, (0, pos, 0, 0), (1, 1, 1, head_dim // 2))

    # Embed
    x = model.wte(token_id).astype(COMPUTE_DTYPE)  # (1, 1, n_embd)
    x = rms_norm(x)

    # Smear with previous embedding
    gate = model.smear_lambda[...].astype(x.dtype) * jax.nn.sigmoid(model.smear_gate(x[:, :, :24]))
    x = x + gate * prev_emb
    new_prev_emb = x

    x0 = x
    backout_layer = n_layer // 2
    x_backout = jnp.zeros_like(x)

    for i, block in enumerate(model.blocks):
        x = model.resid_lambdas[...][i] * x + model.x0_lambdas[...][i] * x0

        # Value embeddings
        ve_key = str(i)
        if ve_key in model.value_embeds:
            ve = model.value_embeds[ve_key](token_id).astype(x.dtype)
        else:
            ve = None

        attn = block.attn
        x_norm = rms_norm(x)

        q = attn.c_q(x_norm).reshape(1, 1, n_head, head_dim)
        k_new = attn.c_k(x_norm).reshape(1, 1, n_kv_head, head_dim)
        v_new = attn.c_v(x_norm).reshape(1, 1, n_kv_head, head_dim)

        if ve is not None:
            ve_r = ve.reshape(1, 1, n_kv_head, head_dim)
            gate_ve = 3.0 * jax.nn.sigmoid(attn.ve_gate(x_norm[..., :12]))
            v_new = v_new + gate_ve[..., None] * ve_r

        q = apply_rotary_emb(q, cos, sin)
        k_new = apply_rotary_emb(k_new, cos, sin)
        q = rms_norm(q) * 1.2
        k_new = rms_norm(k_new) * 1.2

        # Update cache at position `pos` using dynamic_update_slice (static shapes, no recompilation)
        k_cache = jax.lax.dynamic_update_slice(
            k_cache, k_new[None].astype(COMPUTE_DTYPE), (i, 0, pos, 0, 0)
        )
        v_cache = jax.lax.dynamic_update_slice(
            v_cache, v_new[None].astype(COMPUTE_DTYPE), (i, 0, pos, 0, 0)
        )

        # Attention: Q(1,1) x K(1,max_len) -> use full cache with position mask
        k_full = k_cache[i]  # (1, max_len, n_kv_head, head_dim)
        v_full = v_cache[i]

        if n_kv_head < n_head:
            k_full = jnp.repeat(k_full, n_head // n_kv_head, axis=2)
            v_full = jnp.repeat(v_full, n_head // n_kv_head, axis=2)

        q_t = jnp.transpose(q, (0, 2, 1, 3))       # (1, H, 1, D)
        k_t = jnp.transpose(k_full, (0, 2, 1, 3))   # (1, H, max_len, D)
        v_t = jnp.transpose(v_full, (0, 2, 1, 3))

        scale = 1.0 / math.sqrt(head_dim)
        w = jnp.matmul(q_t, jnp.transpose(k_t, (0, 1, 3, 2))) * scale  # (1, H, 1, max_len)

        # Causal mask: only attend to positions <= pos
        k_positions = jnp.arange(max_len)[None, None, None, :]
        mask = k_positions <= pos

        # Sliding window: respect per-layer window pattern
        window_left = model.window_sizes[i][0]
        if window_left > 0 and window_left < max_len:
            window_mask = (pos - k_positions) <= window_left
            mask = mask & window_mask

        w = jnp.where(mask, w, -1e9)
        w = jax.nn.softmax(w, axis=-1)

        y = jnp.matmul(w, v_t)  # (1, H, 1, D)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(1, 1, -1)
        y = attn.c_proj(y)

        x = x + y
        x = x + block.mlp(rms_norm(x))

        # Backout: use lax.cond to avoid Python if inside traced code
        x_backout = jnp.where(i == backout_layer, x, x_backout)

    x = x - model.backout_lambda[...].astype(x.dtype) * x_backout
    x = rms_norm(x)

    logits = model.lm_head(x)
    logits = logits[..., :config.vocab_size].astype(jnp.float32)
    logits = 15.0 * jnp.tanh(logits / 15.0)
    logits = logits[:, 0, :]  # (1, vocab_size)

    return logits, k_cache, v_cache, new_prev_emb


# ---------------------------------------------------------------------------
# Simple padded generation (fallback)
# ---------------------------------------------------------------------------
def generate(model, tokens, max_tokens=256, temperature=1.0, top_k=None, seed=42):
    """Padded generation — simple but slow."""
    sharding = _get_replicated_sharding()
    key = jax.random.key(seed)
    total_len = len(tokens) + max_tokens
    generated = list(tokens)
    for i in range(max_tokens):
        padded = generated + [0] * (total_len - len(generated))
        ids = _to_device(jnp.array([padded[:total_len]], dtype=jnp.int32), sharding)
        logits = model(ids)
        cur = logits[:, len(generated) - 1, :]
        if top_k is not None and top_k > 0:
            tkl, tki = jax.lax.top_k(cur, min(top_k, cur.shape[-1]))
            cur = jnp.full_like(cur, -1e9).at[0, tki[0]].set(tkl[0])
        if temperature > 0:
            key, sk = jax.random.split(key)
            nid = jax.random.categorical(sk, cur / temperature, axis=-1)
        else:
            nid = jnp.argmax(cur, axis=-1)
        generated.append(int(nid[0]))
    return generated


# ---------------------------------------------------------------------------
# Python-loop KV-cached generation (medium speed)
# ---------------------------------------------------------------------------
def generate_with_cache(model, tokens, max_tokens=256, temperature=1.0, top_k=40, seed=42):
    """KV-cached with Python loop. Faster than padded, slower than while_loop."""
    config = model.config
    sharding = _get_replicated_sharding()
    key = jax.random.key(seed)
    total_len = len(tokens) + max_tokens
    n_layer = config.n_layer
    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // config.n_head

    cache_shape = (n_layer, 1, total_len, n_kv_head, head_dim)
    k_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
    v_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
    prev_emb = _to_device(jnp.zeros((1, 1, config.n_embd), dtype=COMPUTE_DTYPE), sharding)

    generated = list(tokens)
    pos = jnp.int32(0)

    # Prefill
    for t in range(len(tokens)):
        tok = _to_device(jnp.array([[tokens[t]]], dtype=jnp.int32), sharding)
        logits, k_cache, v_cache, prev_emb = _single_step_forward(
            model, tok, pos, k_cache, v_cache, prev_emb
        )
        pos = pos + 1

    # Decode
    for _ in range(max_tokens):
        tkl, tki = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
        cur = jnp.full_like(logits, -1e9).at[0, tki[0]].set(tkl[0])
        if temperature > 0:
            key, sk = jax.random.split(key)
            nid = jax.random.categorical(sk, cur / temperature, axis=-1)
        else:
            nid = jnp.argmax(cur, axis=-1)
        token = int(nid[0])
        generated.append(token)

        tok = _to_device(jnp.array([[token]], dtype=jnp.int32), sharding)
        logits, k_cache, v_cache, prev_emb = _single_step_forward(
            model, tok, pos, k_cache, v_cache, prev_emb
        )
        pos = pos + 1

    return generated


# ---------------------------------------------------------------------------
# Fully JIT'd generation via jax.lax.while_loop (fastest)
# ---------------------------------------------------------------------------
def generate_fast(model, tokens, max_tokens=256, temperature=1.0, top_k=40,
                  seed=42, eos_token=0):
    """Fully JIT'd generation using jax.lax.while_loop. Fastest mode.

    All state is packed into a carry tuple so the entire decode loop compiles
    into a single XLA while. No Python-level control flow during decode.

    Args:
        model: GPT model (Flax NNX)
        tokens: list of int — prompt token ids
        max_tokens: maximum tokens to generate after the prompt
        temperature: sampling temperature (0 = greedy)
        top_k: top-k filtering
        seed: random seed
        eos_token: stop generation when this token is produced (default 0)

    Returns:
        list of int — prompt + generated tokens (up to prompt_len + max_tokens)
    """
    config = model.config
    sharding = _get_replicated_sharding()
    prompt_len = len(tokens)
    total_len = prompt_len + max_tokens
    n_layer = config.n_layer
    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // config.n_head

    # Pre-allocate fixed-size buffers
    tokens_buf = jnp.zeros((1, total_len), dtype=jnp.int32)
    tokens_buf = tokens_buf.at[0, :prompt_len].set(jnp.array(tokens, dtype=jnp.int32))
    tokens_buf = _to_device(tokens_buf, sharding)

    cache_shape = (n_layer, 1, total_len, n_kv_head, head_dim)
    k_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
    v_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
    prev_emb = _to_device(jnp.zeros((1, 1, config.n_embd), dtype=COMPUTE_DTYPE), sharding)

    rng_key = jax.random.key(seed)

    # --- Prefill: process prompt tokens one at a time via Python loop ---
    pos = jnp.int32(0)
    for t in range(prompt_len):
        tok = _to_device(jnp.array([[tokens[t]]], dtype=jnp.int32), sharding)
        logits, k_cache, v_cache, prev_emb = _single_step_forward(
            model, tok, pos, k_cache, v_cache, prev_emb
        )
        pos = pos + 1

    # --- Decode via while_loop ---
    def _sample(logits_2d, rng):
        """Top-k sampling, compatible with JAX tracing."""
        k = min(top_k, logits_2d.shape[-1])  # Python min — both are static
        vals, idx = jax.lax.top_k(logits_2d, k)

        def _greedy(_rng):
            return idx[0, jnp.argmax(vals[0])], _rng

        def _stochastic(_rng):
            _rng, sk = jax.random.split(_rng)
            choice = jax.random.categorical(sk, vals[0] / temperature)
            return idx[0, choice], _rng

        token_id, rng_out = jax.lax.cond(
            temperature <= 0.0, _greedy, _stochastic, rng
        )
        return token_id.astype(jnp.int32), rng_out

    # Sample the first decode token from prefill logits
    first_token, rng_key = _sample(logits, rng_key)
    tokens_buf = jax.lax.dynamic_update_slice(
        tokens_buf,
        first_token[None, None],
        (jnp.int32(0), pos),
    )
    first_done = (first_token == eos_token)

    # Carry: (pos, tokens_buf, k_cache, v_cache, prev_emb, rng_key, done)
    init_carry = (pos, tokens_buf, k_cache, v_cache, prev_emb, rng_key, first_done)
    max_pos = jnp.int32(total_len - 1)  # last writable position

    def cond_fn(carry):
        c_pos, _, _, _, _, _, c_done = carry
        return jnp.logical_and(c_pos < max_pos, jnp.logical_not(c_done))

    def body_fn(carry):
        c_pos, c_buf, c_k, c_v, c_prev, c_rng, c_done = carry

        # Read the token at c_pos (just written by previous iteration)
        tok = jax.lax.dynamic_slice(c_buf, (jnp.int32(0), c_pos), (1, 1))

        # Forward
        step_logits, c_k, c_v, c_prev = _single_step_forward(
            model, tok, c_pos, c_k, c_v, c_prev
        )
        new_pos = c_pos + 1

        # Sample
        new_token, c_rng = _sample(step_logits, c_rng)

        # Write token at new_pos
        c_buf = jax.lax.dynamic_update_slice(
            c_buf,
            new_token[None, None],
            (jnp.int32(0), new_pos),
        )

        # Check EOS
        c_done = jnp.logical_or(c_done, new_token == eos_token)

        return (new_pos, c_buf, c_k, c_v, c_prev, c_rng, c_done)

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    final_pos, final_buf, _, _, _, _, _ = final_carry

    # Extract tokens: prompt + generated (up to final_pos + 1)
    all_tokens = final_buf[0]  # (total_len,)
    length = jnp.minimum(final_pos + 1, total_len)
    return [int(all_tokens[i]) for i in range(int(length))]


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------
def _init_kv_cache(model, total_len):
    """Allocate empty KV cache and prev_emb for a model."""
    config = model.config
    sharding = _get_replicated_sharding()
    n_layer = config.n_layer
    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // config.n_head
    cache_shape = (n_layer, 1, total_len, n_kv_head, head_dim)
    k_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
    v_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
    prev_emb = _to_device(jnp.zeros((1, 1, config.n_embd), dtype=COMPUTE_DTYPE), sharding)
    return k_cache, v_cache, prev_emb


def _prefill(model, tokens, k_cache, v_cache, prev_emb):
    """Run prefill for a list of prompt tokens. Returns (logits, k, v, prev_emb, pos)."""
    sharding = _get_replicated_sharding()
    pos = jnp.int32(0)
    logits = None
    for t in range(len(tokens)):
        tok = _to_device(jnp.array([[tokens[t]]], dtype=jnp.int32), sharding)
        logits, k_cache, v_cache, prev_emb = _single_step_forward(
            model, tok, pos, k_cache, v_cache, prev_emb
        )
        pos = pos + 1
    return logits, k_cache, v_cache, prev_emb, pos


def generate_speculative(model, draft_model, tokens, max_tokens=256,
                         temperature=1.0, top_k=40, seed=42, draft_steps=4):
    """Speculative decoding: use a smaller draft model to propose tokens, verify with the main model.

    Algorithm:
        1. Draft model generates ``draft_steps`` tokens autoregressively.
        2. Main model verifies all draft tokens via sequential forward passes.
        3. Compare draft vs main model distributions at each position.
        4. Accept tokens where draft distribution matches main (rejection sampling).
        5. If a token is rejected, resample from the adjusted distribution.
        6. Repeat until max_tokens reached.

    Both models must share the same vocabulary. The draft model should be smaller
    (e.g., 2-layer vs 12-layer) for speedup.

    Args:
        model: main GPT model (larger, more accurate)
        draft_model: draft GPT model (smaller, faster)
        tokens: list of int -- prompt token ids
        max_tokens: maximum number of tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: top-k filtering
        seed: random seed
        draft_steps: number of tokens the draft model proposes per round

    Returns:
        list of int -- prompt + generated tokens
    """
    assert model.config.vocab_size == draft_model.config.vocab_size, \
        "Main and draft models must share the same vocabulary size"

    sharding = _get_replicated_sharding()
    key = jax.random.key(seed)
    vocab_size = model.config.vocab_size
    total_len = len(tokens) + max_tokens + draft_steps  # extra room for draft overshoot

    # Initialize KV caches for both models
    main_k, main_v, main_prev = _init_kv_cache(model, total_len)
    draft_k, draft_v, draft_prev = _init_kv_cache(draft_model, total_len)

    # Prefill both models with the prompt
    main_logits, main_k, main_v, main_prev, main_pos = _prefill(
        model, tokens, main_k, main_v, main_prev
    )
    draft_logits, draft_k, draft_v, draft_prev, draft_pos = _prefill(
        draft_model, tokens, draft_k, draft_v, draft_prev
    )

    generated = list(tokens)
    num_generated = 0

    def _apply_top_k(logits_2d, k):
        """Apply top-k filtering to logits (1, V) -> filtered logits (1, V)."""
        if k is not None and k > 0:
            tkl, tki = jax.lax.top_k(logits_2d, min(k, logits_2d.shape[-1]))
            return jnp.full_like(logits_2d, -1e9).at[0, tki[0]].set(tkl[0])
        return logits_2d

    def _get_probs(logits_2d, temp):
        """Convert logits (1, V) to probability distribution (V,)."""
        filtered = _apply_top_k(logits_2d, top_k)
        if temp > 0:
            return jax.nn.softmax(filtered[0] / temp, axis=-1)
        else:
            # Greedy: one-hot at argmax
            idx = jnp.argmax(filtered[0])
            return jnp.zeros(vocab_size).at[idx].set(1.0)

    def _sample_from_probs(probs, rng):
        """Sample a token from a probability distribution (V,)."""
        rng, sk = jax.random.split(rng)
        token = jax.random.categorical(sk, jnp.log(jnp.maximum(probs, 1e-10)))
        return int(token), rng

    while num_generated < max_tokens:
        # --- Phase 1: Draft model generates draft_steps tokens ---
        draft_tokens = []
        draft_probs_list = []
        steps_this_round = min(draft_steps, max_tokens - num_generated)

        cur_draft_logits = draft_logits

        for _ in range(steps_this_round):
            d_probs = _get_probs(cur_draft_logits, temperature)
            draft_probs_list.append(d_probs)

            if temperature > 0:
                token, key = _sample_from_probs(d_probs, key)
            else:
                token = int(jnp.argmax(d_probs))

            draft_tokens.append(token)

            # Advance draft model by one step
            tok = _to_device(jnp.array([[token]], dtype=jnp.int32), sharding)
            cur_draft_logits, draft_k, draft_v, draft_prev = _single_step_forward(
                draft_model, tok, draft_pos, draft_k, draft_v, draft_prev
            )
            draft_pos = draft_pos + 1

        # --- Phase 2: Main model verifies all draft tokens ---
        main_probs_list = []
        verify_logits = main_logits

        for i, dt in enumerate(draft_tokens):
            m_probs = _get_probs(verify_logits, temperature)
            main_probs_list.append(m_probs)

            # Advance main model
            tok = _to_device(jnp.array([[dt]], dtype=jnp.int32), sharding)
            verify_logits, main_k, main_v, main_prev = _single_step_forward(
                model, tok, main_pos, main_k, main_v, main_prev
            )
            main_pos = main_pos + 1

        # --- Phase 3: Accept/reject draft tokens ---
        accepted = 0
        for i in range(len(draft_tokens)):
            d_probs = draft_probs_list[i]
            m_probs = main_probs_list[i]
            dt = draft_tokens[i]

            if temperature == 0:
                # Greedy: accept if argmax matches
                if int(jnp.argmax(m_probs)) == dt:
                    generated.append(dt)
                    accepted += 1
                else:
                    # Reject: use main model's choice
                    corrected = int(jnp.argmax(m_probs))
                    generated.append(corrected)
                    accepted += 1
                    break
            else:
                # Stochastic: acceptance probability = min(1, p_main / p_draft)
                p_main = float(m_probs[dt])
                p_draft = float(d_probs[dt])

                if p_draft == 0:
                    corrected, key = _sample_from_probs(m_probs, key)
                    generated.append(corrected)
                    accepted += 1
                    break

                accept_prob = min(1.0, p_main / p_draft)
                key, accept_key = jax.random.split(key)
                u = float(jax.random.uniform(accept_key))

                if u < accept_prob:
                    generated.append(dt)
                    accepted += 1
                else:
                    # Resample from adjusted distribution: max(0, p_main - p_draft)
                    adjusted = jnp.maximum(m_probs - d_probs, 0.0)
                    adj_sum = jnp.sum(adjusted)
                    adjusted = jnp.where(adj_sum > 0, adjusted / adj_sum, m_probs)
                    corrected, key = _sample_from_probs(adjusted, key)
                    generated.append(corrected)
                    accepted += 1
                    break

        num_generated += accepted

        if accepted == len(draft_tokens):
            # All accepted: sample bonus token from main model's final logits
            bonus_probs = _get_probs(verify_logits, temperature)
            if temperature > 0:
                bonus_token, key = _sample_from_probs(bonus_probs, key)
            else:
                bonus_token = int(jnp.argmax(bonus_probs))
            generated.append(bonus_token)
            num_generated += 1

            # Advance both models with the bonus token
            tok = _to_device(jnp.array([[bonus_token]], dtype=jnp.int32), sharding)
            main_logits, main_k, main_v, main_prev = _single_step_forward(
                model, tok, main_pos, main_k, main_v, main_prev
            )
            main_pos = main_pos + 1
            draft_logits, draft_k, draft_v, draft_prev = _single_step_forward(
                draft_model, tok, draft_pos, draft_k, draft_v, draft_prev
            )
            draft_pos = draft_pos + 1
        else:
            # Rejection occurred: resync draft model cache from scratch.
            # The draft model is small so this is fast.
            all_tokens_so_far = generated[:]
            draft_k, draft_v, draft_prev = _init_kv_cache(draft_model, total_len)
            draft_logits, draft_k, draft_v, draft_prev, draft_pos = _prefill(
                draft_model, all_tokens_so_far, draft_k, draft_v, draft_prev
            )

            # Resync main model cache too (corrected token may differ from draft)
            main_k, main_v, main_prev = _init_kv_cache(model, total_len)
            main_logits, main_k, main_v, main_prev, main_pos = _prefill(
                model, all_tokens_so_far, main_k, main_v, main_prev
            )

    return generated[:len(tokens) + max_tokens]


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------
def use_calculator(expr):
    """Evaluate a safe math expression or string.count(). Returns result or None."""
    import ast
    expr = expr.replace(",", "")

    # Pure math: only digits and operators
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:
            return None
        try:
            tree = ast.parse(expr, mode='eval')
            # Only allow number literals and binary ops
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                         ast.Constant, ast.Add, ast.Sub, ast.Mult,
                                         ast.Div, ast.FloorDiv, ast.Mod, ast.USub)):
                    return None
            return eval(compile(tree, '<calc>', 'eval'), {"__builtins__": {}}, {})
        except Exception:
            return None

    # String.count(): only allow 'literal'.count('literal')
    import re
    match = re.fullmatch(r"""['"]([\w\s]+)['"]\s*\.\s*count\s*\(\s*['"](\w+)['"]\s*\)""", expr)
    if match:
        string, substr = match.group(1), match.group(2)
        return string.count(substr)

    return None


# ---------------------------------------------------------------------------
# Per-row state for streaming generation with tool use
# ---------------------------------------------------------------------------
class RowState:
    """Per-sample state tracking during streaming generation."""

    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # Current token sequence for this row
        self.forced_tokens = deque()  # Queue of tokens to force inject
        self.in_python_block = False  # Whether we are inside a python block
        self.python_expr_tokens = []  # Tokens of the current python expression
        self.completed = False  # Whether this row has completed generation


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _sample_next_token(self, logits, key, temperature=1.0, top_k=None):
        """Sample next token from logits (B, vocab_size). Returns ((B,) token ids, key)."""
        assert temperature >= 0.0, "temperature must be non-negative"
        if temperature == 0.0:
            return jnp.argmax(logits, axis=-1), key
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.shape[-1])
            vals, idx = jax.lax.top_k(logits, k)
            vals = vals / temperature
            # Sample from top-k using categorical
            key, sk = jax.random.split(key)
            choice = jax.random.categorical(sk, vals, axis=-1)  # (B,)
            # Gather the actual token ids
            return jnp.take_along_axis(idx, choice[:, None], axis=-1)[:, 0], key
        else:
            key, sk = jax.random.split(key)
            token_ids = jax.random.categorical(sk, logits / temperature, axis=-1)
            return token_ids, key

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0,
                 top_k=40, seed=42):
        """
        Streaming generator that yields (token_column, token_masks) per step.

        Ports nanochat's Engine.generate() pattern to JAX:
        - Single prefill, then clone KV cache state for each sample
        - Tool use state machine: <|python_start|>...<|python_end|> triggers
          calculator eval, force-injects <|output_start|>result<|output_end|>
        - Stops on <|assistant_end|> or <|bos|>

        Args:
            tokens: list of int — prompt token ids (already includes BOS etc.)
            num_samples: number of parallel samples to generate
            max_tokens: maximum number of tokens to generate (None = model max)
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering (None = no filtering)
            seed: random seed

        Yields:
            (token_column, token_masks) where:
                token_column: list of int, length num_samples — next token per row
                token_masks: list of int, length num_samples — 1=sampled, 0=forced
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        model = self.model
        config = model.config
        sharding = _get_replicated_sharding()
        key = jax.random.key(seed)

        # Resolve special token ids for the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # KV cache dimensions
        n_layer = config.n_layer
        n_kv_head = config.n_kv_head
        head_dim = config.n_embd // config.n_head
        total_len = len(tokens) + (max_tokens if max_tokens is not None else 256)

        # 1) Prefill with batch=1
        cache_shape = (n_layer, 1, total_len, n_kv_head, head_dim)
        k_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
        v_cache = _to_device(jnp.zeros(cache_shape, dtype=COMPUTE_DTYPE), sharding)
        prev_emb = _to_device(jnp.zeros((1, 1, config.n_embd), dtype=COMPUTE_DTYPE), sharding)

        pos = jnp.int32(0)
        for t in range(len(tokens)):
            tok = _to_device(jnp.array([[tokens[t]]], dtype=jnp.int32), sharding)
            logits, k_cache, v_cache, prev_emb = _single_step_forward(
                model, tok, pos, k_cache, v_cache, prev_emb
            )
            pos = pos + 1

        # 2) Replicate KV cache state for num_samples
        # For num_samples > 1, tile the cache along the batch dimension
        # For num_samples == 1, keep as-is (batch dim stays 1)
        if num_samples > 1:
            # Expand caches: (n_layer, 1, T, H, D) -> (n_layer, B, T, H, D)
            k_caches = [jnp.copy(k_cache) for _ in range(num_samples)]
            v_caches = [jnp.copy(v_cache) for _ in range(num_samples)]
            prev_embs = [jnp.copy(prev_emb) for _ in range(num_samples)]
            positions = [jnp.copy(pos) for _ in range(num_samples)]
        else:
            k_caches = [k_cache]
            v_caches = [v_cache]
            prev_embs = [prev_emb]
            positions = [pos]

        # Expand logits to (num_samples, vocab_size)
        logits = jnp.tile(logits, (num_samples, 1))  # (num_samples, vocab_size)

        # 3) Initialize per-row states
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            # Sample next tokens for all rows
            key, sample_key = jax.random.split(key)
            sampled_tokens, key = self._sample_next_token(logits, key, temperature, top_k)
            sampled_list = [int(sampled_tokens[i]) for i in range(num_samples)]

            # Process each row: choose next token, update state, handle tool use
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                # Select next token: forced or sampled
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_list[i]
                token_column.append(next_token)

                # Update row state
                state.current_tokens.append(next_token)

                # Stop conditions
                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                # Tool use state machine
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is None:
                            # Calculator can't handle it — try sandboxed execution
                            exec_result = execute_code(expr)
                            if exec_result.success:
                                result = exec_result.stdout.rstrip("\n")
                            else:
                                result = "Error"
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column and masks
            yield token_column, token_masks
            num_generated += 1

            # Forward each row through its own KV cache to get next logits
            next_logits = []
            for i in range(num_samples):
                tok = _to_device(
                    jnp.array([[token_column[i]]], dtype=jnp.int32), sharding
                )
                row_logits, k_caches[i], v_caches[i], prev_embs[i] = _single_step_forward(
                    model, tok, positions[i], k_caches[i], v_caches[i], prev_embs[i]
                )
                positions[i] = positions[i] + 1
                next_logits.append(row_logits)

            logits = jnp.concatenate(next_logits, axis=0)  # (num_samples, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that returns final token sequences with masks.

        Terminal tokens (assistant_end, bos) are not included in results.

        Returns:
            (results, masks) where:
                results: list of list of int — token sequences per sample
                masks: list of list of int — 0=forced/prompt, 1=sampled
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks

    def generate_text(self, prompt, max_tokens=256, temperature=1.0, top_k=40,
                      seed=42):
        """Generate text from a prompt string using streaming generation with tool use."""
        tokens = self.tokenizer(prompt, prepend="<|bos|>")
        results, masks = self.generate_batch(
            tokens, num_samples=1, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
        )
        return self.tokenizer.decode(results[0])

    def generate_fast(self, tokens, max_tokens=256, temperature=1.0, top_k=40,
                      seed=42, eos_token=0):
        """Fully JIT'd generation using jax.lax.while_loop. Fastest mode.

        Delegates to the module-level generate_fast() function.
        Does NOT support tool use or streaming — use Engine.generate() for that.

        Args:
            tokens: list of int — prompt token ids
            max_tokens: maximum tokens to generate after the prompt
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering
            seed: random seed
            eos_token: stop generation when this token is produced

        Returns:
            list of int — prompt + generated tokens
        """
        return generate_fast(
            self.model, tokens, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
            eos_token=eos_token,
        )

    def generate_speculative(self, draft_model, tokens, max_tokens=256,
                             temperature=1.0, top_k=40, seed=42, draft_steps=4):
        """Speculative decoding via Engine interface. Delegates to generate_speculative()."""
        return generate_speculative(
            self.model, draft_model, tokens,
            max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, seed=seed, draft_steps=draft_steps,
        )
