"""
Tests for the GPT model.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from flaxchat.gpt import (
    GPT, GPTConfig, rms_norm, precompute_rotary_embeddings,
    apply_rotary_emb, has_ve, CausalSelfAttention, MLP, Block,
)


class TestRMSNorm:
    def test_output_shape(self):
        x = jnp.ones((2, 8, 64))
        y = rms_norm(x)
        assert y.shape == x.shape

    def test_normalization(self):
        key = jax.random.key(0)
        x = jax.random.normal(key, (2, 8, 64))
        y = rms_norm(x)
        # RMS of output should be approximately 1
        rms = jnp.sqrt(jnp.mean(y * y, axis=-1))
        assert jnp.allclose(rms, 1.0, atol=1e-3)

    def test_zero_input(self):
        x = jnp.zeros((2, 8, 64))
        y = rms_norm(x)
        assert jnp.all(jnp.isfinite(y))


class TestRotaryEmbeddings:
    def test_shape(self):
        cos, sin = precompute_rotary_embeddings(128, 64)
        assert cos.shape == (1, 128, 1, 32)
        assert sin.shape == (1, 128, 1, 32)

    def test_apply_shape(self):
        cos, sin = precompute_rotary_embeddings(16, 64)
        x = jnp.ones((2, 16, 4, 64))
        y = apply_rotary_emb(x, cos, sin)
        assert y.shape == x.shape

    def test_deterministic(self):
        cos, sin = precompute_rotary_embeddings(16, 64)
        x = jax.random.normal(jax.random.key(0), (2, 16, 4, 64))
        y1 = apply_rotary_emb(x, cos, sin)
        y2 = apply_rotary_emb(x, cos, sin)
        assert jnp.allclose(y1, y2)


class TestHasVE:
    def test_alternating(self):
        # has_ve(i, n) = i%2 == (n-1)%2
        # n_layer=4: (n-1)%2=1, so VE on odd layers: 1, 3
        assert has_ve(0, 4) is False
        assert has_ve(1, 4) is True
        assert has_ve(2, 4) is False
        assert has_ve(3, 4) is True
        # n_layer=3: (n-1)%2=0, so VE on even layers: 0, 2
        assert has_ve(0, 3) is True
        assert has_ve(1, 3) is False
        assert has_ve(2, 3) is True

    def test_last_layer_included(self):
        # Last layer should always be included
        for n_layer in [2, 3, 4, 6, 8, 12]:
            assert has_ve(n_layer - 1, n_layer) is True


class TestMLP:
    def test_forward_shape(self, tiny_config):
        mlp = MLP(tiny_config, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8, tiny_config.n_embd))
        y = mlp(x)
        assert y.shape == x.shape

    def test_relu_squared(self, tiny_config):
        """Verify ReLU^2 activation is used (negative inputs -> zero)."""
        mlp = MLP(tiny_config, rngs=nnx.Rngs(0))
        # With zero-init c_proj, output should be zero regardless of input
        mlp.c_proj.kernel[...] = jnp.zeros_like(mlp.c_proj.kernel[...])
        x = jax.random.normal(jax.random.key(0), (2, 8, tiny_config.n_embd))
        y = mlp(x)
        assert jnp.allclose(y, 0.0, atol=1e-6)


class TestCausalSelfAttention:
    def test_forward_shape(self, tiny_config):
        attn = CausalSelfAttention(tiny_config, layer_idx=0, rngs=nnx.Rngs(0))
        B, T = 2, 16
        x = jax.random.normal(jax.random.key(0), (B, T, tiny_config.n_embd))
        cos, sin = precompute_rotary_embeddings(T, tiny_config.n_embd // tiny_config.n_head)
        y = attn(x, ve=None, cos=cos, sin=sin, window_size=(T, 0))
        assert y.shape == (B, T, tiny_config.n_embd)

    def test_causal_masking(self, tiny_config):
        """First token output should only depend on first token input."""
        attn = CausalSelfAttention(tiny_config, layer_idx=0, rngs=nnx.Rngs(0))
        B, T = 1, 8
        cos, sin = precompute_rotary_embeddings(T, tiny_config.n_embd // tiny_config.n_head)

        key = jax.random.key(0)
        x = jax.random.normal(key, (B, T, tiny_config.n_embd))

        y1 = attn(x, ve=None, cos=cos, sin=sin, window_size=(T, 0))

        # Modify tokens after position 0
        x_mod = x.at[:, 1:, :].set(jax.random.normal(jax.random.key(1), (B, T - 1, tiny_config.n_embd)))
        y2 = attn(x_mod, ve=None, cos=cos, sin=sin, window_size=(T, 0))

        # First position output should be the same
        assert jnp.allclose(y1[:, 0, :], y2[:, 0, :], atol=1e-5)


class TestBlock:
    def test_forward_shape(self, tiny_config):
        block = Block(tiny_config, layer_idx=0, rngs=nnx.Rngs(0))
        B, T = 2, 16
        x = jax.random.normal(jax.random.key(0), (B, T, tiny_config.n_embd))
        cos, sin = precompute_rotary_embeddings(T, tiny_config.n_embd // tiny_config.n_head)
        y = block(x, ve=None, cos=cos, sin=sin, window_size=(T, 0))
        assert y.shape == x.shape


class TestGPT:
    def test_construction(self, tiny_config):
        model = GPT(tiny_config, rngs=nnx.Rngs(0))
        assert model.config == tiny_config

    def test_forward_logits(self, tiny_model, tiny_config):
        B, T = 2, tiny_config.sequence_len
        idx = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        logits = tiny_model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert logits.dtype == jnp.float32  # logits should be fp32

    def test_forward_loss(self, tiny_model, random_batch):
        inputs, targets = random_batch
        loss = tiny_model(inputs, targets)
        assert loss.shape == ()
        assert loss.dtype == jnp.float32
        assert jnp.isfinite(loss)
        # Loss should be roughly log(vocab_size) for random init
        assert loss > 0

    def test_forward_shakespeare(self, tiny_model, shakespeare_batch):
        inputs, targets = shakespeare_batch
        loss = tiny_model(inputs, targets)
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_num_params(self, tiny_model):
        n = tiny_model.num_params()
        assert n > 0
        assert isinstance(n, int)

    def test_estimate_flops(self, tiny_model):
        flops = tiny_model.estimate_flops()
        assert flops > 0

    def test_window_sizes(self, tiny_config):
        model = GPT(tiny_config, rngs=nnx.Rngs(0))
        assert len(model.window_sizes) == tiny_config.n_layer
        # Last layer should always be full context
        assert model.window_sizes[-1][0] == tiny_config.sequence_len

    def test_value_embeddings_alternating(self, tiny_config):
        model = GPT(tiny_config, rngs=nnx.Rngs(0))
        # Check that VE layers follow the alternating pattern (string keys)
        for i in range(tiny_config.n_layer):
            if has_ve(i, tiny_config.n_layer):
                assert str(i) in model.value_embeds
            else:
                assert str(i) not in model.value_embeds

    def test_ignore_index(self, tiny_model, tiny_config):
        """Test that targets with -1 are ignored in loss."""
        B, T = 2, tiny_config.sequence_len
        inputs = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        targets = jnp.full((B, T), -1, dtype=jnp.int32)
        loss = tiny_model(inputs, targets)
        # With all targets masked, loss should be 0
        assert jnp.allclose(loss, 0.0, atol=1e-6)

    def test_softcap(self, tiny_model, tiny_config):
        """Logits should be bounded by softcap=15."""
        B, T = 2, tiny_config.sequence_len
        idx = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        logits = tiny_model(idx)
        assert jnp.all(logits <= 15.0)
        assert jnp.all(logits >= -15.0)

    def test_jit_compatible(self, tiny_model, random_batch):
        """Model should work under jax.jit."""
        inputs, targets = random_batch

        @jax.jit
        def forward(idx, tgt):
            return tiny_model(idx, tgt)

        loss = forward(inputs, targets)
        assert jnp.isfinite(loss)

    def test_grad_computable(self, tiny_model, random_batch):
        """Should be able to compute gradients."""
        inputs, targets = random_batch

        def loss_fn(model):
            return model(inputs, targets)

        loss, grads = nnx.value_and_grad(loss_fn)(tiny_model)
        assert jnp.isfinite(loss)
        # Check at least one grad is non-zero
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))
        assert has_nonzero


class TestGPTSmall:
    """Tests with the slightly larger model config."""

    def test_forward(self, small_model, small_config):
        B, T = 2, small_config.sequence_len
        idx = jax.random.randint(jax.random.key(0), (B, T), 0, small_config.vocab_size)
        logits = small_model(idx)
        assert logits.shape == (B, T, small_config.vocab_size)

    def test_loss(self, small_model, small_config):
        B, T = 2, small_config.sequence_len
        idx = jax.random.randint(jax.random.key(0), (B, T), 0, small_config.vocab_size)
        targets = jax.random.randint(jax.random.key(1), (B, T), 0, small_config.vocab_size)
        loss = small_model(idx, targets)
        assert jnp.isfinite(loss)
