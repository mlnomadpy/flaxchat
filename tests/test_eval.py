"""
Tests for evaluation utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from flaxchat.eval import forward_model, find_common_length


class TestForwardModel:
    def test_output_shapes(self, tiny_model, tiny_config):
        B, T = 2, tiny_config.sequence_len
        input_ids = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        losses, predictions = forward_model(tiny_model, input_ids)
        assert losses.shape == (B, T)
        assert predictions.shape == (B, T)

    def test_last_position_nan(self, tiny_model, tiny_config):
        B, T = 1, tiny_config.sequence_len
        input_ids = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        losses, _ = forward_model(tiny_model, input_ids)
        assert jnp.isnan(losses[0, -1])

    def test_losses_positive(self, tiny_model, tiny_config):
        B, T = 1, tiny_config.sequence_len
        input_ids = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        losses, _ = forward_model(tiny_model, input_ids)
        # All losses except last should be positive
        assert jnp.all(losses[0, :-1] > 0)


class TestFindCommonLength:
    def test_common_prefix(self):
        seqs = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 7, 8]]
        assert find_common_length(seqs, 'left') == 2

    def test_no_common_prefix(self):
        seqs = [[1, 2], [3, 4]]
        assert find_common_length(seqs, 'left') == 0

    def test_common_suffix(self):
        seqs = [[1, 2, 3], [4, 2, 3], [5, 2, 3]]
        assert find_common_length(seqs, 'right') == 2

    def test_identical_sequences(self):
        seqs = [[1, 2, 3], [1, 2, 3]]
        assert find_common_length(seqs, 'left') == 3


class TestRenderMC:
    def test_basic(self):
        from tasks.common import render_mc
        q = "What is 2+2?"
        letters = ('A', 'B', 'C', 'D')
        choices = ['3', '4', '5', '6']
        rendered = render_mc(q, letters, choices)
        assert "What is 2+2?" in rendered
        assert "=A" in rendered
        assert "=B" in rendered
        assert "4=B" in rendered
