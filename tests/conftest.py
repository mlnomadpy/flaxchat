"""
Shared test fixtures for flaxchat tests.
Uses Shakespeare text for local testing.
"""

import os
import urllib.request
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.gpt import GPT


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "shakespeare.txt")


@pytest.fixture(scope="session")
def shakespeare_text():
    """Download and cache tiny Shakespeare dataset."""
    os.makedirs(os.path.dirname(SHAKESPEARE_CACHE), exist_ok=True)
    if not os.path.exists(SHAKESPEARE_CACHE):
        print(f"Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_CACHE)
    with open(SHAKESPEARE_CACHE, "r") as f:
        return f.read()


@pytest.fixture
def tiny_config():
    """Minimal model config for fast tests."""
    return GPTConfig(
        sequence_len=64,
        vocab_size=256,  # byte-level for simplicity
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="SL",
    )


@pytest.fixture
def small_config():
    """Small but realistic config for integration tests."""
    return GPTConfig(
        sequence_len=128,
        vocab_size=512,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="SSSL",
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Tiny GPT model for unit tests."""
    return GPT(tiny_config, rngs=nnx.Rngs(0))


@pytest.fixture
def small_model(small_config):
    """Small GPT model for integration tests."""
    return GPT(small_config, rngs=nnx.Rngs(0))


@pytest.fixture
def random_batch(tiny_config):
    """Random batch of token IDs for the tiny model."""
    key = jax.random.key(42)
    B, T = 2, tiny_config.sequence_len
    inputs = jax.random.randint(key, (B, T), 0, tiny_config.vocab_size)
    targets = jax.random.randint(jax.random.key(43), (B, T), 0, tiny_config.vocab_size)
    return inputs, targets


@pytest.fixture
def shakespeare_batch(shakespeare_text, tiny_config):
    """Batch from Shakespeare text encoded as bytes."""
    text_bytes = shakespeare_text.encode("utf-8")
    B, T = 2, tiny_config.sequence_len
    total_tokens = B * (T + 1)
    tokens = list(text_bytes[:total_tokens])
    # Pad if needed
    while len(tokens) < total_tokens:
        tokens.append(0)
    import numpy as np
    arr = jnp.array(tokens[:total_tokens]).reshape(B, T + 1)
    inputs = arr[:, :-1]
    targets = arr[:, 1:]
    return inputs, targets
