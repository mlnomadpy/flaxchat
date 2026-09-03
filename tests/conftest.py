"""
Shared test fixtures for flaxchat tests.
Uses Shakespeare text for local testing.
"""

from pathlib import Path
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.gpt import GPT


CORPUS_FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"


@pytest.fixture(scope="session")
def shakespeare_text():
    """Return a committed corpus so tests never depend on the network."""
    return CORPUS_FIXTURE.read_text(encoding="utf-8")


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
